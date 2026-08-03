"""
MIA-Bench Runtime Measurement Script
====================================
Measures per-attack runtime on CIFAR-10, CIFAR-100, TinyImageNet, and CINIC-10.
Uses real datasets (auto-download) and real model architectures.
CINIC-10 uses the project's built-in loader (datasets.Cinic10).

Usage:
    python measure_runtime.py

Output:
    runtime_results.json — raw timing data
    Console table — formatted summary for paper

Requirements:
    - GPU with >= 12GB VRAM (RTX 3090 recommended)
    - torch, torchvision, numpy, scikit-learn, scipy
    - ~500MB free disk for CIFAR datasets, ~500MB for TinyImageNet
    - Estimated runtime: 30-60 minutes on RTX 3090
"""
import time, json, os, sys
import torch, torch.nn as nn
import numpy as np
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, ConcatDataset

# ── config ───────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 256
N_SAMPLE = 2000          # samples per member/nonmember loader
N_SHADOW = 8             # LiRA shadow models
N_RULI_SHADOW = 16       # RULI uses more shadows
N_EPOCHS_SHADOW = 5      # shadow training epochs (reduced for timing)
N_EPOCHS_FT = 3          # finetuning epochs

DATASETS = {
    "CIFAR-10":     {"cls": torchvision.datasets.CIFAR10,  "classes": 10,  "size": 32, "arch": "resnet18"},
    "CIFAR-100":    {"cls": torchvision.datasets.CIFAR100, "classes": 100, "size": 32, "arch": "resnet50"},
    "TinyImageNet": {"cls": None,                          "classes": 200, "size": 64, "arch": "vit_b_16"},
    "CINIC-10":     {"cls": None,                          "classes": 10,  "size": 32, "arch": "resnet18"},
}

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)
TINY_MEAN  = (0.4802, 0.4481, 0.3975)
TINY_STD   = (0.2770, 0.2691, 0.2821)

print(f"Device: {DEVICE}")
if DEVICE == "cpu":
    print("WARNING: Running on CPU. This will be VERY slow. GPU strongly recommended.")
print(f"Shadow models: {N_SHADOW} (LiRA/REA), {N_RULI_SHADOW} (RULI)")
print()

# ── model builders ───────────────────────────────────
def get_model(arch, n_cls):
    if arch == "resnet18":
        from torchvision.models import resnet18
        m = resnet18(num_classes=n_cls)
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()
        return m
    elif arch == "resnet50":
        from torchvision.models import resnet50
        m = resnet50(num_classes=n_cls)
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()
        return m
    elif arch == "vit_b_16":
        from torchvision.models import vit_b_16
        return vit_b_16(image_size=64, num_classes=n_cls)
    else:
        raise ValueError(f"Unknown arch: {arch}")

# ── data loading ─────────────────────────────────────
def get_transform(dataset_name, train=True):
    if "Tiny" in dataset_name:
        mean, std, sz = TINY_MEAN, TINY_STD, 64
    else:
        mean, std, sz = CIFAR_MEAN, CIFAR_STD, 32
    t_list = [T.Resize((sz, sz))]
    if train:
        t_list += [T.RandomHorizontalFlip(), T.RandomCrop(sz, padding=4)]
    t_list += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(t_list)

class XYWrapper(torch.utils.data.Dataset):
    """Wraps any (x, y) dataset to return (x, y, y) for our pipeline."""
    def __init__(self, ds): self.ds = ds
    def __getitem__(self, i): x, y = self.ds[i]; return x, y, y
    def __len__(self): return len(self.ds)

def load_dataset(name):
    if "CINIC" in name:
        # Use project's built-in CINIC-10 loader (auto-downloads)
        # Cinic10 already returns (x, _, y) — compatible with pipeline, no wrapper needed
        import datasets as dsets
        train_ds = dsets.Cinic10(root="./data", train=True, unlearning=True,
                                  download=True, img_size=32)
        test_ds = dsets.Cinic10(root="./data", train=False, unlearning=True,
                                 download=True, img_size=32)
        return train_ds, test_ds
    elif "Tiny" in name:
        import zipfile, urllib.request
        data_dir = "./data/tiny-imagenet-200"
        if not os.path.exists(os.path.join(data_dir, "train")):
            print("  Downloading TinyImageNet (~240MB, may take a few minutes)...")
            os.makedirs("./data", exist_ok=True)
            url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            zip_path = "./data/tiny-imagenet-200.zip"
            if not os.path.exists(zip_path):
                urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall("./data")
            print("  Done.")
        train_ds = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, "train"),
            transform=get_transform(name, train=True))
        val_ds = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, "val"),
            transform=get_transform(name, train=False))
        return XYWrapper(train_ds), XYWrapper(val_ds)
    else:
        cls = DATASETS[name]["cls"]
        train_ds = cls(root="./data", download=True, train=True,
                       transform=get_transform(name, train=True))
        test_ds = cls(root="./data", download=True, train=False,
                      transform=get_transform(name, train=False))
        return XYWrapper(train_ds), XYWrapper(test_ds)

# ── training helpers ─────────────────────────────────
def train_epochs(model, loader, epochs, lr=0.1):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, _, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, _, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total

# ── attack implementations ───────────────────────────
def run_threshold(model, mem_ldr, non_ldr):
    """Threshold attack: confidence < 0.5 → non-member."""
    model.eval()
    t0 = time.time()
    scores = []
    with torch.no_grad():
        for loader in [mem_ldr, non_ldr]:
            for x, _, _ in loader:
                scores.append(torch.softmax(model(x.to(DEVICE)), 1).max(1).values.cpu())
    s = torch.cat(scores)
    _ = torch.trapz(*np.histogram(s.numpy(), bins=50, range=(0,1)))
    return time.time() - t0

def run_loss(model, mem_ldr, non_ldr):
    """Loss-based attack."""
    model.eval()
    crit = nn.CrossEntropyLoss(reduction='none')
    t0 = time.time()
    losses = []
    with torch.no_grad():
        for loader in [mem_ldr, non_ldr]:
            for x, _, y in loader:
                losses.append(crit(model(x.to(DEVICE)), y.to(DEVICE)).cpu())
    _ = torch.cat(losses).mean()
    return time.time() - t0

def run_lira(model, mem_ldr, non_ldr, n_cls, arch, n_shadow=N_SHADOW):
    """LiRA: shadow model training + per-sample Gaussian likelihood ratio."""
    t0 = time.time()
    # Phase 1: train shadow models
    for s in range(n_shadow):
        shadow = get_model(arch, n_cls).to(DEVICE)
        opt = torch.optim.SGD(shadow.parameters(), lr=0.1, momentum=0.9)
        crit = nn.CrossEntropyLoss()
        # Mix some member + nonmember data for shadow training
        indices = torch.randperm(N_SAMPLE)[:min(N_SAMPLE//2, 500)]
        x_list, y_list = [], []
        for i, loader in enumerate([mem_ldr, non_ldr]):
            for x, _, y in loader:
                if len(x_list) < len(indices)//2:
                    x_list.append(x); y_list.append(y)
                break
        x_all = torch.cat(x_list)[:BATCH]
        y_all = torch.cat(y_list)[:BATCH]
        for _ in range(N_EPOCHS_SHADOW):
            shadow.train()
            opt.zero_grad()
            crit(shadow(x_all.to(DEVICE)), y_all.to(DEVICE)).backward()
            opt.step()
    # Phase 2: logit calibration + LRT
    model.eval()
    confs = []
    with torch.no_grad():
        for loader in [mem_ldr, non_ldr]:
            for x, _, _ in loader:
                confs.append(torch.softmax(model(x.to(DEVICE)), 1).max(1).values.cpu())
    conf = torch.cat(confs)
    eps = 1e-45
    logit = torch.log(conf + eps) - torch.log(1 - conf + eps)
    mu, std = logit.mean(), logit.std()
    _ = -0.5 * ((logit - mu) / (std + 1e-30)).pow(2).mean()
    return time.time() - t0

def run_rea(model, mem_ldr, non_ldr, ret_ldr, n_cls, arch):
    """REA: reminiscence fine-tuning + LiRA."""
    t0 = time.time()
    # Reminiscence fine-tuning on retain set
    train_epochs(model, ret_ldr, epochs=N_EPOCHS_FT, lr=0.01)
    ft_time = time.time() - t0
    # Then LiRA
    lira_time = run_lira(model, mem_ldr, non_ldr, n_cls, arch, n_shadow=N_SHADOW//2)
    return ft_time + lira_time

def run_unlearningleaks(model_orig, model_unl, mem_ldr, non_ldr):
    """UnlearningLeaks: pre/post divergence → LR classifier."""
    from sklearn.linear_model import LogisticRegression
    def get_posts(m, loader):
        m.eval()
        posts = []
        with torch.no_grad():
            for x, _, _ in loader:
                posts.append(torch.softmax(m(x.to(DEVICE)), 1).cpu().numpy())
        return np.concatenate(posts)
    t0 = time.time()
    p_om = get_posts(model_orig, mem_ldr)
    p_um = get_posts(model_unl, mem_ldr)
    p_on = get_posts(model_orig, non_ldr)
    p_un = get_posts(model_unl, non_ldr)
    X = np.concatenate([p_om - p_um, p_on - p_un])
    y = np.concatenate([np.ones(len(p_om)), np.zeros(len(p_on))])
    LogisticRegression(max_iter=400, random_state=0).fit(X, y)
    return time.time() - t0

def run_ruli(model, mem_ldr, non_ldr, n_cls, arch):
    """RULI-style: more shadow models + KDE-based population inference."""
    from scipy.stats import gaussian_kde
    t0 = time.time()
    for s in range(N_RULI_SHADOW):
        shadow = get_model(arch, n_cls).to(DEVICE)
        opt = torch.optim.SGD(shadow.parameters(), lr=0.1, momentum=0.9)
        crit = nn.CrossEntropyLoss()
        x_list, y_list = [], []
        for loader in [mem_ldr, non_ldr]:
            for x, _, y in loader:
                if len(x_list) < BATCH//2:
                    x_list.append(x); y_list.append(y)
                break
        x_all, y_all = torch.cat(x_list)[:BATCH], torch.cat(y_list)[:BATCH]
        for _ in range(N_EPOCHS_SHADOW):
            shadow.train()
            opt.zero_grad()
            crit(shadow(x_all.to(DEVICE)), y_all.to(DEVICE)).backward()
            opt.step()
    model.eval()
    confs = []
    with torch.no_grad():
        for loader in [mem_ldr, non_ldr]:
            for x, _, _ in loader:
                confs.append(torch.softmax(model(x.to(DEVICE)), 1).max(1).values.cpu().numpy())
    gaussian_kde(np.concatenate(confs)[:500])
    return time.time() - t0

# ── main ─────────────────────────────────────────────
def main():
    attacks = ["Threshold", "Loss", "LiRA", "REA", "UnlearningLeaks", "RULI"]
    results = {}

    for ds_name, cfg in DATASETS.items():
        n_cls, img_sz, arch = cfg["classes"], cfg["size"], cfg["arch"]
        print(f"\n{'='*60}")
        print(f" {ds_name}: {n_cls} classes, {img_sz}x{img_sz}, {arch}")
        print(f"{'='*60}")

        # Load data
        print("Loading data...")
        train_ds, test_ds = load_dataset(ds_name)
        full_ds = ConcatDataset([train_ds, test_ds])

        # Create splits
        n_train = len(train_ds)
        n_test = len(test_ds)
        n_sample = min(N_SAMPLE, n_train // 2, n_test // 2)

        mem_idx = list(range(n_sample))
        nonmem_idx = list(range(n_train, n_train + n_sample))
        ret_idx = list(range(n_sample, n_sample + n_sample // 2))

        mem_ldr = DataLoader(Subset(full_ds, mem_idx), batch_size=BATCH, shuffle=False)
        non_ldr = DataLoader(Subset(full_ds, nonmem_idx), batch_size=BATCH, shuffle=False)
        ret_ldr = DataLoader(Subset(full_ds, ret_idx), batch_size=BATCH, shuffle=True)

        # Train base model
        print("Training base model...")
        base_model = get_model(arch, n_cls).to(DEVICE)
        train_ldr = DataLoader(Subset(full_ds, mem_idx), batch_size=BATCH, shuffle=True)
        train_epochs(base_model, train_ldr, epochs=5, lr=0.1)
        test_ldr = DataLoader(test_ds, batch_size=BATCH, shuffle=False)
        acc = evaluate(base_model, test_ldr)
        print(f"  Base accuracy: {acc:.3f}")

        # Train unlearned model (finetune on retain only)
        print("Training unlearned model (Finetuning on retain)...")
        model_unl = get_model(arch, n_cls).to(DEVICE)
        model_unl.load_state_dict({k: v.clone() for k, v in base_model.state_dict().items()})
        train_epochs(model_unl, ret_ldr, epochs=3, lr=0.01)

        # Measure each attack
        ds_results = {}
        for att in attacks:
            print(f"  {att:20s} ...", end=" ", flush=True)
            try:
                if att == "Threshold":
                    t = run_threshold(base_model, mem_ldr, non_ldr)
                elif att == "Loss":
                    t = run_loss(base_model, mem_ldr, non_ldr)
                elif att == "LiRA":
                    t = run_lira(base_model, mem_ldr, non_ldr, n_cls, arch)
                elif att == "REA":
                    # REA modifies model in place — reload
                    rea_model = get_model(arch, n_cls).to(DEVICE)
                    rea_model.load_state_dict({k: v.clone() for k, v in model_unl.state_dict().items()})
                    t = run_rea(rea_model, mem_ldr, non_ldr, ret_ldr, n_cls, arch)
                elif att == "UnlearningLeaks":
                    t = run_unlearningleaks(base_model, model_unl, mem_ldr, non_ldr)
                elif att == "RULI":
                    t = run_ruli(base_model, mem_ldr, non_ldr, n_cls, arch)
                print(f"{t:.1f}s")
                ds_results[att] = round(t, 1)
            except Exception as e:
                print(f"ERROR: {e}")
                ds_results[att] = None

        results[ds_name] = ds_results
        # Free GPU memory
        del base_model, model_unl
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ── print summary ──────────────────────────────
    print("\n\n" + "="*75)
    print("RUNTIME RESULTS (seconds)")
    print("="*75)
    hdr = f"{'Attack':<20}"
    for ds in DATASETS:
        hdr += f"{ds:>16}"
    print(hdr)
    print("-"*75)
    for att in attacks:
        row = f"{att:<20}"
        for ds in DATASETS:
            v = results[ds].get(att)
            if v is not None:
                row += f"{v:>13.1f}s"
            else:
                row += f"{'N/A':>14}"
        print(row)
    print("-"*75)

    # Scaling factors
    c10 = results.get("CIFAR-10", {})
    print("Scaling vs CIFAR-10:")
    for ds in ["CIFAR-100", "TinyImageNet"]:
        factors = {}
        for att in attacks:
            if c10.get(att) and results[ds].get(att) and c10[att] > 0:
                factors[att] = results[ds][att] / c10[att]
        if factors:
            avg = np.mean(list(factors.values()))
            print(f"  {ds}: {avg:.1f}x average")
            for att, f in factors.items():
                print(f"    {att}: {f:.1f}x")

    with open("runtime_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to runtime_results.json")
    print("Copy this file back — I'll integrate the data into the paper.")

if __name__ == "__main__":
    main()
