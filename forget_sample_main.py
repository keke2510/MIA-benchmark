"""
Sample-wise unlearning entrypoint.
This file is kept as the main executable for random/sample-wise forgetting.
"""

import random
import os
# import wandb
# import optuna
from typing import Tuple, List
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, dataset, Subset
import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms

# import models
from unlearn import *
from utils import *
import forget_random_strategies
import datasets
import models
import config
from utils.training_utils import *
import os.path as osp
from pathlib import Path

from benchmark.run_layout import (
    forget_index_path,
    method_key as layout_method_key,
    task_dir as layout_task_dir,
    validate_run_tag,
)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
"""
Get Args
"""
parser = argparse.ArgumentParser()
parser.add_argument("-net", type=str, required=True, help="net type")
parser.add_argument(
    "-weight_path",
    type=str,
    required=True,
    help="Path to model weights. If you need to train a new model use pretrain_model.py",
)
parser.add_argument(
    "-dataset",
    type=str,
    required=True,
    nargs="?",
    choices=["Cifar10", "Cifar20", "Cifar100", "Cinic10", "TinyImageNet", "PinsFaceRecognition"],
    help="dataset to train on",
)
parser.add_argument("-classes", type=int, required=True, help="number of classes")
parser.add_argument("-gpu", action="store_true", default=True, help="use gpu or not")
parser.add_argument("-b", type=int, default=256, help="batch size for dataloader")
parser.add_argument("-warm", type=int, default=1, help="warm up training phase")
parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
parser.add_argument(
    "-method",
    type=str,
    required=True,
    nargs="?",
    # choices=[
    #     "baseline",
    #     "retrain",
    #     "finetune",
    #     "blindspot",
    #     "amnesiac",
    #     "negative_grad",
    #     "FisherForgetting",
    #     'Wfisher',
    #     'FT_prune',
    #     "ssd_tuning"
    # ],
    help="select unlearning method from choice set",
) #not to use: "UNSIR", "ssd_tuning"

#TODO "Percentage of trainset to forget"
parser.add_argument(
    "-forget_perc", type=float, default=0.1, help="Percentage of trainset to forget"
)
parser.add_argument(
    "-epochs", type=int, default=1, help="number of epochs of unlearning method to use"
)
parser.add_argument("-seed", type=int, default=0, help="seed for runs")
parser.add_argument(
    "--run-tag",
    type=str,
    default="",
    help="非空时把遗忘模型写入隔离目录（含 run tag/forget_perc/seed），避免新实验覆盖旧结果。",
)

#TODO the para of unlearning method
parser.add_argument("--para1", type=str, default=0)
parser.add_argument("--para2", type=str, default=0)
parser.add_argument("--opt_strategy", type=str, default=None)
parser.add_argument("--mask_path", type=str, default=None)
parser.add_argument("--lambda_align", type=float, default=None)
parser.add_argument("--lambda_stat", type=float, default=None)

args = parser.parse_args()
args.run_tag = validate_run_tag(args.run_tag)


FORGET_INDEX_SEED = 42


def load_or_create_forget_index(index_path: Path, dataset_size: int, forget_perc: float) -> np.ndarray:
    """Load the shared forget index, creating it atomically if absent.

    Multiple workers may start the same (dataset, forget_perc) combination at
    the same time.  Without a lock they would race: each would draw a different
    forget set while writing the same file.  The first process to create the
    lock file generates the index; the others wait and then load it.

    Generation uses a dedicated RNG seeded with ``FORGET_INDEX_SEED`` rather
    than the caller's evaluation seed.  The forget set is part of the benchmark
    definition and is shared by all seeds (see 补实验量清单.md), so it must not
    depend on which worker happens to win the race.  A local generator also
    keeps the caller's global RNG stream identical for winners and losers.
    """
    index_path = Path(index_path)
    if index_path.exists():
        return np.load(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = index_path.with_name(index_path.name + ".lock")
    deadline = time.time() + 1800
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if index_path.exists():
                return np.load(index_path)
            if time.time() > deadline:
                raise TimeoutError(
                    f"Timeout waiting for another worker to create {index_path}"
                )
            # Generating the index is instantaneous, so an old lock without a
            # result means the owner crashed.  Reclaim it instead of hanging.
            try:
                if time.time() - lock_path.stat().st_mtime > 300:
                    print(f"[forget-index] removing stale lock {lock_path}")
                    os.remove(lock_path)
                    continue
            except FileNotFoundError:
                pass
            time.sleep(1)
            continue
        try:
            # Re-check under the lock: another process may have finished
            # between our first check and acquiring the lock.
            if index_path.exists():
                return np.load(index_path)
            rng = np.random.default_rng(FORGET_INDEX_SEED)
            forget_set_idxs = np.sort(
                rng.choice(
                    np.arange(dataset_size),
                    size=int(dataset_size * forget_perc),
                    replace=False,
                )
            )
            tmp_path = index_path.with_name(f"{index_path.name}.tmp.{os.getpid()}.npy")
            np.save(tmp_path, forget_set_idxs)
            os.replace(tmp_path, index_path)
            print(
                f"[forget-index] created {index_path} "
                f"(seed={FORGET_INDEX_SEED}, perc={forget_perc:g}, n={len(forget_set_idxs)})"
            )
            return forget_set_idxs
        finally:
            os.close(fd)
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass


def resolve_img_size(dataset_name):
    if dataset_name == "TinyImageNet":
        return 64
    return 32



def set_seeds(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

# Set seeds
if __name__ == '__main__':
    set_seeds(args)

    batch_size = args.b

    # get network
    net = getattr(models, args.net)(num_classes=args.classes)
    net.load_state_dict(torch.load(args.weight_path))

    # if torch.cuda.device_count() > 1:
    #     print(f"Let's use {torch.cuda.device_count()} GPUs!")
    #     net = nn.DataParallel(net)

    unlearning_teacher = getattr(models, args.net)(num_classes=args.classes)

    if args.gpu and torch.cuda.is_available():
        net = net.cuda()
        unlearning_teacher = unlearning_teacher.cuda()

    root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"
    img_size = resolve_img_size(args.dataset)

    if args.method == 'retrain':
        trainset = getattr(datasets, args.dataset)(root=root, download=True, train=True,
                                                   unlearning=False,
                                                   img_size=img_size)  # TODO if unlearning is true, the data aug is different
    else:
        trainset = getattr(datasets, args.dataset)(root=root, download=True, train=True,
                                                   unlearning=True,
                                                   img_size=img_size)  # TODO if unlearning is true, the data aug is different

    validset = getattr(datasets, args.dataset)(root=root, download=True, train=False, unlearning=True,
                                               img_size=img_size) #unlearning is set to True

    trainloader = DataLoader(trainset, num_workers=0, batch_size=batch_size, shuffle=True)  # num_workers=4
    validloader = DataLoader(validset, num_workers=0, batch_size=batch_size, shuffle=False)  # num_workers=4
    reference_nonmember_dataloader = DataLoader(validset, num_workers=0, batch_size=batch_size, shuffle=True)

    experiment_name = f"{args.net}-{args.dataset}-{args.classes}"
    forget_set_idxs = load_or_create_forget_index(
        forget_index_path(config.CHECKPOINT_PATH, experiment_name, args.forget_perc),
        len(trainset),
        args.forget_perc,
    )

    # forget_set_idxs = np.load(index_set_path_folder + f'/forgetting_dataset_index_{args.forget_perc}.npy')

    remaining_set_idxs = list(set(np.arange(len(trainset))) - set(forget_set_idxs))
    forget_train_set = Subset(trainset, forget_set_idxs)
    retain_train_set = Subset(trainset, remaining_set_idxs)

    aug_trainset = getattr(datasets, args.dataset)(root=root, download=True, train=True,
                                                   unlearning=False,
                                                   img_size=img_size)
    aug_retain_dataloader = DataLoader(Subset(aug_trainset, remaining_set_idxs), batch_size=batch_size, shuffle=True)

    # forget_train, retain_train = torch.utils.data.random_split(trainset, [5000, len(trainset)-5000])
    forget_train_dataloader = DataLoader(list(forget_train_set), batch_size=batch_size, shuffle=True)
    retain_train_dataloader = DataLoader(list(retain_train_set), batch_size=batch_size, shuffle=True)
    full_train_dataloader = DataLoader(
        ConcatDataset((retain_train_dataloader.dataset, forget_train_dataloader.dataset)), batch_size=batch_size)
    valid_poisonedloader = validloader

    if args.method == "rum":
        npz_path = "Cifar10_curvature.npz"
        forget_memorization = construct_forget_memorization(forget_set_idxs, npz_path)

    ###中毒数据训练
    # args.poisoning_rate = 0.1
    # args.trigger_path = "./triggers/trigger_white.png"
    # args.trigger_size = 5
    # args.trigger_label = 1
    # args.dataset = "Cifar10Poison"

    checkpoint_path_folder = str(
        layout_task_dir(
            config.CHECKPOINT_PATH,
            experiment_name,
            "unlearning",
            layout_method_key(args.method, args.para1, args.para2),
            run_tag=args.run_tag,
            forget_perc=args.forget_perc,
            seed=args.seed,
        )
    )

    print("#####", checkpoint_path_folder)
    if not os.path.exists(checkpoint_path_folder):
        os.makedirs(checkpoint_path_folder)
    checkpoint_path = os.path.join(checkpoint_path_folder, "{epoch}-{type}.pth")
    weights_path = checkpoint_path.format(epoch=args.epochs, type="last")

    model_size_scaler = 1
    if args.net == "ViT":
        model_size_scaler = 1
    else:
        model_size_scaler = 1

    kwargs = {
        "model": net,
        "unlearning_teacher": unlearning_teacher,
        "retain_train_dataloader": retain_train_dataloader, #trainloader,
        "forget_train_dataloader": forget_train_dataloader,
        "full_train_dataloader": full_train_dataloader,
        "valid_dataloader": validloader,
        "num_classes": args.classes,
        "dataset_name": args.dataset,
        "device": "cuda:0" if args.gpu else "cpu",
        "weights_path": weights_path,
        "model_name": args.net,
        "para1": args.para1,
        "para2": args.para2,
        "mask_path": args.mask_path,
        "reference_nonmember_dataloader": reference_nonmember_dataloader,
        "lambda_align": args.lambda_align,
        "lambda_stat": args.lambda_stat,
    }
    if args.method == "rum":
        kwargs.update({"forget_dataset": list(forget_train_set),
                       "forget_memorization": forget_memorization})
    if ("orthogonality"  in args.method) or ("our"  in args.method):
        kwargs.update({'aug_retain_dataloader': aug_retain_dataloader})
    #used in ssd tuning
    # "dampening_constant": 1,
    # "selection_weighting": 10 * model_size_scaler,  # used in ssd_tuning

    import time

    start = time.time()

    clean_acc, forgetting_acc, remaining_acc, zrf, mia = getattr(forget_random_strategies, args.method)(
        **kwargs)

    end = time.time()
    time_elapsed = end - start

    # torch.save(net.state_dict(), weights_path)

    logname = osp.join(checkpoint_path_folder, 'log_{}-{}-{}.tsv'.format(args.net, args.dataset, args.classes))
    with open(logname, 'w+') as f:
        columns = ['clean_acc',
                   'forgetting_acc',
                   'remaining_acc',
                   'zrf',
                   'mia',
                   'time'
                   ]
        f.write('\t'.join(columns) + '\n')

    print("d_t = ", clean_acc, "| d_f = ", forgetting_acc, "| d_r = ", remaining_acc,
          "| zrf = ",
          zrf, "| mia = ", mia, "| time = ", time_elapsed)

    with open(logname, 'a') as f:
        columns = [f"{clean_acc}",
                   f"{forgetting_acc}",
                   f"{remaining_acc}",
                   f"{zrf}",
                   f"{mia}",
                   f"{time_elapsed}"
                   ]
        f.write('\t'.join(columns) + '\n')
