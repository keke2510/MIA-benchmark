from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch

import config
from attacks.base import AttackContext, AttackResult
from attacks.registry import register_attack

@register_attack("unlearningleaks")
class UnlearningLeaksAttack:
    supported_datasets = {"Cifar10", "Cifar100", "Cinic10", "TinyImageNet"}
    unsupported_methods = {"salun", "sfron", "rum"}
    shadow_defaults = {
        "Cifar10": {"opt": "sgd", "lr": "1e-1", "bs": "128", "pkeep": "0.83"},
        "Cifar20": {"opt": "adam", "lr": "3e-4", "bs": "128", "pkeep": "0.83"},
        "Cifar100": {"opt": "sgd", "lr": "1e-1", "bs": "128", "pkeep": "0.83"},
        "Cinic10": {"opt": "sgd", "lr": "1e-1", "bs": "128", "pkeep": "0.83"},
        "TinyImageNet": {"opt": "adam", "lr": "3e-4", "bs": "64", "pkeep": "0.83"},
    }
    feature_methods = {
        "direct_diff",
        "sorted_diff",
        "direct_concat",
        "sorted_concat",
        "l2_distance",
        "basic_mia",
    }
    attack_model_choices = {"lr", "dt", "rf", "mlp"}

    @staticmethod
    def repo_root() -> Path:
        return Path(__file__).resolve().parent.parent

    @classmethod
    def result_dir(cls, context: AttackContext) -> Path:
        path = Path(context.attack_result_dir("unlearningleaks")).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def log_dir(cls, context: AttackContext) -> Path:
        path = cls.result_dir(context) / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def shadow_asset_dir(cls, context: AttackContext) -> Path:
        path = cls.result_dir(context) / "shadow_unlearned"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def resolve_feature_method(cls, context: AttackContext) -> str:
        value = str(context.attack_options.get("unlearningleaks_feature", "direct_diff"))
        if value not in cls.feature_methods:
            raise ValueError(f"UnlearningLeaks 不支持 feature `{value}`。")
        return value

    @classmethod
    def resolve_attack_model(cls, context: AttackContext) -> str:
        value = str(context.attack_options.get("unlearningleaks_attack_model", "lr")).lower()
        if value not in cls.attack_model_choices:
            raise ValueError(f"UnlearningLeaks 不支持 attack model `{value}`。")
        return value

    @classmethod
    def shadow_num(cls, context: AttackContext) -> int:
        return int(context.attack_options.get("unlearningleaks_num_shadow", context.num_shadow))

    @classmethod
    def sample_budget(cls, context: AttackContext) -> int:
        return int(context.attack_options.get("unlearningleaks_sample_budget", 500))

    @classmethod
    def target_unlearned_path(cls, context: AttackContext) -> Path:
        return Path(context.checkpoint_dir("unlearning")) / "1-last.pth"

    @classmethod
    def forget_index_path(cls, context: AttackContext) -> Path:
        return (
            Path(config.CHECKPOINT_PATH)
            / "forget_random_main"
            / f"{context.net}-{context.dataset}-{context.classes}"
            / "random_index_set"
            / f"forgetting_dataset_index_{context.forget_perc}.npy"
        )

    @classmethod
    def shadow_original_path(cls, context: AttackContext, shadow_id: int) -> Path:
        return (
            cls.repo_root()
            / "saved_models"
            / context.experiment_name
            / f"{context.model_save_name}_shadow_{shadow_id}_last.pth"
        )

    @classmethod
    def shadow_unlearned_path(cls, context: AttackContext, shadow_id: int) -> Path:
        return cls.shadow_asset_dir(context) / f"shadow_{shadow_id}_{context.machine_unlearning}.pth"

    @classmethod
    def build_shadow_train_command(cls, context: AttackContext, shadow_id: int) -> list[str]:
        defaults = cls.shadow_defaults.get(context.dataset, cls.shadow_defaults["Cifar10"])
        opt = context.attack_options.get("unlearningleaks_shadow_opt") or defaults["opt"]
        bs = context.attack_options.get("unlearningleaks_shadow_bs") or defaults["bs"]
        lr = context.attack_options.get("unlearningleaks_shadow_lr") or defaults["lr"]
        pkeep = context.attack_options.get("unlearningleaks_shadow_pkeep") or defaults["pkeep"]
        return [
            sys.executable,
            str(cls.repo_root() / "train.py"),
            "--name",
            context.experiment_name,
            "--dataset",
            context.dataset,
            "--save_name",
            context.model_save_name,
            "--classes",
            str(context.classes),
            "--net",
            context.net,
            "--opt",
            str(opt),
            "--bs",
            str(bs),
            "--lr",
            str(lr),
            "--pkeep",
            str(pkeep),
            "--num_shadow",
            str(cls.shadow_num(context)),
            "--shadow_id",
            str(shadow_id),
            "--seed",
            str(context.seed),
        ]

    @classmethod
    def run_command(cls, cmd: list[str], *, log_path: Path, verbose: bool, dry_run: bool) -> None:
        print(f"[attack:unlearningleaks] {' '.join(cmd)}")
        if dry_run:
            return
        print(f"[attack:unlearningleaks] 日志文件: {log_path}")
        try:
            if verbose:
                subprocess.run(cmd, cwd=cls.repo_root(), check=True)
            else:
                with open(log_path, "w", encoding="utf-8") as log_file:
                    subprocess.run(
                        cmd,
                        cwd=cls.repo_root(),
                        check=True,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
        except subprocess.CalledProcessError as exc:
            tail = ""
            if log_path.exists():
                lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                tail = "\n".join(lines[-40:])
            raise RuntimeError(
                "UnlearningLeaks 子流程执行失败。"
                + (f"\n\n最近日志:\n{tail}" if tail else "")
            ) from exc

    @classmethod
    def ensure_shadow_originals(cls, context: AttackContext, *, dry_run: bool, verbose: bool) -> None:
        for shadow_id in range(cls.shadow_num(context)):
            path = cls.shadow_original_path(context, shadow_id)
            if path.exists():
                continue
            print(f"[attack:unlearningleaks] 缺少原始 shadow model {shadow_id}，将自动训练。")
            cls.run_command(
                cls.build_shadow_train_command(context, shadow_id),
                log_path=cls.log_dir(context) / f"shadow_train_{shadow_id}.log",
                verbose=verbose,
                dry_run=dry_run,
            )

    @classmethod
    def load_plain_model(cls, context: AttackContext, path: Path, *, checkpoint_dict: bool = False):
        import torch
        import models

        model = getattr(models, context.net)(num_classes=context.classes)
        state = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint_dict:
            state = state["model"]
        model.load_state_dict(state)
        model.eval()
        return model

    @classmethod
    def load_shadow_original(cls, context: AttackContext, shadow_id: int):
        import numpy as np
        import torch
        import models

        path = cls.shadow_original_path(context, shadow_id)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = getattr(models, context.net)(num_classes=context.classes)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model, np.asarray(checkpoint["in_data"], dtype=int)

    @classmethod
    def build_full_datasets(cls, context: AttackContext):
        import datasets
        from torch.utils.data import ConcatDataset

        root = "./data"
        img_size = 64 if context.dataset == "TinyImageNet" else 32
        dataset_cls = getattr(datasets, context.dataset)
        train_det = dataset_cls(root=root, download=True, train=True, unlearning=True, img_size=img_size)
        test_det = dataset_cls(root=root, download=True, train=False, unlearning=True, img_size=img_size)
        full_det = ConcatDataset([train_det, test_det])

        train_aug = dataset_cls(root=root, download=True, train=True, unlearning=False, img_size=img_size)
        test_aug = dataset_cls(
            root=root,
            download=True,
            train=False,
            unlearning=False,
            img_size=img_size,
            data_augmentation=True,
        )
        full_aug = ConcatDataset([train_aug, test_aug])
        return full_det, full_aug, test_det

    @classmethod
    def build_shadow_unlearned_model(cls, context: AttackContext, shadow_id: int):
        import numpy as np
        import torch
        import models
        import forget_random_strategies
        from torch.utils.data import DataLoader, ConcatDataset, Subset

        if context.machine_unlearning in cls.unsupported_methods:
            raise ValueError(
                f"UnlearningLeaks 暂不支持 `{context.machine_unlearning}`，"
                "因为这类方法依赖额外 mask/memorization 资产。"
            )

        weights_path = cls.shadow_unlearned_path(context, shadow_id)
        if weights_path.exists():
            model = getattr(models, context.net)(num_classes=context.classes)
            model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=False))
            model.eval()
            return model

        full_det, full_aug, test_det = cls.build_full_datasets(context)
        model, shadow_in_data = cls.load_shadow_original(context, shadow_id)
        forget_indices = np.load(cls.forget_index_path(context))
        shadow_forget = np.intersect1d(shadow_in_data, forget_indices, assume_unique=False)
        shadow_retain = np.setdiff1d(shadow_in_data, shadow_forget, assume_unique=False)

        if shadow_forget.size == 0:
            torch.save(model.state_dict(), weights_path)
            model.eval()
            return model

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        teacher = deepcopy(model).to(device)
        teacher.eval()

        batch_size = int(context.attack_options.get("unlearningleaks_unlearn_bs", 256))
        retain_loader = DataLoader(Subset(full_det, shadow_retain.tolist()), batch_size=batch_size, shuffle=True)
        forget_loader = DataLoader(Subset(full_det, shadow_forget.tolist()), batch_size=batch_size, shuffle=True)
        full_loader = DataLoader(ConcatDataset((retain_loader.dataset, forget_loader.dataset)), batch_size=batch_size)
        valid_loader = DataLoader(test_det, batch_size=batch_size, shuffle=False)
        aug_retain_loader = DataLoader(Subset(full_aug, shadow_retain.tolist()), batch_size=batch_size, shuffle=True)

        kwargs = {
            "model": model,
            "unlearning_teacher": teacher,
            "retain_train_dataloader": retain_loader,
            "forget_train_dataloader": forget_loader,
            "full_train_dataloader": full_loader,
            "valid_dataloader": valid_loader,
            "reference_nonmember_dataloader": valid_loader,
            "num_classes": context.classes,
            "dataset_name": context.dataset,
            "device": device,
            "weights_path": str(weights_path),
            "model_name": context.net,
            "para1": context.para1,
            "para2": context.para2,
            "mask_path": context.attack_options.get("unlearningleaks_mask_path"),
            "aug_retain_dataloader": aug_retain_loader,
        }
        getattr(forget_random_strategies, context.machine_unlearning)(**kwargs)

        model = getattr(models, context.net)(num_classes=context.classes)
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=False))
        model.eval()
        return model

    @staticmethod
    def softmax_logits(logits: torch.Tensor) -> np.ndarray:
        import numpy as np

        return torch.softmax(logits, dim=1).detach().cpu().numpy()

    @classmethod
    def predict_posteriors(
        cls,
        context: AttackContext,
        model,
        dataset,
        indices: np.ndarray,
        batch_size: int = 256,
    ) -> np.ndarray:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, Subset

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=False)
        outputs = []
        with torch.no_grad():
            for images, _, _ in loader:
                outputs.append(cls.softmax_logits(model(images.to(device))))
        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, context.classes), dtype=np.float32)

    @classmethod
    def construct_features(cls, original: np.ndarray, unlearned: np.ndarray, method: str) -> np.ndarray:
        import numpy as np

        if method == "direct_diff":
            return original - unlearned
        if method == "sorted_diff":
            order = np.argsort(original, axis=1)
            row = np.arange(original.shape[0])[:, None]
            original_sorted = original[row, order]
            unlearned_sorted = unlearned[row, order]
            return original_sorted - unlearned_sorted
        if method == "direct_concat":
            return np.concatenate([original, unlearned], axis=1)
        if method == "sorted_concat":
            order = np.argsort(original, axis=1)
            row = np.arange(original.shape[0])[:, None]
            original_sorted = original[row, order]
            unlearned_sorted = unlearned[row, order]
            return np.concatenate([original_sorted, unlearned_sorted], axis=1)
        if method == "l2_distance":
            return np.linalg.norm(original - unlearned, axis=1, keepdims=True)
        if method == "basic_mia":
            return original
        raise ValueError(f"invalid feature construction method: {method}")

    @classmethod
    def make_attack_model(cls, model_name: str):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.tree import DecisionTreeClassifier

        if model_name == "lr":
            return LogisticRegression(random_state=0, solver="lbfgs", max_iter=400, n_jobs=1)
        if model_name == "dt":
            return DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
        if model_name == "rf":
            return RandomForestClassifier(random_state=0, n_estimators=500, min_samples_leaf=30)
        if model_name == "mlp":
            return MLPClassifier(early_stopping=True, learning_rate_init=0.01)
        raise ValueError(f"invalid attack name: {model_name}")

    @classmethod
    def sample_indices(cls, candidates: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
        if len(candidates) <= budget:
            return np.asarray(candidates, dtype=int)
        return np.asarray(rng.choice(candidates, size=budget, replace=False), dtype=int)

    @classmethod
    def build_shadow_attack_dataset(cls, context: AttackContext):
        import numpy as np

        full_det, _, test_det = cls.build_full_datasets(context)
        feature_method = cls.resolve_feature_method(context)
        batch_size = int(context.attack_options.get("unlearningleaks_eval_bs", 256))
        rng = np.random.default_rng(int(context.attack_options.get("unlearningleaks_seed", context.seed)))
        forget_indices = np.load(cls.forget_index_path(context))
        test_offset = len(full_det) - len(test_det)
        test_indices = np.arange(test_offset, test_offset + len(test_det), dtype=int)

        train_x, train_y = [], []
        shadow_stats = []
        for shadow_id in range(cls.shadow_num(context)):
            original_model, shadow_in_data = cls.load_shadow_original(context, shadow_id)
            unlearned_model = cls.build_shadow_unlearned_model(context, shadow_id)

            pos_candidates = np.intersect1d(shadow_in_data, forget_indices, assume_unique=False)
            neg_candidates = np.setdiff1d(test_indices, shadow_in_data, assume_unique=False)
            pos_idx = cls.sample_indices(pos_candidates, cls.sample_budget(context), rng)
            neg_idx = cls.sample_indices(neg_candidates, cls.sample_budget(context), rng)
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue

            pos_orig = cls.predict_posteriors(context, original_model, full_det, pos_idx, batch_size=batch_size)
            pos_unl = cls.predict_posteriors(context, unlearned_model, full_det, pos_idx, batch_size=batch_size)
            neg_orig = cls.predict_posteriors(context, original_model, full_det, neg_idx, batch_size=batch_size)
            neg_unl = cls.predict_posteriors(context, unlearned_model, full_det, neg_idx, batch_size=batch_size)

            pos_feat = cls.construct_features(pos_orig, pos_unl, feature_method)
            neg_feat = cls.construct_features(neg_orig, neg_unl, feature_method)
            train_x.append(np.concatenate([pos_feat, neg_feat], axis=0))
            train_y.append(np.concatenate([np.ones(len(pos_feat)), np.zeros(len(neg_feat))], axis=0))
            shadow_stats.append({"shadow_id": shadow_id, "pos": int(len(pos_idx)), "neg": int(len(neg_idx))})

        if not train_x:
            raise RuntimeError("UnlearningLeaks 没有构造出任何 shadow attack 样本。")
        return np.concatenate(train_x, axis=0), np.concatenate(train_y, axis=0), shadow_stats

    @classmethod
    def build_target_attack_dataset(cls, context: AttackContext):
        import numpy as np

        full_det, _, test_det = cls.build_full_datasets(context)
        feature_method = cls.resolve_feature_method(context)
        batch_size = int(context.attack_options.get("unlearningleaks_eval_bs", 256))
        rng = np.random.default_rng(int(context.attack_options.get("unlearningleaks_seed", context.seed)))
        forget_indices = np.load(cls.forget_index_path(context))
        test_offset = len(full_det) - len(test_det)
        test_indices = np.arange(test_offset, test_offset + len(test_det), dtype=int)

        budget = int(context.attack_options.get("unlearningleaks_target_budget", cls.sample_budget(context)))
        pos_idx = cls.sample_indices(forget_indices, budget, rng)
        neg_idx = cls.sample_indices(test_indices, budget, rng)

        original_model = cls.load_plain_model(context, Path(context.weight_path), checkpoint_dict=False)
        unlearned_model = cls.load_plain_model(context, cls.target_unlearned_path(context), checkpoint_dict=False)

        pos_orig = cls.predict_posteriors(context, original_model, full_det, pos_idx, batch_size=batch_size)
        pos_unl = cls.predict_posteriors(context, unlearned_model, full_det, pos_idx, batch_size=batch_size)
        neg_orig = cls.predict_posteriors(context, original_model, full_det, neg_idx, batch_size=batch_size)
        neg_unl = cls.predict_posteriors(context, unlearned_model, full_det, neg_idx, batch_size=batch_size)

        pos_feat = cls.construct_features(pos_orig, pos_unl, feature_method)
        neg_feat = cls.construct_features(neg_orig, neg_unl, feature_method)
        test_x = np.concatenate([pos_feat, neg_feat], axis=0)
        test_y = np.concatenate([np.ones(len(pos_feat)), np.zeros(len(neg_feat))], axis=0)
        return test_x, test_y, {"pos": int(len(pos_idx)), "neg": int(len(neg_idx))}

    @classmethod
    def run_attack(cls, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, model_name: str):
        from sklearn.preprocessing import StandardScaler

        from evaluation.effectiveness import summarize_binary_metrics

        classifier = cls.make_attack_model(model_name)
        scaler = None
        if model_name in {"lr", "mlp"}:
            scaler = StandardScaler().fit(train_x)
            train_x_fit = scaler.transform(train_x)
            test_x_fit = scaler.transform(test_x)
        else:
            train_x_fit, test_x_fit = train_x, test_x
        classifier.fit(train_x_fit, train_y)
        scores = classifier.predict_proba(test_x_fit)[:, 1]
        metrics = summarize_binary_metrics(test_y, scores)
        train_scores = classifier.predict_proba(train_x_fit)[:, 1]
        train_metrics = summarize_binary_metrics(train_y, train_scores)
        return classifier, metrics, train_metrics, scores

    @classmethod
    def run(cls, context: AttackContext, dry_run: bool = False) -> AttackResult | None:
        if context.dataset not in cls.supported_datasets:
            raise ValueError(f"UnlearningLeaks 当前只接入了 {sorted(cls.supported_datasets)}。")
        verbose = bool(context.attack_options.get("attack_verbose", False))
        cls.ensure_shadow_originals(context, dry_run=dry_run, verbose=verbose)
        if dry_run:
            return None

        import numpy as np

        from evaluation.cost import build_cost_report
        from evaluation.realism import build_protocol_checklist

        start = time.time()
        train_x, train_y, shadow_stats = cls.build_shadow_attack_dataset(context)
        test_x, test_y, target_stats = cls.build_target_attack_dataset(context)
        model_name = cls.resolve_attack_model(context)
        classifier, test_metrics, train_metrics, test_scores = cls.run_attack(train_x, train_y, test_x, test_y, model_name)
        runtime_sec = time.time() - start

        result_dir = cls.result_dir(context)
        feature_method = cls.resolve_feature_method(context)
        raw_npz = result_dir / "unlearningleaks_arrays.npz"
        np.savez(
            raw_npz,
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            test_scores=test_scores,
        )

        attack_summary = {
            "feature_method": feature_method,
            "attack_model": model_name,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "shadow_stats": shadow_stats,
            "target_stats": target_stats,
        }

        result = AttackResult(
            attack_name="unlearningleaks",
            metrics=test_metrics,
            cost=build_cost_report(
                num_shadow=cls.shadow_num(context),
                attack_sample_number=int(len(test_y)),
                runtime_sec=runtime_sec,
                storage_bytes=raw_npz.stat().st_size if raw_npz.exists() else None,
            ),
            protocol=build_protocol_checklist(
                target_checkpoint=str(cls.target_unlearned_path(context)),
                forget_perc=context.forget_perc,
                shadow_models=cls.shadow_num(context),
                shared_forget_index=True,
                same_target_checkpoint_for_all_attacks=True,
                same_shadow_setup_for_all_attacks=False,
                dataset_split_name="benchmark_forget_index_plus_shadow_membership",
                same_dataset_split_for_all_attacks=False,
                uses_original_unlearning_pipeline=False,
            ),
            artifacts={
                "arrays_npz": str(raw_npz),
            },
            raw=attack_summary,
        )
        summary_json = result_dir / "attack_result.json"
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(result.as_dict(), f, indent=2, ensure_ascii=False)
        result.artifacts["benchmark_summary_json"] = str(summary_json)
        return result
