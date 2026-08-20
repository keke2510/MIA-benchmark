from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import config
from attacks.base import AttackContext, AttackResult
from attacks.registry import register_attack
from evaluation.cost import build_cost_report
from evaluation.effectiveness import summarize_binary_metrics
from evaluation.realism import build_protocol_checklist


@register_attack("threshold")
class ThresholdAttack:
    """Threshold 黑盒基线攻击（Shokri et al., 2017）。

    针对遗忘模型：遗忘集成员由于遗忘不彻底而残留更高的预测置信度，
    测试集非成员从未被训练、置信度偏低。membership score = 正确类置信度。
    """

    @classmethod
    def _unlearned_path(cls, context: AttackContext) -> Path:
        return Path(context.checkpoint_dir("unlearning")) / "1-last.pth"

    @classmethod
    def _forget_index_path(cls, context: AttackContext) -> Path:
        return (
            Path(config.CHECKPOINT_PATH)
            / "forget_random_main"
            / f"{context.net}-{context.dataset}-{context.classes}"
            / "random_index_set"
            / f"forgetting_dataset_index_{context.forget_perc}.npy"
        )

    @classmethod
    def _budget(cls, context: AttackContext) -> int:
        return int(context.attack_options.get("threshold_sample_budget", 500))

    @classmethod
    def _load_model(cls, context: AttackContext):
        import models

        model = getattr(models, context.net)(num_classes=context.classes)
        state = torch.load(cls._unlearned_path(context), map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        return model

    @classmethod
    def _load_datasets(cls, context: AttackContext):
        import datasets

        root = "./data"
        img_size = 64 if context.dataset == "TinyImageNet" else 32
        dataset_cls = getattr(datasets, context.dataset)
        train_det = dataset_cls(root=root, download=True, train=True, unlearning=True, img_size=img_size)
        test_det = dataset_cls(root=root, download=True, train=False, unlearning=True, img_size=img_size)
        return train_det, test_det

    @classmethod
    def run(cls, context: AttackContext, dry_run: bool = False) -> AttackResult | None:
        if dry_run:
            print(f"[attack:threshold] dry-run {context.experiment_name} seed {context.seed}")
            return None

        start = time.time()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = cls._load_model(context).to(device)
        model.eval()

        train_det, test_det = cls._load_datasets(context)
        forget_indices = np.load(cls._forget_index_path(context))
        budget = cls._budget(context)

        n_member = int(min(budget, len(forget_indices)))
        n_nonmember = int(min(budget, len(test_det)))
        rng = np.random.default_rng(context.seed)
        member_idx = rng.choice(forget_indices, n_member, replace=False)
        nonmember_idx = rng.choice(len(test_det), n_nonmember, replace=False)

        labels: list[int] = []
        scores: list[float] = []
        for is_member, indices, src in ((True, member_idx, train_det), (False, nonmember_idx, test_det)):
            loader = DataLoader(Subset(src, indices.tolist()), batch_size=256, shuffle=False, num_workers=0)
            for images, _, targets in loader:
                images = images.to(device)
                targets = targets.to(device)
                with torch.no_grad():
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)
                    conf = probs.gather(1, targets.view(-1, 1)).squeeze(1)
                scores.extend(conf.detach().cpu().numpy().tolist())
                labels.extend([1 if is_member else 0] * int(images.size(0)))

        metrics = summarize_binary_metrics(np.asarray(labels), np.asarray(scores))
        metrics["runtime_sec"] = round(time.time() - start, 4)

        result = AttackResult(
            attack_name="threshold",
            metrics=metrics,
            cost=build_cost_report(
                num_shadow=None,
                num_aug=None,
                attack_sample_number=int(n_member + n_nonmember),
                runtime_sec=metrics["runtime_sec"],
            ),
            protocol=build_protocol_checklist(
                target_checkpoint=str(cls._unlearned_path(context)),
                forget_perc=context.forget_perc,
                shadow_models=0,
                shared_forget_index=True,
            ),
            artifacts={
                "unlearned_checkpoint": str(cls._unlearned_path(context)),
                "forget_index": str(cls._forget_index_path(context)),
            },
            raw={
                "signal": "prediction confidence (correct-class probability)",
                "n_member": n_member,
                "n_nonmember": n_nonmember,
            },
        )

        summary_path = Path(context.attack_result_dir("threshold")) / "attack_result.json"
        summary_path.write_text(json.dumps(result.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        result.artifacts["benchmark_summary_json"] = str(summary_path)
        return result
