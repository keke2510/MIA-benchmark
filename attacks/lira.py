from __future__ import annotations

import json
import os
import subprocess
import sys
from attacks.base import AttackContext, AttackResult
from attacks.registry import register_attack


@register_attack("lira")
class LiRAAttack:
    task_name = "mia_lira"

    @staticmethod
    def _merge_preferred_metrics(metrics: dict, raw: dict) -> dict:
        some_stats = raw.get("some_stats", {})
        if not some_stats:
            return metrics

        merged = dict(metrics)
        merged.update(some_stats)

        if "fix_auc" in some_stats:
            merged["auc"] = some_stats["fix_auc"]
        if "fix_acc" in some_stats:
            merged["accuracy"] = some_stats["fix_acc"]
        if "fix_TPR@0.01FPR" in some_stats:
            merged["TPR@1%FPR"] = some_stats["fix_TPR@0.01FPR"]
        if "fix_TPR@0.1FPR" in some_stats:
            merged["TPR@10%FPR"] = some_stats["fix_TPR@0.1FPR"]

        return merged

    @staticmethod
    def build_command(context: AttackContext) -> list[str]:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return [
            sys.executable,
            os.path.join(repo_root, "mia_lira.py"),
            "--net", context.net,
            "--dataset", context.dataset,
            "--classes", str(context.classes),
            "--name", context.experiment_name,
            "--save_name", context.model_save_name,
            "--num_shadow", str(context.num_shadow),
            "--num_aug", str(context.num_aug),
            "--machine_unlearning", context.machine_unlearning,
            "--para1", context.para1,
            "--para2", context.para2,
            "-forget_perc", str(context.forget_perc),
            "-task", LiRAAttack.task_name,
            "--weight_path", context.weight_path,
            "--seed", str(context.seed),
        ]

    @staticmethod
    def _summary_path(context: AttackContext) -> str:
        return os.path.join(context.checkpoint_dir("unlearning"), "lira_mia_lira_summary.json")

    @staticmethod
    def _curve_paths(context: AttackContext) -> tuple[str, str]:
        base = context.checkpoint_dir("unlearning")
        return (
            os.path.join(base, "lira_mia_lira_tpr.npy"),
            os.path.join(base, "lira_mia_lira_fpr.npy"),
        )

    @classmethod
    def run(cls, context: AttackContext, dry_run: bool = False) -> AttackResult | None:
        cmd = cls.build_command(context)
        print(f"[attack:lira] {' '.join(cmd)}")
        if dry_run:
            return None

        subprocess.run(cmd, check=True)
        result = cls.collect(context)
        summary_path = os.path.join(context.attack_result_dir("lira"), "attack_result.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result.as_dict(), f, indent=2, ensure_ascii=False)
        result.artifacts["benchmark_summary_json"] = summary_path
        return result

    @classmethod
    def collect(cls, context: AttackContext) -> AttackResult:
        import numpy as np

        from evaluation.cost import build_cost_report
        from evaluation.effectiveness import summarize_curve_metrics
        from evaluation.realism import build_protocol_checklist

        tpr_path, fpr_path = cls._curve_paths(context)
        metrics = {}
        if os.path.exists(tpr_path) and os.path.exists(fpr_path):
            tpr = np.load(tpr_path)
            fpr = np.load(fpr_path)
            metrics.update(summarize_curve_metrics(fpr=fpr, tpr=tpr))
        summary_path = cls._summary_path(context)
        raw = {}
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            metrics = cls._merge_preferred_metrics(metrics, raw)

        return AttackResult(
            attack_name="lira",
            metrics=metrics,
            cost=build_cost_report(
                num_shadow=context.num_shadow,
                num_aug=context.num_aug,
                attack_sample_number=context.attack_sample_number,
            ),
            protocol=build_protocol_checklist(
                target_checkpoint=context.weight_path,
                forget_perc=context.forget_perc,
                shadow_models=context.num_shadow,
                shared_forget_index=True,
            ),
            artifacts={
                "summary_json": summary_path,
                "tpr_npy": tpr_path,
                "fpr_npy": fpr_path,
            },
            raw=raw,
        )
