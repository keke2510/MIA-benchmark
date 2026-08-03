from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
import config

from attacks.base import AttackContext, AttackResult
from attacks.registry import register_attack

@register_attack("ruli")
class RULIAttack:
    dataset_train_size = {
        "Cifar10": 50000,
        "Cifar100": 50000,
        "Cinic10": 180000,
        "TinyImageNet": 100000,
    }
    dataset_map = {
        "Cifar10": "cifar10",
        "Cifar100": "cifar100",
        "Cinic10": "cinic10",
        "TinyImageNet": "TinyImageNet",
    }
    arch_map = {
        "ResNet18": "resnet18",
        "ResNet50": "resnet50",
        "ViT": "vit",
    }
    method_map = {
        "retrain": "Retrain",
        "finetune": "FT",
        "negative_grad": "NegGrad",
        "scrub": "Scrub",
            }

    @staticmethod
    def repo_root() -> Path:
        return Path(__file__).resolve().parent.parent

    @classmethod
    def ruli_root(cls) -> Path:
        return cls.repo_root() / "Ruli" / "core"

    @classmethod
    def resolve_dataset(cls, context: AttackContext) -> str:
        if context.dataset not in cls.dataset_map:
            raise ValueError(f"RULI 当前只接入了 {sorted(cls.dataset_map)}，当前是 {context.dataset}。")
        return cls.dataset_map[context.dataset]

    @classmethod
    def resolve_arch(cls, context: AttackContext) -> str:
        value = context.attack_options.get("ruli_arch")
        if value:
            return str(value)
        return cls.arch_map.get(context.net, "resnet18")

    @classmethod
    def resolve_unlearn_method(cls, context: AttackContext) -> str:
        value = context.attack_options.get("ruli_unlearn_method")
        if value:
            return str(value)
        if context.machine_unlearning not in cls.method_map:
            raise ValueError(
                "RULI 当前只自动映射 `retrain/finetune/negative_grad/scrub`。"
                f" 当前 benchmark 方法是 `{context.machine_unlearning}`，请用 `--ruli-unlearn-method` 手动指定。"
            )
        return cls.method_map[context.machine_unlearning]

    @classmethod
    def result_dir(cls, context: AttackContext) -> Path:
        path = Path(context.attack_result_dir("ruli"))
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def resolve_forget_size(cls, context: AttackContext) -> int:
        override = context.attack_options.get("ruli_forget_size")
        if override is not None:
            return int(override)
        train_size = cls.dataset_train_size.get(context.dataset)
        if train_size is None:
            raise ValueError(f"RULI 缺少 `{context.dataset}` 的训练集大小映射。")
        return max(1, int(round(train_size * context.forget_perc)))

    @classmethod
    def log_dir(cls, context: AttackContext) -> Path:
        path = cls.result_dir(context) / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def shadow_result_path(cls, context: AttackContext) -> Path:
        dataset = cls.resolve_dataset(context)
        shadow_num = int(context.attack_options.get("ruli_shadow_num", 8))
        seed = int(context.attack_options.get("ruli_seed", context.seed))
        unlearn_method = cls.resolve_unlearn_method(context)
        task = str(context.attack_options.get("ruli_task", "selective"))
        result_path = cls.ruli_root() / "attack" / "attack_inferences" / dataset
        result_path.mkdir(parents=True, exist_ok=True)
        return result_path / f"shadows_{shadow_num}_{seed}_{unlearn_method}_unlearn_{task}.pth"

    @classmethod
    def shadow_results_ready(cls, context: AttackContext) -> bool:
        return cls.shadow_result_path(context).exists()

    @classmethod
    def shared_forget_index_path(cls, context: AttackContext) -> Path:
        return (
            Path(config.CHECKPOINT_PATH)
            / "forget_random_main"
            / f"{context.net}-{context.dataset}-{context.classes}"
            / "random_index_set"
            / f"forgetting_dataset_index_{context.forget_perc}.npy"
        ).resolve()

    @classmethod
    def build_command(cls, context: AttackContext, *, train_shadow: bool) -> list[str]:
        dataset = cls.resolve_dataset(context)
        arch = cls.resolve_arch(context)
        unlearn_method = cls.resolve_unlearn_method(context)
        task = str(context.attack_options.get("ruli_task", "selective"))
        shadow_num = int(context.attack_options.get("ruli_shadow_num", 8))
        forget_size = cls.resolve_forget_size(context)
        attack_size = int(context.attack_options.get("ruli_attack_size", 24000))
        train_epochs = int(context.attack_options.get("ruli_train_epochs", 50))
        batch_size = int(context.attack_options.get("ruli_batch_size", 128))
        test_batch_size = int(context.attack_options.get("ruli_test_batch_size", 128))
        lr = float(context.attack_options.get("ruli_lr", 0.1))
        weight_decay = float(context.attack_options.get("ruli_weight_decay", 5e-4))
        device = str(context.attack_options.get("ruli_device", "cuda:0"))
        config_path = str(context.attack_options.get("ruli_config_path", "./unlearn_config.json"))
        result_path = str(context.attack_options.get("ruli_result_path", "./attack/attack_inferences"))
        cmd = [
            "unlearn_mia.py",
            "--dataset", dataset,
            "--arch", arch,
            "--task", task,
            "--forget_size", str(forget_size),
            "--forget_index_path", cls.shared_forget_index_path(context).as_posix(),
            "--attack_size", str(attack_size),
            "--train_epochs", str(train_epochs),
            "--batch_size", str(batch_size),
            "--test_batch_size", str(test_batch_size),
            "--lr", str(lr),
            "--weight_decay", str(weight_decay),
            "--shadow_num", str(shadow_num),
            "--device", device,
            "--seed", str(int(context.attack_options.get("ruli_seed", context.seed))),
            "--unlearn_method", unlearn_method,
            "--result_path", result_path,
            "--saved_results", str(cls.shadow_result_path(context)),
            "--config_path", config_path,
        ]
        if train_shadow:
            cmd.append("--train_shadow")
        return cmd

    @classmethod
    def run_python(
        cls,
        cmd: list[str],
        *,
        dry_run: bool,
        verbose: bool,
        log_path: Path,
    ) -> None:
        rendered = [sys.executable, *cmd]
        print(f"[attack:ruli] {' '.join(rendered)}")
        if dry_run:
            return
        print(f"[attack:ruli] 日志文件: {log_path}")
        try:
            if verbose:
                subprocess.run(rendered, cwd=cls.ruli_root(), check=True)
            else:
                with open(log_path, "w", encoding="utf-8") as log_file:
                    subprocess.run(
                        rendered,
                        cwd=cls.ruli_root(),
                        check=True,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
        except subprocess.CalledProcessError as exc:
            log_tail = ""
            if log_path.exists():
                lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                log_tail = "\n".join(lines[-40:])
            raise RuntimeError(
                "RULI 子流程执行失败，请优先检查依赖、数据下载和配置文件路径。"
                + (f"\n\n最近日志:\n{log_tail}" if log_tail else "")
            ) from exc

    @staticmethod
    def parse_metrics_from_log(log_text: str) -> dict[str, float]:
        def extract_block_metrics(block_text: str) -> dict[str, float]:
            patterns = {
                "accuracy": r"Attack accuracy:\s*([0-9.]+)",
                "auc": r"ROC AUC:\s*([0-9.]+)",
                "TPR@10%FPR": r"TPR at 0\.1 FPR:\s*([0-9.]+)",
                "TPR@1%FPR": r"TPR at (?:0\.01 FPR|FPR=1%):\s*([0-9.]+)",
                "TPR@0.1%FPR": r"TPR at 0\.001 FPR:\s*([0-9.]+)",
            }
            metrics: dict[str, float] = {}
            for key, pattern in patterns.items():
                match = re.search(pattern, block_text)
                if match:
                    metrics[key] = float(match.group(1))
            return metrics

        def slice_block(start_marker: str, end_markers: list[str]) -> str:
            start = log_text.find(start_marker)
            if start == -1:
                return ""
            end_positions = [log_text.find(marker, start + len(start_marker)) for marker in end_markers]
            valid_positions = [pos for pos in end_positions if pos != -1]
            end = min(valid_positions) if valid_positions else len(log_text)
            return log_text[start:end]

        blocks = {
            "efficacy": slice_block(
                "** Efficacy Attack Results **",
                ["** Privacy Leakage Attack Results **", "Starting Population Attack with Global Shadow Model Populations..."],
            ),
            "privacy_leakage": slice_block(
                "** Privacy Leakage Attack Results **",
                ["Starting Population Attack with Global Shadow Model Populations..."],
            ),
            "population": slice_block(
                "Population Attack Results:",
                ["Starting Population Attack for Vulnerable Samples"],
            ),
            "vulnerable_population": slice_block(
                "Population Attack Results for Vulnerable Samples:",
                [],
            ),
        }

        parsed = {name: extract_block_metrics(block_text) for name, block_text in blocks.items() if block_text}
        summary = parsed.get("privacy_leakage") or parsed.get("population") or parsed.get("efficacy") or {}
        return {
            "summary": summary,
            "blocks": parsed,
        }

    @classmethod
    def run(cls, context: AttackContext, dry_run: bool = False) -> AttackResult | None:
        log_dir = cls.log_dir(context)
        verbose = bool(context.attack_options.get("attack_verbose", False))
        train_shadow_mode = str(context.attack_options.get("ruli_train_shadow_mode", "auto")).lower()
        if train_shadow_mode == "true":
            train_shadow = True
        elif train_shadow_mode == "false":
            train_shadow = False
        else:
            train_shadow = not cls.shadow_results_ready(context)

        if train_shadow:
            print("[attack:ruli] 将训练并遗忘 shadow models，首次运行会比较久。")
        else:
            print("[attack:ruli] 将复用已有 RULI shadow results。")

        cmd = cls.build_command(context, train_shadow=train_shadow)
        start = time.time()
        cls.run_python(
            cmd,
            dry_run=dry_run,
            verbose=verbose,
            log_path=log_dir / "ruli.log",
        )
        if dry_run:
            return None

        result = cls.collect(context, train_shadow=train_shadow, runtime_sec=time.time() - start)
        summary_path = cls.result_dir(context) / "ruli_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result.as_dict(), f, indent=2, ensure_ascii=False)
        result.artifacts["benchmark_summary_json"] = str(summary_path)
        result.artifacts["log_dir"] = str(log_dir)
        return result

    @classmethod
    def collect(
        cls,
        context: AttackContext,
        *,
        train_shadow: bool,
        runtime_sec: float,
    ) -> AttackResult:
        from evaluation.cost import build_cost_report
        from evaluation.realism import build_protocol_checklist

        log_path = cls.log_dir(context) / "ruli.log"
        log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        parsed = cls.parse_metrics_from_log(log_text)
        metrics = parsed.get("summary", {})
        protocol = build_protocol_checklist(
            target_checkpoint=context.weight_path,
            forget_perc=context.forget_perc,
            shadow_models=int(context.attack_options.get("ruli_shadow_num", 8)),
            shared_forget_index=False,
            same_target_checkpoint_for_all_attacks=False,
            same_shadow_setup_for_all_attacks=False,
            dataset_split_name=f"RULI-{context.attack_options.get('ruli_task', 'selective')}",
            same_dataset_split_for_all_attacks=False,
            uses_original_unlearning_pipeline=False,
        )
        protocol.update(
            {
                "ruli_task": str(context.attack_options.get("ruli_task", "selective")),
                "ruli_train_shadow_this_run": train_shadow,
                "ruli_saved_shadow_results": str(cls.shadow_result_path(context)),
            }
        )
        return AttackResult(
            attack_name="ruli",
            metrics=metrics,
            cost=build_cost_report(
                num_shadow=int(context.attack_options.get("ruli_shadow_num", 8)),
                attack_sample_number=cls.resolve_forget_size(context),
                runtime_sec=runtime_sec,
                storage_bytes=log_path.stat().st_size if log_path.exists() else None,
            ),
            protocol=protocol,
            artifacts={
                "ruli_log": str(log_path),
                "ruli_shadow_results": str(cls.shadow_result_path(context)),
            },
            raw={
                "parsed_from_log": True,
                "ruli_blocks": parsed.get("blocks", {}),
            },
        )
