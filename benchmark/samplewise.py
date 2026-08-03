from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import config
from attacks import AttackContext, get_attack, list_attacks

REPO_ROOT = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class UnlearningMethod:
    name: str
    para1: str
    para2: str

@dataclass(frozen=True)
class DatasetConfig:
    dataset: str
    net: str
    n_classes: int
    weight_path: str
    shadow_opt: str
    shadow_lr: str
    shadow_bs: str = "128"
    shadow_keep: str = "0.83"

DATASET_DEFAULTS = {
    "Cifar10": DatasetConfig(
        dataset="Cifar10",
        net="ResNet18",
        n_classes=10,
        weight_path=f"{config.CHECKPOINT_PATH}/pretrain/ResNet18-Cifar10-10/best.pth",
        shadow_opt="sgd",
        shadow_lr="1e-1",
    ),
    "Cifar20": DatasetConfig(
        dataset="Cifar20",
        net="ViT",
        n_classes=20,
        weight_path=f"{config.CHECKPOINT_PATH}/pretrain/ViT-Cifar20-20/best.pth",
        shadow_opt="adam",
        shadow_lr="3e-4",
    ),
    "Cifar100": DatasetConfig(
        dataset="Cifar100",
        net="ResNet50",
        n_classes=100,
        weight_path=f"{config.CHECKPOINT_PATH}/pretrain/ResNet50-Cifar100-100/best.pth",
        shadow_opt="sgd",
        shadow_lr="1e-1",
    ),
    "Cinic10": DatasetConfig(
        dataset="Cinic10",
        net="ResNet18",
        n_classes=10,
        weight_path=f"{config.CHECKPOINT_PATH}/pretrain/ResNet18-Cinic10-10/best.pth",
        shadow_opt="sgd",
        shadow_lr="1e-1",
    ),
    "TinyImageNet": DatasetConfig(
        dataset="TinyImageNet",
        net="ViT",
        n_classes=200,
        weight_path=f"{config.CHECKPOINT_PATH}/pretrain/ViT-TinyImageNet-200/best.pth",
        shadow_opt="adam",
        shadow_lr="3e-4",
        shadow_bs="64",
    ),
}

ALL_METHODS = {
    "Cifar10": [
        UnlearningMethod("retrain", "0.1", "150"),
        UnlearningMethod("finetune", "0.11", "10"),
        UnlearningMethod("negative_grad", "0.04", "8"),
        UnlearningMethod("relabel", "0.00065", "3"),
        UnlearningMethod("Wfisher", "130", "0"),
        UnlearningMethod("FisherForgetting", "7e-8", "0"),
        UnlearningMethod("scrub", "0.0004", "7"),
        UnlearningMethod("FT_prune", "0.008", "10"),
        UnlearningMethod("salun", "0.00065", "3"),
        UnlearningMethod("sfron", "0.035", "8"),
        UnlearningMethod("rum", "0.12", "0.0005"),
        UnlearningMethod("orthogonality", "0.0012", "0.04"),
    ],
    "Cifar20": [
        UnlearningMethod("retrain", "0.0003", "150"),
        UnlearningMethod("FisherForgetting", "1e-8", "0"),
        UnlearningMethod("finetune", "1.2e-3", "10"),
        UnlearningMethod("negative_grad", "1.5e-3", "15"),
        UnlearningMethod("relabel", "8.5e-4", "8"),
        UnlearningMethod("Wfisher", "40", "0"),
        UnlearningMethod("Wfisher", "45", "0"),
        UnlearningMethod("scrub", "4e-4", "10"),
        UnlearningMethod("FT_prune", "1e-3", "10"),
        UnlearningMethod("salun", "1.2e-3", "15"),
        UnlearningMethod("sfron", "1.1e-4", "6"),
        UnlearningMethod("rum", "3e-3", "1e-3"),
        UnlearningMethod("m_orthogonality", "3e-4", "5e-4"),
    ],
    "Cifar100": [
        UnlearningMethod("retrain", "0.1", "150"),
        UnlearningMethod("finetune", "0.11", "10"),
        UnlearningMethod("negative_grad", "0.04", "8"),
        UnlearningMethod("scrub", "0.0004", "7"),
    ],
    "Cinic10": [
        UnlearningMethod("retrain", "0.1", "150"),
        UnlearningMethod("finetune", "0.11", "10"),
        UnlearningMethod("negative_grad", "0.04", "8"),
        UnlearningMethod("scrub", "0.0004", "7"),
    ],
    "TinyImageNet": [
        UnlearningMethod("retrain", "3e-4", "30"),
        UnlearningMethod("finetune", "8e-4", "10"),
        UnlearningMethod("negative_grad", "1e-3", "12"),
        UnlearningMethod("scrub", "4e-4", "10"),
    ],
}

DEFAULT_ACTIVE_METHODS = {
    "Cifar10": [UnlearningMethod("finetune", "0.11", "10")],
    "Cifar20": [UnlearningMethod("finetune", "1.2e-3", "10")],
    "Cifar100": [UnlearningMethod("finetune", "0.11", "10")],
    "Cinic10": [UnlearningMethod("finetune", "0.11", "10")],
    "TinyImageNet": [UnlearningMethod("finetune", "8e-4", "10")],
}

def _run(cmd: list[str], dry_run: bool) -> None:
    rendered = " ".join(cmd)
    print(f"[benchmark] {rendered}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

def _python_entry(script_name: str) -> list[str]:
    return [sys.executable, str(REPO_ROOT / script_name)]

def _shadow_checkpoint_path(cfg: DatasetConfig, shadow_id: int) -> Path:
    return (
        REPO_ROOT
        / "saved_models"
        / f"{cfg.net}-{cfg.dataset}-{cfg.n_classes}"
        / f"{cfg.net}_shadow_{shadow_id}_last.pth"
    )

def _run_summary_path(cfg: DatasetConfig, seed: int) -> Path:
    path = (
        REPO_ROOT
        / "benchmark_results"
        / "samplewise"
        / f"{cfg.net}-{cfg.dataset}-{cfg.n_classes}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path / f"run_summary_seed_{seed}.json"

SUMMARY_METRIC_KEYS = [
    "auc",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "TPR@0.1%FPR",
    "TPR@1%FPR",
    "TPR@10%FPR",
    "ternary_accuracy_max",
    "ternary_accuracy_mean",
    "ternary_tpr_unlearn",
    "ternary_tpr_retain",
    "ternary_tpr_test",
]

def _format_metric_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)

def print_attack_summary(attack_name: str, result) -> None:
    print(f"[summary:{attack_name}]")
    printed = False
    for key in SUMMARY_METRIC_KEYS:
        if key in result.metrics and result.metrics[key] is not None:
            print(f"  {key}: {_format_metric_value(result.metrics[key])}")
            printed = True
    if "runtime_sec" in result.cost and result.cost["runtime_sec"] is not None:
        print(f"  runtime_sec: {_format_metric_value(result.cost['runtime_sec'])}")
        printed = True
    if "benchmark_summary_json" in result.artifacts:
        print(f"  result_json: {result.artifacts['benchmark_summary_json']}")
        printed = True
    if not printed:
        print("  no standardized metrics available")

def train_shadow_models(cfg: DatasetConfig, *, num_shadow: int, seed: int, dry_run: bool) -> None:
    for shadow_id in range(num_shadow):
        shadow_ckpt = _shadow_checkpoint_path(cfg, shadow_id)
        if shadow_ckpt.exists():
            print(f"[benchmark] skip shadow_{shadow_id}, found existing checkpoint: {shadow_ckpt}")
            continue
        cmd = _python_entry("train.py") + [
            "--name", f"{cfg.net}-{cfg.dataset}-{cfg.n_classes}",
            "--dataset", cfg.dataset,
            "--save_name", cfg.net,
            "--classes", str(cfg.n_classes),
            "--net", cfg.net,
            "--opt", cfg.shadow_opt,
            "--bs", cfg.shadow_bs,
            "--lr", cfg.shadow_lr,
            "--pkeep", cfg.shadow_keep,
            "--num_shadow", str(num_shadow),
            "--shadow_id", str(shadow_id),
            "--seed", str(seed),
        ]
        _run(cmd, dry_run)

def run_pretrain_target(cfg: DatasetConfig, *, seed: int, dry_run: bool) -> None:
    cmd = _python_entry("pretrain_model_sample_wise.py") + [
        "--net", cfg.net,
        "--dataset", cfg.dataset,
        "--classes", str(cfg.n_classes),
        "--bs", cfg.shadow_bs,
        "-lr", cfg.shadow_lr,
        "-seed", str(seed),
    ]
    _run(cmd, dry_run)

def run_sample_unlearning(
    cfg: DatasetConfig,
    methods: Iterable[UnlearningMethod],
    *,
    seed: int,
    forget_perc: float,
    dry_run: bool,
) -> None:
    for method in methods:
        cmd = _python_entry("forget_sample_main.py") + [
            "-net", cfg.net,
            "-dataset", cfg.dataset,
            "-classes", str(cfg.n_classes),
            "-method", method.name,
            "-weight_path", cfg.weight_path,
            "--para1", method.para1,
            "--para2", method.para2,
            "-forget_perc", str(forget_perc),
            "-seed", str(seed),
        ]
        _run(cmd, dry_run)

def run_rea_reminiscence(
    cfg: DatasetConfig,
    methods: Iterable[UnlearningMethod],
    *,
    seed: int,
    forget_perc: float,
    dry_run: bool,
) -> None:
    for method in methods:
        cmd = _python_entry("rea_reminiscence_random.py") + [
            "-net", cfg.net,
            "-dataset", cfg.dataset,
            "-classes", str(cfg.n_classes),
            "-method", method.name,
            "-weight_path", cfg.weight_path,
            "--para1", method.para1,
            "--para2", method.para2,
            "-forget_perc", str(forget_perc),
            "-seed", str(seed),
        ]
        _run(cmd, dry_run)

def run_registered_attacks(
    cfg: DatasetConfig,
    methods: Iterable[UnlearningMethod],
    *,
    attacks: Iterable[str],
    seed: int,
    num_shadow: int,
    num_aug: int,
    attack_options: dict[str, object],
    dry_run: bool,
) -> None:
    for method in methods:
        context = AttackContext(
            dataset=cfg.dataset,
            net=cfg.net,
            classes=cfg.n_classes,
            machine_unlearning=method.name,
            para1=method.para1,
            para2=method.para2,
            seed=seed,
            weight_path=cfg.weight_path,
            num_shadow=num_shadow,
            num_aug=num_aug,
            forget_perc=float(attack_options.get("forget_perc", 0.1)),
            name=f"{cfg.net}-{cfg.dataset}-{cfg.n_classes}",
            save_name=cfg.net,
            attack_options=attack_options,
        )
        for attack_name in attacks:
            attack_cls = get_attack(attack_name)
            result = attack_cls.run(context, dry_run=dry_run)
            if result is not None:
                print_attack_summary(attack_name, result)

def parse_method(method_spec: str, dataset_name: str) -> UnlearningMethod:
    parts = method_spec.split(":")
    if len(parts) == 1:
        method_name = parts[0]
        candidates = [m for m in ALL_METHODS[dataset_name] if m.name == method_name]
        if not candidates:
            raise ValueError(
                f"Unknown method '{method_name}' for dataset '{dataset_name}'. "
                "Use --method-preset all to inspect available defaults or pass method:para1:para2 explicitly."
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Method '{method_name}' has multiple default parameter sets for dataset '{dataset_name}'. "
                "Please pass it explicitly as method:para1:para2."
            )
        return candidates[0]

    if len(parts) != 3:
        raise ValueError(
            f"Invalid method spec '{method_spec}'. Expected format: method or method:para1:para2"
        )
    return UnlearningMethod(parts[0], parts[1], parts[2])

def build_parser() -> argparse.ArgumentParser:
    available_attacks = list_attacks()
    parser = argparse.ArgumentParser(
        description="Sample-wise MIA benchmark wrapper around the original ICCV codebase."
    )
    parser.add_argument(
        "--dataset",
        default="Cifar10",
        choices=sorted(DATASET_DEFAULTS.keys()),
        help="Benchmark dataset preset.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Override unlearning methods with method:para1:para2 entries.",
    )
    parser.add_argument(
        "--method-preset",
        default="default",
        choices=["default", "all"],
        help="`default` 只跑当前激活的方法，`all` 保留并运行原始脚本中的完整 baseline 列表。",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["unlearn", "reminiscence", "attack"],
        choices=["pretrain", "shadow", "unlearn", "reminiscence", "attack", "lira"],
        help="Pipeline stages to run.",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["lira", "rea", "ruli", "unlearningleaks"],
        choices=available_attacks,
        help="要运行的攻击方法。",
    )
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--forget-perc", default=0.1, type=float, help="统一的 sample-wise 遗忘比例。")
    parser.add_argument("--num-shadow", default=8, type=int)
    parser.add_argument("--num-aug", default=10, type=int)
    parser.add_argument(
        "--dhattack-classifier-type",
        default=None,
        choices=["vgg", "resnet", "mobilenet"],
        help="DHAttack 使用的目标/本地模型类型。默认会根据当前 benchmark 主模型自动映射。",
    )
    parser.add_argument(
        "--dhattack-disturb-num",
        default=30,
        type=int,
        help="DHAttack 的查询次数，对应论文里的 disturb_num。",
    )
    parser.add_argument(
        "--dhattack-ref-epochs",
        default=100,
        type=int,
        help="DHAttack 训练目标模型和 reference models 使用的 epoch 数。",
    )
    parser.add_argument(
        "--dhattack-num-local-models",
        default=8,
        type=int,
        help="DHAttack 使用的 local/reference model 数量。",
    )
    parser.add_argument(
        "--dhattack-train-ref-mode",
        default="auto",
        choices=["auto", "true", "false"],
        help="`auto` 表示如果本地模型不存在就训练，否则直接复用。",
    )
    parser.add_argument(
        "--ruli-task",
        default="selective",
        choices=["selective", "class-wise", "mixed", "canary", "privacy", "vulnerable"],
        help="RULI 的任务类型。做 sample-wise benchmark 时优先用 `selective`。",
    )
    parser.add_argument(
        "--ruli-unlearn-method",
        default=None,
        choices=["Retrain", "FT", "GA", "GA+", "NegGrad", "Scrub"],
        help="手动指定 RULI 的 unlearning method；默认会根据当前 benchmark 方法自动映射。",
    )
    parser.add_argument("--ruli-shadow-num", default=8, type=int, help="RULI shadow model 数量。")
    parser.add_argument("--ruli-forget-size", default=None, type=int, help="RULI selective forget size。默认会按数据集自动选择。")
    parser.add_argument("--ruli-attack-size", default=24000, type=int, help="RULI attack set size。")
    parser.add_argument("--ruli-train-epochs", default=50, type=int, help="RULI target/shadow 训练 epoch。")
    parser.add_argument("--ruli-batch-size", default=128, type=int, help="RULI batch size。")
    parser.add_argument("--ruli-test-batch-size", default=128, type=int, help="RULI test batch size。")
    parser.add_argument("--ruli-lr", default=0.1, type=float, help="RULI 学习率。")
    parser.add_argument("--ruli-weight-decay", default=5e-4, type=float, help="RULI 权重衰减。")
    parser.add_argument("--ruli-device", default="cuda:0", help="RULI 使用的设备。")
    parser.add_argument("--ruli-seed", default=None, type=int, help="RULI 自己的随机种子；默认跟 benchmark seed 一致。")
    parser.add_argument(
        "--ruli-train-shadow-mode",
        default="auto",
        choices=["auto", "true", "false"],
        help="`auto` 表示如果已有 shadow results 就复用，否则训练。",
    )
    parser.add_argument("--ruli-config-path", default="./unlearn_config.json", help="RULI config 路径，相对 `Ruli/core/`。")
    parser.add_argument("--ruli-result-path", default="./attack/attack_inferences", help="RULI 结果目录，相对 `Ruli/core/`。")
    parser.add_argument(
        "--ruli-arch",
        default=None,
        choices=["resnet18", "resnet50", "vgg16_bn", "wrn28_10", "vit"],
        help="RULI 使用的模型结构；默认根据当前 benchmark 模型自动映射。",
    )
    parser.add_argument(
        "--apollo-unlearn-method",
        default=None,
        choices=["Baseline", "Retrain", "Finetune", "GradAscent", "BadTeacher", "RandomLabel", "SCRUB", "SalUn", "SFRon"],
        help="手动指定 Apollo 使用的遗忘方法；默认会根据当前 benchmark 方法自动映射。",
    )
    parser.add_argument("--apollo-size-train", default=2500, type=int, help="Apollo target 训练集大小。")
    parser.add_argument("--apollo-size-shadow", default=2500, type=int, help="Apollo 每个 shadow 的训练集大小。")
    parser.add_argument("--apollo-num-shadow", default=8, type=int, help="Apollo shadow model 数量。")
    parser.add_argument("--apollo-pretrain-epochs", default=200, type=int, help="Apollo target pretrain epoch。")
    parser.add_argument("--apollo-shadow-epochs", default=200, type=int, help="Apollo shadow training epoch。")
    parser.add_argument("--apollo-batch-size", default=128, type=int, help="Apollo batch size。")
    parser.add_argument("--apollo-lr", default=0.1, type=float, help="Apollo 学习率。")
    parser.add_argument("--apollo-weight-decay", default=5e-4, type=float, help="Apollo 权重衰减。")
    parser.add_argument("--apollo-opt", default="sgd", choices=["sgd", "adamw"], help="Apollo 优化器。")
    parser.add_argument("--apollo-forget-perc", default=0.1, type=float, help="Apollo sample-wise forget ratio。")
    parser.add_argument("--apollo-shadow-split", default="full", choices=["full", "limited"], help="Apollo shadow 采样方式。")
    parser.add_argument("--apollo-attack-type", default="Apollo", choices=["Apollo", "Apollo_Offline", "ULiRA", "UMIA"], help="Apollo 仓库里的攻击类型。")
    parser.add_argument("--apollo-attack-N", default=200, type=int, help="Apollo 攻击采样数，最终会自动截断到不超过 forget set 大小。")
    parser.add_argument("--apollo-atk-lr", default=0.1, type=float, help="Apollo 对抗优化学习率。")
    parser.add_argument("--apollo-atk-epochs", default=30, type=int, help="Apollo 对抗优化 epoch。")
    parser.add_argument("--apollo-eps", default=10.0, type=float, help="Apollo 扰动约束 eps。")
    parser.add_argument(
        "--apollo-attack-weights",
        nargs=2,
        type=float,
        default=[1.0, 1.0],
        help="Apollo 的两个攻击损失权重。",
    )
    parser.add_argument(
        "--unlearningleaks-feature",
        default="direct_diff",
        choices=["direct_diff", "sorted_diff", "direct_concat", "sorted_concat", "l2_distance", "basic_mia"],
        help="UnlearningLeaks 使用的特征构造方式。",
    )
    parser.add_argument(
        "--unlearningleaks-attack-model",
        default="lr",
        choices=["lr", "dt", "rf", "mlp"],
        help="UnlearningLeaks 使用的攻击分类器。",
    )
    parser.add_argument("--unlearningleaks-num-shadow", default=8, type=int, help="UnlearningLeaks 使用的 shadow 数量。")
    parser.add_argument("--unlearningleaks-sample-budget", default=500, type=int, help="每个 shadow/target 正负样本的采样上限。")
    parser.add_argument("--unlearningleaks-target-budget", default=500, type=int, help="target 侧正负样本采样上限。")
    parser.add_argument("--unlearningleaks-eval-bs", default=256, type=int, help="UnlearningLeaks posterior 提取 batch size。")
    parser.add_argument("--unlearningleaks-unlearn-bs", default=256, type=int, help="生成 shadow unlearned models 时的 batch size。")
    parser.add_argument("--unlearningleaks-seed", default=None, type=int, help="UnlearningLeaks 内部采样种子；默认跟 benchmark seed 一致。")
    parser.add_argument("--unlearningleaks-shadow-opt", default=None, choices=["sgd", "adam"], help="自动补训 shadow model 时使用的优化器。")
    parser.add_argument("--unlearningleaks-shadow-lr", default=None, help="自动补训 shadow model 时使用的学习率。")
    parser.add_argument("--unlearningleaks-shadow-bs", default=None, help="自动补训 shadow model 时使用的 batch size。")
    parser.add_argument("--unlearningleaks-shadow-pkeep", default=None, help="自动补训 shadow model 时使用的 pkeep。")
    parser.add_argument(
        "--attack-verbose",
        action="store_true",
        help="显示攻击方法原始脚本的完整输出；默认只显示统一摘要，并把详细日志写入结果目录。",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    total_start = time.time()

    cfg = DATASET_DEFAULTS[args.dataset]
    methods = (
        [parse_method(spec, args.dataset) for spec in args.methods]
        if args.methods
        else (
            DEFAULT_ACTIVE_METHODS[args.dataset]
            if args.method_preset == "default"
            else ALL_METHODS[args.dataset]
        )
    )
    stage_runtime_sec: dict[str, float] = {}

    def timed_stage(stage_name: str, fn) -> None:
        stage_start = time.time()
        fn()
        stage_runtime_sec[stage_name] = time.time() - stage_start

    if "pretrain" in args.stages:
        timed_stage("pretrain", lambda: run_pretrain_target(cfg, seed=args.seed, dry_run=args.dry_run))
    if "shadow" in args.stages:
        timed_stage(
            "shadow",
            lambda: train_shadow_models(cfg, num_shadow=args.num_shadow, seed=args.seed, dry_run=args.dry_run),
        )
    if "unlearn" in args.stages:
        timed_stage(
            "unlearn",
            lambda: run_sample_unlearning(
                cfg,
                methods,
                seed=args.seed,
                forget_perc=args.forget_perc,
                dry_run=args.dry_run,
            ),
        )
    if "reminiscence" in args.stages:
        timed_stage(
            "reminiscence",
            lambda: run_rea_reminiscence(
                cfg,
                methods,
                seed=args.seed,
                forget_perc=args.forget_perc,
                dry_run=args.dry_run,
            ),
        )
    if "attack" in args.stages or "lira" in args.stages:
        attack_options = {
            "forget_perc": args.forget_perc,
            "dhattack_classifier_type": args.dhattack_classifier_type,
            "dhattack_disturb_num": args.dhattack_disturb_num,
            "dhattack_ref_epochs": args.dhattack_ref_epochs,
            "dhattack_num_local_models": args.dhattack_num_local_models,
            "dhattack_train_ref_mode": args.dhattack_train_ref_mode,
            "ruli_task": args.ruli_task,
            "ruli_unlearn_method": args.ruli_unlearn_method,
            "ruli_shadow_num": args.ruli_shadow_num,
            "ruli_forget_size": args.ruli_forget_size,
            "ruli_attack_size": args.ruli_attack_size,
            "ruli_train_epochs": args.ruli_train_epochs,
            "ruli_batch_size": args.ruli_batch_size,
            "ruli_test_batch_size": args.ruli_test_batch_size,
            "ruli_lr": args.ruli_lr,
            "ruli_weight_decay": args.ruli_weight_decay,
            "ruli_device": args.ruli_device,
            "ruli_seed": args.ruli_seed if args.ruli_seed is not None else args.seed,
            "ruli_train_shadow_mode": args.ruli_train_shadow_mode,
            "ruli_config_path": args.ruli_config_path,
            "ruli_result_path": args.ruli_result_path,
            "ruli_arch": args.ruli_arch,
            "apollo_unlearn_method": args.apollo_unlearn_method,
            "apollo_size_train": args.apollo_size_train,
            "apollo_size_shadow": args.apollo_size_shadow,
            "apollo_num_shadow": args.apollo_num_shadow,
            "apollo_pretrain_epochs": args.apollo_pretrain_epochs,
            "apollo_shadow_epochs": args.apollo_shadow_epochs,
            "apollo_batch_size": args.apollo_batch_size,
            "apollo_lr": args.apollo_lr,
            "apollo_weight_decay": args.apollo_weight_decay,
            "apollo_opt": args.apollo_opt,
            "apollo_forget_perc": args.apollo_forget_perc,
            "apollo_shadow_split": args.apollo_shadow_split,
            "apollo_attack_type": args.apollo_attack_type,
            "apollo_attack_N": args.apollo_attack_N,
            "apollo_atk_lr": args.apollo_atk_lr,
            "apollo_atk_epochs": args.apollo_atk_epochs,
            "apollo_eps": args.apollo_eps,
            "apollo_attack_weights": tuple(args.apollo_attack_weights),
            "unlearningleaks_feature": args.unlearningleaks_feature,
            "unlearningleaks_attack_model": args.unlearningleaks_attack_model,
            "unlearningleaks_num_shadow": args.unlearningleaks_num_shadow,
            "unlearningleaks_sample_budget": args.unlearningleaks_sample_budget,
            "unlearningleaks_target_budget": args.unlearningleaks_target_budget,
            "unlearningleaks_eval_bs": args.unlearningleaks_eval_bs,
            "unlearningleaks_unlearn_bs": args.unlearningleaks_unlearn_bs,
            "unlearningleaks_seed": args.unlearningleaks_seed if args.unlearningleaks_seed is not None else args.seed,
            "unlearningleaks_shadow_opt": args.unlearningleaks_shadow_opt,
            "unlearningleaks_shadow_lr": args.unlearningleaks_shadow_lr,
            "unlearningleaks_shadow_bs": args.unlearningleaks_shadow_bs,
            "unlearningleaks_shadow_pkeep": args.unlearningleaks_shadow_pkeep,
            "attack_verbose": args.attack_verbose,
        }
        timed_stage(
            "attack",
            lambda: run_registered_attacks(
                cfg,
                methods,
                attacks=args.attacks,
                seed=args.seed,
                num_shadow=args.num_shadow,
                num_aug=args.num_aug,
                attack_options=attack_options,
                dry_run=args.dry_run,
            ),
        )

    total_runtime_sec = time.time() - total_start
    run_summary = {
        "dataset": cfg.dataset,
        "net": cfg.net,
        "classes": cfg.n_classes,
        "seed": args.seed,
        "forget_perc": args.forget_perc,
        "methods": [f"{m.name}:{m.para1}:{m.para2}" for m in methods],
        "stages": list(args.stages),
        "attacks": list(args.attacks),
        "num_shadow": args.num_shadow,
        "num_aug": args.num_aug,
        "stage_runtime_sec": stage_runtime_sec,
        "total_runtime_sec": total_runtime_sec,
    }

    summary_path = _run_summary_path(cfg, args.seed)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)

    for stage_name, stage_runtime in stage_runtime_sec.items():
        print(f"[benchmark:runtime] {stage_name}_runtime_sec={stage_runtime:.3f}")
    print(f"[benchmark:runtime] total_runtime_sec={total_runtime_sec:.3f}")
    print(f"[benchmark:runtime] summary_json={summary_path}")

if __name__ == "__main__":
    main()
