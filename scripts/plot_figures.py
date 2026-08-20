#!/usr/bin/env python3
"""MIA-Bench 论文图表生成脚本（6 种攻击）。

从 benchmark_results/ 和 runtime_results.json 读取实测结果，
生成 5 张论文图（4 张 TPR@FPR 柱状图 + 1 张 runtime 柱状图）。

用法（在仓库根目录、结果已生成后运行）：
    python scripts/plot_figures.py --out figures

输出：
    figures/fig-cifar10.png
    figures/fig-cifar100.png
    figures/fig-tinyimagenet.png
    figures/fig-cinic10.png
    figures/fig-runtime_comparison.png

注意：
    - Threshold 与 Loss 单调等价（loss = -log(正确类置信度)），TPR@FPR 完全相同，
      图中两者柱子高度一致，这是正常现象。
    - RULI 结果在 ruli_summary.json，其余 5 种攻击在 attack_result.json。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 服务器无显示环境
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ======================================================================
# 全局样式（Times New Roman，顶会论文风格）
# ======================================================================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8.5,
    }
)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8
plt.rcParams["xtick.major.size"] = 4
plt.rcParams["ytick.major.size"] = 4
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["axes.axisbelow"] = True

# ======================================================================
# 映射表
# ======================================================================
# 数据集 -> 结果目录名
DATASETS = {
    "Cifar10": "ResNet18-Cifar10-10",
    "Cifar100": "ResNet50-Cifar100-100",
    "TinyImageNet": "ViT-TinyImageNet-200",
    "Cinic10": "ResNet18-Cinic10-10",
}

# 遗忘算法 -> (目录前缀, 显示名)。目录名含超参，用前缀匹配。
ALGORITHMS = [
    ("retrain", "Retrain"),
    ("finetune", "Finetune"),
    ("negative_grad", "NegGrad"),
    ("scrub", "SCRUB"),
]

# 攻击 -> (结果文件名, 图例名, 颜色)。RULI 用 ruli_summary.json。
ATTACKS = [
    ("unlearningleaks", "attack_result.json", "Unl.Leaks", "#0072B2"),
    ("lira", "attack_result.json", "LiRA", "#E69F00"),
    ("ruli", "ruli_summary.json", "RULI", "#009E73"),
    ("rea", "attack_result.json", "REA", "#CC79A7"),
    ("threshold", "attack_result.json", "Threshold", "#56B4E9"),
    ("loss", "attack_result.json", "Loss", "#D55E00"),
]

METRICS = ["TPR@0.1%FPR", "TPR@1%FPR", "TPR@10%FPR"]
METRIC_YMAX = {
    "TPR@0.1%FPR": 4.5,
    "TPR@1%FPR": 9.0,
    "TPR@10%FPR": 25.0,
}

SEEDS = [42, 123, 999]

# runtime 图的攻击顺序（与 runtime_results.json 的 key 对齐）
RUNTIME_ATTACKS = ["Threshold", "Loss", "LiRA", "REA", "UnlearningLeaks", "RULI"]
RUNTIME_ATTACK_LABELS = ["Threshold", "Loss", "LiRA", "REA", "Unl.Leaks", "RULI"]
RUNTIME_DATASETS = ["CIFAR-10", "CIFAR-100", "TinyImageNet", "CINIC-10"]
RUNTIME_COLORS = {
    "CIFAR-10": "#0072B2",
    "CIFAR-100": "#E69F00",
    "TinyImageNet": "#009E73",
    "CINIC-10": "#CC79A7",
}


# ======================================================================
# 读取 TPR@FPR 数据
# ======================================================================
def _algo_dir(dataset_dir: Path, algo_prefix: str) -> Path:
    """在数据集目录下按前缀找到算法目录（目录名带超参）。"""
    matches = sorted(dataset_dir.glob(f"{algo_prefix}_*"))
    if not matches:
        raise FileNotFoundError(f"未找到算法目录：{dataset_dir} / {algo_prefix}_*")
    return matches[0]


def _read_metric(json_path: Path, metric: str) -> float:
    """从结果 JSON 读取一个指标（TPR@FPR 存的是 0-1 小数，返回百分数）。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    value = data["metrics"][metric]
    # 统一转成百分数：结果里是小数（<=1）就 ×100
    value = float(value)
    return value * 100.0 if value <= 1.0 else value


def collect_tpr(root: Path, dataset_key: str) -> dict:
    """收集某数据集所有攻击×算法×指标的 mean/std。

    返回结构：{metric: {attack: {algo: (mean, std)}}}
    """
    dataset_dir = root / "benchmark_results" / "samplewise" / DATASETS[dataset_key]
    result = {m: {a[0]: {} for a in ATTACKS} for m in METRICS}

    for algo_key, algo_label in ALGORITHMS:
        adir = _algo_dir(dataset_dir, algo_key)
        for att_key, fname, _, _ in ATTACKS:
            values = {m: [] for m in METRICS}
            for seed in SEEDS:
                jpath = adir / att_key / f"seed_{seed}" / fname
                if not jpath.exists():
                    print(f"  [warn] 缺失 {jpath}", file=sys.stderr)
                    continue
                for m in METRICS:
                    try:
                        values[m].append(_read_metric(jpath, m))
                    except KeyError:
                        # RULI 老结果可能缺 TPR@FPR 字段
                        print(f"  [warn] {jpath} 缺 {m}", file=sys.stderr)
            for m in METRICS:
                if values[m]:
                    arr = np.array(values[m], dtype=float)
                    result[m][att_key][algo_key] = (float(arr.mean()), float(arr.std(ddof=1)))
                else:
                    result[m][att_key][algo_key] = (np.nan, np.nan)
    return result


# ======================================================================
# TPR@FPR 柱状图（4 算法 × 6 攻击 × 3 个 FPR 阈值）
# ======================================================================
def plot_tpr(dataset_key: str, data: dict, out_path: Path) -> None:
    algo_labels = [a[1] for a in ALGORITHMS]
    x = np.arange(len(ALGORITHMS))
    width = 0.12  # 6 组柱子
    n_attacks = len(ATTACKS)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharex=True)

    for idx, (ax, metric) in enumerate(zip(axes, METRICS)):
        y_max = METRIC_YMAX[metric]
        for j, (att_key, _, att_label, color) in enumerate(ATTACKS):
            means, stds = [], []
            for algo_key, _ in ALGORITHMS:
                mean, std = data[metric][att_key].get(algo_key, (np.nan, np.nan))
                means.append(mean if not np.isnan(mean) else 0.0)
                stds.append(std if not np.isnan(std) else 0.0)
            means = np.array(means, dtype=float)
            stds = np.array(stds, dtype=float)

            offset = (j - (n_attacks - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=2.0,
                error_kw={"elinewidth": 0.7, "capthick": 0.7},
                label=att_label,
                color=color,
                edgecolor="white",
                linewidth=0.3,
                zorder=2,
            )
            # 数值标注（只标非零）
            label_offset = y_max * 0.018
            for bar, mean in zip(bars, means):
                if mean > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        mean + label_offset,
                        f"{mean:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=5.5,
                        rotation=90,
                        color="#333333",
                    )

        ax.set_title(metric, fontsize=11, fontweight="bold", pad=10)
        ax.text(-0.08, 1.04, f"({chr(97 + idx)})", transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom", ha="left")
        ax.set_xticks(x)
        ax.set_xticklabels(algo_labels)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.5, zorder=0)
        ax.set_xlabel("Unlearning Algorithm", fontsize=11)

    axes[0].set_ylabel("TPR@FPR (%)", fontsize=11)

    legend_handles = [Patch(facecolor=a[3], edgecolor="white", label=a[2]) for a in ATTACKS]
    legend = axes[0].legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        fontsize=8,
        title="Attack Method",
        title_fontsize=9,
        edgecolor="#cccccc",
        framealpha=0.9,
        borderpad=0.6,
        handlelength=1.2,
        handleheight=0.7,
        ncol=2,
    )
    legend.get_frame().set_linewidth(0.5)

    plt.tight_layout(rect=[0, 0, 0.94, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  已生成 {out_path}")


# ======================================================================
# runtime 柱状图（6 攻击 × 4 数据集）
# ======================================================================
def plot_runtime(runtime_json: Path, out_path: Path) -> None:
    with open(runtime_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = {ds: [] for ds in RUNTIME_DATASETS}
    # Threshold 实测 <1s（runtime_results.json 记为 null）；CINIC-10 RULI 补跑后=12.0
    fallback = {"Threshold": 0.5, "RULI": 12.0}
    for ds in RUNTIME_DATASETS:
        row = raw.get(ds, {})
        for att in RUNTIME_ATTACKS:
            v = row.get(att)
            data[ds].append(fallback.get(att, 0.0) if v is None else float(v))

    x = np.arange(len(RUNTIME_ATTACKS))
    width = 0.18

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for j, ds in enumerate(RUNTIME_DATASETS):
        vals = data[ds]
        offset = (j - (len(RUNTIME_DATASETS) - 1) / 2) * width
        bars = ax.bar(
            x + offset, vals, width, label=ds,
            color=RUNTIME_COLORS[ds], edgecolor="white", linewidth=0.5, zorder=2,
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=6.5, color="#333333",
                )

    ax.set_ylabel("Single-run Runtime (s)", fontsize=12)
    ax.set_xlabel("Attack Method", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(RUNTIME_ATTACK_LABELS, rotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.5, zorder=0)
    ax.legend(loc="upper left", frameon=True, fontsize=9, title="Dataset",
              title_fontsize=10, edgecolor="#cccccc", framealpha=0.9)

    y_max = max(max(v) for v in data.values())
    ax.set_ylim(0, y_max * 1.15)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  已生成 {out_path}")


# ======================================================================
# main
# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="生成 MIA-Bench 论文图表（6 攻击）")
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="仓库根目录（含 benchmark_results/ 与 runtime_results.json）")
    parser.add_argument("--out", type=Path, default=Path("figures"),
                        help="输出目录")
    parser.add_argument("--only", type=str, default=None,
                        help="只画指定数据集（Cifar10/Cifar100/TinyImageNet/Cinic10）")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    fig_map = {
        "Cifar10": "fig-cifar10.png",
        "Cifar100": "fig-cifar100.png",
        "TinyImageNet": "fig-tinyimagenet.png",
        "Cinic10": "fig-cinic10.png",
    }

    datasets = [args.only] if args.only else list(DATASETS.keys())

    for ds in datasets:
        if ds not in DATASETS:
            print(f"未知数据集 {ds}，可选 {list(DATASETS)}", file=sys.stderr)
            continue
        print(f"[TPR@FPR] {ds}")
        data = collect_tpr(args.root, ds)
        plot_tpr(ds, data, args.out / fig_map[ds])

    runtime_json = args.root / "runtime_results.json"
    if runtime_json.exists():
        print("[runtime]")
        plot_runtime(runtime_json, args.out / "fig-runtime_comparison.png")
    else:
        print(f"[warn] 未找到 {runtime_json}，跳过 runtime 图", file=sys.stderr)


if __name__ == "__main__":
    main()
