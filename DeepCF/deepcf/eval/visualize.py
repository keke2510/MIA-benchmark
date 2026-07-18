"""t-SNE 嵌入可视化、影响力分布图、模型分析."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Optional, List, Dict
import os


def plot_tsne_embeddings(
    z: np.ndarray, labels: np.ndarray, scores: Optional[np.ndarray] = None,
    top_k_indices: Optional[List[int]] = None,
    save_path: str = "tsne_embeddings.png", figsize: tuple = (12, 5),
):
    """t-SNE 嵌入可视化（类别着色 + 影响力着色）."""
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, z.shape[0] - 1))
    z_2d = tsne.fit_transform(z)

    n_panels = 2 if scores is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    scatter1 = axes[0].scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap="tab10", s=30, alpha=0.7)
    if top_k_indices:
        axes[0].scatter(z_2d[top_k_indices, 0], z_2d[top_k_indices, 1],
                       s=120, edgecolors="red", facecolors="none", linewidths=2, marker="o")
    axes[0].set_title("t-SNE by Category", fontsize=13)
    axes[0].set_xlabel("Dim 1"); axes[0].set_ylabel("Dim 2")
    plt.colorbar(scatter1, ax=axes[0], label="Category")

    if scores is not None and len(axes) > 1:
        scatter2 = axes[1].scatter(z_2d[:, 0], z_2d[:, 1], c=scores, cmap="YlOrRd", s=30, alpha=0.7)
        if top_k_indices:
            axes[1].scatter(z_2d[top_k_indices, 0], z_2d[top_k_indices, 1],
                           s=120, edgecolors="blue", facecolors="none", linewidths=2, marker="o")
        axes[1].set_title("t-SNE by Radiation Score", fontsize=13)
        axes[1].set_xlabel("Dim 1"); axes[1].set_ylabel("Dim 2")
        plt.colorbar(scatter2, ax=axes[1], label="Score")

    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


def plot_influence_distribution(
    scores: np.ndarray, degrees: np.ndarray,
    top_k_indices: Optional[List[int]] = None,
    save_path: str = "influence_distribution.png", figsize: tuple = (14, 10),
):
    """影响力分布图：直方图 + 度-影响力散点图 + Top-N柱状图."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    axes[0, 0].hist(scores, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0, 0].axvline(x=np.median(scores), color="red", linestyle="--",
                       label=f"Median={np.median(scores):.3f}")
    axes[0, 0].set_title("Radiation Score Distribution")
    axes[0, 0].set_xlabel("Score"); axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()

    axes[0, 1].scatter(degrees, scores, alpha=0.5, s=20)
    if top_k_indices:
        axes[0, 1].scatter(degrees[top_k_indices], scores[top_k_indices],
                          color="red", s=80, marker="*", label="Top-K", edgecolors="black")
    axes[0, 1].set_title("Degree vs. Radiation Score")
    axes[0, 1].set_xlabel("Degree"); axes[0, 1].set_ylabel("Radiation Score")
    axes[0, 1].legend()

    axes[1, 0].scatter(degrees, scores, alpha=0.5, s=20)
    if top_k_indices:
        axes[1, 0].scatter(degrees[top_k_indices], scores[top_k_indices],
                          color="red", s=80, marker="*", label="Top-K", edgecolors="black")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title("Degree (log) vs. Radiation Score")
    axes[1, 0].set_xlabel("Degree (log scale)"); axes[1, 0].set_ylabel("Radiation Score")
    axes[1, 0].legend()

    N = min(20, len(scores))
    top_indices = np.argsort(scores)[::-1][:N]
    top_scores = scores[top_indices]
    colors = plt.cm.YlOrRd(top_scores / max(top_scores))
    axes[1, 1].bar(range(N), top_scores, color=colors, edgecolor="black", linewidth=0.5)
    axes[1, 1].set_title(f"Top-{N} Radiation Scores")
    axes[1, 1].set_xlabel("Rank"); axes[1, 1].set_ylabel("Score")

    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


def plot_training_curves(history: Dict[str, list], save_path: str = "training_curves.png"):
    """训练曲线图."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["epoch"], history["train_loss"], color="blue", linewidth=1.5)
    ax1.set_title("Training Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax2.plot(history["epoch"], history["lr"], color="green", linewidth=1.5)
    ax2.set_title("Learning Rate"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("LR")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
