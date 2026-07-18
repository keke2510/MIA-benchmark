#!/usr/bin/env python
"""生成 DeepCF 模型架构流程图 — 大幅改进版：更大间距、无重叠、分区清晰."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def box(ax, x, y, w, h, text, color, fontsize=10, tc="white", lw=1.2):
    """绘制圆角矩形."""
    b = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.12",
                       facecolor=color, edgecolor="#333", linewidth=lw, alpha=0.95)
    ax.add_patch(b)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=tc, weight="bold")


def zone(ax, x, y, w, h, color, alpha=0.35, lw=2):
    """绘制背景分区."""
    z = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.2",
                       facecolor=color, edgecolor=color, linewidth=lw, alpha=alpha)
    ax.add_patch(z)


def arrow(ax, x1, y1, x2, y2, color="#555", lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, connectionstyle="arc3,rad=0"))


def label(ax, x, y, text, fontsize=13, color="#1a1a2e", bold=True):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color, weight="bold" if bold else "normal")


def main():
    # 更大画布，更多间距
    fig, ax = plt.subplots(1, 1, figsize=(20, 24))
    ax.set_xlim(-8, 8)
    ax.set_ylim(-11, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    # =====================================================================
    # TITLE
    # =====================================================================
    ax.text(0, 9.6, "DeepCF  Model Architecture", ha="center", fontsize=26, weight="bold", color="#1a1a2e")
    ax.text(0, 9.1, "Variational Graph Autoencoder for Urban Merchant Influence Modeling",
            ha="center", fontsize=11, color="#666", style="italic")
    ax.plot([-6, 6], [8.85, 8.85], color="#ccc", lw=0.8)

    # =====================================================================
    # GRAPH DEFINITION BAR
    # =====================================================================
    box(ax, 0, 8.5, 12, 0.55,
        "Nodes = Merchants   |   Edges = Consumption Flow   |   Weights = Amount + Frequency   |   Undirected Graph",
        "#4A5568", fontsize=9, tc="#E2E8F0", lw=1)

    # =====================================================================
    # ZONE 1: INPUT (light blue)
    # =====================================================================
    zone(ax, 0, 7.3, 11, 1.7, "#BEE3F8", alpha=0.25)
    label(ax, 0, 8.05, "INPUT", fontsize=14, color="#2B6CB0")

    in_y = 7.1
    box(ax, -2.8, in_y, 2.6, 0.95, "Feature Matrix\nX  (N x F)", "#3182CE", fontsize=10)
    box(ax, 0, in_y, 2.6, 0.95, "Adjacency Matrix\nA  (N x N)", "#3182CE", fontsize=10)
    box(ax, 2.8, in_y, 2.6, 0.95, "Weight Matrix\nW  (N x N)", "#3182CE", fontsize=10)

    for sx in [-2.8, 0, 2.8]:
        arrow(ax, sx, 6.6, sx, 5.45, "#3182CE")

    # =====================================================================
    # ZONE 2: GCN ENCODER (blue)
    # =====================================================================
    zone(ax, 0, 4.1, 11, 2.5, "#90CDF4", alpha=0.20)
    label(ax, 0, 5.3, "GCN  ENCODER", fontsize=14, color="#2B6CB0")

    # Layer 1
    box(ax, 0, 4.55, 8.5, 0.8,
        "Layer 1:  GCNConv(F -> 128)  ->  ReLU  ->  Dropout(0.3)",
        "#2B6CB0", fontsize=10)
    # Layer 2
    box(ax, 0, 3.5, 8.5, 0.8,
        "Layer 2:  GCNConv(128 -> 128, shared weights)  ->  ReLU",
        "#2B6CB0", fontsize=10)
    arrow(ax, 0, 4.12, 0, 3.92, "#2B6CB0")

    # mu and logvar projections
    box(ax, -2.0, 2.68, 3.2, 0.65, "mu  =  Linear(128 -> 64)", "#2C5282", fontsize=9)
    box(ax, 2.0, 2.68, 3.2, 0.65, "log-var  =  Linear(128 -> 64)", "#2C5282", fontsize=9)

    # =====================================================================
    # ZONE 3: REPARAMETERIZATION (green)
    # =====================================================================
    arrow(ax, -2.0, 2.33, 0, 1.55, "#2B6CB0")
    arrow(ax, 2.0, 2.33, 0, 1.55, "#2B6CB0")

    zone(ax, 0, 1.0, 7, 1.0, "#A3E4D7", alpha=0.25)
    label(ax, -4.2, 1.0, "REPARAMETERIZATION", fontsize=12, color="#0E6251")
    box(ax, 0, 1.0, 5.5, 0.65, "z  =  mu  +  sigma * epsilon  ,   epsilon ~ N(0, I)",
        "#0E6251", fontsize=10)
    label(ax, 3.3, 1.0, "64-dim latent space", fontsize=9, color="#0E6251", bold=False)

    # =====================================================================
    # ZONE 4: DECODER (orange)
    # =====================================================================
    zone(ax, 0, -1.6, 11, 2.7, "#FBD38D", alpha=0.20)
    label(ax, 0, -0.3, "THREE-HEAD  DECODER", fontsize=14, color="#C05621")

    arrow(ax, 0, 0.65, 0, 0.05, "#C05621")

    hy = -0.65
    # Link Head
    box(ax, -3.0, hy, 2.8, 1.1,
        "Link Head\n\nsigmoid(zi^T zj)\n-> A_hat in [0,1]",
        "#DD6B20", fontsize=9, tc="#FFF5F5")
    # Rank Head
    box(ax, 0, hy, 2.8, 1.1,
        "Rank Head\n\nMLP([zi || zj])\n-> score sij",
        "#DD6B20", fontsize=9, tc="#FFF5F5")
    # Weight Head
    box(ax, 3.0, hy, 2.8, 1.1,
        "Weight Head\n\nMLP([zi || zj]) + Softplus\n-> w_hat >= 0",
        "#DD6B20", fontsize=9, tc="#FFF5F5")

    # =====================================================================
    # ZONE 5: LOSS FUNCTIONS (red + purple)
    # =====================================================================
    zone(ax, 0, -3.8, 11, 2.2, "#FEB2B2", alpha=0.18)
    label(ax, 0, -2.75, "LOSS  FUNCTIONS", fontsize=14, color="#9B2C2C")

    ly = -3.45
    for sx in [-3.0, 0, 3.0]:
        arrow(ax, sx, -1.22, sx, ly + 0.32, "#DD6B20")

    box(ax, -3.0, ly, 2.5, 0.6, "BCE(A_hat, A)", "#C53030", fontsize=10)
    box(ax, 0, ly, 2.5, 0.6, "BPR(s_pos, s_neg)", "#C53030", fontsize=10)
    box(ax, 3.0, ly, 2.5, 0.6, "MSE(w_hat, w)", "#C53030", fontsize=10)

    # KL divergence on the right, connected from encoder
    box(ax, 6.0, 2.68, 2.0, 0.65, "KL Divergence", "#805AD5", fontsize=10, tc="#FAF5FF")
    # arrow from mu/logvar area to KL
    ax.annotate("", xy=(5.2, 2.68), xytext=(3.4, 2.68),
                arrowprops=dict(arrowstyle="->", color="#805AD5", lw=1.8, linestyle="dashed"))
    ax.annotate("", xy=(6.0, 2.33), xytext=(6.0, -3.5),
                arrowprops=dict(arrowstyle="->", color="#805AD5", lw=1.5, linestyle="dashed"))

    # =====================================================================
    # JOINT LOSS
    # =====================================================================
    for sx in [-3.0, 0, 3.0]:
        arrow(ax, sx, ly - 0.32, sx, -4.9, "#C53030")

    # Plus signs
    for px in [-1.5, 1.5]:
        label(ax, px, -4.55, "+", fontsize=16, color="#666")

    box(ax, 0, -5.35, 9, 0.7,
        "L  =  lambda1 * BCE  +  lambda2 * BPR  +  lambda3 * MSE  +  beta * KL",
        "#1A202C", fontsize=11, tc="#E2E8F0")

    # =====================================================================
    # TRAINING
    # =====================================================================
    arrow(ax, 0, -5.72, 0, -6.45, "#1A202C", lw=2.2)
    zone(ax, 0, -6.8, 10, 0.9, "#CBD5E0", alpha=0.25)
    box(ax, 0, -6.8, 6, 0.65, "End-to-End Training  (Adam + ReduceLROnPlateau)",
        "#2D3748", fontsize=10, tc="#E2E8F0")

    # =====================================================================
    # ZONE 6: OUTPUTS
    # =====================================================================
    arrow(ax, 0, -7.15, 0, -8.1, "#2D3748", lw=2.2)
    zone(ax, 0, -9.3, 12, 2.0, "#A0AEC0", alpha=0.18)
    label(ax, 0, -8.35, "MULTI-DIMENSIONAL  ANALYSIS  OUTPUTS", fontsize=14, color="#2D3748")

    oy = -9.1
    outputs = [
        ("t-SNE Embedding\nVisualization", "by Category\n& Influence Score"),
        ("Top-K Key\nMerchant Ranking", "Radiation Score\nDescending Order"),
        ("Influence\nDistribution Plots", "Histogram + Degree\nvs. Score Scatter"),
        ("Training Curves\n& Ablation Study", "Loss / AUC per\nEpoch Comparison"),
    ]
    for i, (title, desc) in enumerate(outputs):
        ox = -4.5 + i * 3.0
        box(ax, ox, oy, 2.6, 1.2, title, "#4A5568", fontsize=9, tc="#F7FAFC")
        ax.text(ox, oy - 0.58, desc, ha="center", va="center", fontsize=7, color="#CBD5E0")

    # =====================================================================
    # FOOTER
    # =====================================================================
    ax.text(0, -10.5, "Default Hyperparameters:  lambda1=1.0   lambda2=0.5   lambda3=0.3   beta=0.001   |   Optimizer: Adam(lr=0.001, weight_decay=5e-4)",
            ha="center", fontsize=8, color="#A0AEC0", style="italic")

    # =====================================================================
    # SAVE
    # =====================================================================
    plt.tight_layout(pad=0.3)
    plt.savefig("docs/architecture.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print("Architecture diagram saved to docs/architecture.png")


if __name__ == "__main__":
    main()
