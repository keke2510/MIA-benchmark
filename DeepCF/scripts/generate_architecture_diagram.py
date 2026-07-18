#!/usr/bin/env python
"""生成 DeepCF 模型架构流程图 — ASCII-only 版本，避免 Unicode 缺字."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def draw_box(ax, x, y, w, h, text, color, text_color="white", fontsize=10):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor="#333333", linewidth=1.2, alpha=0.95,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, weight="bold")


def draw_sub(ax, x, y, text, fontsize=9, color="white"):
    """副标题文字."""
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color)


def draw_arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.5, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def main():
    fig, ax = plt.subplots(1, 1, figsize=(16, 18))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-8.5, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#FAFBFC")
    ax.set_facecolor("#FAFBFC")

    # ===== 标题 =====
    ax.text(0, 8.0, "DeepCF Model Architecture", ha="center", va="center",
            fontsize=22, weight="bold", color="#1a1a2e")
    ax.text(0, 7.45, "Variational Graph Autoencoder for Urban Consumption Flow Network",
            ha="center", va="center", fontsize=10, color="#666666", style="italic")

    # ===== 图定义横幅 =====
    draw_box(ax, 0, 7.0, 10.5, 0.5,
             "Graph: Nodes = Merchants  |  Edges = Consumption Flow  |  Weights = Amount + Frequency  |  Undirected",
             "#5D6D7E", fontsize=8)

    # ===== 输入层 =====
    in_y = 6.1
    draw_box(ax, -2.5, in_y, 2.2, 0.85, "Features\nX (N x F)", "#2E86AB", fontsize=9)
    draw_box(ax, 0, in_y, 2.2, 0.85, "Adjacency\nA (N x N)", "#2E86AB", fontsize=9)
    draw_box(ax, 2.5, in_y, 2.2, 0.85, "Weights\nW (N x N)", "#2E86AB", fontsize=9)
    ax.text(0, 6.7, "INPUT", ha="center", va="center", fontsize=12, weight="bold", color="#1a1a2e")

    for sx in [-2.5, 0, 2.5]:
        draw_arrow(ax, sx, 5.62, sx, 5.15, "#2E86AB")

    # ===== GCN Encoder =====
    enc = FancyBboxPatch((-4.5, 2.0), 9, 3.05, boxstyle="round,pad=0.15",
                         facecolor="#E8F4F8", edgecolor="#2E86AB", linewidth=2.0, alpha=0.5)
    ax.add_patch(enc)
    ax.text(0, 4.90, "GCN Encoder", ha="center", va="center", fontsize=14, weight="bold", color="#1a5276")

    draw_box(ax, 0, 4.1, 7.2, 0.75,
             "Layer 1: GCNConv(F->128) -> ReLU -> Dropout(0.3)",
             "#3498DB", fontsize=9)
    draw_box(ax, 0, 3.1, 7.2, 0.75,
             "Layer 2: GCNConv(128->128, shared) -> ReLU",
             "#3498DB", fontsize=9)
    draw_arrow(ax, 0, 3.72, 0, 3.5, "#3498DB")

    draw_box(ax, -1.6, 2.35, 2.8, 0.55, "mean: Linear(128->64)", "#2980B9", fontsize=8)
    draw_box(ax, 1.6, 2.35, 2.8, 0.55, "log-var: Linear(128->64)", "#2980B9", fontsize=8)

    # ===== Reparameterization =====
    draw_arrow(ax, -1.6, 2.05, 0, 1.55, "#2E86AB")
    draw_arrow(ax, 1.6, 2.05, 0, 1.55, "#2E86AB")

    draw_box(ax, 0, 1.25, 4.5, 0.55,
             "z = mu + sigma * eps,  eps ~ N(0,I)", "#16A085", fontsize=10)
    ax.text(2.65, 1.25, "[64-dim]", ha="center", va="center", fontsize=8, color="#0D6B5D")
    ax.text(2.65, 0.95, "latent space", ha="center", va="center", fontsize=8, color="#0D6B5D")

    # ===== Three-Head Decoder =====
    dec = FancyBboxPatch((-4.5, -2.85), 9, 3.95, boxstyle="round,pad=0.15",
                         facecolor="#FEF9E7", edgecolor="#F39C12", linewidth=2.0, alpha=0.5)
    ax.add_patch(dec)
    ax.text(0, 0.95, "Three-Head Decoder", ha="center", va="center",
            fontsize=14, weight="bold", color="#7D6608")

    draw_arrow(ax, 0, 0.95, 0, 0.35, "#F39C12")

    # 三个解码头
    hy = -0.05
    draw_box(ax, -3.0, hy, 2.6, 0.85, "Link Head\nsigmoid(zi * zj)", "#E67E22", fontsize=9)
    draw_sub(ax, -3.0, hy - 0.42, "output: A_hat (N x N) in [0,1]", fontsize=7, color="#FFEAA7")

    draw_box(ax, 0, hy, 2.6, 0.85, "Rank Head\nMLP([zi || zj]) -> score", "#E67E22", fontsize=9)
    draw_sub(ax, 0, hy - 0.42, "output: sij (pairwise influence)", fontsize=7, color="#FFEAA7")

    draw_box(ax, 3.0, hy, 2.6, 0.85, "Weight Head\nMLP([zi || zj]) -> Softplus", "#E67E22", fontsize=9)
    draw_sub(ax, 3.0, hy - 0.42, "output: w_hat_ij >= 0", fontsize=7, color="#FFEAA7")

    # ===== 损失函数 =====
    ly = -1.65
    for sx in [-3.0, 0, 3.0]:
        draw_arrow(ax, sx, -0.92, sx, ly + 0.3, "#E67E22")

    draw_box(ax, -3.0, ly, 2.2, 0.5, "BCE(A_hat, A)", "#C0392B", fontsize=9)
    draw_box(ax, 0, ly, 2.2, 0.5, "BPR(s+, s-)", "#C0392B", fontsize=9)
    draw_box(ax, 3.0, ly, 2.2, 0.5, "MSE(w_hat, w)", "#C0392B", fontsize=9)

    # KL Loss
    draw_box(ax, 5.6, 2.35, 1.6, 0.55, "KL Divergence", "#8E44AD", fontsize=10)
    draw_arrow(ax, 2.0, 2.55, 5.0, 2.35, "#8E44AD", lw=1.5)

    # ===== Joint Loss =====
    jl_y = -2.8
    for sx in [-3.0, 0, 3.0]:
        draw_arrow(ax, sx, ly - 0.28, sx, jl_y + 0.35, "#C0392B")
    draw_arrow(ax, 5.6, 2.05, 0, jl_y + 0.35, "#8E44AD", lw=1.2)

    draw_box(ax, 0, jl_y, 7.5, 0.7,
             "L = lambda1*BCE  +  lambda2*BPR  +  lambda3*MSE  +  beta*KL",
             "#1a1a2e", fontsize=11)

    # ===== Training =====
    tr_y = -4.1
    draw_arrow(ax, 0, jl_y - 0.38, 0, tr_y + 0.35, "#333333", lw=2.2)
    draw_box(ax, 0, tr_y, 3.5, 0.55, "End-to-End Training", "#2C3E50", fontsize=10)
    draw_sub(ax, 2.1, tr_y, "Adam + ReduceLROnPlateau", fontsize=8, color="#BDC3C7")

    # ===== Output =====
    out = FancyBboxPatch((-4.5, -7.6), 9, 3.3, boxstyle="round,pad=0.15",
                         facecolor="#EAECEE", edgecolor="#7F8C8D", linewidth=2.0, alpha=0.7)
    ax.add_patch(out)
    ax.text(0, -4.35, "Multi-Dimensional Analysis Outputs", ha="center", va="center",
            fontsize=14, weight="bold", color="#2C3E50")
    draw_arrow(ax, 0, tr_y - 0.3, 0, -4.5, "#333333", lw=2.2)

    outs = [
        ("t-SNE Embeddings", "by Category & Influence"),
        ("Top-K Key Merchants", "Radiation Score Ranking"),
        ("Influence Distribution", "Histogram + Scatter"),
        ("Training Curves", "Loss + AUC + Ablation"),
    ]
    for i, (title, desc) in enumerate(outs):
        ox = -3.0 + i * 2.0
        oy = -5.3
        draw_box(ax, ox, oy, 1.8, 0.82, f"{title}\n{desc}", "#5D6D7E", fontsize=7)

    # ===== 底部信息 =====
    ax.text(0, -7.9, "Hyperparameters: lambda1=1.0  lambda2=0.5  lambda3=0.3  beta=0.001  |  Optimizer: Adam(lr=0.001)",
            ha="center", va="center", fontsize=8, color="#999999", style="italic")

    plt.tight_layout(pad=0.5)
    plt.savefig("docs/architecture.png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print("Architecture diagram saved to docs/architecture.png")


if __name__ == "__main__":
    main()
