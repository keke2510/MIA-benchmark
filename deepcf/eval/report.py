"""综合评估报告生成."""

import os
import numpy as np
import torch
from typing import Dict, List, Optional
from datetime import datetime

from deepcf.eval.ranking import compute_radiation_scores, top_k_merchants, rank_all_merchants
from deepcf.eval.visualize import plot_tsne_embeddings, plot_influence_distribution, plot_training_curves
from deepcf.train.metrics import compute_auc_ap


def generate_report(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    adjacency: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    history: Dict[str, list],
    test_edges: np.ndarray,
    test_edges_neg: np.ndarray,
    output_dir: str = "outputs",
) -> Dict:
    """生成综合评估报告."""
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    x = x.to(device)
    edge_index = edge_index.to(device)
    model.eval()

    # 获取嵌入
    with torch.no_grad():
        z_tensor = model.get_embeddings(x, edge_index)
        z = z_tensor.cpu().numpy()

    # 链接预测评估
    test_pos_src = torch.tensor(test_edges[:, 0], dtype=torch.long, device=device)
    test_pos_dst = torch.tensor(test_edges[:, 1], dtype=torch.long, device=device)
    with torch.no_grad():
        pos_scores = torch.sigmoid((z_tensor[test_pos_src] * z_tensor[test_pos_dst]).sum(dim=1)).cpu().numpy()

    test_neg_src = torch.tensor(test_edges_neg[:, 0], dtype=torch.long, device=device)
    test_neg_dst = torch.tensor(test_edges_neg[:, 1], dtype=torch.long, device=device)
    with torch.no_grad():
        neg_scores = torch.sigmoid((z_tensor[test_neg_src] * z_tensor[test_neg_dst]).sum(dim=1)).cpu().numpy()

    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([pos_scores, neg_scores])
    auc, ap = compute_auc_ap(y_true, y_score)

    # 辐射力评分与 Top-K
    rank_scores = rank_all_merchants(model, x, edge_index)
    radiation_scores = compute_radiation_scores(z, adjacency, weights, rank_scores)
    top_k = top_k_merchants(radiation_scores, labels, k=20)

    topk_indices = [item["id"] for item in top_k[:10]]

    # 生成可视化
    plot_tsne_embeddings(z, labels, radiation_scores, topk_indices,
                         save_path=os.path.join(output_dir, "tsne_embeddings.png"))
    plot_influence_distribution(radiation_scores, adjacency.sum(axis=1), topk_indices,
                                save_path=os.path.join(output_dir, "influence_distribution.png"))
    plot_training_curves(history,
                         save_path=os.path.join(output_dir, "training_curves.png"))

    report = {
        "timestamp": datetime.now().isoformat(),
        "link_prediction": {"auc": auc, "ap": ap},
        "top_10_merchants": top_k[:10],
        "top_20_merchants": top_k[:20],
        "radiation_stats": {
            "mean": float(radiation_scores.mean()), "std": float(radiation_scores.std()),
            "max": float(radiation_scores.max()), "min": float(radiation_scores.min()),
        },
        "output_files": [
            os.path.join(output_dir, "tsne_embeddings.png"),
            os.path.join(output_dir, "influence_distribution.png"),
            os.path.join(output_dir, "training_curves.png"),
        ],
    }
    return report
