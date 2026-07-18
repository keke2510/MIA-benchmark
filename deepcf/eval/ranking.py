"""Top-K 关键商户识别与影响力排序."""

import numpy as np
import torch
from typing import List, Dict


def compute_radiation_scores(
    z: np.ndarray,
    adjacency: np.ndarray,
    weights: np.ndarray,
    rank_scores: np.ndarray,
    alpha: float = 0.3,
    beta: float = 0.4,
    gamma: float = 0.3,
) -> np.ndarray:
    """综合辐射力评分: Score = alpha*degree_norm + beta*rank_norm + gamma*weight_sum_norm."""
    degrees = adjacency.sum(axis=1)
    degree_norm = degrees / (degrees.max() + 1e-8)
    weight_sums = weights.sum(axis=1)
    weight_norm = weight_sums / (weight_sums.max() + 1e-8)
    rank_norm = (rank_scores - rank_scores.min()) / (rank_scores.max() - rank_scores.min() + 1e-8)
    return alpha * degree_norm + beta * rank_norm + gamma * weight_norm


def top_k_merchants(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> List[Dict]:
    """识别 Top-K 关键商户."""
    indices = np.argsort(scores)[::-1][:k]
    return [{"id": int(idx), "score": float(scores[idx]), "category": int(labels[idx])} for idx in indices]


def rank_all_merchants(model, x, edge_index, edge_weight=None) -> np.ndarray:
    """使用排序头为所有商户对打分并汇总每个商户的总辐射力."""
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    edge_index = edge_index.to(device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    with torch.no_grad():
        z = model.get_embeddings(x, edge_index, edge_weight=edge_weight)

    N = z.shape[0]
    all_scores = np.zeros(N)
    edge_index_np = edge_index.cpu().numpy()

    for i in range(edge_index_np.shape[1]):
        u = edge_index_np[0, i]
        v = edge_index_np[1, i]
        with torch.no_grad():
            score = model.decoder.rank(
                z,
                torch.tensor([u], device=device),
                torch.tensor([v], device=device),
            )
        all_scores[u] += score.cpu().item()
        all_scores[v] += score.cpu().item()

    degrees = np.bincount(edge_index_np.flatten(), minlength=N)
    degrees[degrees == 0] = 1
    all_scores /= degrees
    return all_scores
