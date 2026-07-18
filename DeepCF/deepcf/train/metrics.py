"""评估指标：AUC, AP, Precision@K, Recall@K, NDCG, MAE, RMSE."""

import numpy as np
from typing import List, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error


def compute_auc_ap(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """计算 AUC-ROC 和 Average Precision."""
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return float(auc), float(ap)


def compute_precision_recall_at_k(
    topk_lists: List[List[int]],
    true_items: List[List[int]],
    k: int = 10,
) -> Tuple[float, float]:
    """计算 Precision@K 和 Recall@K."""
    precisions = []
    recalls = []
    for topk, truths in zip(topk_lists, true_items):
        topk_k = topk[:k]
        truths_set = set(truths)
        hits = len(set(topk_k) & truths_set)
        precisions.append(hits / k if k > 0 else 0)
        recalls.append(hits / len(truths_set) if len(truths_set) > 0 else 0)
    return float(np.mean(precisions)), float(np.mean(recalls))


def compute_ndcg_at_k(
    topk_lists: List[List[int]],
    true_items: List[List[int]],
    k: int = 10,
) -> float:
    """计算 NDCG@K."""
    ndcgs = []
    for topk, truths in zip(topk_lists, true_items):
        topk_k = topk[:k]
        truths_set = set(truths)
        dcg = 0.0
        for rank, item in enumerate(topk_k, start=1):
            if item in truths_set:
                dcg += 1.0 / np.log2(rank + 1)
        ideal_hits = min(len(truths_set), k)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs))


def compute_mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """计算 MAE 和 RMSE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return float(mae), float(rmse)
