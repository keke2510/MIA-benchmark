"""图数据处理工具函数."""

import numpy as np
from typing import Dict, Tuple, List


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """对称归一化邻接矩阵: D^(-1/2) * (A+I) * D^(-1/2).

    Args:
        adjacency: (N, N) 邻接矩阵

    Returns:
        归一化后的邻接矩阵
    """
    A = adjacency + np.eye(adjacency.shape[0], dtype=adjacency.dtype)
    D = np.array(A.sum(axis=1)).flatten()
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(D_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def split_edges(
    adjacency: np.ndarray,
    train_ratio: float = 0.85,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """将边划分为训练/验证/测试集.

    Args:
        adjacency: (N, N) 邻接矩阵
        train_ratio: 训练边比例
        val_ratio: 验证边比例
        seed: 随机种子

    Returns:
        Dict with:
            - train_adj: 训练邻接矩阵
            - val_edges: (M_val, 2) 验证边列表
            - test_edges: (M_test, 2) 测试边列表
            - val_edges_neg: (M_val, 2) 验证负样本
            - test_edges_neg: (M_test, 2) 测试负样本
    """
    rng = np.random.default_rng(seed)
    N = adjacency.shape[0]

    # 提取所有无向边（上三角）
    edges = []
    non_edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if adjacency[i, j] > 0:
                edges.append((i, j))
            else:
                non_edges.append((i, j))

    edges = np.array(edges)
    non_edges = np.array(non_edges)

    # 随机打乱
    perm = rng.permutation(len(edges))
    edges = edges[perm]

    n_total = len(edges)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_edges = edges[:n_train]
    val_edges = edges[n_train:n_train + n_val]
    test_edges = edges[n_train + n_val:]

    # 负采样（与验证/测试边数量一致）
    n_val_neg = min(len(val_edges), len(non_edges))
    n_test_neg = min(len(test_edges), len(non_edges))
    neg_perm = rng.permutation(len(non_edges))
    val_edges_neg = non_edges[neg_perm[:n_val_neg]]
    test_edges_neg = non_edges[neg_perm[n_val_neg:n_val_neg + n_test_neg]]

    # 构建训练邻接矩阵
    train_adj = np.zeros_like(adjacency)
    for i, j in train_edges:
        train_adj[i, j] = adjacency[i, j]
        train_adj[j, i] = adjacency[j, i]

    return {
        "train_adj": train_adj,
        "val_edges": val_edges,
        "test_edges": test_edges,
        "val_edges_neg": val_edges_neg,
        "test_edges_neg": test_edges_neg,
    }


def scale_features(features: np.ndarray) -> np.ndarray:
    """StandardScaler 特征标准化.

    Args:
        features: (N, F) 特征矩阵

    Returns:
        标准化后的特征矩阵
    """
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (features - mean) / std
