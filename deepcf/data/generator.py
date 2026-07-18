"""合成消费网络数据生成器."""

import numpy as np
from typing import Dict


def generate_synthetic_data(
    num_nodes: int = 500,
    num_features: int = 16,
    edge_density: float = 0.05,
    community_k: int = 5,
    weight_range: tuple = (0.1, 10.0),
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """生成合成城市消费网络数据.

    生成逻辑:
    1. 使用随机块模型(SBM)生成社区结构
    2. 为每个节点分配商户类别和位置坐标
    3. 按类别偏置调整边连接概率
    4. 按距离衰减和类别关联度为边分配权重

    Returns:
        Dict with keys:
            - features: (N, F) 节点特征矩阵
            - adjacency: (N, N) 邻接矩阵
            - weights: (N, N) 边权重矩阵
            - labels: (N,) 商户类别
            - positions: (N, 2) 商户坐标
    """
    rng = np.random.default_rng(seed)

    # 1. 分配社区归属和类别
    nodes_per_community = num_nodes // community_k
    labels = np.zeros(num_nodes, dtype=np.int64)
    for k in range(community_k):
        start = k * nodes_per_community
        end = start + nodes_per_community if k < community_k - 1 else num_nodes
        labels[start:end] = k

    # 2. 生成节点位置（每个社区聚集在一个中心附近）
    community_centers = rng.uniform(0, 10, size=(community_k, 2))
    positions = np.zeros((num_nodes, 2))
    for i in range(num_nodes):
        c = labels[i]
        positions[i] = community_centers[c] + rng.normal(0, 0.8, size=2)

    # 3. 构建 SBM 块概率矩阵（社区内连接概率 > 社区间）
    block_probs = np.zeros((community_k, community_k))
    for i in range(community_k):
        for j in range(community_k):
            if i == j:
                block_probs[i, j] = edge_density * 3.0  # 社区内高概率
            else:
                block_probs[i, j] = edge_density * 0.3  # 社区间低概率

    # 4. 生成邻接矩阵
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            prob = block_probs[labels[i], labels[j]]
            if rng.random() < prob:
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0

    # 5. 分配边权重（距离衰减 + 类别关联度 + 随机噪声）
    weights = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency[i, j] > 0:
                dist = np.linalg.norm(positions[i] - positions[j])
                dist_factor = np.exp(-0.3 * dist)
                same_category = 1.0 if labels[i] == labels[j] else 0.3
                base_weight = (dist_factor + same_category) / 2
                noise = rng.uniform(0.8, 1.2)
                w = base_weight * (weight_range[1] - weight_range[0]) + weight_range[0]
                w *= noise
                w = np.clip(w, weight_range[0], weight_range[1])
                weights[i, j] = w
                weights[j, i] = w

    # 6. 生成节点特征
    features = np.zeros((num_nodes, num_features), dtype=np.float32)
    # 前 community_k 维: 类别 one-hot
    for i in range(num_nodes):
        features[i, labels[i]] = 1.0
    # 坐标（2维）
    features[:, community_k:community_k + 2] = positions
    # 度（1维）
    degrees = adjacency.sum(axis=1)
    features[:, community_k + 2] = degrees / max(degrees.max(), 1)
    # 平均消费（1维）
    features[:, community_k + 3] = rng.uniform(30, 500, size=num_nodes)
    # 评分（1维）
    features[:, community_k + 4] = rng.uniform(1.0, 5.0, size=num_nodes)
    # 剩余维度：随机噪声
    remaining = num_features - (community_k + 5)
    if remaining > 0:
        features[:, community_k + 5:] = rng.normal(0, 0.1, size=(num_nodes, remaining))

    return {
        "features": features.astype(np.float32),
        "adjacency": adjacency.astype(np.float32),
        "weights": weights.astype(np.float32),
        "labels": labels,
        "positions": positions.astype(np.float32),
    }
