"""数据模块测试."""

import numpy as np
from deepcf.data.generator import generate_synthetic_data


def test_generate_synthetic_data_shape():
    """测试合成数据的形状正确."""
    data = generate_synthetic_data(num_nodes=100, num_features=8,
                                    edge_density=0.05, community_k=3,
                                    seed=42)
    # 特征矩阵
    assert data["features"].shape == (100, 8)
    # 邻接矩阵
    assert data["adjacency"].shape == (100, 100)
    # 权重矩阵
    assert data["weights"].shape == (100, 100)
    # 商户类别
    assert data["labels"].shape == (100,)


def test_adjacency_symmetric():
    """测试邻接矩阵对称（无向图）."""
    data = generate_synthetic_data(num_nodes=50, seed=42)
    A = data["adjacency"]
    assert np.allclose(A, A.T)


def test_adjacency_no_self_loops():
    """测试邻接矩阵无自环."""
    data = generate_synthetic_data(num_nodes=50, seed=42)
    assert np.all(np.diag(data["adjacency"]) == 0)


def test_weights_match_adjacency():
    """测试权重与边对应：无边处权重为0，有边处权重大于0."""
    data = generate_synthetic_data(num_nodes=50, seed=42)
    A = data["adjacency"]
    W = data["weights"]
    # 有边处权重大于0
    edge_mask = A > 0
    assert np.all(W[edge_mask] > 0)
    # 无边处权重为0
    assert np.all(W[~edge_mask] == 0)


def test_labels_range():
    """测试类别标签在合理范围内."""
    data = generate_synthetic_data(num_nodes=100, community_k=5, seed=42)
    labels = data["labels"]
    assert labels.min() >= 0
    assert labels.max() <= 4  # community_k=5 → 0..4
