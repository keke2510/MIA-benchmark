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


def test_normalize_adjacency():
    """测试邻接矩阵归一化."""
    from deepcf.data.utils import normalize_adjacency
    import numpy as np

    A = np.array([[0, 1, 1],
                  [1, 0, 0],
                  [1, 0, 0]], dtype=np.float32)
    A_norm = normalize_adjacency(A)

    assert A_norm.shape == (3, 3)
    # 对称归一化后矩阵仍对称
    assert np.allclose(A_norm, A_norm.T)
    # 所有条目应在 [0, 1] 范围内
    assert np.all(A_norm >= 0) and np.all(A_norm <= 1.0)


def test_split_edges():
    """测试边划分比例."""
    from deepcf.data.utils import split_edges
    import numpy as np

    # 构造简单无向图
    A = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)

    splits = split_edges(A, train_ratio=0.7, val_ratio=0.15, seed=42)

    assert "train_adj" in splits
    assert "val_edges" in splits
    assert "test_edges" in splits
    # 训练邻接矩阵应包含原始边的一部分
    assert splits["train_adj"].sum() > 0
    assert splits["train_adj"].sum() < A.sum()


def test_bpr_dataset():
    """测试 BPR 数据集采样."""
    from deepcf.data.dataset import BPRDataset
    import numpy as np

    A = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.float32)
    W = A * 2.0

    dataset = BPRDataset(A, W, num_negatives=1)

    assert len(dataset) > 0
    u, i_pos, i_neg, w = dataset[0]
    # 正样本应有边
    assert A[u, i_pos] > 0
    # 负样本应无边
    assert A[u, i_neg] == 0
    # 权重应匹配
    assert w == W[u, i_pos]
