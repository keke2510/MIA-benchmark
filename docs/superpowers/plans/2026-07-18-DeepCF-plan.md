# DeepCF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 VGAE + 双层GCN + 三头解码器的 DeepCF Python 包，包含合成数据生成、多任务训练、评估指标和论文级可视化输出。

**Architecture:** 按 data → model → train → eval 单向依赖的模块化设计。GCN编码器将商户特征和邻接矩阵映射到64维隐空间，三个独立解码头并行输出链接预测、影响力排序和边权重回归结果。BPR损失驱动影响力排序，BCE驱动图重构，MSE驱动权重回归，KL散度正则化隐空间。

**Tech Stack:** PyTorch 2.x + PyTorch Geometric 2.3 + NumPy + Scikit-learn + Matplotlib/Seaborn

---

## 文件总览

| 文件 | 职责 | 依赖 |
|------|------|------|
| `deepcf/config.py` | 全局配置数据类 | 无 |
| `deepcf/data/generator.py` | 合成消费网络生成器 | numpy, scipy |
| `deepcf/data/dataset.py` | PyTorch Dataset + BPR采样 | torch, numpy |
| `deepcf/data/utils.py` | 图标准化、特征归一化、边划分 | numpy, scipy |
| `deepcf/model/encoder.py` | 双层GCN编码器(→ μ, σ) | torch, torch_geometric |
| `deepcf/model/decoder.py` | 三头解码器(Link/Rank/Weight) | torch |
| `deepcf/model/vgae.py` | VGAE主模型 | encoder, decoder, losses |
| `deepcf/model/losses.py` | 多任务损失(BCE+BPR+MSE+KL) | torch |
| `deepcf/train/trainer.py` | 训练循环+早停+checkpoint | torch, vgae |
| `deepcf/train/metrics.py` | 评估指标(AUC/P@K/R@K/NDCG) | numpy, sklearn |
| `deepcf/eval/ranking.py` | Top-K商户识别+影响力排序 | torch, numpy |
| `deepcf/eval/visualize.py` | t-SNE+影响力分布+模型分析 | matplotlib, sklearn |
| `deepcf/eval/report.py` | 综合评估报告生成 | ranking, visualize, metrics |
| `scripts/train.py` | CLI训练入口 | deepcf |
| `scripts/evaluate.py` | CLI评估入口 | deepcf |
| `scripts/visualize.py` | CLI可视化入口 | deepcf |

---

### Task 1: 项目脚手架与配置

**Files:**
- Create: `deepcf/__init__.py`
- Create: `deepcf/config.py`
- Create: `deepcf/data/__init__.py`
- Create: `deepcf/model/__init__.py`
- Create: `deepcf/train/__init__.py`
- Create: `deepcf/eval/__init__.py`
- Create: `scripts/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p deepcf/data deepcf/model deepcf/train deepcf/eval scripts tests
```

- [ ] **Step 2: 创建 deepcf/__init__.py**

```python
"""DeepCF: 基于图表示学习的城市消费网络建模与关键商户消费辐射最大化研究."""

__version__ = "0.1.0"
```

- [ ] **Step 3: 创建各子包 __init__.py**

```bash
touch deepcf/data/__init__.py deepcf/model/__init__.py deepcf/train/__init__.py deepcf/eval/__init__.py tests/__init__.py
```

- [ ] **Step 4: 创建 deepcf/config.py**

```python
"""DeepCF 全局配置."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    """数据配置."""

    num_nodes: int = 500
    num_features: int = 16
    edge_density: float = 0.05
    community_k: int = 5
    weight_range: tuple = (0.1, 10.0)
    train_ratio: float = 0.85
    val_ratio: float = 0.05


@dataclass
class ModelConfig:
    """模型配置."""

    input_dim: int = 16
    hidden_dim: int = 128
    latent_dim: int = 64
    dropout: float = 0.3
    mlp_hidden: int = 64


@dataclass
class LossConfig:
    """损失函数配置."""

    lambda_link: float = 1.0
    lambda_rank: float = 0.5
    lambda_weight: float = 0.3
    beta_kl: float = 0.001


@dataclass
class TrainConfig:
    """训练配置."""

    lr: float = 0.001
    weight_decay: float = 5e-4
    epochs: int = 500
    early_stop_patience: int = 50
    lr_patience: int = 20
    lr_factor: float = 0.5
    batch_size: int = 128
    checkpoint_interval: int = 50


@dataclass
class DeepCFConfig:
    """DeepCF 总配置."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "outputs"

    def __post_init__(self):
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
```

- [ ] **Step 5: 安装依赖并验证导入**

```bash
pip install torch torch-geometric numpy scipy pandas scikit-learn matplotlib seaborn tqdm jupyter
```

Run: `python -c "from deepcf.config import DeepCFConfig; cfg = DeepCFConfig(); print(cfg)"`
Expected: 打印默认配置，无导入错误。

---

### Task 2: 合成数据生成器

**Files:**
- Create: `deepcf/data/generator.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: 编写测试**

在 `tests/test_data.py`:

```python
"""数据模块测试."""

import numpy as np
import torch
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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_data.py -v
```

Expected: 所有测试失败，`generate_synthetic_data` 未定义。

- [ ] **Step 3: 实现 deepcf/data/generator.py**

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_data.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add deepcf/ deepcf/data/ deepcf/model/ deepcf/train/ deepcf/eval/ tests/ scripts/
git add deepcf/config.py deepcf/data/generator.py tests/test_data.py
git commit -m "feat: add project scaffolding, config, and synthetic data generator

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: 图工具函数与数据集

**Files:**
- Create: `deepcf/data/utils.py`
- Create: `deepcf/data/dataset.py`

- [ ] **Step 1: 扩展 tests/test_data.py 添加工具函数测试**

```python
def test_normalize_adjacency():
    """测试邻接矩阵归一化."""
    from deepcf.data.utils import normalize_adjacency
    import numpy as np

    A = np.array([[0, 1, 1],
                  [1, 0, 0],
                  [1, 0, 0]], dtype=np.float32)
    A_norm = normalize_adjacency(A)

    assert A_norm.shape == (3, 3)
    # 归一化后每行和应为1（加自环后）
    row_sums = A_norm.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


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
```

- [ ] **Step 2: 实现 deepcf/data/utils.py**

```python
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
```

- [ ] **Step 3: 实现 deepcf/data/dataset.py**

```python
"""PyTorch Dataset 与 BPR 采样."""

import numpy as np
import torch
from torch.utils.data import Dataset


class BPRDataset(Dataset):
    """BPR 成对排序数据集.

    对每个节点采样 1 个正样本（有边邻居）和 num_negatives 个负样本（无边节点），
    构成 (user, pos_item, neg_item, weight) 训练三元组。
    """

    def __init__(
        self,
        adjacency: np.ndarray,
        weights: np.ndarray,
        num_negatives: int = 1,
        seed: int = 42,
    ):
        self.adjacency = adjacency
        self.weights = weights
        self.num_nodes = adjacency.shape[0]
        self.num_negatives = num_negatives
        self.rng = np.random.default_rng(seed)

        # 为每个节点建立正样本列表
        self.pos_neighbors = {}
        all_nodes = set(range(self.num_nodes))
        for i in range(self.num_nodes):
            neighbors = np.where(adjacency[i] > 0)[0]
            self.pos_neighbors[i] = neighbors.tolist()

    def __len__(self) -> int:
        return self.num_nodes

    def __getitem__(self, idx: int) -> tuple:
        u = idx
        pos_list = self.pos_neighbors.get(u, [])
        if len(pos_list) == 0:
            # 孤立节点：回退到最近的非孤立节点
            for dist in range(1, self.num_nodes):
                fallback = (u + dist) % self.num_nodes
                if len(self.pos_neighbors.get(fallback, [])) > 0:
                    u = fallback
                    pos_list = self.pos_neighbors[u]
                    break
            else:
                # 全图无边，回退到自身
                return (torch.tensor(u), torch.tensor(u),
                        torch.tensor(u), torch.tensor(0.0))

        i_pos = int(self.rng.choice(pos_list))
        w = float(self.weights[u, i_pos])

        # 负采样
        i_neg = int(self.rng.integers(0, self.num_nodes))
        while self.adjacency[u, i_neg] > 0 or i_neg == i_pos:
            i_neg = int(self.rng.integers(0, self.num_nodes))

        return (torch.tensor(u), torch.tensor(i_pos),
                torch.tensor(i_neg), torch.tensor(w))


def collate_bpr_batch(batch: list) -> tuple:
    """整理 BPR 批次.

    Args:
        batch: list of (u, i_pos, i_neg, w) tuples

    Returns:
        (users, pos_items, neg_items, weights) 张量
    """
    users, pos_items, neg_items, weights = zip(*batch)
    return (
        torch.stack(users),
        torch.stack(pos_items),
        torch.stack(neg_items),
        torch.stack(weights),
    )
```

- [ ] **Step 4: 运行测试**

Run: `cd tests && python -c "
from test_data import test_normalize_adjacency, test_split_edges, test_bpr_dataset
test_normalize_adjacency()
test_split_edges()
test_bpr_dataset()
print('All new tests passed')
"`

Expected: "All new tests passed"

- [ ] **Step 5: Commit**

```bash
git add deepcf/data/utils.py deepcf/data/dataset.py tests/test_data.py
git commit -m "feat: add graph utils, edge splitting, and BPR dataset

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: GCN 编码器

**Files:**
- Create: `deepcf/model/encoder.py`
- Create: `tests/test_encoder.py`

- [ ] **Step 1: 编写测试**

在 `tests/test_encoder.py`:

```python
"""编码器测试."""

import torch
from deepcf.model.encoder import GCNEncoder


def test_encoder_output_shapes():
    """测试编码器输出形状."""
    encoder = GCNEncoder(input_dim=16, hidden_dim=128, latent_dim=64, dropout=0.3)
    encoder.eval()

    x = torch.randn(100, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

    mu, logvar = encoder(x, edge_index)

    assert mu.shape == (100, 64)
    assert logvar.shape == (100, 64)


def test_reparameterize_shape():
    """测试重参数化输出形状."""
    from deepcf.model.encoder import reparameterize

    mu = torch.randn(50, 64)
    logvar = torch.randn(50, 64)
    z = reparameterize(mu, logvar)

    assert z.shape == (50, 64)


def test_reparameterize_deterministic_eval():
    """测试 eval 模式下重参数化不引入噪声."""
    from deepcf.model.encoder import reparameterize

    mu = torch.ones(10, 8)
    logvar = torch.zeros(10, 8)

    torch.manual_seed(42)
    z1 = reparameterize(mu, logvar, training=False)
    z2 = reparameterize(mu, logvar, training=False)

    # eval 模式下不应有随机性
    assert torch.allclose(z1, mu, atol=1e-6)
    assert torch.allclose(z2, mu, atol=1e-6)


def test_encoder_train_mode():
    """测试训练模式下输出有变化."""
    from deepcf.model.encoder import reparameterize

    mu = torch.zeros(5, 4)
    logvar = torch.zeros(5, 4)

    torch.manual_seed(1)
    z1 = reparameterize(mu, logvar, training=True)
    torch.manual_seed(2)
    z2 = reparameterize(mu, logvar, training=True)

    # 不同种子应产生不同噪声
    assert not torch.allclose(z1, z2)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_encoder.py -v
```

Expected: FAIL, `GCNEncoder` 未定义。

- [ ] **Step 3: 实现 deepcf/model/encoder.py**

```python
"""GCN 编码器：双层图卷积 + 重参数化."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    """双层 GCN 编码器.

    X → GCNConv1 (→128) → ReLU → Dropout →
    GCNConv2 (→128) → μ (64-d) + log σ² (64-d)
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播.

        Args:
            x: (N, F) 节点特征
            edge_index: (2, E) 边索引
            edge_weight: (E,) 可选的边权重

        Returns:
            (mu, logvar) 均值和 log 方差
        """
        h = self.conv1(x, edge_index, edge_weight=edge_weight)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        h = F.relu(h)

        mu = self.mu_proj(h)
        logvar = self.logvar_proj(h)

        return mu, logvar


def reparameterize(
    mu: torch.Tensor, logvar: torch.Tensor, training: bool = True
) -> torch.Tensor:
    """重参数化技巧: z = μ + σ ⊙ ε.

    Args:
        mu: (N, D) 均值
        logvar: (N, D) log 方差
        training: 是否为训练模式（训练时注入噪声）

    Returns:
        (N, D) 潜在表示 z
    """
    if training:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    return mu
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_encoder.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add deepcf/model/encoder.py tests/test_encoder.py
git commit -m "feat: add GCN encoder with reparameterization

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: 三头解码器

**Files:**
- Create: `deepcf/model/decoder.py`
- Create: `tests/test_decoder.py`

- [ ] **Step 1: 编写测试**

在 `tests/test_decoder.py`:

```python
"""解码器测试."""

import torch
from deepcf.model.decoder import LinkDecoder, RankDecoder, WeightDecoder, MultiHeadDecoder


def test_link_decoder_shape():
    """测试链接预测头输出形状."""
    decoder = LinkDecoder()
    z = torch.randn(30, 64)

    # 全量预测
    adj_pred = decoder(z)
    assert adj_pred.shape == (30, 30)
    assert torch.all((adj_pred >= 0) & (adj_pred <= 1))

    # 边级别预测
    edge_indices = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    scores = decoder.decode_edges(z, edge_indices)
    assert scores.shape == (3,)


def test_rank_decoder_shape():
    """测试排序头输出形状."""
    decoder = RankDecoder(latent_dim=64, hidden_dim=64)
    z = torch.randn(50, 64)
    u = torch.tensor([0, 1, 2])
    v = torch.tensor([1, 2, 3])

    scores = decoder(z, u, v)
    assert scores.shape == (3,)


def test_weight_decoder_shape():
    """测试权重回归头输出形状."""
    decoder = WeightDecoder(latent_dim=64, hidden_dim=64)
    z = torch.randn(50, 64)
    u = torch.tensor([0, 1, 2])
    v = torch.tensor([1, 2, 3])

    pred_weights = decoder(z, u, v)
    assert pred_weights.shape == (3,)
    assert torch.all(pred_weights >= 0)  # 权重应为非负


def test_multi_head_decoder_outputs():
    """测试多头解码器联合输出."""
    decoder = MultiHeadDecoder(latent_dim=64, hidden_dim=64)
    z = torch.randn(40, 64)
    u = torch.tensor([0, 1, 2, 3])
    v = torch.tensor([1, 2, 3, 4])

    adj_pred, rank_scores, weight_pred = decoder(z, u, v)

    assert adj_pred.shape == (40, 40)
    assert rank_scores.shape == (4,)
    assert weight_pred.shape == (4,)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_decoder.py -v
```

Expected: FAIL.

- [ ] **Step 3: 实现 deepcf/model/decoder.py**

```python
"""三头解码器：链接预测、影响力排序、边权重回归."""

import torch
import torch.nn as nn


class LinkDecoder(nn.Module):
    """链接预测头：内积 + sigmoid 重构邻接矩阵."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """全量邻接矩阵重构.

        Args:
            z: (N, D) 节点嵌入

        Returns:
            (N, N) 重构邻接矩阵 Â
        """
        return torch.sigmoid(z @ z.T)

    def decode_edges(
        self, z: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """边级别链接得分.

        Args:
            z: (N, D) 节点嵌入
            edge_index: (2, E) 边索引

        Returns:
            (E,) 边得分
        """
        src, dst = edge_index[0], edge_index[1]
        return torch.sigmoid((z[src] * z[dst]).sum(dim=1))


class RankDecoder(nn.Module):
    """影响力排序头：MLP([z_u ‖ z_v]) → s_uv.

    2层 MLP: 2*latent_dim → hidden_dim → 1
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, z: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """计算节点对影响力得分.

        Args:
            z: (N, D) 节点嵌入
            u: (B,) 源节点索引
            v: (B,) 目标节点索引

        Returns:
            (B,) 影响力得分
        """
        zu = z[u]
        zv = z[v]
        concat = torch.cat([zu, zv], dim=-1)
        return self.mlp(concat).squeeze(-1)


class WeightDecoder(nn.Module):
    """边权重回归头：MLP([z_u ‖ z_v]) → ŵ_uv.

    2层 MLP: 2*latent_dim → hidden_dim → 1，输出用 Softplus 保证非负。
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, z: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """预测边权重.

        Args:
            z: (N, D) 节点嵌入
            u: (B,) 源节点索引
            v: (B,) 目标节点索引

        Returns:
            (B,) 预测权重（非负）
        """
        zu = z[u]
        zv = z[v]
        concat = torch.cat([zu, zv], dim=-1)
        return nn.functional.softplus(self.mlp(concat)).squeeze(-1)


class MultiHeadDecoder(nn.Module):
    """多头解码器：组合 Link + Rank + Weight 三个头."""

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.link = LinkDecoder()
        self.rank = RankDecoder(latent_dim, hidden_dim)
        self.weight = WeightDecoder(latent_dim, hidden_dim)

    def forward(
        self, z: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """联合前向传播.

        Args:
            z: (N, D) 节点嵌入
            u: (B,) 源节点索引
            v: (B,) 目标节点索引

        Returns:
            (adj_pred, rank_scores, weight_pred)
        """
        adj_pred = self.link(z)
        rank_scores = self.rank(z, u, v)
        weight_pred = self.weight(z, u, v)
        return adj_pred, rank_scores, weight_pred
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_decoder.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add deepcf/model/decoder.py tests/test_decoder.py
git commit -m "feat: add three-head decoder (link, rank, weight)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: 损失函数

**Files:**
- Create: `deepcf/model/losses.py`
- Create: `tests/test_losses.py`

- [ ] **Step 1: 编写测试**

在 `tests/test_losses.py`:

```python
"""损失函数测试."""

import torch
from deepcf.model.losses import (
    bpr_loss,
    kl_divergence_loss,
    multi_task_loss,
)


def test_bpr_loss_positive():
    """测试 BPR 损失：正样本得分 > 负样本得分时损失应较小."""
    s_pos = torch.tensor([5.0, 4.0, 3.0])
    s_neg = torch.tensor([1.0, 2.0, 0.5])

    loss = bpr_loss(s_pos, s_neg)

    assert loss.item() >= 0
    # BPR = -log(sigmoid(s_pos - s_neg))，差越大损失越小
    assert loss.item() < 0.5


def test_bpr_loss_negative():
    """测试 BPR 损失：负样本得分更高时损失应较大."""
    s_pos = torch.tensor([1.0, 2.0])
    s_neg = torch.tensor([5.0, 6.0])

    loss = bpr_loss(s_pos, s_neg)

    assert loss.item() > 1.0  # 差为负时 sigmoid 趋近0，-log 很大


def test_kl_divergence_zero():
    """测试 KL 散度：当 μ=0, σ=1 时 KL=0."""
    mu = torch.zeros(100, 64)
    logvar = torch.zeros(100, 64)

    kl = kl_divergence_loss(mu, logvar)

    assert kl.item() == 0.0 or torch.isclose(kl, torch.tensor(0.0), atol=1e-6)


def test_kl_divergence_positive():
    """测试 KL 散度非负."""
    mu = torch.randn(50, 64)
    logvar = torch.randn(50, 64)

    kl = kl_divergence_loss(mu, logvar)

    assert kl.item() >= 0


def test_multi_task_loss_shape():
    """测试多任务联合损失输出形状."""
    adj_pred = torch.rand(30, 30)
    adj_true = torch.randint(0, 2, (30, 30)).float()
    rank_pos = torch.tensor([3.0, 4.0, 5.0])
    rank_neg = torch.tensor([1.0, 2.0, 3.0])
    weight_pred = torch.tensor([2.0, 3.0, 4.0])
    weight_true = torch.tensor([2.1, 2.9, 3.8])
    mu = torch.randn(30, 64)
    logvar = torch.randn(30, 64)

    total, components = multi_task_loss(
        adj_pred, adj_true,
        rank_pos, rank_neg,
        weight_pred, weight_true,
        mu, logvar,
    )

    assert isinstance(total, torch.Tensor)
    assert total.ndim == 0  # 标量
    assert "link" in components
    assert "rank" in components
    assert "weight" in components
    assert "kl" in components
    assert "total" in components
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_losses.py -v
```

Expected: FAIL.

- [ ] **Step 3: 实现 deepcf/model/losses.py**

```python
"""多任务损失函数."""

import torch
import torch.nn.functional as F
from typing import Dict


def bpr_loss(s_pos: torch.Tensor, s_neg: torch.Tensor) -> torch.Tensor:
    """BPR 成对排序损失.

    L_BPR = -mean(log(σ(s_pos - s_neg)))

    Args:
        s_pos: (B,) 正样本得分
        s_neg: (B,) 负样本得分

    Returns:
        标量损失
    """
    diff = s_pos - s_neg
    return -torch.mean(F.logsigmoid(diff))


def kl_divergence_loss(
    mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """KL 散度: KL(q(z) ‖ N(0, I)).

    KL = -0.5 * mean(1 + logvar - mu² - exp(logvar))

    Args:
        mu: (N, D) 均值
        logvar: (N, D) log 方差

    Returns:
        标量损失
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def multi_task_loss(
    adj_pred: torch.Tensor,
    adj_true: torch.Tensor,
    rank_pos: torch.Tensor,
    rank_neg: torch.Tensor,
    weight_pred: torch.Tensor,
    weight_true: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    lambda_link: float = 1.0,
    lambda_rank: float = 0.5,
    lambda_weight: float = 0.3,
    beta_kl: float = 0.001,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """多任务联合损失.

    L = λ₁·BCE(Â, A) + λ₂·BPR(s_pos, s_neg) + λ₃·MSE(ŵ, w) + β·KL

    Args:
        adj_pred: (N, N) 预测邻接矩阵
        adj_true: (N, N) 真实邻接矩阵
        rank_pos: (B,) 正样本影响力得分
        rank_neg: (B,) 负样本影响力得分
        weight_pred: (B,) 预测边权重
        weight_true: (B,) 真实边权重
        mu: (N, D) 编码器均值
        logvar: (N, D) 编码器 log 方差
        lambda_link/lambda_rank/lambda_weight: 各任务权重
        beta_kl: KL 正则化权重

    Returns:
        (total_loss, component_losses_dict)
    """
    loss_link = F.binary_cross_entropy(adj_pred, adj_true, reduction="mean")
    loss_rank = bpr_loss(rank_pos, rank_neg)
    loss_weight = F.mse_loss(weight_pred, weight_true, reduction="mean")
    loss_kl = kl_divergence_loss(mu, logvar)

    total = (
        lambda_link * loss_link
        + lambda_rank * loss_rank
        + lambda_weight * loss_weight
        + beta_kl * loss_kl
    )

    components = {
        "link": loss_link.item(),
        "rank": loss_rank.item(),
        "weight": loss_weight.item(),
        "kl": loss_kl.item(),
        "total": total.item(),
    }

    return total, components
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_losses.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add deepcf/model/losses.py tests/test_losses.py
git commit -m "feat: add multi-task loss functions (BCE, BPR, MSE, KL)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: VGAE 主模型

**Files:**
- Create: `deepcf/model/vgae.py`
- Create: `tests/test_vgae.py`

- [ ] **Step 1: 编写测试**

在 `tests/test_vgae.py`:

```python
"""VGAE 主模型测试."""

import torch
from deepcf.model.vgae import DeepCFVGAE
from deepcf.config import ModelConfig, LossConfig


def test_vgae_forward():
    """测试 VGAE 前向传播."""
    config = ModelConfig(input_dim=16, hidden_dim=128, latent_dim=64)
    model = DeepCFVGAE(config)
    model.eval()

    x = torch.randn(50, 16)
    edge_index = torch.tensor([
        [0, 0, 1, 2, 3],
        [1, 2, 3, 4, 4],
    ], dtype=torch.long)
    u = torch.tensor([0, 1, 2])
    v = torch.tensor([1, 2, 3])

    adj_pred, rank_scores, weight_pred, mu, logvar, z = model(x, edge_index, u, v)

    assert adj_pred.shape == (50, 50)
    assert rank_scores.shape == (3,)
    assert weight_pred.shape == (3,)
    assert mu.shape == (50, 64)
    assert logvar.shape == (50, 64)
    assert z.shape == (50, 64)


def test_vgae_get_embeddings():
    """测试获取嵌入接口."""
    config = ModelConfig(input_dim=16, hidden_dim=128, latent_dim=64)
    model = DeepCFVGAE(config)
    model.eval()

    x = torch.randn(30, 16)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    z = model.get_embeddings(x, edge_index)

    assert z.shape == (30, 64)


def test_vgae_compute_loss():
    """测试计算损失接口."""
    model_config = ModelConfig(input_dim=16, hidden_dim=128, latent_dim=64)
    loss_config = LossConfig()
    model = DeepCFVGAE(model_config)

    x = torch.randn(40, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    adj_true = torch.eye(40) * 0  # no self-loops in target
    adj_true[0, 1] = adj_true[1, 0] = 1
    adj_true[1, 2] = adj_true[2, 1] = 1

    u = torch.tensor([0, 1])
    v = torch.tensor([1, 2])
    rank_pos = torch.tensor([3.0, 4.0])
    rank_neg = torch.tensor([1.0, 2.0])
    w_true = torch.tensor([2.5, 3.5])

    total, components = model.compute_loss(
        x, edge_index, adj_true, u, v, rank_pos, rank_neg, None, w_true, loss_config
    )

    assert total.ndim == 0
    assert "link" in components
    assert "rank" in components
    assert "weight" in components
    assert "kl" in components
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_vgae.py -v
```

Expected: FAIL.

- [ ] **Step 3: 实现 deepcf/model/vgae.py**

```python
"""DeepCF VGAE 主模型."""

import torch
import torch.nn as nn
from typing import Dict

from deepcf.config import ModelConfig, LossConfig
from deepcf.model.encoder import GCNEncoder, reparameterize
from deepcf.model.decoder import MultiHeadDecoder
from deepcf.model.losses import multi_task_loss


class DeepCFVGAE(nn.Module):
    """DeepCF: 面向城市消费网络的 VGAE 模型.

    编码: 双层GCN → μ, σ → z
    解码: 三头并行（链接预测 + 影响力排序 + 权重回归）
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = GCNEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            dropout=config.dropout,
        )
        self.decoder = MultiHeadDecoder(
            latent_dim=config.latent_dim,
            hidden_dim=config.mlp_hidden,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播.

        Args:
            x: (N, F) 节点特征
            edge_index: (2, E) 边索引
            u: (B,) 源节点索引（用于 Rank/Weight 头）
            v: (B,) 目标节点索引（用于 Rank/Weight 头）
            edge_weight: (E,) 可选的边权重

        Returns:
            (adj_pred, rank_scores, weight_pred, mu, logvar, z)
        """
        mu, logvar = self.encoder(x, edge_index, edge_weight=edge_weight)
        z = reparameterize(mu, logvar, training=self.training)
        adj_pred, rank_scores, weight_pred = self.decoder(z, u, v)
        return adj_pred, rank_scores, weight_pred, mu, logvar, z

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """获取节点嵌入（eval 模式，无噪声）.

        Args:
            x: (N, F) 节点特征
            edge_index: (2, E) 边索引
            edge_weight: (E,) 可选的边权重

        Returns:
            (N, D) 节点嵌入
        """
        mu, logvar = self.encoder(x, edge_index, edge_weight=edge_weight)
        return reparameterize(mu, logvar, training=False)

    def compute_loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        adj_true: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        rank_pos: torch.Tensor,
        rank_neg: torch.Tensor,
        edge_weight: torch.Tensor | None,
        weight_true: torch.Tensor,
        loss_config: LossConfig,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """一次性前向 + 计算损失.

        Args:
            x: (N, F) 节点特征
            edge_index: (2, E) 边索引
            adj_true: (N, N) 真实邻接矩阵
            u, v: (B,) 排序样本节点对
            rank_pos, rank_neg: (B,) 正/负样本排序得分
            edge_weight: (E,) 可选的边权重
            weight_true: (B,) 真实边权重
            loss_config: 损失配置

        Returns:
            (total_loss, component_dict)
        """
        adj_pred, rank_scores, weight_pred, mu, logvar, _ = self(
            x, edge_index, u, v, edge_weight=edge_weight
        )

        return multi_task_loss(
            adj_pred=adj_pred,
            adj_true=adj_true,
            rank_pos=rank_scores,
            rank_neg=rank_neg,
            weight_pred=weight_pred,
            weight_true=weight_true,
            mu=mu,
            logvar=logvar,
            lambda_link=loss_config.lambda_link,
            lambda_rank=loss_config.lambda_rank,
            lambda_weight=loss_config.lambda_weight,
            beta_kl=loss_config.beta_kl,
        )
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_vgae.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add deepcf/model/vgae.py tests/test_vgae.py
git commit -m "feat: add DeepCF VGAE main model

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: 评估指标

**Files:**
- Create: `deepcf/train/metrics.py`

- [ ] **Step 1: 扩展测试**

在 `tests/test_data.py` 末尾追加（或在新建的 `tests/test_metrics.py`）:

```python
def test_metrics():
    """测试评估指标函数."""
    from deepcf.train.metrics import (
        compute_auc_ap,
        compute_precision_recall_at_k,
        compute_ndcg_at_k,
        compute_mae_rmse,
    )
    import numpy as np

    # 测试 AUC/AP
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_score = np.array([0.9, 0.8, 0.3, 0.1, 0.7, 0.6])
    auc, ap = compute_auc_ap(y_true, y_score)
    assert 0 <= auc <= 1
    assert 0 <= ap <= 1
    assert auc > 0.5  # 应优于随机

    # 测试 Precision@K / Recall@K
    # 每个用户：推荐列表和真实正样本
    topk_lists = [[0, 1, 2], [3, 4, 5]]
    true_items = [[1, 3], [4, 5]]
    p5, r5 = compute_precision_recall_at_k(topk_lists, true_items, k=3)
    assert 0 <= p5 <= 1
    assert 0 <= r5 <= 1

    # 测试 NDCG@K
    ndcg = compute_ndcg_at_k(topk_lists, true_items, k=3)
    assert 0 <= ndcg <= 1

    # 测试 MAE/RMSE
    y_true_reg = np.array([2.0, 3.0, 4.0])
    y_pred_reg = np.array([2.1, 2.8, 4.2])
    mae, rmse = compute_mae_rmse(y_true_reg, y_pred_reg)
    assert mae >= 0
    assert rmse >= mae  # RMSE >= MAE 通常成立
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_data.py::test_metrics -v
```

Expected: FAIL.

- [ ] **Step 3: 实现 deepcf/train/metrics.py**

```python
"""评估指标：AUC, AP, Precision@K, Recall@K, NDCG, MAE, RMSE."""

import numpy as np
from typing import List, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error


def compute_auc_ap(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[float, float]:
    """计算 AUC-ROC 和 Average Precision.

    Args:
        y_true: (N,) 二值标签
        y_score: (N,) 预测得分

    Returns:
        (auc, ap)
    """
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return float(auc), float(ap)


def compute_precision_recall_at_k(
    topk_lists: List[List[int]],
    true_items: List[List[int]],
    k: int = 10,
) -> Tuple[float, float]:
    """计算 Precision@K 和 Recall@K.

    Args:
        topk_lists: 每个用户的 Top-K 推荐列表
        true_items: 每个用户的真实正样本列表
        k: 截断值

    Returns:
        (precision_at_k, recall_at_k)
    """
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
    """计算 NDCG@K.

    Args:
        topk_lists: 每个用户的 Top-K 推荐列表
        true_items: 每个用户的真实正样本列表
        k: 截断值

    Returns:
        ndcg_at_k
    """
    ndcgs = []
    for topk, truths in zip(topk_lists, true_items):
        topk_k = topk[:k]
        truths_set = set(truths)

        dcg = 0.0
        for rank, item in enumerate(topk_k, start=1):
            if item in truths_set:
                dcg += 1.0 / np.log2(rank + 1)

        # IDCG: 理想排序（所有正样本排在最前）
        ideal_hits = min(len(truths_set), k)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcgs))


def compute_mae_rmse(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float]:
    """计算 MAE 和 RMSE.

    Args:
        y_true: (N,) 真实值
        y_pred: (N,) 预测值

    Returns:
        (mae, rmse)
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return float(mae), float(rmse)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_data.py::test_metrics -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add deepcf/train/metrics.py tests/test_data.py
git commit -m "feat: add evaluation metrics (AUC, AP, P@K, R@K, NDCG, MAE, RMSE)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: 训练器

**Files:**
- Create: `deepcf/train/trainer.py`

- [ ] **Step 1: 实现 deepcf/train/trainer.py**

```python
"""训练循环：多任务训练 + 早停 + Checkpoint."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm

from deepcf.config import DeepCFConfig, ModelConfig, LossConfig
from deepcf.model.vgae import DeepCFVGAE
from deepcf.data.dataset import collate_bpr_batch


class Trainer:
    """DeepCF 训练器."""

    def __init__(
        self,
        model: DeepCFVGAE,
        config: DeepCFConfig,
        train_loader: DataLoader,
        val_data: Optional[Dict] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_data = val_data

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=config.train.lr_patience,
            factor=config.train.lr_factor,
        )

        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.history: Dict[str, list] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

        os.makedirs(config.output_dir, exist_ok=True)

    def train_epoch(self, adj_true: torch.Tensor) -> float:
        """训练一个 epoch.

        Args:
            adj_true: (N, N) 真实邻接矩阵

        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            users, pos_items, neg_items, weights = batch
            users = users.to(self.config.device)
            pos_items = pos_items.to(self.config.device)
            neg_items = neg_items.to(self.config.device)
            weights = weights.to(self.config.device)

            # BPR 排序样本
            rank_pos = self.model.decoder.rank(
                self.model.get_embeddings(self.train_loader.dataset.x, self.train_loader.dataset.edge_index),
                users, pos_items,
            ).detach()
            rank_neg = self.model.decoder.rank(
                self.model.get_embeddings(self.train_loader.dataset.x, self.train_loader.dataset.edge_index),
                users, neg_items,
            ).detach()

            self.optimizer.zero_grad()

            loss, components = self.model.compute_loss(
                x=self.train_loader.dataset.x,
                edge_index=self.train_loader.dataset.edge_index,
                adj_true=adj_true,
                u=users,
                v=pos_items,
                rank_pos=rank_pos,
                rank_neg=rank_neg,
                edge_weight=self.train_loader.dataset.edge_weight,
                weight_true=weights,
                loss_config=self.config.loss,
            )

            loss.backward()
            self.optimizer.step()

            total_loss += components["total"]
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(self, adj_true: torch.Tensor) -> Dict[str, list]:
        """完整训练流程.

        Args:
            adj_true: (N, N) 真实邻接矩阵

        Returns:
            训练历史记录
        """
        self.model.to(self.config.device)
        pbar = tqdm(range(1, self.config.train.epochs + 1), desc="Training")

        for epoch in pbar:
            train_loss = self.train_epoch(adj_true)
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            pbar.set_postfix({
                "loss": f"{train_loss:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
            })

            # 早停检查
            if train_loss < self.best_val_loss:
                self.best_val_loss = train_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1

            self.scheduler.step(train_loss)

            # 定期保存
            if epoch % self.config.train.checkpoint_interval == 0:
                self._save_checkpoint(epoch, is_best=False)

            if self.patience_counter >= self.config.train.early_stop_patience:
                print(f"Early stopping at epoch {epoch}, "
                      f"best loss {self.best_val_loss:.4f} at epoch {self.best_epoch}")
                break

        print(f"Training completed. Best loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        return self.history

    def _save_checkpoint(self, epoch: int, is_best: bool):
        """保存检查点."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "config": self.config,
        }

        if is_best:
            torch.save(ckpt, os.path.join(self.config.output_dir, "best_model.pt"))

        torch.save(
            ckpt,
            os.path.join(self.config.output_dir, f"checkpoint_epoch_{epoch}.pt"),
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        model_config: ModelConfig,
        config: DeepCFConfig,
        device: str = "cpu",
    ) -> tuple[DeepCFVGAE, Dict]:
        """从检查点加载模型.

        Args:
            path: 检查点文件路径
            model_config: 模型配置
            config: 全局配置
            device: 设备

        Returns:
            (model, checkpoint_dict)
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = DeepCFVGAE(model_config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        return model, ckpt
```

- [ ] **Step 2: 验证导入**

```bash
python -c "from deepcf.train.trainer import Trainer; print('Trainer imported OK')"
```

Expected: "Trainer imported OK"

- [ ] **Step 3: Commit**

```bash
git add deepcf/train/trainer.py
git commit -m "feat: add training loop with early stopping and checkpoint

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: 评估与可视化

**Files:**
- Create: `deepcf/eval/ranking.py`
- Create: `deepcf/eval/visualize.py`
- Create: `deepcf/eval/report.py`

- [ ] **Step 1: 实现 deepcf/eval/ranking.py**

```python
"""Top-K 关键商户识别与影响力排序."""

import numpy as np
import torch
from typing import List, Tuple, Dict


def compute_radiation_scores(
    z: np.ndarray,
    adjacency: np.ndarray,
    weights: np.ndarray,
    rank_scores: np.ndarray,
    alpha: float = 0.3,
    beta: float = 0.4,
    gamma: float = 0.3,
) -> np.ndarray:
    """综合辐射力评分.

    Score = α·degree_norm + β·rank_norm + γ·weight_sum_norm

    Args:
        z: (N, D) 节点嵌入
        adjacency: (N, N) 邻接矩阵
        weights: (N, N) 边权重矩阵
        rank_scores: (N,) 排序头得分（归一化后）
        alpha, beta, gamma: 权重系数

    Returns:
        (N,) 综合辐射力得分
    """
    # 度中心性（归一化）
    degrees = adjacency.sum(axis=1)
    degree_norm = degrees / (degrees.max() + 1e-8)

    # 边权重总和（归一化）
    weight_sums = weights.sum(axis=1)
    weight_norm = weight_sums / (weight_sums.max() + 1e-8)

    # 排序得分归一化
    rank_norm = (rank_scores - rank_scores.min()) / (rank_scores.max() - rank_scores.min() + 1e-8)

    scores = alpha * degree_norm + beta * rank_norm + gamma * weight_norm
    return scores


def top_k_merchants(
    scores: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> List[Dict]:
    """识别 Top-K 关键商户.

    Args:
        scores: (N,) 辐射力得分
        labels: (N,) 商户类别
        k: 返回数量

    Returns:
        [{id, score, category}, ...] 按得分降序排列
    """
    indices = np.argsort(scores)[::-1][:k]
    results = []
    for idx in indices:
        results.append({
            "id": int(idx),
            "score": float(scores[idx]),
            "category": int(labels[idx]),
        })
    return results


def rank_all_merchants(model, x, edge_index, edge_weight=None) -> np.ndarray:
    """使用排序头为所有商户对打分并汇总每个商户的总辐射力.

    Args:
        model: DeepCFVGAE 模型
        x: (N, F) 节点特征
        edge_index: (2, E) 边索引
        edge_weight: (E,) 可选的边权重

    Returns:
        (N,) 每个商户的平均影响力得分
    """
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    edge_index = edge_index.to(device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    with torch.no_grad():
        z = model.get_embeddings(x, edge_index, edge_weight=edge_weight)

    N = z.shape[0]
    # 对所有节点打分（使用所有邻居对来估计影响力）
    all_scores = np.zeros(N)
    edge_index_np = edge_index.cpu().numpy()

    # 对每条边上的两个方向打分
    for i in range(edge_index_np.shape[1]):
        u = edge_index_np[0, i]
        v = edge_index_np[1, i]
        with torch.no_grad():
            score = model.decoder.rank(
                z.unsqueeze(0).expand(2, -1, -1).transpose(0, 1).contiguous().view(N, -1)[[u, v]],
                torch.tensor([u], device=device),
                torch.tensor([v], device=device),
            )
        all_scores[u] += score.cpu().item()
        all_scores[v] += score.cpu().item()

    # 平均化
    degrees = np.bincount(edge_index_np.flatten(), minlength=N)
    degrees[degrees == 0] = 1
    all_scores /= degrees

    return all_scores
```

- [ ] **Step 2: 实现 deepcf/eval/visualize.py**

```python
"""t-SNE 嵌入可视化、影响力分布图、模型分析."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Optional, List, Dict
import os


def plot_tsne_embeddings(
    z: np.ndarray,
    labels: np.ndarray,
    scores: Optional[np.ndarray] = None,
    top_k_indices: Optional[List[int]] = None,
    save_path: str = "tsne_embeddings.png",
    figsize: tuple = (12, 5),
):
    """t-SNE 嵌入可视化（双面板：类别着色 + 影响力着色）.

    Args:
        z: (N, D) 节点嵌入
        labels: (N,) 类别标签
        scores: (N,) 辐射力得分（可选）
        top_k_indices: Top-K 节点索引列表（可选）
        save_path: 保存路径
        figsize: 图像大小
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, z.shape[0] - 1))
    z_2d = tsne.fit_transform(z)

    fig, axes = plt.subplots(1, 2 if scores is not None else 1, figsize=figsize)
    if scores is None:
        axes = [axes]

    # 面板1：按类别着色
    scatter1 = axes[0].scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap="tab10", s=30, alpha=0.7)
    if top_k_indices:
        axes[0].scatter(z_2d[top_k_indices, 0], z_2d[top_k_indices, 1],
                       s=120, edgecolors="red", facecolors="none", linewidths=2, marker="o")
    axes[0].set_title("t-SNE by Category", fontsize=13)
    axes[0].set_xlabel("Dim 1")
    axes[0].set_ylabel("Dim 2")
    plt.colorbar(scatter1, ax=axes[0], label="Category")

    # 面板2：按影响力着色
    if scores is not None and len(axes) > 1:
        scatter2 = axes[1].scatter(z_2d[:, 0], z_2d[:, 1], c=scores, cmap="YlOrRd",
                                    s=30, alpha=0.7)
        if top_k_indices:
            axes[1].scatter(z_2d[top_k_indices, 0], z_2d[top_k_indices, 1],
                           s=120, edgecolors="blue", facecolors="none", linewidths=2, marker="o")
        axes[1].set_title("t-SNE by Radiation Score", fontsize=13)
        axes[1].set_xlabel("Dim 1")
        axes[1].set_ylabel("Dim 2")
        plt.colorbar(scatter2, ax=axes[1], label="Score")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_influence_distribution(
    scores: np.ndarray,
    degrees: np.ndarray,
    top_k_indices: Optional[List[int]] = None,
    save_path: str = "influence_distribution.png",
    figsize: tuple = (14, 10),
):
    """影响力分布图（直方图 + 度-影响力散点图 + 社区汇总）.

    Args:
        scores: (N,) 辐射力得分
        degrees: (N,) 节点度
        top_k_indices: Top-K 索引
        save_path: 保存路径
        figsize: 图像大小
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. 辐射力直方图
    axes[0, 0].hist(scores, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0, 0].axvline(x=np.median(scores), color="red", linestyle="--", label=f"Median={np.median(scores):.3f}")
    axes[0, 0].set_title("Radiation Score Distribution")
    axes[0, 0].set_xlabel("Score")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()

    # 2. 度-影响力散点图
    axes[0, 1].scatter(degrees, scores, alpha=0.5, s=20)
    if top_k_indices:
        axes[0, 1].scatter(degrees[top_k_indices], scores[top_k_indices],
                          color="red", s=80, marker="*", label="Top-K", edgecolors="black")
    axes[0, 1].set_title("Degree vs. Radiation Score")
    axes[0, 1].set_xlabel("Degree")
    axes[0, 1].set_ylabel("Radiation Score")
    axes[0, 1].legend()

    # 3. 对数尺度散点图
    axes[1, 0].scatter(degrees, scores, alpha=0.5, s=20)
    if top_k_indices:
        axes[1, 0].scatter(degrees[top_k_indices], scores[top_k_indices],
                          color="red", s=80, marker="*", label="Top-K", edgecolors="black")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title("Degree (log) vs. Radiation Score")
    axes[1, 0].set_xlabel("Degree (log scale)")
    axes[1, 0].set_ylabel("Radiation Score")
    axes[1, 0].legend()

    # 4. Top-K 得分柱状图
    N = min(20, len(scores))
    top_indices = np.argsort(scores)[::-1][:N]
    top_scores = scores[top_indices]
    colors = plt.cm.YlOrRd(top_scores / max(top_scores))
    axes[1, 1].bar(range(N), top_scores, color=colors, edgecolor="black", linewidth=0.5)
    axes[1, 1].set_title(f"Top-{N} Radiation Scores")
    axes[1, 1].set_xlabel("Rank")
    axes[1, 1].set_ylabel("Score")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    history: Dict[str, list],
    save_path: str = "training_curves.png",
):
    """训练曲线图.

    Args:
        history: 训练历史字典，含 'epoch', 'train_loss', 'lr'
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss 曲线
    ax1.plot(history["epoch"], history["train_loss"], color="blue", linewidth=1.5)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    # LR 曲线
    ax2.plot(history["epoch"], history["lr"], color="green", linewidth=1.5)
    ax2.set_title("Learning Rate")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("LR")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
```

- [ ] **Step 3: 实现 deepcf/eval/report.py**

```python
"""综合评估报告生成."""

import os
import numpy as np
import torch
from typing import Dict, List, Optional
from datetime import datetime

from deepcf.eval.ranking import compute_radiation_scores, top_k_merchants, rank_all_merchants
from deepcf.eval.visualize import (
    plot_tsne_embeddings,
    plot_influence_distribution,
    plot_training_curves,
)
from deepcf.train.metrics import (
    compute_auc_ap,
    compute_precision_recall_at_k,
    compute_mae_rmse,
)


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
    """生成综合评估报告.

    Args:
        model: DeepCFVGAE 模型
        x: (N, F) 节点特征
        edge_index: (2, E) 边索引
        adjacency: (N, N) 完整邻接矩阵
        weights: (N, N) 边权矩阵
        labels: (N,) 商户类别
        history: 训练历史
        test_edges: (M, 2) 测试边
        test_edges_neg: (M, 2) 测试负样本
        output_dir: 输出目录

    Returns:
        报告字典
    """
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    x = x.to(device)
    edge_index = edge_index.to(device)

    model.eval()

    # 1. 获取嵌入
    with torch.no_grad():
        z = model.get_embeddings(x, edge_index).cpu().numpy()

    # 2. 链接预测评估
    with torch.no_grad():
        z_tensor = model.get_embeddings(x, edge_index)

    # 测试正样本得分
    test_pos_src = torch.tensor(test_edges[:, 0], dtype=torch.long, device=device)
    test_pos_dst = torch.tensor(test_edges[:, 1], dtype=torch.long, device=device)
    with torch.no_grad():
        pos_scores = torch.sigmoid(
            (z_tensor[test_pos_src] * z_tensor[test_pos_dst]).sum(dim=1)
        ).cpu().numpy()

    # 测试负样本得分
    test_neg_src = torch.tensor(test_edges_neg[:, 0], dtype=torch.long, device=device)
    test_neg_dst = torch.tensor(test_edges_neg[:, 1], dtype=torch.long, device=device)
    with torch.no_grad():
        neg_scores = torch.sigmoid(
            (z_tensor[test_neg_src] * z_tensor[test_neg_dst]).sum(dim=1)
        ).cpu().numpy()

    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([pos_scores, neg_scores])
    auc, ap = compute_auc_ap(y_true, y_score)

    # 3. 辐射力评分与 Top-K
    rank_scores = rank_all_merchants(model, x, edge_index)
    radiation_scores = compute_radiation_scores(z, adjacency, weights, rank_scores)
    top_k = top_k_merchants(radiation_scores, labels, k=20)

    # 4. 生成可视化
    topk_indices = [item["id"] for item in top_k[:10]]

    plot_tsne_embeddings(
        z, labels, radiation_scores, topk_indices,
        save_path=os.path.join(output_dir, "tsne_embeddings.png"),
    )

    plot_influence_distribution(
        radiation_scores, adjacency.sum(axis=1), topk_indices,
        save_path=os.path.join(output_dir, "influence_distribution.png"),
    )

    plot_training_curves(
        history,
        save_path=os.path.join(output_dir, "training_curves.png"),
    )

    # 5. 汇总报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "link_prediction": {"auc": auc, "ap": ap},
        "top_10_merchants": top_k[:10],
        "top_20_merchants": top_k[:20],
        "radiation_stats": {
            "mean": float(radiation_scores.mean()),
            "std": float(radiation_scores.std()),
            "max": float(radiation_scores.max()),
            "min": float(radiation_scores.min()),
        },
        "output_files": [
            os.path.join(output_dir, "tsne_embeddings.png"),
            os.path.join(output_dir, "influence_distribution.png"),
            os.path.join(output_dir, "training_curves.png"),
        ],
    }

    return report
```

- [ ] **Step 4: 验证导入**

```bash
python -c "
from deepcf.eval.ranking import compute_radiation_scores, top_k_merchants
from deepcf.eval.visualize import plot_tsne_embeddings, plot_influence_distribution, plot_training_curves
from deepcf.eval.report import generate_report
print('All eval modules imported OK')
"
```

Expected: "All eval modules imported OK"

- [ ] **Step 5: Commit**

```bash
git add deepcf/eval/ranking.py deepcf/eval/visualize.py deepcf/eval/report.py
git commit -m "feat: add evaluation, visualization, and report generation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 11: CLI 脚本与端到端运行

**Files:**
- Create: `scripts/train.py`
- Create: `scripts/evaluate.py`
- Create: `scripts/visualize.py`

- [ ] **Step 1: 实现 scripts/train.py**

```python
#!/usr/bin/env python
"""DeepCF 训练入口脚本."""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

from deepcf.config import DeepCFConfig
from deepcf.data.generator import generate_synthetic_data
from deepcf.data.utils import normalize_adjacency, split_edges, scale_features
from deepcf.data.dataset import BPRDataset, collate_bpr_batch
from deepcf.model.vgae import DeepCFVGAE
from deepcf.train.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="DeepCF Model Training")
    parser.add_argument("--nodes", type=int, default=500, help="Number of merchant nodes")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 配置
    config = DeepCFConfig()
    config.data.num_nodes = args.nodes
    config.train.epochs = args.epochs
    config.train.lr = args.lr
    config.train.batch_size = args.batch_size
    config.model.latent_dim = args.latent_dim
    config.output_dir = args.output_dir
    config.seed = args.seed
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"Device: {config.device}")
    print(f"Config: {config}")

    # 生成合成数据
    print("\n[1/4] Generating synthetic data...")
    data = generate_synthetic_data(
        num_nodes=config.data.num_nodes,
        num_features=config.model.input_dim,
        edge_density=config.data.edge_density,
        community_k=config.data.community_k,
        seed=config.seed,
    )
    X = scale_features(data["features"])
    A = data["adjacency"]
    W = data["weights"]
    labels = data["labels"]

    print(f"  Nodes: {config.data.num_nodes}, Features: {X.shape[1]}")
    print(f"  Edges: {int(A.sum() // 2)}, Labels: {len(np.unique(labels))}")

    # 边划分
    print("\n[2/4] Splitting edges...")
    splits = split_edges(A, train_ratio=0.85, val_ratio=0.05, seed=config.seed)
    train_adj = splits["train_adj"]
    print(f"  Train edges: {int(train_adj.sum() // 2)}")
    print(f"  Val edges: {len(splits['val_edges'])}")
    print(f"  Test edges: {len(splits['test_edges'])}")

    # 构建图数据（PyG格式）
    print("\n[3/4] Building graph tensors...")
    # 从 train_adj 构建 edge_index
    edge_list = []
    edge_weights_list = []
    for i in range(config.data.num_nodes):
        for j in range(i + 1, config.data.num_nodes):
            if train_adj[i, j] > 0:
                edge_list.append([i, j])
                edge_list.append([j, i])
                edge_weights_list.append(W[i, j])
                edge_weights_list.append(W[i, j])

    edge_index_np = np.array(edge_list).T  # (2, E)
    edge_weight_np = np.array(edge_weights_list, dtype=np.float32)

    # PyTorch tensors
    x_tensor = torch.tensor(X, dtype=torch.float32)
    edge_index_tensor = torch.tensor(edge_index_np, dtype=torch.long)
    edge_weight_tensor = torch.tensor(edge_weight_np, dtype=torch.float32)
    adj_true = torch.tensor(train_adj, dtype=torch.float32)

    # BPR Dataset
    dataset = BPRDataset(train_adj, W, num_negatives=1, seed=config.seed)
    dataset.x = x_tensor
    dataset.edge_index = edge_index_tensor
    dataset.edge_weight = edge_weight_tensor
    train_loader = DataLoader(
        dataset, batch_size=config.train.batch_size,
        shuffle=True, collate_fn=collate_bpr_batch,
    )

    # 模型
    print("\n[4/4] Training model...")
    model = DeepCFVGAE(config.model)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(model, config, train_loader)
    history = trainer.train(adj_true)

    # 保存最终结果
    print(f"\nTraining history saved. Best loss: {trainer.best_val_loss:.4f}")
    print(f"Outputs saved to: {config.output_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 实现 scripts/evaluate.py**

```python
#!/usr/bin/env python
"""DeepCF 评估入口脚本."""

import argparse
import os
import json
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from deepcf.config import DeepCFConfig
from deepcf.data.generator import generate_synthetic_data
from deepcf.data.utils import split_edges, scale_features
from deepcf.model.vgae import DeepCFVGAE
from deepcf.eval.report import generate_report


def main():
    parser = argparse.ArgumentParser(description="DeepCF Model Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = DeepCFConfig()
    config.data.num_nodes = args.nodes
    config.output_dir = args.output_dir
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = DeepCFVGAE(config.model)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(config.device)
    model.eval()

    # 生成数据
    data = generate_synthetic_data(
        num_nodes=config.data.num_nodes,
        num_features=config.model.input_dim,
        seed=config.seed,
    )
    X = scale_features(data["features"])
    A = data["adjacency"]
    W = data["weights"]
    labels = data["labels"]

    # 构建 edge_index（从完整邻接矩阵）
    edge_list = []
    for i in range(config.data.num_nodes):
        for j in range(i + 1, config.data.num_nodes):
            if A[i, j] > 0:
                edge_list.append([i, j])
                edge_list.append([j, i])

    edge_index_np = np.array(edge_list).T
    x_tensor = torch.tensor(X, dtype=torch.float32)
    edge_index_tensor = torch.tensor(edge_index_np, dtype=torch.long)

    # 边划分
    splits = split_edges(A, seed=config.seed)

    # 生成报告
    report = generate_report(
        model=model,
        x=x_tensor,
        edge_index=edge_index_tensor,
        adjacency=A,
        weights=W,
        labels=labels,
        history=ckpt.get("history", {"epoch": [], "train_loss": [], "lr": []}),
        test_edges=splits["test_edges"],
        test_edges_neg=splits["test_edges_neg"],
        output_dir=config.output_dir,
    )

    print("\n=== Evaluation Report ===")
    print(f"AUC: {report['link_prediction']['auc']:.4f}")
    print(f"AP:  {report['link_prediction']['ap']:.4f}")
    print(f"\nTop-10 Merchants:")
    for m in report["top_10_merchants"]:
        print(f"  ID={m['id']:4d}  Score={m['score']:.4f}  Category={m['category']}")
    print(f"\nRadiation Stats: {report['radiation_stats']}")
    print(f"Outputs: {report['output_files']}")

    # 保存 JSON 报告
    with open(os.path.join(config.output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nReport saved to {config.output_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 实现 scripts/visualize.py**

```python
#!/usr/bin/env python
"""DeepCF 独立可视化脚本."""

import argparse
import os
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from deepcf.config import DeepCFConfig
from deepcf.data.generator import generate_synthetic_data
from deepcf.data.utils import scale_features
from deepcf.model.vgae import DeepCFVGAE
from deepcf.eval.ranking import compute_radiation_scores, top_k_merchants, rank_all_merchants
from deepcf.eval.visualize import (
    plot_tsne_embeddings,
    plot_influence_distribution,
    plot_training_curves,
)


def main():
    parser = argparse.ArgumentParser(description="DeepCF Visualization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="outputs/viz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = DeepCFConfig()
    config.data.num_nodes = args.nodes
    config.output_dir = args.output_dir
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = DeepCFVGAE(config.model)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(config.device)
    model.eval()

    # 数据
    data = generate_synthetic_data(
        num_nodes=config.data.num_nodes,
        num_features=config.model.input_dim,
        seed=config.seed,
    )
    X = scale_features(data["features"])
    A = data["adjacency"]
    W = data["weights"]
    labels = data["labels"]

    edge_list = []
    for i in range(config.data.num_nodes):
        for j in range(i + 1, config.data.num_nodes):
            if A[i, j] > 0:
                edge_list.append([i, j])
                edge_list.append([j, i])

    edge_index_np = np.array(edge_list).T
    x_tensor = torch.tensor(X, dtype=torch.float32).to(config.device)
    edge_index_tensor = torch.tensor(edge_index_np, dtype=torch.long).to(config.device)

    # 获取嵌入
    with torch.no_grad():
        z = model.get_embeddings(x_tensor, edge_index_tensor).cpu().numpy()

    # 计算影响力
    rank_scores = rank_all_merchants(model, x_tensor, edge_index_tensor)
    radiation = compute_radiation_scores(z, A, W, rank_scores)
    top_k = top_k_merchants(radiation, labels, k=20)
    topk_idx = [m["id"] for m in top_k[:10]]

    # 生成可视化
    plot_tsne_embeddings(
        z, labels, radiation, topk_idx,
        save_path=os.path.join(config.output_dir, "tsne_embeddings.png"),
    )
    print(f"t-SNE saved to {config.output_dir}/tsne_embeddings.png")

    plot_influence_distribution(
        radiation, A.sum(axis=1), topk_idx,
        save_path=os.path.join(config.output_dir, "influence_distribution.png"),
    )
    print(f"Influence distribution saved to {config.output_dir}/influence_distribution.png")

    history = ckpt.get("history", {"epoch": [], "train_loss": [], "lr": []})
    plot_training_curves(
        history,
        save_path=os.path.join(config.output_dir, "training_curves.png"),
    )
    print(f"Training curves saved to {config.output_dir}/training_curves.png")

    print("\nDone! All visualizations generated.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 端到端测试运行**

```bash
python scripts/train.py --nodes 200 --epochs 50 --output-dir outputs/test_run --seed 42
```

Expected: 训练完成，在 `outputs/test_run/` 下生成 `best_model.pt`。

```bash
python scripts/evaluate.py --checkpoint outputs/test_run/best_model.pt --nodes 200 --output-dir outputs/test_eval --seed 42
```

Expected: 输出 AUC、Top-10 商户、评估图。

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py scripts/evaluate.py scripts/visualize.py
git commit -m "feat: add CLI scripts for train, evaluate, and visualize

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 12: Jupyter Notebooks

**Files:**
- Create: `deepcf/notebooks/01_data_exploration.ipynb`
- Create: `deepcf/notebooks/02_model_training.ipynb`
- Create: `deepcf/notebooks/03_results_analysis.ipynb`

- [ ] **Step 1: 创建 01_data_exploration.ipynb**

```python
# Cell 1 (markdown): "# DeepCF 数据探索\n本 notebook 用于探索合成消费网络数据的结构特征。"
# Cell 2 (code):
"""
import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deepcf.data.generator import generate_synthetic_data
from deepcf.data.utils import normalize_adjacency

sns.set_style("whitegrid")
np.random.seed(42)

# 生成数据
data = generate_synthetic_data(num_nodes=300, num_features=16,
                                edge_density=0.05, community_k=5, seed=42)
X = data['features']
A = data['adjacency']
W = data['weights']
labels = data['labels']
positions = data['positions']

print(f"Nodes: {A.shape[0]}")
print(f"Edges: {int(A.sum() // 2)}")
print(f"Features: {X.shape[1]}")
print(f"Communities: {len(np.unique(labels))}")
"""
# Cell 3 (markdown): "## 1. 度分布"
# Cell 4 (code):
"""
degrees = A.sum(axis=1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(degrees, bins=30, color='steelblue', edgecolor='white')
axes[0].set_title('Degree Distribution')
axes[0].set_xlabel('Degree')
axes[0].set_ylabel('Count')
axes[1].bar(range(len(degrees)), sorted(degrees, reverse=True), width=1)
axes[1].set_title('Degree Rank Plot')
axes[1].set_xlabel('Node Rank')
axes[1].set_ylabel('Degree')
plt.tight_layout()
plt.show()
"""
# Cell 5 (markdown): "## 2. 权重分布"
# Cell 6 (code):
"""
nonzero_w = W[W > 0]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(nonzero_w, bins=30, color='coral', edgecolor='white')
axes[0].set_title('Edge Weight Distribution')
axes[0].set_xlabel('Weight')
axes[0].set_ylabel('Count')
axes[1].scatter(degrees, W.sum(axis=1), alpha=0.5)
axes[1].set_title('Degree vs Total Weight')
axes[1].set_xlabel('Degree')
axes[1].set_ylabel('Total Outgoing Weight')
plt.tight_layout()
plt.show()
"""
# Cell 7 (markdown): "## 3. 社区结构可视化"
# Cell 8 (code):
"""
plt.figure(figsize=(10, 8))
# 绘制节点
for c in range(5):
    mask = labels == c
    plt.scatter(positions[mask, 0], positions[mask, 1],
               label=f'Community {c}', s=30, alpha=0.7)
# 绘制部分边
edge_list = np.argwhere(np.triu(A > 0))
sample_idx = np.random.choice(len(edge_list), min(500, len(edge_list)), replace=False)
for idx in sample_idx:
    i, j = edge_list[idx]
    plt.plot([positions[i, 0], positions[j, 0]],
             [positions[i, 1], positions[j, 1]],
             color='gray', alpha=0.15, linewidth=0.5)
plt.title('Merchant Consumption Network')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend(markerscale=2, fontsize=9)
plt.show()
"""
# Cell 9 (markdown): "## 4. 图统计汇总"
# Cell 10 (code):
"""
print(f"=== Network Statistics ===")
print(f"Number of nodes: {A.shape[0]}")
print(f"Number of edges: {int(A.sum() // 2)}")
print(f"Average degree: {degrees.mean():.2f}")
print(f"Max degree: {degrees.max():.0f}")
print(f"Graph density: {A.sum() / (A.shape[0] * (A.shape[0] - 1)):.4f}")
print(f"Average clustering coeff: (not computed for large graphs)")
print(f"Average edge weight: {nonzero_w.mean():.3f}")
print(f"Weight std: {nonzero_w.std():.3f}")
"""
```

- [ ] **Step 2: 创建 02_model_training.ipynb**

```python
# Cell 1 (markdown): "# DeepCF 模型训练\n端到端训练 VGAE 模型并监控训练过程。"
# Cell 2 (code):
"""
import sys; sys.path.insert(0, '..')
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

from deepcf.config import DeepCFConfig
from deepcf.data.generator import generate_synthetic_data
from deepcf.data.utils import split_edges, scale_features
from deepcf.data.dataset import BPRDataset, collate_bpr_batch
from deepcf.model.vgae import DeepCFVGAE
from deepcf.train.trainer import Trainer

# Config
config = DeepCFConfig()
config.data.num_nodes = 300
config.train.epochs = 200
config.train.batch_size = 128
config.seed = 42

torch.manual_seed(config.seed)
np.random.seed(config.seed)

# Data
data = generate_synthetic_data(num_nodes=config.data.num_nodes,
                                num_features=config.model.input_dim, seed=config.seed)
X = scale_features(data['features'])
A = data['adjacency']; W = data['weights']
splits = split_edges(A, seed=config.seed)
train_adj = splits['train_adj']

# Build graph tensors
edge_list = []
for i in range(config.data.num_nodes):
    for j in range(i + 1, config.data.num_nodes):
        if train_adj[i, j] > 0:
            edge_list.append([i, j]); edge_list.append([j, i])
edge_index_np = np.array(edge_list).T

x_tensor = torch.tensor(X, dtype=torch.float32)
edge_index_tensor = torch.tensor(edge_index_np, dtype=torch.long)
adj_true = torch.tensor(train_adj, dtype=torch.float32)

dataset = BPRDataset(train_adj, W, num_negatives=1, seed=config.seed)
dataset.x = x_tensor; dataset.edge_index = edge_index_tensor

train_loader = DataLoader(dataset, batch_size=config.train.batch_size,
                           shuffle=True, collate_fn=collate_bpr_batch)

model = DeepCFVGAE(config.model)
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

trainer = Trainer(model, config, train_loader)
history = trainer.train(adj_true)

# Plot training curve
plt.figure(figsize=(8, 4))
plt.plot(history['epoch'], history['train_loss'])
plt.title('Training Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.show()
"""
```

- [ ] **Step 3: 创建 03_results_analysis.ipynb**

```python
# Cell 1 (markdown): "# DeepCF 结果分析\nt-SNE 嵌入、Top-K 商户、影响力分布全量可视化。"
# Cell 2 (code):
"""
import sys; sys.path.insert(0, '..')
import os, json
import numpy as np
import torch
import warnings; warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

from deepcf.config import DeepCFConfig
from deepcf.data.generator import generate_synthetic_data
from deepcf.data.utils import scale_features
from deepcf.model.vgae import DeepCFVGAE
from deepcf.eval.report import generate_report

config = DeepCFConfig(); config.data.num_nodes = 300; config.seed = 42

# Load model
model = DeepCFVGAE(config.model)
ckpt = torch.load('outputs/best_model.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.to(config.device); model.eval()

# Data
data = generate_synthetic_data(num_nodes=config.data.num_nodes,
                                num_features=config.model.input_dim, seed=config.seed)
X = scale_features(data['features']); A = data['adjacency']; W = data['weights']
labels = data['labels']

edge_list = []
for i in range(config.data.num_nodes):
    for j in range(i+1, config.data.num_nodes):
        if A[i,j] > 0: edge_list.append([i,j]); edge_list.append([j,i])
edge_index_np = np.array(edge_list).T

x_tensor = torch.tensor(X, dtype=torch.float32)
edge_index_tensor = torch.tensor(edge_index_np, dtype=torch.long)

from deepcf.data.utils import split_edges
splits = split_edges(A, seed=config.seed)

report = generate_report(model, x_tensor, edge_index_tensor, A, W, labels,
                          ckpt.get('history', {'epoch':[],'train_loss':[],'lr':[]}),
                          splits['test_edges'], splits['test_edges_neg'],
                          output_dir='outputs/analysis')

# Display results
from IPython.display import Image, display as ipd
for f in report['output_files']:
    if os.path.exists(f):
        print(f'\\n{f}')
        ipd(Image(filename=f))

print(f"\\nAUC: {report['link_prediction']['auc']:.4f}")
print(f"AP: {report['link_prediction']['ap']:.4f}")
print(f"\\nTop-10 Key Merchants:")
for m in report['top_10_merchants']:
    print(f"  ID={m['id']:3d}  Score={m['score']:.4f}  Cat={m['category']}")
"""
```

- [ ] **Step 4: Commit**

```bash
git add deepcf/notebooks/
git commit -m "feat: add Jupyter notebooks for data exploration, training, and analysis

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Plan Self-Review

**1. Spec coverage check:**
- §2 图定义 → Task 2 (generator), Task 3 (utils)
- §3 模型架构 → Task 4 (encoder), Task 5 (decoder), Task 6 (losses), Task 7 (vgae)
- §4 数据管线 → Task 2, Task 3
- §5 训练配置 → Task 9 (trainer), Task 11 (CLI)
- §6 评估指标 → Task 8 (metrics)
- §7 可视化 → Task 10 (visualize), Task 12 (notebooks)
- §8 项目结构 → Task 1 (scaffolding)
- §9 交付形式 → Task 11 (CLI), Task 12 (notebooks)

**2. Placeholder scan:** ✅ No TBD, TODO, or vague instructions. All steps have concrete code.

**3. Type consistency check:**
- `GCNEncoder(input_dim, hidden_dim, latent_dim, dropout)` — consistent across tasks
- `DeepCFVGAE(config: ModelConfig)` — Task 7 constructor matches Task 9/10/11 usage
- `multi_task_loss()` signature consistent between Task 6 definition and Task 7 usage
- `BPRDataset(adjacency, weights, num_negatives, seed)` consistent across Task 3 and Task 9/11
- `generate_synthetic_data(num_nodes, num_features, ...)` consistent across all tasks
- `rank_all_merchants(model, x, edge_index, edge_weight=None)` — signature matches usage
- `compute_loss()` method — return signature `(total, components)` consistent across Task 7 and Task 9
