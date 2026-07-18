# DeepCF

> **基于图表示学习的城市消费网络建模与关键商户消费辐射最大化研究**
>
> DeepCF: **Deep** **C**onsumption **F**low Network — A Variational Graph Autoencoder for Urban Merchant Influence Modeling

---

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3+-3baea0.svg)](https://pyg.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 目录

- [项目简介](#项目简介)
- [核心思想](#核心思想)
- [模型架构](#模型架构)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [CLI 使用指南](#cli-使用指南)
- [API 文档](#api-文档)
- [实验结果](#实验结果)
- [引用](#引用)

---

## 项目简介

城市消费网络中，不同商户对消费流的辐射/吸引能力差异巨大。**识别网络中具有最大消费辐射影响力的关键商户**，对商业选址、营销资源分配、城市商业规划等具有重要价值。

DeepCF 基于 **变分图自编码器（Variational Graph Autoencoder, VGAE）** 框架，使用双层 GCN 编码器学习商户节点的 64 维潜在表示，并行训练三个解码头同时完成：

| 任务 | 目标 | 损失函数 |
|------|------|----------|
| **链接预测** | 重构商户间消费转移关系 | Binary Cross-Entropy |
| **影响力排序** | 商户辐射力的成对比较 | Bayesian Personalized Ranking |
| **权重回归** | 预测消费转移的金额/频次大小 | Mean Squared Error |

最终基于学到的嵌入识别 **Top-K 关键商户**，并通过 t-SNE 嵌入可视化、影响力分布图等多维分析输出论文级结果。

---

## 核心思想

```text
城市消费网络（无向加权图）
│
│  节点 = 商户
│  边   = 消费转移关系
│  权重 = 消费金额 + 转移频次
│  特征 = 商户类别、位置、人均消费、评分等
│
▼
┌─────────────────────────────────────────┐
│            DeepCF VGAE 模型              │
│                                         │
│  输入: X(N×F) 特征, A(N×N) 邻接矩阵      │
│         │                               │
│         ▼                               │
│  ┌──────────────────┐                   │
│  │  GCN 编码器       │                   │
│  │  第1层: F → 128  │  ReLU + Dropout  │
│  │  第2层: 128 → μ,σ│  共享权重         │
│  └────────┬─────────┘                   │
│           │                             │
│           ▼                             │
│  ┌──────────────────┐                   │
│  │  重参数化         │  z = μ + σ ⊙ ε  │
│  │  隐空间: 64 维    │  ε ~ N(0, I)    │
│  └────────┬─────────┘                   │
│           │                             │
│     ┌─────┼─────┬──────────┐            │
│     ▼     ▼     ▼          ▼            │
│  ┌────┐┌────┐┌────┐   ┌────────┐      │
│  │链接││排序││权重│   │  KL    │      │
│  │预测││打分││回归│   │  损失  │      │
│  └──┬─┘└──┬─┘└──┬─┘   └────────┘      │
│     │     │     │                      │
│     ▼     ▼     ▼                      │
│  BCE(A) BPR(s) MSE(w)  ← 联合损失     │
│                                         │
│  L = λ₁·BCE + λ₂·BPR + λ₃·MSE + β·KL  │
└─────────────────────────────────────────┘
         │
         ▼
   ┌──────────────┐
   │  多维分析输出  │
   │              │
   │  • t-SNE 嵌入可视化      │
   │  • Top-K 关键商户识别    │
   │  • 影响力分布图           │
   │  • 训练曲线 + 消融实验    │
   └──────────────┘
```

---

## 模型架构

### 编码器：双层 GCN

```
X(N×F) ──► GCNConv₁ ──► ReLU ──► Dropout ──► GCNConv₂ ──► ReLU ──┬──► μ (N×64)
                                                                     └──► log σ² (N×64)
```

- **第1层**：输入维度 F → 隐藏维度 128，ReLU 激活，Dropout=0.3
- **第2层**：隐藏维度 128 → 隐藏维度 128，共享权重
- **μ / log σ²**：各自独立的线性投影到 64 维
- **重参数化**：`z = μ + exp(½·logσ²) ⊙ ε`，ε ~ N(0, I)

### 解码器：三头并行

| 解码头 | 计算方式 | 架构 | 输出 |
|--------|----------|------|------|
| **Link Head** | σ(zᵢᵀ zⱼ) | 内积 + Sigmoid | Âᵢⱼ ∈ [0, 1] |
| **Rank Head** | MLP([zᵢ ‖ zⱼ]) | Linear(128→64) → ReLU → Linear(64→1) | sᵢⱼ ∈ ℝ |
| **Weight Head** | MLP([zᵢ ‖ zⱼ]) | Linear(128→64) → ReLU → Linear(64→1) → Softplus | ŵᵢⱼ ≥ 0 |

### 联合损失函数

```
L = λ₁ × BCE(Â, A)       # 链接预测：图结构重构
  + λ₂ × BPR(s⁺, s⁻)     # 影响力排序：成对偏好学习
  + λ₃ × MSE(ŵ, w)       # 权重回归：消费流大小
  + β  × KL(q‖p)         # 变分正则化：隐空间平滑
```

默认超参：`λ₁=1.0`, `λ₂=0.5`, `λ₃=0.3`, `β=0.001`

---

## 项目结构

```
deepcf/
├── config.py                  # 全局配置（5个dataclass）
├── data/
│   ├── generator.py           # SBM合成消费网络生成器
│   ├── dataset.py             # PyTorch Dataset + BPR三元组采样
│   └── utils.py               # 图标准化 / 边划分 / 特征归一化
├── model/
│   ├── encoder.py             # 双层GCN编码器 → μ, σ
│   ├── decoder.py             # 三头解码器（Link/Rank/Weight）
│   ├── vgae.py                # DeepCF VGAE 主模型
│   └── losses.py              # 联合损失（BCE + BPR + MSE + KL）
├── train/
│   ├── trainer.py             # 训练循环 + 早停 + Checkpoint
│   └── metrics.py             # 评估指标（AUC/AP/P@K/R@K/NDCG/MAE/RMSE）
├── eval/
│   ├── ranking.py             # Top-K商户识别 + 辐射力评分
│   ├── visualize.py           # t-SNE / 影响力分布 / 训练曲线
│   └── report.py              # 综合评估报告生成
├── notebooks/
│   ├── 01_data_exploration.ipynb   # 数据探索与网络统计
│   ├── 02_model_training.ipynb     # 交互式模型训练
│   └── 03_results_analysis.ipynb   # 结果分析与全量可视化
scripts/
├── train.py                   # 训练入口（CLI）
├── evaluate.py                # 评估入口（CLI）
└── visualize.py               # 可视化入口（CLI）
tests/
├── test_data.py               # 数据层测试（8个）
├── test_encoder.py            # 编码器测试（4个）
├── test_decoder.py            # 解码器测试（4个）
├── test_losses.py             # 损失函数测试（5个）
└── test_vgae.py               # VGAE主模型测试（3个）
```

---

## 环境要求

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | ≥ 3.10 | 运行环境 |
| PyTorch | ≥ 2.0 | 深度学习框架 |
| PyTorch Geometric | ≥ 2.3 | 图神经网络 |
| NumPy | — | 数值计算 |
| SciPy | — | 稀疏矩阵 |
| scikit-learn | — | t-SNE、评估指标 |
| Matplotlib | — | 可视化 |
| Seaborn | — | 统计图表 |
| tqdm | — | 进度条 |

### 安装

```bash
# 克隆项目
git clone <repo-url>
cd DeepCF

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "from deepcf.config import DeepCFConfig; print('OK')"
```

---

## 快速开始

### 1. 训练模型

```bash
# 基本训练：500节点，100轮
python scripts/train.py --nodes 500 --epochs 100 --output-dir outputs/run1

# 完整训练：500节点，500轮，自定义学习率
python scripts/train.py \
    --nodes 500 \
    --epochs 500 \
    --lr 0.001 \
    --latent-dim 64 \
    --batch-size 128 \
    --output-dir outputs/experiment \
    --seed 42
```

**命令行参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--nodes` | 500 | 商户数量 |
| `--epochs` | 500 | 训练轮数 |
| `--lr` | 0.001 | 学习率 |
| `--latent-dim` | 64 | 潜在空间维度 |
| `--batch-size` | 128 | 批次大小 |
| `--output-dir` | outputs | 输出目录 |
| `--seed` | 42 | 随机种子 |

**训练输出：**

```
outputs/
├── best_model.pt              # 最佳模型（按验证损失）
├── checkpoint_epoch_50.pt     # 每50轮的检查点
├── checkpoint_epoch_100.pt
└── ...
```

### 2. 评估模型

```bash
python scripts/evaluate.py \
    --checkpoint outputs/experiment/best_model.pt \
    --nodes 500 \
    --output-dir outputs/eval \
    --seed 42
```

**评估输出：**

```
=== Evaluation Report ===
AUC: 0.7642
AP:  0.7623

Top-10 Merchants:
  ID=  24  Score=0.8284  Category=0
  ID= 162  Score=0.8278  Category=1
  ...

Radiation Stats:
  mean=0.578, std=0.102, max=0.828, min=0.153
```

### 3. 生成可视化

```bash
python scripts/visualize.py \
    --checkpoint outputs/experiment/best_model.pt \
    --nodes 500 \
    --output-dir outputs/viz \
    --seed 42
```

**可视化输出：**

| 文件 | 内容 |
|------|------|
| `tsne_embeddings.png` | 双面板：按类别着色 + 按辐射力着色（Top-K 高亮） |
| `influence_distribution.png` | 四面板：直方图 + 度-影响力散点图 + 对数散点 + Top-20柱状图 |
| `training_curves.png` | 双面板：Loss曲线 + 学习率调度曲线 |

### 4. 运行测试

```bash
python -m pytest tests/ -v
```

预期输出：**24 passed**

---

## CLI 使用指南

### train.py — 模型训练

```bash
python scripts/train.py [OPTIONS]
```

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--nodes` | int | 500 | 合成数据商户数量 |
| `--epochs` | int | 500 | 训练轮数，配合早停自动结束 |
| `--lr` | float | 0.001 | Adam 初始学习率 |
| `--latent-dim` | int | 64 | 商户嵌入维度 |
| `--batch-size` | int | 128 | BPR 三元组批次大小 |
| `--output-dir` | str | outputs | 模型和 checkpoints 保存目录 |
| `--seed` | int | 42 | 随机种子（保证可复现） |

**训练流程：**

1. 生成合成消费网络数据（SBM 社区结构 + 距离衰减权重）
2. 边划分：训练 85% / 验证 5% / 测试 10%
3. 构建 BPR 三元组 Dataset（正样本=有边，负样本=无边）
4. 端到端训练 VGAE 模型（Adam + ReduceLROnPlateau）
5. 每 50 epochs 保存 checkpoint，保留最佳模型

### evaluate.py — 模型评估

```bash
python scripts/evaluate.py --checkpoint PATH [OPTIONS]
```

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--checkpoint` | str | **必填** | 模型检查点路径 |
| `--nodes` | int | 500 | 合成数据商户数量 |
| `--output-dir` | str | outputs/eval | 评估结果输出目录 |
| `--seed` | int | 42 | 随机种子 |

**评估指标说明：**

| 指标 | 范围 | 含义 |
|------|------|------|
| **AUC-ROC** | [0, 1] | 区分真实边 vs 非边：>0.5 优于随机，>0.8 良好 |
| **AP** | [0, 1] | 精确率-召回率曲线下面积 |
| **辐射力得分** | [0, 1] | 综合度中心性 + 排序头得分 + 边权重归一化 |
| **Top-K** | — | 按辐射力降序排列的关键商户列表 |

### visualize.py — 可视化生成

```bash
python scripts/visualize.py --checkpoint PATH [OPTIONS]
```

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--checkpoint` | str | **必填** | 模型检查点路径 |
| `--nodes` | int | 500 | 合成数据商户数量 |
| `--output-dir` | str | outputs/viz | 可视化输出目录 |
| `--seed` | int | 42 | 随机种子 |

---

## API 文档

### 数据生成

```python
from deepcf.data.generator import generate_synthetic_data

data = generate_synthetic_data(
    num_nodes=500,         # 商户数量
    num_features=16,       # 节点特征维度
    edge_density=0.05,     # 边密度
    community_k=5,         # 社区数（商业区）
    weight_range=(0.1, 10.0),  # 边权重范围
    seed=42,
)
# Returns: {"features", "adjacency", "weights", "labels", "positions"}
```

### 边划分与特征处理

```python
from deepcf.data.utils import split_edges, normalize_adjacency, scale_features

X_scaled = scale_features(data["features"])
A_norm = normalize_adjacency(data["adjacency"])
splits = split_edges(data["adjacency"], train_ratio=0.85, val_ratio=0.05)
# Returns: {"train_adj", "val_edges", "test_edges", "val_edges_neg", "test_edges_neg"}
```

### BPR 数据集

```python
from deepcf.data.dataset import BPRDataset, collate_bpr_batch
from torch.utils.data import DataLoader

dataset = BPRDataset(train_adj, weights, num_negatives=1, seed=42)
loader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_bpr_batch)
# Each batch: (users, pos_items, neg_items, weights)
```

### 模型构建

```python
from deepcf.config import ModelConfig
from deepcf.model.vgae import DeepCFVGAE

config = ModelConfig(input_dim=16, hidden_dim=128, latent_dim=64, dropout=0.3)
model = DeepCFVGAE(config)

# 前向传播
adj_pred, rank_scores, weight_pred, mu, logvar, z = model(x, edge_index, u, v)

# 获取节点嵌入（eval模式，无噪声）
embeddings = model.get_embeddings(x, edge_index)
```

### 训练

```python
from deepcf.config import DeepCFConfig
from deepcf.train.trainer import Trainer

config = DeepCFConfig()
trainer = Trainer(model, config, train_loader)
history = trainer.train(adj_true)
# history = {"epoch": [...], "train_loss": [...], "lr": [...]}
```

### 评估与可视化

```python
from deepcf.eval.report import generate_report

report = generate_report(
    model, x_tensor, edge_index_tensor,
    adjacency, weights, labels,
    history, test_edges, test_edges_neg,
    output_dir="outputs/eval",
)
# report["link_prediction"] → {"auc": 0.76, "ap": 0.76}
# report["top_10_merchants"] → [{id, score, category}, ...]
```

### 评估指标

```python
from deepcf.train.metrics import (
    compute_auc_ap,              # (AUC, AP)
    compute_precision_recall_at_k,  # (P@K, R@K)
    compute_ndcg_at_k,           # NDCG@K
    compute_mae_rmse,            # (MAE, RMSE)
)
```

### Jupyter Notebook

```bash
jupyter notebook deepcf/notebooks/
```

| Notebook | 内容 |
|----------|------|
| `01_data_exploration.ipynb` | 图统计、度分布、权重分布、社区结构可视化 |
| `02_model_training.ipynb` | 交互配置 → 数据生成 → 训练 → 实时 Loss 曲线 |
| `03_results_analysis.ipynb` | 加载模型 → t-SNE → Top-K → 全量可视化 |

---

## 实验结果

> 实验配置：500 商户 / 5 商业区 / 5,312 条边 / 51,842 参数量 / 100 epochs / CPU

### 训练收敛

```
Loss:  17.81 (epoch  1)
  →    5.34 (epoch 15)  快速下降期：学到图基本结构
  →    3.81 (epoch 30)  平稳优化期：三任务逐步平衡
  →    3.14 (epoch 89)  精细调优期 ★ 最佳模型
  →    3.27 (epoch 100) 收敛平台期
```

### 链接预测

| 指标 | 数值 |
|------|------|
| AUC-ROC | **0.7642** |
| Average Precision | **0.7623** |

### Top-10 关键商户

| 排名 | 商户 ID | 社区 | 辐射力得分 |
|------|---------|------|------------|
| 1 | 24 | Community 0 | 0.8284 |
| 2 | 162 | Community 1 | 0.8278 |
| 3 | 83 | Community 0 | 0.8248 |
| 4 | 159 | Community 1 | 0.8231 |
| 5 | 33 | Community 0 | 0.8105 |
| 6 | 61 | Community 0 | 0.7936 |
| 7 | 104 | Community 1 | 0.7898 |
| 8 | 21 | Community 0 | 0.7792 |
| 9 | 77 | Community 0 | 0.7686 |
| 10 | 150 | Community 1 | 0.7623 |

### 社区影响力层级

```
  Community 0 ── 主导层 (Top-20 占 45%)
  Community 1 ── 主导层 (Top-20 占 30%)
  Community 3 ── 中间层 (Top-20 占 15%)
  Community 2 ── 边缘层 (无 Top-20 商户)
  Community 4 ── 边缘层 (无 Top-20 商户)
```

### 辐射力统计

| 统计量 | 数值 |
|--------|------|
| 均值 | 0.578 |
| 标准差 | 0.102 |
| 最大值 | 0.828 |
| 最小值 | 0.153 |
| **极差倍数** | **5.4×** |

---

## 引用

本项目基于以下经典文献：

- **VGAE**: Kipf, T. N., & Welling, M. (2016). *Variational Graph Auto-Encoders.* NIPS 2016.
- **BPR**: Rendle, S., et al. (2009). *BPR: Bayesian Personalized Ranking from Implicit Feedback.* UAI 2009.
- **GCN**: Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR 2017.

```bibtex
@misc{deepcf2026,
  title   = {DeepCF: Deep Consumption Flow Network for Urban Merchant Influence Modeling},
  author  = {DeepCF Team},
  year    = {2026},
  url     = {https://github.com/your-repo/deepcf}
}
```

---

## License

MIT License © 2026
