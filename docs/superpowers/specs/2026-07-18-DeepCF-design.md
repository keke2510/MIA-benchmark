# DeepCF Design Specification

> **项目名称**: DeepCF — 基于图表示学习的城市消费网络建模与关键商户消费辐射最大化研究
>
> **Date**: 2026-07-18
>
> **Status**: Approved

---

## 1. Overview

### 1.1 Problem Statement

城市消费网络中，不同商户对消费流的辐射/吸引能力差异巨大。识别网络中具有最大消费辐射影响力的关键商户，对商业选址、营销资源分配、城市商业规划等具有重要意义。

### 1.2 Proposed Method

DeepCF 基于 **Variational Graph Autoencoder (VGAE)** 框架，使用双层 GCN 编码器学习商户节点的 64 维潜在表示，并通过三头解码器同时完成：(A) 商户间链接预测、(B) 影响力成对排序、(C) 边权重回归。最终基于学到的嵌入识别 Top-K 关键商户并生成多维可视化分析。

### 1.3 Key References

- VGAE: Kipf & Welling, "Variational Graph Auto-Encoders" (NIPS 2016)
- BPR: Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI 2009)
- GCN: Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks" (ICLR 2017)

---

## 2. Graph Definition

| Property | Value |
|----------|-------|
| Node | Merchant (商户) |
| Edge | Consumption transfer relationship (消费转移关系) |
| Edge Weight | Consumption amount + transfer frequency (消费金额 + 转移频次) |
| Type | Undirected weighted graph (无向加权图) |
| Node Features | Rich merchant attributes: category, location coordinates, avg spend, rating, etc. |

---

## 3. Model Architecture

### 3.1 Overall: Multi-task VGAE

```
Input: X(N×F) features, A(N×N) adjacency, W(N×N) weights
       │
       ▼
┌──────────────────────┐
│  GCN Encoder         │
│  Layer 1: N×16→N×128 │  ReLU + Dropout(0.3)
│  Layer 2: N×128→μ,σ  │  Shared weight + independent projection
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Reparameterization  │  z = μ + σ ⊙ ε, ε ~ N(0,I), dim=64
└──────────┬───────────┘
           │
     ┌─────┼─────┬──────────┐
     ▼     ▼     ▼          ▼
  ┌────┐┌────┐┌────┐   ┌────────┐
  │Link││Rank││Wgt │   │  KL    │
  │Head││Head││Head│   │  Loss  │
  └──┬─┘└──┬─┘└──┬─┘   └────────┘
     │     │     │
     ▼     ▼     ▼
  BCE(A) BPR(s) MSE(w)   ← Joint Loss
```

### 3.2 Encoder (GCN)

- **GCN Layer 1**: `H₁ = ReLU(D^(-½) Â D^(-½) X W₀)`, output dim = 128, Dropout = 0.3
- **GCN Layer 2 (shared weight)**: `H₂ = D^(-½) Â D^(-½) H₁ W₁`, output dim = 128
- **μ head**: `μ = H₂ W_μ`, output dim = 64
- **log σ² head**: `log σ² = H₂ W_σ`, output dim = 64
- **Reparameterization**: `z = μ + exp(½ log σ²) ⊙ ε`

### 3.3 Decoder (Three Heads)

| Head | Computation | Output | Loss |
|------|-------------|--------|------|
| **Link** | σ(zᵢᵀ zⱼ) — inner product + sigmoid | Âᵢⱼ ∈ [0,1] | BCE(Â, A) |
| **Rank** | MLP([zᵢ ‖ zⱼ]) → sᵢⱼ — 2-layer MLP(128→64→1) | sᵢⱼ ∈ ℝ | BPR(s_pos, s_neg) |
| **Weight** | MLP([zᵢ ‖ zⱼ]) → ŵᵢⱼ — 2-layer MLP(128→64→1) | ŵᵢⱼ ∈ ℝ | MSE(ŵ, w) |

### 3.4 Joint Loss Function

```
L = λ₁ · BCE(Â, A)          # Graph structure reconstruction
  + λ₂ · BPR(s_pos, s_neg)  # Pairwise influence ranking
  + λ₃ · MSE(ŵ, w)          # Edge weight regression
  + β  · KL(q(z|X,A) ‖ p(z))  # Variational regularization
```

Default hyperparameters: `λ₁ = 1.0, λ₂ = 0.5, λ₃ = 0.3, β = 0.001`

---

## 4. Data Pipeline

### 4.1 Data Strategy

| Phase | Data Source | Purpose |
|-------|-------------|---------|
| Prototype & tuning | Synthetic data (generator) | Fast iteration, controlled experiments |
| Main experiments | Public datasets (Yelp / Foursquare) | Real-world credibility |
| Ablation studies | Synthetic data | Controlled parameter analysis |

### 4.2 Synthetic Data Generator

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_nodes` | 500 | Number of merchants |
| `num_features` | 16 | Raw node features (category one-hot + coords + spend + rating) |
| `edge_density` | 0.05 | ~5% of node pairs have edges |
| `community_k` | 5 | Number of business districts (SBM communities) |
| `weight_range` | [0.1, 10.0] | Edge weight range |

Generation logic: SBM → community structure → category-bias edge probability → weight assignment (distance decay + category affinity)

### 4.3 Data Loading

1. **Graph construction**: Feature matrix X(N×F) + Adjacency A(N×N) + Weight matrix W(N×N)
2. **Edge split**: Train 85% / Validation 5% / Test 10%
3. **BPR sampling**: Per node per epoch: 1 positive edge + 1 negative edge → (u, i⁺, i⁻) triplet
4. **Batching**: Mini-batch = 128 triplets, full adjacency matrix for GCN forward pass

---

## 5. Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.001, weight_decay=5e-4) |
| LR Schedule | ReduceLROnPlateau (patience=20, factor=0.5) |
| Epochs | 500 (early stopping patience=50) |
| Batch Size | 128 (BPR triplet batch) |
| Device | GPU (CUDA) / CPU auto-detect |
| Checkpoint | Every 50 epochs + best validation loss |

---

## 6. Evaluation Metrics

| Task | Metrics |
|------|---------|
| Link Prediction | AUC-ROC, AP (Average Precision) |
| Influence Ranking | Precision@K, Recall@K, NDCG@K (K=5,10,20) |
| Weight Regression | MAE, RMSE, R² |
| Comprehensive | Radiation score = α·degree_centrality + β·rank_score + γ·avg_weight |

---

## 7. Visualization & Analysis Outputs

### 7.1 t-SNE Embedding Visualization
- 2D/3D projection of 64-dim merchant embeddings
- Color by merchant category → semantic clustering validation
- Color by radiation score (heatmap) → highlight Top-K high-influence nodes
- Before/after training comparison

### 7.2 Influence Distribution
- Radiation score histogram + long-tail analysis
- Degree vs. Influence scatter plot → identify "low-degree high-influence" nodes
- Geographic heatmap (if coordinates available)
- Community-level influence bar chart

### 7.3 Top-K Key Merchant Identification
- Composite score ranking → Top-10 / Top-20 tables
- Graph visualization with Top-K nodes highlighted
- Cascade propagation simulation: random vs. Top-K seed comparison

### 7.4 Model Analysis
- Training curves (loss + metrics over epochs)
- Ablation: pure GCN vs. GCN+VAE vs. DeepCF full model
- Parameter sensitivity: β, λ ratios impact heatmap
- Embedding dimension analysis: dim=16/32/64/128 comparison

---

## 8. Project Structure

```
deepcf/
├── __init__.py
├── config.py                 # Global config (model, training, paths)
├── data/
│   ├── __init__.py
│   ├── generator.py          # Synthetic consumption network generator
│   ├── dataset.py            # PyTorch Dataset + BPR sampling
│   └── utils.py              # Graph normalization, feature scaling, train/test split
├── model/
│   ├── __init__.py
│   ├── encoder.py            # GCN encoder → μ, σ
│   ├── decoder.py            # Three-head decoder (Link / Rank / Weight)
│   ├── vgae.py               # VGAE main model
│   └── losses.py             # Multi-task loss (BCE + BPR + MSE + KL)
├── train/
│   ├── __init__.py
│   ├── trainer.py            # Training loop (early stop, checkpoint)
│   └── metrics.py            # Evaluation metrics (AUC, P@K, R@K, NDCG)
├── eval/
│   ├── __init__.py
│   ├── ranking.py            # Top-K identification & influence ranking
│   ├── visualize.py          # t-SNE, influence distribution, model analysis
│   └── report.py             # Comprehensive evaluation report
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_results_analysis.ipynb
tests/
├── test_encoder.py
├── test_decoder.py
├── test_vgae.py
├── test_losses.py
└── test_data.py
scripts/
├── train.py                  # Training CLI
├── evaluate.py               # Evaluation CLI
└── visualize.py              # Visualization CLI
```

### Design Principles

- **Single Responsibility**: Each module has one clear purpose
- **Unidirectional Dependencies**: data → model → train → eval
- **Config-driven**: All hyperparameters in config.py
- **Testable**: Each module independently unit-testable

---

## 9. Deliverable Format

**Python package** (`deepcf/`) + **Jupyter Notebooks** (3 notebooks) + **CLI scripts** (train/evaluate/visualize).

---

## 10. Non-Goals (for this iteration)

- Consumer node modeling (graph is merchant-only)
- Real-time / streaming inference
- Distributed training (multi-GPU)
- Web UI or API server
- Temporal dynamics (static graph snapshot)

---

## 11. Dependencies

```
torch >= 2.0
torch-geometric >= 2.3
numpy, scipy, pandas
scikit-learn (t-SNE, metrics)
matplotlib, seaborn
jupyter
tqdm
```

---

## Specification Self-Review

- **Placeholder scan**: ✅ No TBDs or TODOs remain. All key parameters have concrete defaults.
- **Internal consistency**: ✅ Architecture (§3) aligns with pipeline (§5) and evaluation (§6). VGAE structure matches the project description.
- **Scope check**: ✅ Focused and implementable. Non-goals clearly stated. No scope creep.
- **Ambiguity check**: ✅ GCN vs GAT resolved (GCN). VAE role clarified (VGAE encoder). Multi-task approach explicitly defined as shared-encoder + three-head decoder.
