<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.7+-red?style=flat-square&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-success?style=flat-square" alt="Status">
</p>

<h1 align="center">🔬 MIA-benchmark</h1>

<p align="center">
  <strong>Machine Unlearning Membership Inference Attack Benchmark</strong>
</p>

<p align="center">
  <i>A sample-wise machine unlearning benchmark with unified evaluation across multiple privacy attacks and an integrated unlearning method, MUSE</i>
</p>

---

## 📊 Overview

MIA-benchmark is a comprehensive evaluation framework for **privacy-preserving machine learning** and **machine unlearning**, focusing on:

- 🎯 **Unified Evaluation Protocol**: Five-dimensional unified evaluation for sample-wise forget index
- 🔬 **Diverse Attack Methods**: Integration of LiRA, REA, RULI, UnlearningLeaks, and other mainstream attacks
- 🛡️ **MUSE Defense Method**: Novel Membership-Undistinguishable Sample Erasure approach
- 📈 **Comprehensive Metrics**: Unified output for utility and privacy metrics

---

## ✨ Key Features

### 🏛️ Unified Architecture
- **Single Benchmark Entry**: Run full pipeline via [`run_benchmark.py`](./run_benchmark.py)
- **Modular Design**: Lightweight integration of new unlearning methods
- **Reproducibility**: Multi-round experiments with shared `seed + sample` per round

### 🔐 Supported Attack Methods

| Attack Method | Description | Characteristics |
|---------------|-------------|-----------------|
| **LiRA** | Likelihood Ratio Attack | Likelihood ratio-based membership inference |
| **REA** | Raisin Evaluation Attack | Requires additional reminiscence stage |
| **RULI** | Re-supplemented Unlearning Inference | Maintains independent shadow attack assets |
| **UnlearningLeaks** | Unlearning Information Leakage | Generates shadow-unlearned assets |

### 🎯 Supported Datasets/Models

| Dataset | Backbone | Notes |
|---------|----------|-------|
| **CIFAR-10** | ResNet18 | Classic image classification |
| **CIFAR-100** | ResNet50 | Fine-grained classification |
| **TinyImageNet** | ViT | Vision Transformer |

### 🔬 Supported Unlearning Methods

- `retrain` - Retraining baseline
- `finetune` - Fine-tuning method
- `negative_grad` - Negative gradient method
- `scrub` - Scrub method
- **`muse`** - Our proposed method ⭐

---

## 🛡️ About MISE

### What is MISE?

**MISE (Membership-Indistinguishable Sample Erasure)** is a novel unlearning method implemented on top of the existing sample-wise framework.

### Core Idea

Maintain the base retain-set objective while adding two regularizers to make **forgotten samples behave more like reference non-member samples**, thus reducing membership inference risk.

### Loss Function

For each training step, MUSE samples:
- forget batch `D_f`
- retain batch `D_r`
- reference non-member batch `D_t`

Total loss:

```text
L_total = L_base + λ_align × L_align + λ_stat × L_stat
```

Where:
- `L_base = CE(model(x_r), y_r)` - Base classification loss
- `L_align` - Batch-mean softmax distribution alignment between forget and non-member batches
- `L_stat` - Attack-sensitive statistics alignment:
  - Maximum softmax confidence
  - Top1-top2 logit margin

### Implementation Files

- [`muse/trainer.py`](./muse/trainer.py) - MUSE trainer
- [`forget_random_strategies.py`](./forget_random_strategies.py) - Forgetting strategies
- [`forget_sample_main.py`](./forget_sample_main.py) - Sample-wise unlearning entry

---

## 📁 Repository Structure

```
MIA-benchmark/
├── run_benchmark.py              # Main benchmark entry ⭐
├── benchmark/
│   └── samplewise.py             # Dataset presets, stage orchestration, attack dispatch
├── forget_sample_main.py         # Sample-wise unlearning entry
├── rea_reminiscence_random.py    # REA-specific reminiscence stage
├── attacks/                       # Attack method adapters
├── mise/                          # MISE implementation ⭐
├── Ruli/                          # RULI attack code
├── scripts/                       # Batch experiment scripts
├── models/                        # Model definitions
├── evaluation/                     # Evaluation metrics
└── utils/                         # Utility functions
```

---

## 🚀 Quick Start

### Environment Setup

Current project environment:

```bash
# Create conda environment
conda create -n exp python=3.10 -y
conda activate exp

# Install dependencies
pip install -r requirements.txt
```

**Verified environment:**
- Python: `3.10.0`
- PyTorch: `2.7.1+cu128`
- TorchVision: `0.22.1+cu128`
- GPU: `NVIDIA GeForce RTX 5090`

### Example 1: Run finetune + REA attack on CIFAR-10

**Full pipeline (with pretrain and shadow):**

```bash
python run_benchmark.py \
  --dataset Cifar10 \
  --seed 1 \
  --methods finetune \
  --attacks rea \
  --stages pretrain shadow unlearn reminiscence attack
```

**Unlearn and attack only (reuse existing shadow):**

```bash
python run_benchmark.py \
  --dataset Cifar10 \
  --seed 1 \
  --methods finetune \
  --attacks rea \
  --stages unlearn reminiscence attack
```

### Example 2: One method + four attacks

```bash
python run_benchmark.py \
  --dataset Cifar10 \
  --seed 1 \
  --methods finetune \
  --attacks lira rea ruli unlearningleaks \
  --stages unlearn attack
```

### Example 3: Dry-run to check command

```bash
python run_benchmark.py \
  --dataset Cifar10 \
  --seed 1 \
  --methods finetune \
  --attacks lira rea ruli unlearningleaks \
  --stages unlearn reminiscence attack \
  --dry-run
```

---

## 📊 Benchmark Pipeline

### Supported Stages

| Stage | Description | Reusable |
|-------|-------------|----------|
| `pretrain` | Train target model | ❌ |
| `shadow` | Prepare shadow models | ✅ |
| `unlearn` | Run chosen unlearning method | ❌ |
| `reminiscence` | REA-specific additional stage | ❌ |
| `attack` | Run selected attacks | ❌ |

**Important behavior:**
- `shadow` is reusable if checkpoints exist
- `pretrain`, `unlearn`, `reminiscence`, and `attack` can be rerun per seed/sample
- REA requires additional `reminiscence` stage

---

## ⚙️ Main Parameters

### Basic Parameters

```bash
--dataset Cifar10|Cifar100|TinyImageNet|Cinic10
--methods method or method:para1:para2
--stages pretrain shadow unlearn reminiscence attack
--attacks lira rea ruli unlearningleaks
--seed 1
--forget-perc 0.1          # Forgetting ratio
--num-shadow 8              # Number of shadow models
--num-aug 10                # Data augmentation count
--dry-run                   # Trial run check
```

### Attack-Specific Parameters

**RULI:**
```bash
--ruli-task selective
--ruli-shadow-num 8
--ruli-train-shadow-mode auto
```

**UnlearningLeaks:**
```bash
--unlearningleaks-feature direct_diff
--unlearningleaks-attack-model lr
--unlearningleaks-num-shadow 8
```

### MUSE Hyperparameters

Dataset-specific defaults in [`config.py`](./config.py):

| Dataset | lr | epochs | λ_align | λ_stat |
|----------|-----|--------|----------|--------|
| CIFAR-10 | 4e-4 | 7 | 1.0 | 0.5 |
| CIFAR-100 | 3e-4 | 8 | 1.0 | 0.75 |
| CINIC-10 | 4e-4 | 8 | 1.0 | 0.5 |
| TinyImageNet | 3e-4 | 10 | 0.5 | 0.25 |

**Manual override:**

```bash
python run_benchmark.py \
  --dataset Cifar10 \
  --seed 1 \
  --methods muse:4e-4:7 \
  --attacks rea \
  --stages unlearn reminiscence attack
```

---

## 📈 Output Results

### Unlearning Metrics

Per-method unlearning results stored at:

```
log_files/model/forget_random_main/<experiment>/unlearning/<method_key>/
```

**TSV file records:**
- `clean_acc` → **TA** (Test Accuracy)
- `forgetting_acc` → **UA** (Unlearning Accuracy)
- `remaining_acc` → **RA** (Retain Accuracy)
- `zrf` (Zero-Residual Forgetting)
- `mia` (MIA attack success rate)
- `time` (Runtime)

### Attack Metrics

Attack results stored at:

```
benchmark_results/samplewise/<experiment>/<method_key>/<attack>/seed_<seed>/
```

**Output files:**
- `attack_result.json` - Attack result summary
- `ruli_summary.json` - RULI-specific results
- Attack log files
- ROC-related arrays for LiRA/REA

### Run Summary

Each benchmark run saves:

```
benchmark_results/samplewise/<experiment>/run_summary_seed_<seed>.json
```

---

## 📊 Metrics Explanation

### Unlearning Utility Metrics

| Metric | Full Name | Meaning |
|--------|-----------|---------|
| **TA** | Test Accuracy | Accuracy on test set |
| **UA** | Unlearning Accuracy | Accuracy on forget set |
| **RA** | Retain Accuracy | Accuracy on retain set |

### Privacy Attack Metrics

Standardized output:
- **AUC** - Area under ROC curve
- **Accuracy** - Attack accuracy
- **TPR@0.1%FPR** - True positive rate at 0.1% FPR
- **TPR@1%FPR** - True positive rate at 1% FPR
- **TPR@10%FPR** - True positive rate at 10% FPR ⭐

**Key metric:** Following the stricter legacy definition with fixed 0.1% FPR, the most critical privacy metric is **TPR@10%FPR**.

---

## 🧪 Multi-Round Experiments

### Batch Script Example

Run five rounds on CIFAR-10 with shared seed/sample per round, comparing four attacks and five unlearning methods (including `muse`):

```bash
PYTHON_BIN=/root/miniconda3/envs/python \
bash ./scripts/run_cifar10_five_rounds_muse.sh
```

**Script features:**
- ✅ One random seed per round
- ✅ One shared forget sample per round
- ✅ Reuse existing shadow checkpoints
- ✅ Rerun pretrain/unlearn/attack each round
- ✅ Skip failed sub-experiments and continue
- ✅ Output multi-round `mean/std`

---

## 📝 Notes

- `REA` requires additional `reminiscence` stage
- `RULI` maintains its own internal shadow attack assets
- `UnlearningLeaks` uses shared shadows and builds shadow-unlearned assets
- Contains several third-party components; benchmark wrapper is recommended for new experiments

---

## 📧 Contact

- **Author:** Jingchao Hu
- **Email:** [08235752@cumt.edu.cn](mailto:08235752@cumt.edu.cn)
- **University:** China University of Mining and Technology (211)

---

<p align="center">
  <sub> built with ❤️ for privacy-preserving machine learning research </sub>
</p>
