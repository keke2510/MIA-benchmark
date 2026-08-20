# MIA-Bench

**Benchmarking Membership Inference Attacks on Machine Unlearning**

[![TKDE](https://img.shields.io/badge/IEEE-TKDE-blue)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=69)

MIA-Bench is a comprehensive benchmarking framework for systematically evaluating membership inference attacks (MIAs) on machine unlearning. It establishes a standardized five-dimensional evaluation protocol—**E**ffectiveness, **S**tability, **A**pplicability, **C**ost, and **P**racticality—and integrates representative attacks, unlearning algorithms, and datasets into a unified, reproducible platform.

> 📄 **Paper**: submitted to IEEE TKDE (CCF-A)

---

## Highlights

- **Five-dimensional evaluation**: Effectiveness, Stability, Applicability, Cost, Practicality
- **6 attack methods**: LiRA, REA, RULI, UnlearningLeaks + Threshold, Loss baselines
- **4 unlearning algorithms**: Retrain, Finetuning, NegGrad, SCRUB
- **4 benchmark datasets**: CIFAR-10, CIFAR-100, TinyImageNet, CINIC-10
- **288 experimental settings** across all dimensions
- **Unified API** for attacks and unlearning methods — extensible for future work
- **Automated multi-dimensional reporting**: AUC, Accuracy, TPR@FPR, σ, CV, runtime

---

## Supported Attacks

| Attack | Threat Model | Key Signal | Paper |
|--------|-------------|------------|-------|
| **LiRA** | Gray-box (shadow models) | Per-sample confidence distribution | Carlini et al., IEEE S&P 2022 |
| **REA** (Reminiscence Attack) | Algorithm-specific | Residual parameter gradients | Xiao et al., ICCV 2025 |
| **RULI** | Gray-box (shadow models) | Population-level likelihood inference | Naderloui et al., USENIX Security 2025 |
| **UnlearningLeaks** | Algorithm-specific | Pre/post-unlearning posterior divergence | Chen et al., ACM CCS 2021 |
| **Threshold** (baseline) | Black-box | Prediction confidence thresholding | Shokri et al., IEEE S&P 2017 |
| **Loss** (baseline) | Black-box | Per-sample loss values | Yeom et al., IEEE CSF 2018 |

## Supported Unlearning Algorithms

| Algorithm | Type | Description |
|-----------|------|-------------|
| **Retrain** | Exact (gold standard) | Train from scratch on retain set |
| **Finetuning** | Approximate | Fine-tune on retain set for a few epochs |
| **NegGrad** | Approximate | Gradient ascent on forget set + descent on retain set |
| **SCRUB** | Approximate | Teacher-student distillation with selective obedience |

## Repository Structure

```
mia-benchmark-main/
├── run_benchmark.py              # Main benchmark entry point
├── benchmark/samplewise.py       # Dataset presets, stage orchestration, attack dispatch
├── attacks/                      # Attack method adapters (lira, rea, ruli, unlearningleaks, threshold, loss)
├── evaluation/                   # Multi-dimensional metrics (effectiveness, stability, cost, realism)
├── models/                       # Model architectures (ResNet, ViT)
├── datasets/                     # Dataset loaders (CIFAR-10/100, TinyImageNet, CINIC-10)
├── forget_random_strategies.py   # Unlearning algorithm implementations
├── Ruli/                         # RULI attack engine
├── scripts/                      # Batch experiment scripts
├── config.py                     # Global configuration
└── utils/                        # Training and evaluation utilities
```

---

## Quick Start

### Requirements

- Python 3.8+
- PyTorch 1.10+ with CUDA
- RTX 3090 or better (24GB+ VRAM recommended)

```bash
pip install -r requirements.txt
```

### Run a single experiment

```bash
python run_benchmark.py \
  --dataset Cifar10 \
  --seed 1 \
  --methods finetune \
  --attacks lira rea \
  --stages pretrain shadow unlearn reminiscence attack
```

### Run all attacks on one unlearning method

```bash
python run_benchmark.py \
  --dataset Cifar10 \
  --seed 1 \
  --methods finetune \
  --attacks lira rea ruli unlearningleaks \
  --stages unlearn attack
```

### Dry-run (check commands without execution)

```bash
python run_benchmark.py \
  --dataset Cifar10 \
  --seed 1 \
  --methods finetune \
  --attacks lira rea ruli unlearningleaks \
  --dry-run
```

---

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| `pretrain` | Train the base target model on full training data |
| `shadow` | Train reusable shadow models for LiRA/RULI |
| `unlearn` | Apply the chosen unlearning algorithm |
| `reminiscence` | REA-specific: fine-tune on retain set to amplify residual signals |
| `attack` | Run selected MIA methods and collect metrics |

---

## Evaluation Metrics

### Effectiveness
- AUC (Area Under ROC Curve)
- Accuracy
- TPR@FPR (0.1%, 1%, 10%)

### Stability
- Standard deviation (σ) across 3 random seeds
- Coefficient of variation (CV) across datasets and algorithms

### Cost
- Training time (hours)
- Query budget (queries per sample)
- Storage (GB)

### Output Structure

```
benchmark_results/samplewise/<experiment>/<method>/<attack>/seed_<seed>/
├── attack_result.json
├── ruli_summary.json
├── attack logs
└── ROC arrays (LiRA/REA)

log_files/model/forget_random_main/<experiment>/unlearning/<method>/
└── log_<net>-<dataset>-<classes>.tsv   # TA, UA, RA, ZRF, MIA, time
```

---

## CLI Reference

```
--dataset      Cifar10 | Cifar100 | TinyImageNet | Cinic10
--methods      retrain | finetune | negative_grad | scrub
--attacks      lira | rea | ruli | unlearningleaks | threshold | loss
--stages       pretrain shadow unlearn reminiscence attack
--seed         <int>
--forget-perc  0.1
--num-shadow   8
--num-aug      10
--dry-run
```

### Attack-specific options

**RULI**:
```
--ruli-task selective
--ruli-shadow-num 8
--ruli-train-shadow-mode auto
```

**UnlearningLeaks**:
```
--unlearningleaks-feature direct_diff
--unlearningleaks-attack-model lr
--unlearningleaks-num-shadow 8
```

---

## Citation

```bibtex
@article{liu2026miabench,
  title={MIA-Bench: Benchmarking Membership Inference Attacks on Machine Unlearning},
  author={Liu, Shang and Hu, Jingchao and Yang, Zhan and Li, Yonggang and Liu, Jinfei and Cao, Yang},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2026},
  note={Under review}
}
```

## License

This project is released under the [MIT License](LICENSE).
