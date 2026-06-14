<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.7+-red?style=flat-square&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-success?style=flat-square" alt="Status">
</p>

<h1 align="center">🔬 MIA-benchmark</h1>

<p align="center">
  <strong>机器遗忘成员推理攻击统一评测框架</strong>
</p>

<p align="center">
  <i>面向 sample-wise machine unlearning 的 benchmark 项目，统一接入多种隐私攻击评估方法，并实现新的遗忘方法 MUSE</i>
</p>

---

## 📊 项目简介

MIA-benchmark 是一个专注于 **机器遗忘隐私安全** 的综合评测框架，旨在：

- 🎯 **统一评测协议**：五维度统一评测 sample-wise forget index
- 🔬 **多样化攻击方法**：集成 LiRA、REA、RULI、UnlearningLeaks 等主流攻击
- 🛡️ **MUSE 防御方法**：提出 Membership-Undistinguishable Sample Erasure 新方法
- 📈 **全面指标评估**：统一输出 utility 和 privacy 指标

---

## ✨ 主要特性

### 🏛️ 统一架构
- **统一 benchmark 入口**：[`run_benchmark.py`](./run_benchmark.py) 一键运行全流程
- **模块化设计**：在尽量少改原框架的前提下接入新遗忘方法
- **可复现性保证**：支持多轮实验，每轮内部共享同一组 `seed + sample`

### 🔐 已接入攻击方法

| 攻击方法 | 描述 | 特点 |
|---------|------|------|
| **LiRA** | Likelihood Ratio Attack | 基于似然比的成员推理 |
| **REA** | Raisin Evaluation Attack | 需要额外 reminiscence 阶段 |
| **RULI** | Re-supplemented Unlearning Inference | 维护独立 shadow attack 资产 |
| **UnlearningLeaks** | 遗忘信息泄露攻击 | 生成 shadow-unlearned assets |

### 🎯 支持的数据集/模型

| 数据集 | 骨干网络 | 备注 |
|--------|---------|------|
| **CIFAR-10** | ResNet18 | 经典图像分类 |
| **CIFAR-100** | ResNet50 | 细粒度分类 |
| **TinyImageNet** | ViT | Vision Transformer |

### 🔬 已接入遗忘方法

- `retrain` - 重新训练基线
- `finetune` - 微调方法
- `negative_grad` - 负梯度方法
- `scrub` - Scrub 方法
- **`muse`** - 本项目提出的新方法 ⭐

---

## 🛡️ MUSE 方法详解

### 什么是 MUSE？

**MUSE (Membership-Undistinguishable Sample Erasure)** 是在现有 sample-wise 遗忘主线上的一个新方法。

### 核心思想

保留 retain set 上的基础目标，同时加入两个正则项，让 **forget 样本的输出行为更接近 reference non-member 样本**，从而降低 membership inference 风险。

### 损失函数

每个 step 同时采样：
- forget batch `D_f`
- retain batch `D_r`  
- reference non-member batch `D_t`

总损失为：

```text
L_total = L_base + λ_align × L_align + λ_stat × L_stat
```

其中：
- `L_base = CE(model(x_r), y_r)` - 基础分类损失
- `L_align` - forget batch 与 non-member batch 的 batch-mean softmax 分布对齐
- `L_stat` - forget batch 与 non-member batch 的攻击敏感统计量对齐
  - 最大 softmax confidence
  - top1-top2 logit margin

### 主要实现文件

- [`muse/trainer.py`](./muse/trainer.py) - MUSE 训练器
- [`forget_random_strategies.py`](./forget_random_strategies.py) - 遗忘策略
- [`forget_sample_main.py`](./forget_sample_main.py) - sample-wise 遗忘主入口

---

## 📁 仓库结构

```
MIA-benchmark/
├── run_benchmark.py              # 主 benchmark 入口 ⭐
├── benchmark/
│   └── samplewise.py             # 数据集预设、stage 编排、攻击分发
├── forget_sample_main.py         # sample-wise 遗忘主入口
├── rea_reminiscence_random.py    # REA 专用 reminiscence 阶段
├── attacks/                       # 攻击方法适配层
├── muse/                          # MUSE 实现 ⭐
├── Ruli/                          # RULI 攻击代码
├── scripts/                       # 批量实验脚本
├── models/                        # 模型定义
├── evaluation/                     # 评估指标
└── utils/                         # 工具函数
```

---

## 🚀 快速开始

### 环境配置

当前项目使用的主要环境：

```bash
# 创建 conda 环境
conda create -n exp python=3.10 -y
conda activate exp

# 安装依赖
pip install -r requirements.txt
```

**验证环境：**
- Python: `3.10.0`
- PyTorch: `2.7.1+cu128`
- TorchVision: `0.22.1+cu128`
- GPU: `NVIDIA GeForce RTX 5090`

### 示例 1：在 CIFAR-10 上运行 finetune + REA 攻击

**完整流程（含 pretrain 和 shadow）：**

```bash
python run_benchmark.py \
  --dataset Cifar10 \
  --seed 1 \
  --methods finetune \
  --attacks rea \
  --stages pretrain shadow unlearn reminiscence attack
```

**仅运行遗忘和攻击（复用已有 shadow）：**

```bash
python run_benchmark.py \
  --dataset Cifar10 \
  --seed 1 \
  --methods finetune \
  --attacks rea \
  --stages unlearn reminiscence attack
```

### 示例 2：一个遗忘方法 + 四种攻击

```bash
python run_benchmark.py \
  --dataset Cifar10 \
  --seed 1 \
  --methods finetune \
  --attacks lira rea ruli unlearningleaks \
  --stages unlearn attack
```

### 示例 3：Dry-run 检查命令链路

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

## 📊 Benchmark 流程

### 支持的 Stages

| Stage | 描述 | 可复用 |
|-------|------|--------|
| `pretrain` | 训练 target model | ❌ |
| `shadow` | 准备 shadow models | ✅ |
| `unlearn` | 运行指定遗忘方法 | ❌ |
| `reminiscence` | REA 专用附加阶段 | ❌ |
| `attack` | 运行攻击方法 | ❌ |

**重要行为：**
- `shadow` 如果已有 checkpoint，会直接复用
- `pretrain`、`unlearn`、`reminiscence`、`attack` 可以针对不同 seed/sample 反复重跑
- `REA` 需要额外的 `reminiscence`，其他攻击不需要

---

## ⚙️ 常用参数

### 基础参数

```bash
--dataset Cifar10|Cifar100|TinyImageNet|Cinic10
--methods method 或 method:para1:para2
--stages pretrain shadow unlearn reminiscence attack
--attacks lira rea ruli unlearningleaks
--seed 1
--forget-perc 0.1          # 遗忘比例
--num-shadow 8              # shadow 模型数量
--num-aug 10                # 数据增强数量
--dry-run                   # 试运行检查
```

### 攻击方法特定参数

**RULI：**
```bash
--ruli-task selective
--ruli-shadow-num 8
--ruli-train-shadow-mode auto
```

**UnlearningLeaks：**
```bash
--unlearningleaks-feature direct_diff
--unlearningleaks-attack-model lr
--unlearningleaks-num-shadow 8
```

### MUSE 超参配置

数据集默认超参定义在 [`config.py`](./config.py)：

| 数据集 | lr | epochs | λ_align | λ_stat |
|--------|-----|--------|----------|--------|
| CIFAR-10 | 4e-4 | 7 | 1.0 | 0.5 |
| CIFAR-100 | 3e-4 | 8 | 1.0 | 0.75 |
| CINIC-10 | 4e-4 | 8 | 1.0 | 0.5 |
| TinyImageNet | 3e-4 | 10 | 0.5 | 0.25 |

**手动覆盖超参：**

```bash
python run_benchmark.py \
  --dataset Cifar10 \
  --seed 1 \
  --methods muse:4e-4:7 \
  --attacks rea \
  --stages unlearn reminiscence attack
```

---

## 📈 输出结果

### 遗忘阶段指标

每个方法的 `unlearning` 结果存放路径：

```
log_files/model/forget_random_main/<experiment>/unlearning/<method_key>/
```

**TSV 文件记录指标：**
- `clean_acc` → **TA** (测试集准确率)
- `forgetting_acc` → **UA** (forget set 准确率)
- `remaining_acc` → **RA** (retain set 准确率)
- `zrf` (Zero-Residual Forgetting)
- `mia` (MIA 攻击成功率)
- `time` (运行时间)

### 攻击阶段指标

攻击结果存放路径：

```
benchmark_results/samplewise/<experiment>/<method_key>/<attack>/seed_<seed>/
```

**输出文件：**
- `attack_result.json` - 攻击结果汇总
- `ruli_summary.json` - RULI 特定结果
- 攻击日志文件
- LiRA/REA 的 ROC 相关数组

### 顶层运行摘要

每次 benchmark 运行会保存：

```
benchmark_results/samplewise/<experiment>/run_summary_seed_<seed>.json
```

---

## 📊 指标说明

### 遗忘效能指标

| 指标 | 全称 | 含义 |
|------|------|------|
| **TA** | Test Accuracy | 测试集准确率 |
| **UA** | Unlearning Accuracy | forget set 上的准确率 |
| **RA** | Retain Accuracy | retain set 上的准确率 |

### 隐私攻击指标

统一输出为：
- **AUC** - ROC 曲线下面积
- **Accuracy** - 攻击准确率
- **TPR@0.1%FPR** - 0.1% FPR 时的真正率
- **TPR@1%FPR** - 1% FPR 时的真正率
- **TPR@10%FPR** - 10% FPR 时的真正率 ⭐

**关键指标：** 采用较严格的 legacy 定义，固定 0.1 FPR，最关键的隐私指标是 **TPR@10%FPR**。

---

## 🧪 多轮实验

### 批量实验脚本示例

在 CIFAR-10 上运行五轮实验，每轮内部共享同一组 `seed + sample`，同时比较四种攻击和五种遗忘方法（含 `muse`）：

```bash
PYTHON_BIN=/root/miniconda3/envs/python \
bash ./scripts/run_cifar10_five_rounds_muse.sh
```

**脚本功能：**
- ✅ 每轮生成一个随机 `seed`
- ✅ 每轮生成一份共享的 forget sample
- ✅ 复用已有的 `shadow` checkpoint
- ✅ 每轮重新跑 `pretrain / unlearn / attack`
- ✅ 中途某个实验失败时自动跳过并继续
- ✅ 最后输出多轮 `mean/std`

---

## 📝 备注

- `REA` 需要额外的 `reminiscence` 阶段
- `RULI` 除了复用 benchmark 主线信息，还会维护它自己的一套 shadow attack 资产
- `UnlearningLeaks` 会复用主线 shadow，同时生成自己的 shadow-unlearned assets
- 仓库里有多个第三方子模块；如果只是想做统一实验，建议优先使用 benchmark 入口

---

## 📧 联系方式

- **作者:** 胡景超
- **邮箱:** [08235752@cumt.edu.cn](mailto:08235752@cumt.edu.cn)
- **学校:** 中国矿业大学（211）
- **论文状态:** 《机器遗忘学习的成员推理攻击与防御方法研究》 - Cybersecurity 期刊在投

---

<p align="center">
  <sub> built with ❤️ for privacy-preserving machine learning research </sub>
</p>
