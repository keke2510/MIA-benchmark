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
