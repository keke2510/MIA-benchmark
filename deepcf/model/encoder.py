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
