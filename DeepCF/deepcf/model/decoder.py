"""三头解码器：链接预测、影响力排序、边权重回归."""

import torch
import torch.nn as nn


class LinkDecoder(nn.Module):
    """链接预测头：内积 + sigmoid 重构邻接矩阵."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(z @ z.T)

    def decode_edges(
        self, z: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        return torch.sigmoid((z[src] * z[dst]).sum(dim=1))


class RankDecoder(nn.Module):
    """影响力排序头：MLP([z_u ‖ z_v]) → s_uv."""

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
        zu = z[u]
        zv = z[v]
        concat = torch.cat([zu, zv], dim=-1)
        return self.mlp(concat).squeeze(-1)


class WeightDecoder(nn.Module):
    """边权重回归头：MLP([z_u ‖ z_v]) → ŵ_uv. 用 Softplus 保证非负."""

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
        adj_pred = self.link(z)
        rank_scores = self.rank(z, u, v)
        weight_pred = self.weight(z, u, v)
        return adj_pred, rank_scores, weight_pred
