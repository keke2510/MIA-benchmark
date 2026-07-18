"""DeepCF VGAE 主模型."""

import torch
import torch.nn as nn
from typing import Dict

from deepcf.config import ModelConfig, LossConfig
from deepcf.model.encoder import GCNEncoder, reparameterize
from deepcf.model.decoder import MultiHeadDecoder
from deepcf.model.losses import multi_task_loss


class DeepCFVGAE(nn.Module):
    """DeepCF: 面向城市消费网络的 VGAE 模型."""

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
    ) -> tuple:
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
        mu, logvar = self.encoder(x, edge_index, edge_weight=edge_weight)
        return reparameterize(mu, logvar, training=False)

    def compute_loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        adj_true: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        neg_u: torch.Tensor,
        neg_v: torch.Tensor,
        edge_weight: torch.Tensor | None,
        weight_true: torch.Tensor,
        loss_config: LossConfig,
    ) -> tuple:
        # 一次编码，两次解码：正样本 + 负样本各算一次 BPR 分数
        mu, logvar = self.encoder(x, edge_index, edge_weight=edge_weight)
        z = reparameterize(mu, logvar, training=self.training)

        adj_pred, rank_pos, weight_pred = self.decoder(z, u, v)
        _, rank_neg, _ = self.decoder(z, neg_u, neg_v)

        return multi_task_loss(
            adj_pred=adj_pred,
            adj_true=adj_true,
            rank_pos=rank_pos,
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
