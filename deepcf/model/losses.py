"""多任务损失函数."""

import torch
import torch.nn.functional as F
from typing import Dict


def bpr_loss(s_pos: torch.Tensor, s_neg: torch.Tensor) -> torch.Tensor:
    """BPR成对排序损失: -mean(log(σ(s_pos - s_neg)))."""
    diff = s_pos - s_neg
    return -torch.mean(F.logsigmoid(diff))


def kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL散度: -0.5 * mean(1 + logvar - mu² - exp(logvar))."""
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
    """多任务联合损失."""
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
