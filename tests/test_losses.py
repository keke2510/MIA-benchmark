"""损失函数测试."""

import torch
from deepcf.model.losses import (
    bpr_loss,
    kl_divergence_loss,
    multi_task_loss,
)


def test_bpr_loss_positive():
    """BPR 损失：正样本得分>负样本时损失应较小."""
    s_pos = torch.tensor([5.0, 4.0, 3.0])
    s_neg = torch.tensor([1.0, 2.0, 0.5])
    loss = bpr_loss(s_pos, s_neg)
    assert loss.item() >= 0
    assert loss.item() < 0.5


def test_bpr_loss_negative():
    """BPR 损失：负样本得分更高时损失应较大."""
    s_pos = torch.tensor([1.0, 2.0])
    s_neg = torch.tensor([5.0, 6.0])
    loss = bpr_loss(s_pos, s_neg)
    assert loss.item() > 1.0


def test_kl_divergence_zero():
    """KL散度：μ=0, σ=1 时 KL=0."""
    mu = torch.zeros(100, 64)
    logvar = torch.zeros(100, 64)
    kl = kl_divergence_loss(mu, logvar)
    assert torch.isclose(kl, torch.tensor(0.0), atol=1e-6)


def test_kl_divergence_positive():
    """KL散度非负."""
    mu = torch.randn(50, 64)
    logvar = torch.randn(50, 64)
    kl = kl_divergence_loss(mu, logvar)
    assert kl.item() >= 0


def test_multi_task_loss_shape():
    """多任务联合损失输出."""
    adj_pred = torch.rand(30, 30)
    adj_true = torch.randint(0, 2, (30, 30)).float()
    rank_pos = torch.tensor([3.0, 4.0, 5.0])
    rank_neg = torch.tensor([1.0, 2.0, 3.0])
    weight_pred = torch.tensor([2.0, 3.0, 4.0])
    weight_true = torch.tensor([2.1, 2.9, 3.8])
    mu = torch.randn(30, 64)
    logvar = torch.randn(30, 64)

    total, components = multi_task_loss(
        adj_pred, adj_true, rank_pos, rank_neg,
        weight_pred, weight_true, mu, logvar,
    )

    assert isinstance(total, torch.Tensor)
    assert total.ndim == 0
    assert "link" in components
    assert "rank" in components
    assert "weight" in components
    assert "kl" in components
    assert "total" in components
