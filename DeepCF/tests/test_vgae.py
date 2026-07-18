"""VGAE 主模型测试."""

import torch
from deepcf.model.vgae import DeepCFVGAE
from deepcf.config import ModelConfig, LossConfig


def test_vgae_forward():
    """测试 VGAE 前向传播."""
    config = ModelConfig(input_dim=16, hidden_dim=128, latent_dim=64)
    model = DeepCFVGAE(config)
    model.eval()

    x = torch.randn(50, 16)
    edge_index = torch.tensor([
        [0, 0, 1, 2, 3],
        [1, 2, 3, 4, 4],
    ], dtype=torch.long)
    u = torch.tensor([0, 1, 2])
    v = torch.tensor([1, 2, 3])

    adj_pred, rank_scores, weight_pred, mu, logvar, z = model(x, edge_index, u, v)

    assert adj_pred.shape == (50, 50)
    assert rank_scores.shape == (3,)
    assert weight_pred.shape == (3,)
    assert mu.shape == (50, 64)
    assert logvar.shape == (50, 64)
    assert z.shape == (50, 64)


def test_vgae_get_embeddings():
    """测试获取嵌入接口."""
    config = ModelConfig(input_dim=16, hidden_dim=128, latent_dim=64)
    model = DeepCFVGAE(config)
    model.eval()

    x = torch.randn(30, 16)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    z = model.get_embeddings(x, edge_index)
    assert z.shape == (30, 64)


def test_vgae_compute_loss():
    """测试计算损失接口."""
    model_config = ModelConfig(input_dim=16, hidden_dim=128, latent_dim=64)
    loss_config = LossConfig()
    model = DeepCFVGAE(model_config)

    x = torch.randn(40, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    adj_true = torch.zeros(40, 40)
    adj_true[0, 1] = adj_true[1, 0] = 1
    adj_true[1, 2] = adj_true[2, 1] = 1

    u = torch.tensor([0, 1])
    v = torch.tensor([1, 2])
    neg_u = torch.tensor([0, 1])
    neg_v = torch.tensor([3, 3])
    w_true = torch.tensor([2.5, 3.5])

    total, components = model.compute_loss(
        x, edge_index, adj_true, u, v, neg_u, neg_v, None, w_true, loss_config
    )

    assert total.ndim == 0
    assert "link" in components
    assert "rank" in components
    assert "weight" in components
    assert "kl" in components
