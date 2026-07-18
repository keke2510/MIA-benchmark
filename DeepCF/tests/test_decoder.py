"""解码器测试."""

import torch
from deepcf.model.decoder import LinkDecoder, RankDecoder, WeightDecoder, MultiHeadDecoder


def test_link_decoder_shape():
    """测试链接预测头输出形状."""
    decoder = LinkDecoder()
    z = torch.randn(30, 64)

    # 全量预测
    adj_pred = decoder(z)
    assert adj_pred.shape == (30, 30)
    assert torch.all((adj_pred >= 0) & (adj_pred <= 1))

    # 边级别预测
    edge_indices = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    scores = decoder.decode_edges(z, edge_indices)
    assert scores.shape == (3,)


def test_rank_decoder_shape():
    """测试排序头输出形状."""
    decoder = RankDecoder(latent_dim=64, hidden_dim=64)
    z = torch.randn(50, 64)
    u = torch.tensor([0, 1, 2])
    v = torch.tensor([1, 2, 3])

    scores = decoder(z, u, v)
    assert scores.shape == (3,)


def test_weight_decoder_shape():
    """测试权重回归头输出形状."""
    decoder = WeightDecoder(latent_dim=64, hidden_dim=64)
    z = torch.randn(50, 64)
    u = torch.tensor([0, 1, 2])
    v = torch.tensor([1, 2, 3])

    pred_weights = decoder(z, u, v)
    assert pred_weights.shape == (3,)
    assert torch.all(pred_weights >= 0)  # 权重应为非负


def test_multi_head_decoder_outputs():
    """测试多头解码器联合输出."""
    decoder = MultiHeadDecoder(latent_dim=64, hidden_dim=64)
    z = torch.randn(40, 64)
    u = torch.tensor([0, 1, 2, 3])
    v = torch.tensor([1, 2, 3, 4])

    adj_pred, rank_scores, weight_pred = decoder(z, u, v)

    assert adj_pred.shape == (40, 40)
    assert rank_scores.shape == (4,)
    assert weight_pred.shape == (4,)
