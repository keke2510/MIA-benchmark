"""编码器测试."""

import torch
from deepcf.model.encoder import GCNEncoder


def test_encoder_output_shapes():
    """测试编码器输出形状."""
    encoder = GCNEncoder(input_dim=16, hidden_dim=128, latent_dim=64, dropout=0.3)
    encoder.eval()

    x = torch.randn(100, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

    mu, logvar = encoder(x, edge_index)

    assert mu.shape == (100, 64)
    assert logvar.shape == (100, 64)


def test_reparameterize_shape():
    """测试重参数化输出形状."""
    from deepcf.model.encoder import reparameterize

    mu = torch.randn(50, 64)
    logvar = torch.randn(50, 64)
    z = reparameterize(mu, logvar)

    assert z.shape == (50, 64)


def test_reparameterize_deterministic_eval():
    """测试 eval 模式下重参数化不引入噪声."""
    from deepcf.model.encoder import reparameterize

    mu = torch.ones(10, 8)
    logvar = torch.zeros(10, 8)

    torch.manual_seed(42)
    z1 = reparameterize(mu, logvar, training=False)
    z2 = reparameterize(mu, logvar, training=False)

    # eval 模式下不应有随机性
    assert torch.allclose(z1, mu, atol=1e-6)
    assert torch.allclose(z2, mu, atol=1e-6)


def test_encoder_train_mode():
    """测试训练模式下输出有变化."""
    from deepcf.model.encoder import reparameterize

    mu = torch.zeros(5, 4)
    logvar = torch.zeros(5, 4)

    torch.manual_seed(1)
    z1 = reparameterize(mu, logvar, training=True)
    torch.manual_seed(2)
    z2 = reparameterize(mu, logvar, training=True)

    # 不同种子应产生不同噪声
    assert not torch.allclose(z1, z2)
