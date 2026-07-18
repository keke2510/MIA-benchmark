#!/usr/bin/env python
"""DeepCF 训练入口脚本."""

import argparse, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepcf.config import DeepCFConfig
from deepcf.data.generator import generate_synthetic_data
from deepcf.data.utils import split_edges, scale_features
from deepcf.data.dataset import BPRDataset, collate_bpr_batch
from deepcf.model.vgae import DeepCFVGAE
from deepcf.train.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="DeepCF Model Training")
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = DeepCFConfig()
    config.data.num_nodes = args.nodes
    config.train.epochs = args.epochs
    config.train.lr = args.lr
    config.train.batch_size = args.batch_size
    config.model.latent_dim = args.latent_dim
    config.output_dir = args.output_dir
    config.seed = args.seed
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"Device: {config.device}")
    print(f"Nodes: {config.data.num_nodes}, Features: {config.model.input_dim}")

    # Generate data
    print("\n[1/4] Generating synthetic data...")
    data = generate_synthetic_data(
        num_nodes=config.data.num_nodes, num_features=config.model.input_dim,
        edge_density=config.data.edge_density, community_k=config.data.community_k,
        seed=config.seed,
    )
    X = scale_features(data["features"])
    A = data["adjacency"]
    W = data["weights"]
    labels = data["labels"]
    print(f"  Edges: {int(A.sum() // 2)}, Communities: {len(np.unique(labels))}")

    # Split edges
    print("\n[2/4] Splitting edges...")
    splits = split_edges(A, train_ratio=0.85, val_ratio=0.05, seed=config.seed)
    train_adj = splits["train_adj"]

    # Build graph tensors
    print("\n[3/4] Building graph tensors...")
    edge_list = []
    for i in range(config.data.num_nodes):
        for j in range(i + 1, config.data.num_nodes):
            if train_adj[i, j] > 0:
                edge_list.append([i, j])
                edge_list.append([j, i])

    edge_index_np = np.array(edge_list).T
    x_tensor = torch.tensor(X, dtype=torch.float32)
    edge_index_tensor = torch.tensor(edge_index_np, dtype=torch.long)
    adj_true = torch.tensor(train_adj, dtype=torch.float32)

    dataset = BPRDataset(train_adj, W, num_negatives=1, seed=config.seed)
    dataset.x = x_tensor
    dataset.edge_index = edge_index_tensor
    train_loader = DataLoader(
        dataset, batch_size=config.train.batch_size,
        shuffle=True, collate_fn=collate_bpr_batch,
    )

    # Train
    print("\n[4/4] Training model...")
    model = DeepCFVGAE(config.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    trainer = Trainer(model, config, train_loader)
    history = trainer.train(adj_true)

    print(f"\nTraining complete. Best loss: {trainer.best_val_loss:.4f}")
    print(f"Outputs saved to: {config.output_dir}/")


if __name__ == "__main__":
    main()
