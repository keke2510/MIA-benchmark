#!/usr/bin/env python
"""DeepCF 独立可视化脚本."""

import argparse, os, sys
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepcf.config import DeepCFConfig
from deepcf.data.generator import generate_synthetic_data
from deepcf.data.utils import scale_features
from deepcf.model.vgae import DeepCFVGAE
from deepcf.eval.ranking import compute_radiation_scores, top_k_merchants, rank_all_merchants
from deepcf.eval.visualize import plot_tsne_embeddings, plot_influence_distribution, plot_training_curves


def main():
    parser = argparse.ArgumentParser(description="DeepCF Visualization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="outputs/viz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = DeepCFConfig()
    config.data.num_nodes = args.nodes
    config.output_dir = args.output_dir
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = DeepCFVGAE(config.model)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(config.device)
    model.eval()

    data = generate_synthetic_data(
        num_nodes=config.data.num_nodes, num_features=config.model.input_dim, seed=config.seed,
    )
    X = scale_features(data["features"])
    A = data["adjacency"]
    W = data["weights"]
    labels = data["labels"]

    edge_list = []
    for i in range(config.data.num_nodes):
        for j in range(i + 1, config.data.num_nodes):
            if A[i, j] > 0:
                edge_list.append([i, j])
                edge_list.append([j, i])

    edge_index_np = np.array(edge_list).T
    x_tensor = torch.tensor(X, dtype=torch.float32).to(config.device)
    edge_index_tensor = torch.tensor(edge_index_np, dtype=torch.long).to(config.device)

    with torch.no_grad():
        z = model.get_embeddings(x_tensor, edge_index_tensor).cpu().numpy()

    rank_scores = rank_all_merchants(model, x_tensor, edge_index_tensor)
    radiation = compute_radiation_scores(z, A, W, rank_scores)
    top_k = top_k_merchants(radiation, labels, k=20)
    topk_idx = [m["id"] for m in top_k[:10]]

    plot_tsne_embeddings(z, labels, radiation, topk_idx,
                         save_path=os.path.join(config.output_dir, "tsne_embeddings.png"))
    print(f"t-SNE saved to {config.output_dir}/tsne_embeddings.png")

    plot_influence_distribution(radiation, A.sum(axis=1), topk_idx,
                                save_path=os.path.join(config.output_dir, "influence_distribution.png"))
    print(f"Influence distribution saved to {config.output_dir}/influence_distribution.png")

    history = ckpt.get("history", {"epoch": [], "train_loss": [], "lr": []})
    plot_training_curves(history, save_path=os.path.join(config.output_dir, "training_curves.png"))
    print(f"Training curves saved to {config.output_dir}/training_curves.png")
    print("\nDone!")


if __name__ == "__main__":
    main()
