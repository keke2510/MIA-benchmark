#!/usr/bin/env python
"""DeepCF 评估入口脚本."""

import argparse, os, sys, json
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepcf.config import DeepCFConfig
from deepcf.data.generator import generate_synthetic_data
from deepcf.data.utils import split_edges, scale_features
from deepcf.model.vgae import DeepCFVGAE
from deepcf.eval.report import generate_report


def main():
    parser = argparse.ArgumentParser(description="DeepCF Model Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
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
    x_tensor = torch.tensor(X, dtype=torch.float32)
    edge_index_tensor = torch.tensor(edge_index_np, dtype=torch.long)

    splits = split_edges(A, seed=config.seed)

    report = generate_report(
        model=model, x=x_tensor, edge_index=edge_index_tensor,
        adjacency=A, weights=W, labels=labels,
        history=ckpt.get("history", {"epoch": [], "train_loss": [], "lr": []}),
        test_edges=splits["test_edges"], test_edges_neg=splits["test_edges_neg"],
        output_dir=config.output_dir,
    )

    print(f"\n=== Evaluation Report ===")
    print(f"AUC: {report['link_prediction']['auc']:.4f}")
    print(f"AP:  {report['link_prediction']['ap']:.4f}")
    print(f"\nTop-10 Merchants:")
    for m in report["top_10_merchants"]:
        print(f"  ID={m['id']:4d}  Score={m['score']:.4f}  Category={m['category']}")
    print(f"\nRadiation Stats: {report['radiation_stats']}")

    with open(os.path.join(config.output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nReport saved to {config.output_dir}/")


if __name__ == "__main__":
    main()
