"""PyTorch Dataset 与 BPR 采样."""

import numpy as np
import torch
from torch.utils.data import Dataset


class BPRDataset(Dataset):
    """BPR 成对排序数据集.

    对每个节点采样 1 个正样本（有边邻居）和 num_negatives 个负样本（无边节点），
    构成 (user, pos_item, neg_item, weight) 训练三元组。
    """

    def __init__(
        self,
        adjacency: np.ndarray,
        weights: np.ndarray,
        num_negatives: int = 1,
        seed: int = 42,
    ):
        self.adjacency = adjacency
        self.weights = weights
        self.num_nodes = adjacency.shape[0]
        self.num_negatives = num_negatives
        self.rng = np.random.default_rng(seed)

        # 为每个节点建立正样本列表
        self.pos_neighbors = {}
        all_nodes = set(range(self.num_nodes))
        for i in range(self.num_nodes):
            neighbors = np.where(adjacency[i] > 0)[0]
            self.pos_neighbors[i] = neighbors.tolist()

    def __len__(self) -> int:
        return self.num_nodes

    def __getitem__(self, idx: int) -> tuple:
        u = idx
        pos_list = self.pos_neighbors.get(u, [])
        if len(pos_list) == 0:
            # 孤立节点：回退到最近的非孤立节点
            for dist in range(1, self.num_nodes):
                fallback = (u + dist) % self.num_nodes
                if len(self.pos_neighbors.get(fallback, [])) > 0:
                    u = fallback
                    pos_list = self.pos_neighbors[u]
                    break
            else:
                # 全图无边，回退到自身
                return (torch.tensor(u), torch.tensor(u),
                        torch.tensor(u), torch.tensor(0.0))

        i_pos = int(self.rng.choice(pos_list))
        w = float(self.weights[u, i_pos])

        # 负采样
        i_neg = int(self.rng.integers(0, self.num_nodes))
        while self.adjacency[u, i_neg] > 0 or i_neg == i_pos:
            i_neg = int(self.rng.integers(0, self.num_nodes))

        return (torch.tensor(u), torch.tensor(i_pos),
                torch.tensor(i_neg), torch.tensor(w))


def collate_bpr_batch(batch: list) -> tuple:
    """整理 BPR 批次.

    Args:
        batch: list of (u, i_pos, i_neg, w) tuples

    Returns:
        (users, pos_items, neg_items, weights) 张量
    """
    users, pos_items, neg_items, weights = zip(*batch)
    return (
        torch.stack(users),
        torch.stack(pos_items),
        torch.stack(neg_items),
        torch.stack(weights),
    )
