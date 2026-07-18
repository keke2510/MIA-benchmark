"""训练循环：多任务训练 + 早停 + Checkpoint."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm

from deepcf.config import DeepCFConfig, ModelConfig, LossConfig
from deepcf.model.vgae import DeepCFVGAE
from deepcf.data.dataset import collate_bpr_batch


class Trainer:
    """DeepCF 训练器."""

    def __init__(
        self,
        model: DeepCFVGAE,
        config: DeepCFConfig,
        train_loader: DataLoader,
        val_data: Optional[Dict] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_data = val_data

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=config.train.lr_patience,
            factor=config.train.lr_factor,
        )

        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.history: Dict[str, list] = {
            "epoch": [],
            "train_loss": [],
            "lr": [],
        }

        os.makedirs(config.output_dir, exist_ok=True)

    def train_epoch(self, adj_true: torch.Tensor) -> float:
        """训练一个 epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        x_all = self.train_loader.dataset.x.to(self.config.device)
        edge_index_all = self.train_loader.dataset.edge_index.to(self.config.device)
        adj_all = adj_true.to(self.config.device)

        for batch in self.train_loader:
            users, pos_items, neg_items, weights = batch
            users = users.to(self.config.device)
            pos_items = pos_items.to(self.config.device)
            neg_items = neg_items.to(self.config.device)
            weights = weights.to(self.config.device)

            self.optimizer.zero_grad()

            loss, components = self.model.compute_loss(
                x=x_all,
                edge_index=edge_index_all,
                adj_true=adj_all,
                u=users,
                v=pos_items,
                neg_u=users,
                neg_v=neg_items,
                edge_weight=None,
                weight_true=weights,
                loss_config=self.config.loss,
            )

            loss.backward()
            self.optimizer.step()

            total_loss += components["total"]
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(self, adj_true: torch.Tensor) -> Dict[str, list]:
        """完整训练流程."""
        self.model.to(self.config.device)
        pbar = tqdm(range(1, self.config.train.epochs + 1), desc="Training")

        for epoch in pbar:
            train_loss = self.train_epoch(adj_true)
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            pbar.set_postfix({
                "loss": f"{train_loss:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
            })

            if train_loss < self.best_val_loss:
                self.best_val_loss = train_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1

            self.scheduler.step(train_loss)

            if epoch % self.config.train.checkpoint_interval == 0:
                self._save_checkpoint(epoch, is_best=False)

            if self.patience_counter >= self.config.train.early_stop_patience:
                print(f"Early stopping at epoch {epoch}, "
                      f"best loss {self.best_val_loss:.4f} at epoch {self.best_epoch}")
                break

        print(f"Training completed. Best loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        return self.history

    def _save_checkpoint(self, epoch: int, is_best: bool):
        """保存检查点."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }
        if is_best:
            torch.save(ckpt, os.path.join(self.config.output_dir, "best_model.pt"))
        torch.save(
            ckpt,
            os.path.join(self.config.output_dir, f"checkpoint_epoch_{epoch}.pt"),
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        model_config: ModelConfig,
        config: DeepCFConfig,
        device: str = "cpu",
    ) -> tuple:
        """从检查点加载模型."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = DeepCFVGAE(model_config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        return model, ckpt
