# From https://github.com/vikram2000b/bad-teaching-unlearning
import csv

import torch
from torch.utils.data import Subset

from datasets import Dataset
from torch import nn
from torch.nn import functional as F
from utils.training_utils import *
import numpy as np

class MemorizationDataset(Dataset):
    def __init__(self, base_dataset, mem_values):
        """
        :param base_dataset: 原始数据集（例如返回 (image, label)）
        :param mem_values: 与 base_dataset 中样本一一对应的 memorization 值（数组或 list）
        """
        self.base_dataset = base_dataset
        self.mem_values = mem_values

    def __getitem__(self, index):
        # 从基础数据集中取样本，同时附加该样本的 memorization 数值
        img, _, label = self.base_dataset[index]
        mem = self.mem_values[index]
        return img, label, mem

    def __len__(self):
        return len(self.base_dataset)

def construct_forget_dataset(base_dataset, forget_dataset_index, npz_path):
    """
    根据 forget_dataset_index 选取 base_dataset 中的样本，并从 npz_path 文件中提取相应的 mem 数值，
    构造一个包含 (image, label, mem) 的新数据集返回。

    :param base_dataset: 原始的 forget_dataset（例如 CIFAR-10 数据集的一部分）
    :param forget_dataset_index: 一个索引列表，指明哪些样本需要被“遗忘”
    :param npz_path: npz 文件的路径，此文件中存有键 'mem'，对应所有样本的 memorization 数值
    :return: 一个 MemorizationDataset 类型的数据集
    """
    # 加载 npz 数据
    npz_data = np.load(npz_path)
    mem_all = npz_data['mem']  # 假定 mem_all 的顺序与合并数据集顺序一致
    # 按给定的索引顺序提取需要的 memorization 数值
    # 注意：如果 forget_dataset_index 是列表或 numpy 数组，下面这样取值即可
    mem_forget = mem_all[forget_dataset_index]

    # 利用 Subset 选取对应的样本
    subset_forget = Subset(base_dataset, forget_dataset_index)
    # 构造包装数据集，使得 __getitem__ 返回 (img, label, mem)
    new_forget_dataset = MemorizationDataset(subset_forget, mem_forget)
    return new_forget_dataset

def construct_valid_dataset(base_dataset, valid_index_in_mem, npz_path):
    """
    根据 forget_dataset_index 选取 base_dataset 中的样本，并从 npz_path 文件中提取相应的 mem 数值，
    构造一个包含 (image, label, mem) 的新数据集返回。

    :param base_dataset: 原始的 forget_dataset（例如 CIFAR-10 数据集的一部分）
    :param forget_dataset_index: 一个索引列表，指明哪些样本需要被“遗忘”
    :param npz_path: npz 文件的路径，此文件中存有键 'mem'，对应所有样本的 memorization 数值
    :return: 一个 MemorizationDataset 类型的数据集
    """
    # 加载 npz 数据
    npz_data = np.load(npz_path)
    mem_all = npz_data['mem']  # 假定 mem_all 的顺序与合并数据集顺序一致
    # 按给定的索引顺序提取需要的 memorization 数值
    # 注意：如果 forget_dataset_index 是列表或 numpy 数组，下面这样取值即可
    mem_forget = mem_all[valid_index_in_mem]

    # 利用 Subset 选取对应的样本
    # 构造包装数据集，使得 __getitem__ 返回 (img, label, mem)
    new_valid_dataset = MemorizationDataset(base_dataset, mem_forget)
    return new_valid_dataset

def construct_forget_memorization(forget_dataset_index, npz_path):
    """
    根据 forget_dataset_index 选取 base_dataset 中的样本，并从 npz_path 文件中提取相应的 mem 数值，
    构造一个包含 (image, label, mem) 的新数据集返回。

    :param base_dataset: 原始的 forget_dataset（例如 CIFAR-10 数据集的一部分）
    :param forget_dataset_index: 一个索引列表，指明哪些样本需要被“遗忘”
    :param npz_path: npz 文件的路径，此文件中存有键 'mem'，对应所有样本的 memorization 数值
    :return: 一个 MemorizationDataset 类型的数据集
    """
    # 加载 npz 数据
    npz_data = np.load(npz_path)
    mem_all = npz_data['mem']  # 假定 mem_all 的顺序与合并数据集顺序一致
    # 按给定的索引顺序提取需要的 memorization 数值
    # 注意：如果 forget_dataset_index 是列表或 numpy 数组，下面这样取值即可
    mem_forget = mem_all[forget_dataset_index]
    return mem_forget

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)) * 100


def training_step(model, batch, device):
    images, labels, clabels = batch
    images, clabels = images.to(device), clabels.to(device)
    out = model(images)  # Generate predictions
    loss = F.cross_entropy(out, clabels)  # Calculate loss
    return loss

def validation_step(model, batch, device):
    images, labels, clabels = batch   # labels 100, clabels 20
    images, clabels = images.to(device), clabels.to(device)
    out = model(images)  # Generate predictions
    loss = F.cross_entropy(out, clabels)  # Calculate loss
    acc = accuracy(out, clabels)  # Calculate accuracy
    return {"Loss": loss.detach(), "Acc": acc}

def validation_step_rum(model, batch, device):
    images, labels, _ = batch   # labels 100, clabels 20
    images, labels = images.to(device), labels.to(device)
    out = model(images)  # Generate predictions
    loss = F.cross_entropy(out, labels)  # Calculate loss
    acc = accuracy(out, labels)  # Calculate accuracy
    return {"Loss": loss.detach(), "Acc": acc}

def validation_epoch_end(model, outputs):
    batch_losses = [x["Loss"] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
    batch_accs = [x["Acc"] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
    return {"Loss": epoch_loss.item(), "Acc": epoch_acc.item()}


def epoch_end(model, epoch, result):
    print(
        "Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch,
            result["lrs"][-1],
            result["train_loss"],
            result["Loss"],
            result["Acc"],
        )
    )

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    outputs = [validation_step(model, batch, device) for batch in val_loader]
    return validation_epoch_end(model, outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

@torch.no_grad()
def eval_training(epoch, net, testloader, tb=True):
    loss_function = nn.CrossEntropyLoss
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for images, _, labels in testloader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)

        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print("Evaluating Network.....")
    print(
        "Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}".format(
            epoch,
            test_loss / len(testloader.dataset),
            correct.float() / len(testloader.dataset),
        )
    )
    return correct.float() / len(testloader.dataset)

def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)

def fit_one_cycle(
    epochs, model, train_loader, val_loader, device, lr=0.01, milestones=None, mask=None, model_name='ResNet18', l1=False
):
    torch.cuda.empty_cache()
    history = []

    if model_name=='ViT':
        optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)

    if milestones:
        train_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.2
        )  # learning rate decay
        warmup_scheduler = WarmUpLR(optimizer, len(train_loader))

    for epoch in range(epochs):
        if epoch > 1 and milestones:
            train_scheduler.step(epoch)

        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = training_step(model, batch, device)
            if l1:
                loss += 5e-5*l1_regularization(model)
            train_losses.append(loss)
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))

            if epoch <= 1 and milestones:
                warmup_scheduler.step()

        # Validation phase
        result = evaluate(model, val_loader, device)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        result["lrs"] = lrs
        epoch_end(model, epoch, result)
        history.append(result)

        #acc = eval_training(epoch, model, val_loader)
    return history

def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for index, row in enumerate(csv_reader):
            if index > 0:
                data.append([float(x) for x in row[0].split('\t')])
    return np.asarray(data)
