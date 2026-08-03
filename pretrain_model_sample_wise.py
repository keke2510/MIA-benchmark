import logging
import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import datasets
from torch.utils.data import DataLoader

import models
import config
from utils.training_utils import *

# Original code from https://github.com/weiaicunzai/pytorch-cifar100 <- refer to this repo for comments

def train(epochs):
    start = time.time()
    net.train()
    for batch_index, (images, _, labels) in enumerate(trainloader):
        labels = labels.to(device)
        images = images.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch <= args.warm:
            warmup_scheduler.step()

    finish = time.time()

    print("epoch {} training time consumed: {:.2f}s".format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start = time.time()
    net.eval()

    test_loss = 0.0  # summed per-example loss
    correct = 0.0

    for images, _, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print("Evaluating Network.....")
    print(
        "Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s".format(
            epoch,
            test_loss / len(testloader.dataset),
            correct.float() / len(testloader.dataset),
            finish - start,
        )
    )

    # 记录测试信息
    logger.info(
        "Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed: {:.2f}s".format(
            epoch,
            test_loss / len(testloader.dataset),
            correct.float() / len(testloader.dataset),
            finish - start,
        )
    )

    return correct.float() / len(testloader.dataset)


def resolve_img_size(dataset_name):
    if dataset_name == "TinyImageNet":
        return 64
    return 32

parser = argparse.ArgumentParser()
parser.add_argument("--net", type=str, default='ViT', help="net type, ResNet18, ViT")
parser.add_argument("--dataset", type=str, default='Cifar20', help="dataset to train on: Cifar10, Cifar20, Cifar100, Cinic10, TinyImageNet")
parser.add_argument("--classes", type=int, default=20, help="number of classes")
parser.add_argument("--bs", type=int, default=256, help="batch size for dataloader")
parser.add_argument("-warm", type=int, default=1, help="warm up training phase")
parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
parser.add_argument("-seed", type=int, default=0, help="seed for runs")
args = parser.parse_args()

MILESTONES = (
    getattr(config, f"{args.dataset}_MILESTONES")
    if args.net != "ViT"
    else getattr(config, f"{args.dataset}_ViT_MILESTONES")
)
EPOCHS = (
    getattr(config, f"{args.dataset}_EPOCHS")
    if args.net != "ViT"
    else getattr(config, f"{args.dataset}_ViT_EPOCHS")
)

# get network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = getattr(models, args.net)(num_classes=args.classes)
net = net.to(device)

# checkpoint_path = os.path.join(config.CHECKPOINT_PATH, args.net, config.TIME_NOW)
output_path = os.path.join(config.CHECKPOINT_PATH, "{task}".format(task="pretrain"),
                               "{net}-{dataset}-{classes}".format(net=args.net,dataset=args.dataset,classes=args.classes))
if not os.path.exists(output_path):
    os.makedirs(output_path)

checkpoint_path = os.path.join(output_path, "{type}.pth")

weights_path = checkpoint_path.format(
    type="best"
)
# net.load_state_dict(torch.load(weights_path))

# dataloaders
root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"
img_size = resolve_img_size(args.dataset)

trainset = getattr(datasets, args.dataset)(
    root=root, download=True, train=True, unlearning=False, img_size=img_size
) #TODO we don't want to use too many data augmentation; but here, we use some
testset = getattr(datasets, args.dataset)(
    root=root, download=True, train=False, unlearning=False, img_size=img_size
)

trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True)
testloader = DataLoader(testset, batch_size=args.bs, shuffle=False)

loss_function = nn.CrossEntropyLoss()
if args.net == "ViT":
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.2)  # learning rate decay
iter_per_epoch = len(trainloader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

logger = logging.getLogger("TrainingLogger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(output_path+"/training.log")
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

best_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    if epoch > args.warm:
        train_scheduler.step(epoch)

    train(epoch)
    acc = eval_training(epoch)

    # start to save best performance model after learning rate decay to 0.01
    if best_acc < acc:  # and epoch > MILESTONE
    #
        weights_path = checkpoint_path.format(
         type="best"
        )
        print("saving weights file to {}".format(weights_path))
        torch.save(net.state_dict(), weights_path)
        best_acc = acc
        continue
