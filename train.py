# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import logging

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torchvision.transforms as transforms

import os
import argparse
import csv
import time

from torch.utils.data import ConcatDataset

import config
import datasets
import models
from lira_utils import progress_bar, set_random_seed
from randomaug import RandAugment
from utils.training_utils import WarmUpLR

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')  # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--classes', default=10, type=int, help='the number of classes')
parser.add_argument('--opt', default='sgd')
parser.add_argument('--resume_checkpoint', '-r', default=None, help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='res18')
parser.add_argument('--dataset', default='Cifar10')
parser.add_argument('--bs', default=512, type=int)
parser.add_argument('--size', default=32, type=int)
parser.add_argument('--n_epochs', default=150, type=int)  # TODO epochs
parser.add_argument('--num_total', default=None, type=int)
parser.add_argument('--patch', default=4, type=int, help="patch for ViT")
parser.add_argument('--dimhead', default=512, type=int)
parser.add_argument('--convkernel', default=8, type=int, help="parameter for convmixer")
parser.add_argument('--name', default='test')
parser.add_argument('--save_name', default='test')
parser.add_argument('--num_shadow', default=None, type=int)
parser.add_argument('--shadow_id', default=None, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--pkeep', default=0.5, type=float)

parser.add_argument("-warm", type=int, default=1, help="warm up training phase")

args = parser.parse_args()

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

if args.num_shadow is not None:
    args.job_name = args.save_name + f'_shadow_{args.shadow_id}'
    log_file = os.path.join(log_dir, f"shadow_model_{args.shadow_id}.log")
else:
    args.job_name = args.save_name + '_target'
    log_file = os.path.join(log_dir, f"target_model.log")  # this one, we have another version

# take in args
# usewandb = not args.nowandb
# name = args.job_name
# if usewandb:
#     import wandb
#     wandb.init(project='canary_shadow_model', name=name)
#     wandb.config.update(args)

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,  # 记录 INFO 级别以上的日志
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def log_message(message):
    print(message)  # 终端打印日志
    logging.info(message)  # 记录到文件


def resolve_schedule(dataset_name, net_name):
    if net_name == "ViT":
        # Some environments may still carry an older config.py that only defines
        # the non-ViT schedule. Fall back gracefully instead of crashing.
        milestones = getattr(
            config,
            f"{dataset_name}_ViT_MILESTONES",
            getattr(config, f"{dataset_name}_MILESTONES"),
        )
        epochs = getattr(
            config,
            f"{dataset_name}_ViT_EPOCHS",
            getattr(config, f"{dataset_name}_EPOCHS"),
        )
        return milestones, epochs

    milestones = getattr(config, f"{dataset_name}_MILESTONES")
    epochs = getattr(config, f"{dataset_name}_EPOCHS")
    return milestones, epochs


def resolve_img_size(dataset_name):
    if dataset_name == "TinyImageNet":
        return 64
    return 32


MILESTONES, EPOCHS = resolve_schedule(args.dataset, args.net)

bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
aug = args.aug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.net == "vit_timm":
    size = 384
else:
    size = imsize

# TODO prepare data
root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"
img_size = resolve_img_size(args.dataset)

trainset = getattr(datasets, args.dataset)(
    root=root, download=True, train=True, unlearning=False, img_size=img_size
)
testset = getattr(datasets, args.dataset)(
    root=root, download=True, train=False, unlearning=False, img_size=img_size, data_augmentation=True
)  # here use some data augmentation

trainset = ConcatDataset([trainset, testset])
dataset_size = len(trainset)

if args.num_total:
    dataset_size = args.num_total

# set random seed
set_random_seed(args.seed)

# get shadow dataset
if args.num_shadow is not None:
    # get shadow dataset
    keep = np.random.uniform(0, 1, size=(args.num_shadow, dataset_size))
    order = keep.argsort(0)
    keep = order < int(args.pkeep * args.num_shadow)  # TODO half dataset, right?
    keep = np.array(keep[args.shadow_id], dtype=bool)
    keep = keep.nonzero()[0]
else:
    # get target dataset
    keep = np.random.choice(dataset_size, size=int(args.pkeep * dataset_size), replace=False)
    keep.sort()

keep_bool = np.full((dataset_size), False)
keep_bool[keep] = True

trainset = torch.utils.data.Subset(trainset, keep)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

# Model factory..
print('==> Building model..')

net = getattr(models, args.net)(num_classes=args.classes)
net = net.to(device)

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# use cosine scheduling
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.2)  # learning rate decay
iter_per_epoch = len(trainloader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)


##### Training
# scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, _, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        # with torch.cuda.amp.autocast(enabled=use_amp):
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        if epoch <= args.warm:
            warmup_scheduler.step()

    return train_loss / (batch_idx + 1)


##### Validation
# TODO validset is in the training split: the first 50000 is training data, 50000-60000 is test data
def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, _, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total

    # os.makedirs('loglog', exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    # with open(f'loglog/{name}.txt', 'a') as appender:
    #     appender.write(content + "\n")
    log_message(content)
    return test_loss, acc


list_loss = []
list_acc = []

# if usewandb:
#     wandb.watch(net)

net.cuda()

for epoch in range(start_epoch, start_epoch + EPOCHS):
    start = time.time()
    if epoch > args.warm:
        train_scheduler.step(epoch)

    trainloss = train(epoch)
    val_loss, acc = test(epoch)

    # scheduler.step() # step cosine scheduling

    list_loss.append(val_loss)
    list_acc.append(acc)

    # Log training..
    # if usewandb:
    #     wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, 'val_acc': acc, 'lr': optimizer.param_groups[0]['lr'],
    #     'epoch_time': time.time()-start})
    log_message({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, 'val_acc': acc,
                 'lr': optimizer.param_groups[0]['lr'],
                 'epoch_time': time.time() - start})

    # Write out csv..
    # with open(f'loglog/{name}.csv', 'w') as f:
    #     writer = csv.writer(f, lineterminator='\n')
    #     writer.writerow(list_loss)
    #     writer.writerow(list_acc)

state = {"model": net.state_dict(),
         "in_data": keep,
         "keep_bool": keep_bool,
         "model_arch": args.net}
os.makedirs('saved_models/' + args.name, exist_ok=True)
torch.save(state, './saved_models/' + args.name + '/' + args.job_name + '_last.pth')
