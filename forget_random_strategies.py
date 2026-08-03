"""
Strategy implementations for sample-wise unlearning.
"""

import random
import numpy as np
from typing import Tuple, List
from copy import deepcopy

import torch
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset, dataset, Subset
from tqdm import tqdm

from sklearn import linear_model, model_selection

from thirdparty.repdistiller.distiller_zoo import DistillKL
from thirdparty.repdistiller.helper.loops import train_distill, validate_scrub
from thirdparty.repdistiller.helper.util import adjust_learning_rate
from unlearn import *
from utils.metrics import UnLearningScore, get_membership_attack_prob
import utils.ssd
import config

import copy
import torch.nn as nn
from itertools import cycle
from collections import defaultdict
from utils.overall_utils import *

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_metric_scores(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader,
                      valid_dataloader, device, fast=True):
    d_t_acc_dict = evaluate(model, valid_dataloader, device) #测试集
    d_f_acc_dict = evaluate(model, forget_train_dataloader, device)  # 遗忘集
    d_r_acc_dict = evaluate(model, retain_train_dataloader, device)  # 保留集
    zrf = UnLearningScore(model, unlearning_teacher, forget_train_dataloader, 128, device) #遗忘集
    if fast is False:
        mia = get_membership_attack_prob(retain_train_dataloader, forget_train_dataloader, valid_dataloader, model)
    else:
        mia = 0.0
    return (d_t_acc_dict["Acc"], d_f_acc_dict["Acc"], d_r_acc_dict["Acc"], zrf, mia)

def get_metric_scores_simple(model, retain_train_dataloader, forget_train_dataloader,
                      valid_dataloader, device):
    d_t_acc_dict = evaluate(model, valid_dataloader, device) #测试集
    d_f_acc_dict = evaluate(model, forget_train_dataloader, device)  # 遗忘集
    d_r_acc_dict = evaluate(model, retain_train_dataloader, device)  # 保留集
    mia = get_membership_attack_prob(retain_train_dataloader, forget_train_dataloader, valid_dataloader, model)

    return (d_t_acc_dict["Acc"],d_f_acc_dict["Acc"], d_r_acc_dict["Acc"], mia)

##############################
def baseline(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader,
             valid_dataloader, device, **kwargs):
    return get_metric_scores(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader, valid_dataloader, device)

##############################
import models
def retrain(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader,
            valid_dataloader,  dataset_name, model_name, device,
            num_classes, weights_path,
            para1=0.1,
            para2=50,
            **kwargs):
    # for layer in model.children():
    #     if hasattr(layer, "reset_parameters"):
    #         layer.reset_parameters()
    model = getattr(models, model_name)(num_classes=num_classes)
    # model.load_state_dict(torch.load(weights_path))
    model.to(device)
    if model_name == "ViT":
        epochs = getattr(config, f"{dataset_name}_{model_name}_EPOCHS")
        milestones = getattr(config, f"{dataset_name}_{model_name}_MILESTONES")
    else:
        epochs = getattr(config, f"{dataset_name}_EPOCHS")
        milestones = getattr(config, f"{dataset_name}_MILESTONES")

    _ = fit_one_cycle(epochs, model, retain_train_dataloader, valid_dataloader, lr=float(para1), model_name=model_name,
                      milestones=milestones, device=device)

    # loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)  # 5e-4
    # train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)  # learning rate decay
    # iter_per_epoch = len(retain_train_dataloader)
    # warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch)
    #
    # best_acc = 0.0
    # for epoch in range(1, 20 + 1):
    #     if epoch > 1:
    #         train_scheduler.step(epoch)
    #
    #     model.train()
    #     for batch_index, (images, _, labels) in enumerate(retain_train_dataloader):
    #         labels = labels.cuda()
    #         images = images.cuda()
    #
    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         loss = loss_function(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         if epoch <= 1:
    #             warmup_scheduler.step()
    #     acc = eval_training(epoch, model, valid_dataloader)

    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader,
                             valid_dataloader,  device, fast=True)

##############################
def finetune(model, unlearning_teacher, retain_train_dataloader,
             forget_train_dataloader, valid_dataloader,  device, weights_path,
             para1=0.02, para2=5, rum=False, model_name='ResNet18', **kwargs):
    _ = fit_one_cycle(int(para2), model, retain_train_dataloader, valid_dataloader, lr=float(para1), device=device, model_name=model_name)

    if not rum:
        torch.save(model.state_dict(), weights_path)
        return get_metric_scores(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader,
                                 valid_dataloader, device, fast=True)
    else:
        return model

##############################
def l2_penalty(model, model_init, weight_decay):
    l2_loss = 0
    for (k, p), (k_init, p_init) in zip(model.named_parameters(), model_init.named_parameters()):
        if p.requires_grad:
            l2_loss += (p - p_init).pow(2).sum()
    l2_loss *= (weight_decay / 2.)
    return l2_loss

def train_negrad_1(model, model_init, retain_loader, forget_loader, loss_fn, optimizer, alpha, mask):
    model.train()

    for idx, (batch_retain, batch_forget) in enumerate(zip(retain_loader, cycle(forget_loader))):
        batch_retain = [tensor.to(next(model.parameters()).device) for tensor in batch_retain]
        batch_forget = [tensor.to(next(model.parameters()).device) for tensor in batch_forget]
        input_r, _, target_r = batch_retain
        input_f, _, target_f = batch_forget
        output_r = model(input_r)
        output_f = model(input_f)
        loss = alpha * (loss_fn(output_r, target_r) +
                        l2_penalty(model, model_init, weight_decay=0.1)) - \
               (1 - alpha) * loss_fn(output_f, target_f)

        optimizer.zero_grad()
        loss.backward()

        if mask:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]
        optimizer.step()

    return

def train_negrad_2(model, model_init, retain_loader, forget_loader, loss_fn, optimizer, mask):
    model.train()

    for idx, batch_forget in enumerate(forget_loader):
        batch_forget = [tensor.to(next(model.parameters()).device) for tensor in batch_forget]
        input_f, _, target_f = batch_forget
        output_f = model(input_f)
        loss = -loss_fn(output_f, target_f)

        optimizer.zero_grad()
        loss.backward()

        if mask:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]
        optimizer.step()

    return

def train_negrad(model, retain_loader, forget_loader, loss_fn, optimizer, alpha):
    model.train()

    for idx, (batch_retain, batch_forget) in enumerate(zip(retain_loader, cycle(forget_loader))):
        batch_retain = [tensor.to(next(model.parameters()).device) for tensor in batch_retain]
        batch_forget = [tensor.to(next(model.parameters()).device) for tensor in batch_forget]
        input_r,_,target_r = batch_retain
        input_f,_,target_f = batch_forget
        output_r = model(input_r)
        output_f = model(input_f)
        loss = alpha * loss_fn(output_r, target_r) - (1 - alpha) * loss_fn(output_f, target_f)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return

#adjust_parameters
def negative_grad(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader, valid_dataloader,
                  device, weights_path, para1=0.0001, para2=3,
                  opt_strategy=None,
                  mask_path=None,
                  model_name='ResNet18',
                  **kwargs):
    #TODO strategy
    if opt_strategy == 'weight_saliency':
        mask = torch.load(mask_path)
    else:
        mask = None
    alpha =0.7#0.84 #TODO 0.95
    epochs = int(para2)#10f
    lr = float(para1)#0.01
    loss_fn = nn.CrossEntropyLoss()
    if model_name == 'ViT':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0)
    model_init = copy.deepcopy(model)
    for epoch in range(epochs):
        train_negrad_1(model, model_init, retain_train_dataloader, forget_train_dataloader, loss_fn, optimizer, alpha, mask)
        # train_negrad(model, retain_train_dataloader, forget_train_dataloader, loss_fn, optimizer,  alpha)

        d_t, d_f, d_r, zrf, mia = get_metric_scores(model, unlearning_teacher, retain_train_dataloader,
                                                          forget_train_dataloader, valid_dataloader, device, fast=True)
        print("d_t = ", d_t, "| d_tp = ", "| d_f = ", d_f, "| d_r = ", d_r, "| zrf = ", zrf, "| mia = ", mia)

    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader,
                             valid_dataloader, device, fast=True)

#adjust_parameters
def negative_grad_2(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader, valid_dataloader,
                  device, weights_path, para1=0.0001, para2=3,
                  opt_strategy=None,
                  mask_path=None,
                  **kwargs):
    #TODO strategy
    if opt_strategy == 'weight_saliency':
        mask = torch.load(mask_path)
    else:
        mask = None
    epochs = int(para2)#10
    lr = float(para1)#0.01
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0)
    model_init = copy.deepcopy(model)
    for epoch in range(epochs):
        train_negrad_2(model, model_init, retain_train_dataloader, forget_train_dataloader, loss_fn, optimizer, mask)
        # train_negrad(model, retain_train_dataloader, forget_train_dataloader, loss_fn, optimizer,  alpha)

        d_t, d_f, d_r, zrf, mia = get_metric_scores(model, unlearning_teacher, retain_train_dataloader,
                                                          forget_train_dataloader, valid_dataloader, device, fast=True)
        print("d_t = ", d_t, "| d_tp = ", "| d_f = ", d_f, "| d_r = ", d_r, "| zrf = ", zrf, "| mia = ", mia)

    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader,
                             valid_dataloader, device)

##############################
def amnesiac(model,
             unlearning_teacher,
             retain_train_dataloader,
             forget_train_dataloader,
             valid_dataloader,
             num_classes,
             device, weights_path, para1='0.0001', para2='2',
                opt_strategy=None,
                mask_path=None,
             **kwargs):
    # TODO strategy
    if opt_strategy == 'weight_saliency':
        mask = torch.load(mask_path)
    else:
        mask = None

    unlearninglabels = list(range(num_classes))
    unlearning_trainset = []

    #change the labels in unlearning_trainset
    for x, _, clabel in forget_train_dataloader.dataset:
        rnd = random.choice(unlearninglabels)
        while rnd == clabel:
            rnd = random.choice(unlearninglabels)
        unlearning_trainset.append((x, _, rnd))

    for x, _, y in retain_train_dataloader.dataset:
        unlearning_trainset.append((x, _, y))

    unlearning_train_dataloader = DataLoader(unlearning_trainset, batch_size=128, pin_memory=True, shuffle=True)

    #unlearn process of the amnesiac
    _ = fit_one_unlearning_cycle(int(para2), model, unlearning_train_dataloader, valid_dataloader, device=device, lr=float(para1), mask=mask)

    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(model, unlearning_teacher, retain_train_dataloader,
                             forget_train_dataloader, valid_dataloader,  device, fast=True)

def relabel(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader,
             valid_dataloader,  num_classes, device, weights_path, para1='0.0001', para2='2',
             **kwargs):
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=float(para1))

    for epoch in range(int(para2)):
        model.train()
        start = time.time()
        losses = AverageMeter()
        top1 = AverageMeter()
        loader_len = len(forget_train_dataloader) + len(retain_train_dataloader)

        for i, (image, _, target) in enumerate(forget_train_dataloader):
            image = image.cuda()
            target = torch.randint(0, num_classes, target.shape).cuda()

            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for i, (image, _, target) in enumerate(retain_train_dataloader):
            image = image.cuda()
            target = target.cuda()

            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % 100 == 0:
                end = time.time()
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Time {3:.2f}'.format(
                    epoch, i, loader_len, end - start, loss=losses, top1=top1))
                start = time.time()

    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(model, unlearning_teacher, retain_train_dataloader,
                             forget_train_dataloader, valid_dataloader,  device, fast=True)

def salun(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader,
             valid_dataloader,  num_classes, device, weights_path, para1='0.0001', para2='2',
             mask_path=None, rum=False,
             **kwargs):
    # TODO salun mask
    mask = torch.load(mask_path)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(para1))

    for epoch in range(int(para2)):
        model.train()
        start = time.time()
        losses = AverageMeter()
        top1 = AverageMeter()
        loader_len = len(forget_train_dataloader) + len(retain_train_dataloader)

        for i, batch in enumerate(forget_train_dataloader):
            # if rum:
            #     image, target, _ = batch
            # else:
            #     image, _, target = batch
            image, _, target = batch
            image = image.cuda()
            target = torch.randint(0, num_classes, target.shape).cuda()

            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.module.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()

        for i, batch in enumerate(retain_train_dataloader):
            # if rum:
            #     image, target, _ = batch
            # else:
            #     image, _, target = batch
            image, _, target = batch

            image = image.cuda()
            target = target.cuda()

            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.module.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()
            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

        if (i + 1) % 100 == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Time {3:.2f}'.format(
                epoch, i, loader_len, end - start, loss=losses, top1=top1))
            start = time.time()
    if not rum:
        torch.save(model.state_dict(), weights_path)
        return get_metric_scores(model, unlearning_teacher, retain_train_dataloader,
                             forget_train_dataloader, valid_dataloader,  device, fast=True)
    else:
        return model

##############################
def save_hessian(model, filepath):
    """
    将模型每个参数的二阶导数（即 p.grad2_acc）保存到文件中。
    保存时构造一个字典，键为参数名称，值为对应的 p.grad2_acc。
    """
    hessian_dict = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            # 转移到 CPU 保存，避免 GPU 张量保存后再加载时问题
            hessian_dict[name] = p.grad2_acc.cpu()
    torch.save(hessian_dict, filepath)
    print(f"Hessian (second order derivatives) saved to {filepath}")

def load_hessian(model, filepath, device):
    """
    加载 Hessian 信息，并将每个参数的二阶导数数据赋值给 p.grad2_acc。

    参数：
        model: 待赋值的模型
        filepath: Hessian 信息保存的文件路径（例如 "fisher_hessian.pt"）
        device: 指定设备（如 "cuda" 或 "cpu"）
    """
    # 从文件中加载 Hessian 字典
    hessian_dict = torch.load(filepath, map_location=device)

    # 遍历模型参数，将对应的 Hessian 信息赋值给 p.grad2_acc
    for name, p in model.named_parameters():
        if p.requires_grad:
            if name in hessian_dict:
                # 注意：如果需要，转换到参数所在设备
                p.grad2_acc = hessian_dict[name].to(p.device)
            else:
                print(f"Warning: {name} 在 Hessian 数据中未找到！")

    print("Hessian 数据加载完成。")

def FisherForgetting(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader, valid_dataloader,
                      num_classes, device, weights_path, para1=1e-7, **kwargs):
    load = True #TODO

    def hessian(dataset, model):
        model.eval()
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        loss_fn = nn.CrossEntropyLoss()

        for p in model.parameters():
            p.grad_acc = 0
            p.grad2_acc = 0

        for data, _, orig_target in tqdm(train_loader):
            data, orig_target = data.to(device), orig_target.to(device)
            output = model(data)
            prob = F.softmax(output, dim=-1).data

            for y in range(output.shape[1]):
                target = torch.empty_like(orig_target).fill_(y)
                loss = loss_fn(output, target)
                model.zero_grad()
                loss.backward(retain_graph=True)
                for p in model.parameters():
                    if p.requires_grad:
                        p.grad_acc += (orig_target == target).float() * p.grad.data
                        p.grad2_acc += prob[:, y] * p.grad.data.pow(2)

        for p in model.parameters():
            p.grad_acc /= len(train_loader)
            p.grad2_acc /= len(train_loader)

    for p in model.parameters():
        p.data0 = deepcopy(p.data.clone())
    if load:
        # 加载 Hessian 信息文件替代重新计算
        load_hessian(model, "fisher_hessian.pt", device)
    else:
        hessian(retain_train_dataloader.dataset, model)
        save_hessian(model, "fisher_hessian.pt")

    fisher_dir = []

    alpha = float(para1)#1e-6
    def get_mean_var(p, is_base_dist=False, alpha=3e-6):
        var = deepcopy(1.0 / (p.grad2_acc + 1e-8))
        var = var.clamp(max=1e3)
        if p.size(0) == num_classes:
            var = var.clamp(max=1e2)
        var = alpha * var

        if p.ndim > 1:
            var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
        if not is_base_dist:
            mu = deepcopy(p.data0.clone())
        else:
            mu = deepcopy(p.data0.clone())
        if p.ndim == 1:
            # BatchNorm
            var *= 10
        #         var*=1
        return mu, var

    for i, p in enumerate(model.parameters()):
        #print("i", i)
        mu, var = get_mean_var(p, False, alpha=alpha)#mean, variance
        p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()
        fisher_dir.append(var.sqrt().view(-1).cpu().detach().numpy())

    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(model, unlearning_teacher, retain_train_dataloader,
                             forget_train_dataloader, valid_dataloader,  device)

############################## no
def ssd_tuning(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader, valid_dataloader,
               full_train_dataloader, device, weights_path, para1=1, para2=10, **kwargs):
    dampening_constant = float(para1)
    selection_weighting = float(para2)
    parameters = {"lower_bound": 1, "exponent": 1, "magnitude_diff": None, "min_layer": -1,
                  "max_layer": -1, "forget_threshold": 1,
        "dampening_constant": dampening_constant, "selection_weighting": selection_weighting}

    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    pdr = ssd.ParameterPerturber(model, optimizer, device, parameters) #here trained?
    model = model.eval()

    sample_importances = pdr.calc_importance(forget_train_dataloader)
    original_importances = pdr.calc_importance(full_train_dataloader)

    pdr.modify_weight(original_importances, sample_importances)

    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader, valid_dataloader,  device)

def validate(val_loader, model, criterion, print_freq):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, _, target) in enumerate(val_loader):
        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                    i, len(val_loader), loss=losses, top1=top1
                )
            )

    print("valid_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg

def warmup_lr(epoch, step, optimizer, one_epoch_step, warmup, lr0):
    overall_steps = warmup * one_epoch_step
    current_steps = epoch * one_epoch_step + step

    lr = lr0 * current_steps / overall_steps
    lr = min(lr, lr0)

    for p in optimizer.param_groups:
        p["lr"] = lr

def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)

import time
def FT_prune(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader,
             valid_dataloader,  device, weights_path, para1='0.01', para2='10', model_name='ResNet18',
             **kwargs):
    ##################################### Training setting #################################################
    # lr = 0.1  # "initial learning rate"
    warmup = 6  # "warm up epochs"
    # decreasing_lr = "91,136"  # "decreasing strategy"
    # decreasing_lr = list(map(int, decreasing_lr.split(",")))
    momentum = 0.9  # "momentum"
    weight_decay = 5e-4  # "weight decay"
    print_freq = 200  # "print frequency"
    ##################################### Unlearn setting #################################################
    with_l1 = True
    no_l1_epochs = 0  # "non l1 epochs"
    unlearn_lr = float(para1)  # "initial learning rate"
    unlearn_epochs = int(para2)  # "number of total epochs for unlearn to run"
    alpha = 5e-5  # "unlearn noise"

    criterion = nn.CrossEntropyLoss()
    if model_name == 'ViT':
        optimizer = torch.optim.AdamW(model.parameters(), lr=unlearn_lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), unlearn_lr, momentum=momentum, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)  # 0.1 is fixed

    for epoch in range(0, unlearn_epochs):
        start_time = time.time()
        print("Epoch #{}, Learning rate: {}".format(epoch, optimizer.state_dict()["param_groups"][0]["lr"]))

        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to train mode
        model.train()

        start = time.time()

        for i, (image, _, target) in enumerate(retain_train_dataloader):
            if epoch < warmup:
                warmup_lr(epoch, i + 1, optimizer, len(retain_train_dataloader), warmup, unlearn_lr)

            image = image.cuda()
            target = target.cuda()
            if epoch < unlearn_epochs - no_l1_epochs:
                current_alpha = alpha * (1 - epoch / (unlearn_epochs - no_l1_epochs))  # decaying
                ## current_alpha = args.alpha * (epoch / (args.unlearn_epochs-args.no_l1_epochs))  # increasing
            elif unlearn_epochs - no_l1_epochs == 0:
                current_alpha = alpha
            else:
                current_alpha = 0
            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)
            if with_l1:
                loss += current_alpha * l1_regularization(model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % print_freq == 0:
                end = time.time()
                print("Epoch: [{0}][{1}/{2}]\t"
                      "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                      "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                      "Time {3:.2f}".format(epoch, i, len(retain_train_dataloader), end - start, loss=losses, top1=top1))
                start = time.time()

        print("train_accuracy {top1.avg:.3f}".format(top1=top1))

        # scheduler.step()

        print("one epoch duration:{}".format(time.time() - start_time))

    # val
    validate(valid_dataloader, model, criterion, print_freq)
    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader, valid_dataloader,
                              device)

########
from torch.autograd import grad

def get_x_y_from_data_dict(data, device):
    x, y = data.values()
    if isinstance(x, list):
        x, y = x[0].to(device), y[0].to(device)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def sam_grad(model, loss):
    params = []
    for param in model.parameters():
        params.append(param)
    sample_grad = grad(loss, params)
    sample_grad = [x.view(-1) for x in sample_grad]
    return torch.cat(sample_grad)

def apply_perturb(model, v):
    curr = 0
    with torch.no_grad():
        for param in model.parameters():
            length = param.view(-1).shape[0]
            param += v[curr: curr + length].view(param.shape)
            curr += length

def woodfisher(model, train_dl, device, criterion, v):
    model.eval()
    k_vec = torch.clone(v)
    N = 1000
    o_vec = None
    for idx, batch in enumerate(tqdm(train_dl)):
        data, labels, clabels = batch
        model.zero_grad()
        data = data.to(device)
        label = clabels.to(device)
        output = model(data)
        loss = criterion(output, label)
        sample_grad = sam_grad(model, loss)
        with torch.no_grad():
            if o_vec is None:
                o_vec = torch.clone(sample_grad)
            else:
                tmp = torch.dot(o_vec, sample_grad)
                k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
                o_vec -= (tmp / (N + tmp)) * o_vec
        if idx > N:
            return k_vec
    return k_vec

def woodfisher_im(model, train_dl, device, criterion, v):
    model.eval()
    k_vec = torch.clone(v)
    N = 300000
    o_vec = None
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    for idx, batch in enumerate(tqdm(train_dl)):
        data, labels, clabels = batch
        model.zero_grad()
        data = data.to(device)
        label = clabels.to(device)
        output = model(data)
        loss = criterion(output, label)
        sample_grad = sam_grad(model, loss)
        with torch.no_grad():
            if o_vec is None:
                o_vec = torch.clone(sample_grad)
            else:
                tmp = torch.dot(o_vec, sample_grad)
                k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
                o_vec -= (tmp / (N + tmp)) * o_vec
        if idx > N:
            return k_vec
    return k_vec

def Wfisher(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader, valid_dataloader,
                      device, weights_path, para1=3,
            **kwargs):
    alpha = float(para1)
    batch_size = 64
    criterion = nn.CrossEntropyLoss()

    retain_grad_loader = torch.utils.data.DataLoader(
        retain_train_dataloader.dataset, batch_size=batch_size, shuffle=False
    )
    retain_loader = torch.utils.data.DataLoader(
        retain_train_dataloader.dataset, batch_size=1, shuffle=False
    )
    forget_loader = torch.utils.data.DataLoader(
        forget_train_dataloader.dataset, batch_size=batch_size, shuffle=False
    )

    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    forget_grad = torch.zeros_like(torch.cat(params)).to(device)
    retain_grad = torch.zeros_like(torch.cat(params)).to(device)
    total = 0
    model.eval()

    for i, batch in enumerate(tqdm(forget_loader)):
        data, labels, clabels = batch
        model.zero_grad()
        real_num = data.shape[0]
        data = data.to(device)
        label = clabels.to(device)
        output = model(data)
        loss = criterion(output, label)
        f_grad = sam_grad(model, loss) * real_num
        forget_grad += f_grad
        total += real_num

    total_2 = 0
    for i, batch in enumerate(tqdm(retain_grad_loader)):
        data, labels, clabels = batch
        model.zero_grad()
        real_num = data.shape[0]
        data = data.to(device)
        label = clabels.to(device)
        output = model(data)
        loss = criterion(output, label)
        r_grad = sam_grad(model, loss) * real_num
        retain_grad += r_grad
        total_2 += real_num

    retain_grad *= total / ((total + total_2) * total_2)
    forget_grad /= total + total_2

    perturb = woodfisher(
        model,
        retain_loader,
        device=device,
        criterion=criterion,
        v=forget_grad - retain_grad,
    )
    apply_perturb(model, alpha * perturb)

    # _ = fit_one_cycle(
    #     1, model, retain_train_dataloader, valid_dataloader, lr=0.01, device=next(model.parameters()).device
    # )

    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader, valid_dataloader,
                              device)

class Args:
    pass



def scrub(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader, valid_dataloader,
            device, weights_path, para1=0.001, para2=5,
            **kwargs):
    model_t = copy.deepcopy(model)

    args = Args()
    args.optim = 'adam'#'sgd'
    args.gamma = 1
    args.alpha = 0.1
    args.beta = 0
    args.smoothing = 0.5
    args.msteps = 2 #first 5 epochs, maximize the training acc of forget dataset; after that, minimize
    args.clip = 0.2
    args.sstart = 10
    args.kd_T = 4
    args.distill = 'kd'
    args.print_freq = 50

    args.sgda_epochs = int(para2)
    args.sgda_learning_rate = float(para1)#0.0005
    args.lr_decay_epochs = [3, 5, 9]
    args.lr_decay_rate = 0.1
    args.sgda_weight_decay = 5e-4
    args.sgda_momentum = 0.9

    model_s = copy.deepcopy(model)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    criterion_kd = DistillKL(args.kd_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    # optimizer
    if args.optim == "sgd":
        optimizer = optim.SGD(trainable_list.parameters(),
                              lr=args.sgda_learning_rate,
                              momentum=args.sgda_momentum,
                              weight_decay=args.sgda_weight_decay)
    elif args.optim == "adam":
        optimizer = optim.Adam(trainable_list.parameters(),
                               lr=args.sgda_learning_rate,
                               weight_decay=args.sgda_weight_decay)
    elif args.optim == "rmsp":
        optimizer = optim.RMSprop(trainable_list.parameters(),
                                  lr=args.sgda_learning_rate,
                                  momentum=args.sgda_momentum,
                                  weight_decay=args.sgda_weight_decay)

    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    t1 = time.time()
    acc_rs = []
    acc_fs = []
    acc_vs = []
    acc_fvs = []

    forget_validation_loader = copy.deepcopy(valid_dataloader)
    # fgt_cls = list(np.unique(forget_train_dataloader.dataset.targets))
    # indices = [i in fgt_cls for i in forget_validation_loader.dataset.targets]
    # forget_validation_loader.dataset.data = forget_validation_loader.dataset.data[indices]
    # forget_validation_loader.dataset.targets = forget_validation_loader.dataset.targets[indices]

    # scrub_name = "checkpoints/scrub_{}_{}_seed{}_step".format(args.model, args.dataset, args.seed)
    for epoch in range(1, args.sgda_epochs + 1):

        lr = adjust_learning_rate(epoch, args, optimizer)

        acc_r, acc5_r, loss_r = validate_scrub(retain_train_dataloader, model_s, criterion_cls, args, True)
        acc_f, acc5_f, loss_f = validate_scrub(forget_train_dataloader, model_s, criterion_cls, args, True)
        acc_v, acc5_v, loss_v = validate_scrub(valid_dataloader, model_s, criterion_cls, args, True)
        acc_fv, acc5_fv, loss_fv = validate_scrub(forget_validation_loader, model_s, criterion_cls, args, True)
        acc_rs.append(100 - acc_r.item())
        acc_fs.append(100 - acc_f.item())
        acc_vs.append(100 - acc_v.item())
        acc_fvs.append(100 - acc_fv.item())

        maximize_loss = 0
        if epoch <= args.msteps: #first three epochs
            maximize_loss = train_distill(epoch, forget_train_dataloader, module_list, None, criterion_list, optimizer, args,
                                          "maximize")
        #last two epoch
        train_acc, train_loss = train_distill(epoch, retain_train_dataloader, module_list, None, criterion_list, optimizer, args,
                                              "minimize")
        # torch.save(model_s.state_dict(), scrub_name + str(epoch) + ".pt")
        print("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(maximize_loss, train_loss,
                                                                                     train_acc))
    t2 = time.time()
    print(t2 - t1)

    acc_r, acc5_r, loss_r = validate_scrub(retain_train_dataloader, model_s, criterion_cls, args, True)
    acc_f, acc5_f, loss_f = validate_scrub(forget_train_dataloader, model_s, criterion_cls, args, True)
    acc_v, acc5_v, loss_v = validate_scrub(valid_dataloader, model_s, criterion_cls, args, True)
    acc_fv, acc5_fv, loss_fv = validate_scrub(forget_validation_loader, model_s, criterion_cls, args, True)
    acc_rs.append(100 - acc_r.item())
    acc_fs.append(100 - acc_f.item())
    acc_vs.append(100 - acc_v.item())
    acc_fvs.append(100 - acc_fv.item())

    from matplotlib import pyplot as plt
    # indices = list(range(0, len(acc_rs)))
    # plt.plot(indices, acc_rs, marker='*', color=u'#1f77b4', alpha=1, label='retain-set')
    # plt.plot(indices, acc_fs, marker='o', color=u'#ff7f0e', alpha=1, label='forget-set')
    # plt.plot(indices, acc_vs, marker='^', color=u'#2ca02c', alpha=1, label='validation-set')
    # plt.plot(indices, acc_fvs, marker='.', color='red', alpha=1, label='forget-validation-set')
    # plt.legend(prop={'size': 14})
    # plt.tick_params(labelsize=12)
    # plt.xlabel('epoch', size=14)
    # plt.ylabel('error', size=14)
    # plt.grid()
    # plt.show()

    try:
        selected_idx, _ = min(enumerate(acc_fs), key=lambda x: abs(x[1] - acc_fvs[-1]))
    except:
        selected_idx = len(acc_fs) - 1
    print("the selected index is {}".format(selected_idx))
    # selected_model = "checkpoints/scrub_{}_{}_seed{}_step{}.pt".format(args.model, args.dataset, args.seed,
    #                                                                    int(selected_idx))
    model = copy.deepcopy(model_s)
    # model_s.load_state_dict(torch.load(selected_model))
    # return model_s, model_s_final
    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader,
                             valid_dataloader,
                             device, fast=True)

def split_forget_dataset_by_mem(forget_dataset, forget_memorization, part_num=3):
    """
    将 forget_dataset 中的每个样本根据其 mem 值（第三个返回值）
    从低到高排序后，均分为三个子数据集：
      - low_forget_dataset：mem 较低的样本
      - mid_forget_dataset：mem 中间的样本
      - high_forget_dataset：mem 较高的样本
    返回值为 (low_forget_dataset, mid_forget_dataset, high_forget_dataset)
    """
    all_indices = list(range(len(forget_dataset)))
    mem_values = forget_memorization

    # 2. 对所有索引按照 mem 值从低到高排序
    sorted_indices = [idx for _, idx in sorted(zip(mem_values, all_indices), key=lambda pair: pair[0])]

    # 3. 均分索引列表
    n = len(sorted_indices)
    # 计算每一份的大小（余数部分归入最后一份）
    if part_num == 3:
        third = n // 3
        # 为了确保三份包含所有样本，可以这样划分：
        low_indices = sorted_indices[:third]
        mid_indices = sorted_indices[third:2 * third]
        high_indices = sorted_indices[2 * third:]

        # 4. 利用 Subset 构造三个子数据集
        low_forget_dataset = Subset(forget_dataset, low_indices)
        mid_forget_dataset = Subset(forget_dataset, mid_indices)
        high_forget_dataset = Subset(forget_dataset, high_indices)

        return low_forget_dataset, mid_forget_dataset, high_forget_dataset
    elif part_num == 2:
        split_num = int(n *0.7)
        # 为了确保三份包含所有样本，可以这样划分：
        low_indices = sorted_indices[:split_num]
        high_indices = sorted_indices[split_num:]

        # 4. 利用 Subset 构造三个子数据集
        low_forget_dataset = Subset(forget_dataset, low_indices)
        high_forget_dataset = Subset(forget_dataset, high_indices)

        return low_forget_dataset, high_forget_dataset

def rum(model,
        unlearning_teacher,
        retain_train_dataloader,
        forget_train_dataloader,
        valid_dataloader,
            device, weights_path, forget_dataset, forget_memorization, num_classes, para1=0.1, para2=0.00005, mask_path=None, forget_dataset_memorization=None,
            **kwargs):
    # meta unlearn
    # nothing-Finetune-SalUn (low-medium-high memorization order)
    # Step1: according to the memorization to seperate the dataloader into three subsets
    low_forget_dataset, mid_forget_dataset, high_forget_dataset = split_forget_dataset_by_mem(forget_dataset, forget_memorization)

    # 下面可以对 low_forget_dataset、mid_forget_dataset、high_forget_dataset 分别进行后续操作
    # print("low_forget_dataset size:", len(low_forget_dataset))
    # print("mid_forget_dataset size:", len(mid_forget_dataset))
    # print("high_forget_dataset size:", len(high_forget_dataset))

    #TODO notice the structure of the batch

    # Step2: do noting to the low; do fientune to the medium forget dataset; do salun to the high forget dataset
    mid_forget_dataloader = DataLoader(mid_forget_dataset,128, shuffle=True)
    model = finetune(model, unlearning_teacher, retain_train_dataloader, mid_forget_dataloader, valid_dataloader,
            device, weights_path, para1=para1, para2=5, mask_path=None, rum=True,
            **kwargs)

    high_forget_dataloader = DataLoader(high_forget_dataset,128, shuffle=True) #high_forget_dataset
    model = salun(model, unlearning_teacher, retain_train_dataloader, high_forget_dataloader, valid_dataloader,
            num_classes=num_classes, device=device, weights_path=weights_path, para1=str(para2),
                  para2=3, mask_path=mask_path,  rum=True,
            **kwargs)

    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader,
                             valid_dataloader,
                             device)

def sfron(model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader,
             valid_dataloader, num_classes, device, weights_path, para1='0.0001', para2='2',
             mask_path=None, model_name='ResNet18',
             **kwargs):
    forget_freq = 5
    # TODO salun mask
    mask = torch.load(mask_path)
    criterion = torch.nn.CrossEntropyLoss()
    if model_name == 'ViT':
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(para1), weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=float(para1))

    for epoch in range(int(para2)):
        model.train()
        start = time.time()
        losses = AverageMeter()
        top1 = AverageMeter()
        loader_len = len(forget_train_dataloader) + len(retain_train_dataloader)
        iter_forget_loader = iter(forget_train_dataloader)
        iter_retain_loader = iter(retain_train_dataloader)

        for step in range(len(retain_train_dataloader)):
            if step % forget_freq == 0:
                try:
                    image, _, target = next(iter_forget_loader)
                except:
                    iter_forget_loader = iter(forget_train_dataloader)
                    image, _, target = next(iter_forget_loader)

                image = image.cuda()
                target = target.cuda()

                # compute output
                output_clean = model(image)
                loss = -criterion(output_clean, target)

                optimizer.zero_grad()
                loss.backward()

                if mask:
                    for name, param in model.module.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]

                optimizer.step()

            try:
                image, _, target = next(iter_retain_loader)
            except:
                iter_retain_loader = iter(retain_train_dataloader)
                image, _, target = next(iter_retain_loader)

            image = image.cuda()
            target = target.cuda()

            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.module.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()
            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (step + 1) % 100 == 0:
                end = time.time()
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Time {3:.2f}'.format(
                    epoch, step, loader_len, end - start, loss=losses, top1=top1))
                start = time.time()

    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(model, unlearning_teacher, retain_train_dataloader,
                             forget_train_dataloader, valid_dataloader,  device, fast=True)

def orthogonality(model,
                  unlearning_teacher,
                  retain_train_dataloader,
                  forget_train_dataloader,
                  valid_dataloader,
                  num_classes,
                  device,
                  weights_path,
                  para1 ='0.0015',
                  para2 ='0.01',
                    model_name = 'ResNet18',
                    aug_retain_dataloader=None,
                    dataset_name='CIFAR10',
                    **kwargs):
    start = time.time()

    unlearninglabels = list(range(num_classes))
    unlearning_trainset = []

    ori_model = copy.deepcopy(model)
    ori_model.eval()

    forget_dataloader = DataLoader(
        forget_train_dataloader.dataset, 64, pin_memory=True, shuffle=True
    )

    ortho_gamma = 1.0 #2.5 for resnet18
    alpha = 0.0
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            # print(f"Activation of {name} - Shape: {output.shape}")
            # Assuming you want to store the output in a dictionary
            batch_size = output.size(0)
            activation[name] = output.view(batch_size, -1)
        return hook

    def orthogonality_loss(features1, features2, eps=1e-6, batch_size=32):
        # Normalize the features to unit vectors
        features1 = features1 / (features1.norm(p=2, dim=1, keepdim=True) + eps)
        features2 = features2 / (features2.norm(p=2, dim=1, keepdim=True) + eps)

        # Initialize loss variable
        loss = 0.0

        # Process the features in smaller batches to reduce memory consumption
        num_batches = features1.size(0) // batch_size + (1 if features1.size(0) % batch_size != 0 else 0)

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, features1.size(0))

            # Select the current batch of features
            batch_features1 = features1[start_idx:end_idx]
            batch_features2 = features2[start_idx:end_idx]

            # Compute the dot product for the current batch
            dot_product = torch.sum(batch_features1 * batch_features2, dim=1)  # Dot product along the feature dimension

            # Add the squared dot product to the loss
            loss += (dot_product ** 2).mean()  # Mean of the squared dot product

        return loss / num_batches

    def fit_one_unlearning_cycle_orthogonal(epochs, model, train_loader, lr, device, retain_loader=None, num_classes=20, mask=None):
        history = []

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        for epoch in range(epochs):
            model.train()
            train_losses = []
            lrs = []
            if retain_loader is not None:
                iter_retain = iter(retain_loader)
            for batch in train_loader:
                images, _, ori_labels = batch
                labels = torch.randint(0, num_classes, ori_labels.shape)

                images, labels, ori_labels = images.to(device), labels.to(device), ori_labels.to(device)
                out = model(images)  # Generate predictions
                _ = ori_model(images)  # Generate predictions

                # batch_features0 = activation.get('features0', None)
                batch_features1 = activation.get('features1', None)
                batch_features2 = activation.get('features2', None)
                batch_features3 = activation.get('features2', None)

                ori_features1 = activation.get('ori_features1', None)
                ori_features2 = activation.get('ori_features2', None)
                ori_features3 = activation.get('ori_features2', None)

                loss_ortho = 0.0
                if batch_features1 is not None:
                    loss_ortho = (orthogonality_loss(batch_features1.to(device), ori_features1.to(device)) +
                    orthogonality_loss(batch_features2.to(device), ori_features2.to(device)) +
                    orthogonality_loss(batch_features3.to(device), ori_features3.to(device)))

                loss = ortho_gamma * loss_ortho + 0.1 * F.cross_entropy(out, labels)

                loss.backward()
                train_losses.append(loss.detach().cpu())

                optimizer.step()
                optimizer.zero_grad()

                if retain_loader is not None:
                    try:
                        images, _, labels = next(iter_retain)
                    except:
                        iter_retain = iter(retain_loader)
                        images, _, labels = next(iter_retain)

                    images, labels = images.to(device), labels.to(device)
                    out = model(images)  # Generate predictions
                    loss = F.cross_entropy(out, labels)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # if mask:
                #     for name, param in model.named_parameters():
                #         if param.grad is not None:
                #             param.grad *= mask[name]

                lrs.append(get_lr(optimizer))

            result = evaluate(model, valid_dataloader, device)
            result["train_loss"] = torch.stack(train_losses).mean()
            result["lrs"] = lrs
            epoch_end(model, epoch, result)
            history.append(result)
        return history

    # 在 feature_extractor 中的 ReLU 层注册 forward hook，捕获激活值
    #for resnet18
    if model_name == 'ResNet18':
        hook_handle1 = model.conv3_x[-1].register_forward_hook(get_activation('features1'))
        hook_handle2 = model.conv4_x[-1].register_forward_hook(get_activation('features2'))
        hook_handle3 = model.conv5_x[-1].register_forward_hook(get_activation('features3'))
    elif model_name == 'ViT':
        hook_handle1 = model.transformer.layers[0][1].fn.net[0].register_forward_hook(
           get_activation('features1'))
        hook_handle2 = model.transformer.layers[3][0].fn.to_qkv.register_forward_hook(
            get_activation('features2'))
        hook_handle3 = model.mlp_head[0].register_forward_hook(get_activation('features3'))

        hook_handle_ori_1 = ori_model.transformer.layers[0][1].fn.net[0].register_forward_hook(
            get_activation('ori_features1'))
        hook_handle_ori_2 = ori_model.transformer.layers[3][0].fn.to_qkv.register_forward_hook(
            get_activation('ori_features2'))
        hook_handle_ori_3 = ori_model.mlp_head[0].register_forward_hook(get_activation('ori_features3'))

    #unlearn process of the amnesiac
    _ = fit_one_unlearning_cycle_orthogonal(6, model, forget_dataloader, #retain_loader=retain_train_dataloader,
                                            num_classes=num_classes, device=device, lr=float(para1)) #TODO 10

    # hook_handle0.remove()
    hook_handle1.remove()
    hook_handle2.remove()
    hook_handle3.remove()
    hook_handle_ori_1.remove()
    hook_handle_ori_2.remove()
    hook_handle_ori_3.remove()

    d_t, d_f, d_r, zrf, mia =get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dataloader,
        forget_train_dataloader,
        valid_dataloader,
        device,
        fast=True)
    print("d_t = ", d_t, "| d_f = ", d_f, "| d_r = ", d_r, "| mia = ", mia)

    if dataset_name == 'cifar20':
        _ = fit_one_cycle(
            3, model, aug_retain_dataloader, valid_dataloader, lr=float(para2), device=next(model.parameters()).device,
            model_name=model_name, l1=True
        )
        _ = fit_one_cycle(
            4, model, aug_retain_dataloader, valid_dataloader, lr=float(para2), device=next(model.parameters()).device,
            model_name=model_name
        )

        _ = fit_one_cycle(
            3, model, retain_train_dataloader, valid_dataloader, lr=float(para2), device=next(model.parameters()).device,
            model_name=model_name
        )
        _ = fit_one_cycle(
            5, model, retain_train_dataloader, valid_dataloader, lr=1e-4, device=next(model.parameters()).device,
            model_name=model_name
        )
    elif dataset_name == 'cifar10':
        _ = fit_one_cycle(
            3, model, aug_retain_dataloader, valid_dataloader, lr=float(para2), device=next(model.parameters()).device,
            model_name=model_name, l1=True
        )
        _ = fit_one_cycle(
            1, model, aug_retain_dataloader, valid_dataloader, lr=0.5*float(para2), device=next(model.parameters()).device,
            model_name=model_name
        )

        _ = fit_one_cycle(
            2, model, retain_train_dataloader, valid_dataloader, lr=0.2*float(para2),
            device=next(model.parameters()).device,
            model_name=model_name
        )

    end = time.time()
    time_elapsed = end - start
    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(
        model, unlearning_teacher, retain_train_dataloader, forget_train_dataloader,
        valid_dataloader,
        device,
        fast=True
    )
