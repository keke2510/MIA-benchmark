import argparse
import copy
import os
from collections import OrderedDict
import sys

# import arg_parser
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader, ConcatDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import datasets
import forget_full_class_strategies
import models
import unlearn
# import saliency_utils as utils
import random
import numpy as np

from utils.utils import get_classwise_ds, build_retain_sets_in_unlearning


def save_gradient_ratio(data_loaders, model, criterion, args):
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.unlearn_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    gradients = {}

    forget_loader = data_loaders["forget"]
    model.eval()

    for name, param in model.named_parameters():
        gradients[name] = 0

    for i, (image, _, target) in enumerate(forget_loader):
        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = - criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for i in threshold_list:
        print(i)
        sorted_dict_positions = {}
        hard_dict = {}

        # Concatenate all tensors into a single tensor
        all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()])

        # Calculate the threshold index for the top 10% elements
        threshold_index = int(len(all_elements) * i)

        # Calculate positions of all elements
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            # tensor_positions = positions[start_index: start_index + num_elements]
            tensor_ranks = ranks[start_index : start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements

        torch.save(hard_dict, os.path.join(args.save_dir, "with_{}.pt".format(i)))

def main(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        # Set seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    seed = args.seed

    root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"

    forget_class = int(args.forget_class)

    batch_size = args.batch_size
    img_size = 224 if args.net == "ViT" else 32
    trainset = getattr(datasets, args.dataset)(root=root, download=True, train=True, unlearning=True, img_size=img_size)
    validset = getattr(datasets, args.dataset)(root=root, download=True, train=False, unlearning=True,
                                               img_size=img_size)

    classwise_train, classwise_test = get_classwise_ds(trainset, num_classes=20), \
                                      get_classwise_ds(validset, num_classes=20)

    (retain_train, retain_valid) = build_retain_sets_in_unlearning(classwise_train, classwise_test, 20,
                                                                   int(args.forget_class), config.ood_classes)

    forget_train, forget_valid = classwise_train[int(args.forget_class)], classwise_test[int(args.forget_class)]

    forget_valid_dl = DataLoader(forget_valid, batch_size)
    retain_valid_dl = DataLoader(retain_valid, batch_size)
    forget_train_dl = DataLoader(forget_train, batch_size)
    retain_train_dl = DataLoader(retain_train, batch_size, shuffle=True)
    full_train_dl = DataLoader(ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)),
                               batch_size=batch_size, )
    # full_valid_dl = DataLoader(ConcatDataset((retain_valid_dl.dataset, forget_valid_dl.dataset)),
    #                            batch_size=batch_size, )

    ood_valid_ds = {}
    ood_train_ds = {}
    ood_valid_dl = []
    ood_train_dl = []
    for cls in config.ood_classes:
        ood_valid_ds[cls] = []
        ood_train_ds[cls] = []

        for img, label, clabel in classwise_test[cls]:
            ood_valid_ds[cls].append((img, label, int(args.forget_class)))

        for img, label, clabel in classwise_train[cls]:
            ood_train_ds[cls].append((img, label, int(args.forget_class)))

        ood_valid_dl.append(DataLoader(ood_valid_ds[cls], batch_size))
        ood_train_dl.append(DataLoader(ood_train_ds[cls], batch_size))

    # get network
    model = getattr(models, args.net)(num_classes=args.classes)
    model.load_state_dict(torch.load(args.mask))
    model.cuda()

    unlearn_data_loaders = OrderedDict(
        retain=retain_train_dl, forget=forget_train_dl, val=full_train_dl, test=retain_train_dl
    )

    criterion = nn.CrossEntropyLoss()

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        checkpoint = torch.load(args.mask, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]

        if args.unlearn != "retrain":
            model.load_state_dict(checkpoint, strict=False)

        save_gradient_ratio(unlearn_data_loaders, model, criterion, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Lottery Tickets Experiments")

    ##################################### Dataset #################################################
    parser.add_argument(
        "--data", type=str, default="../data", help="location of the data corpus"
    )
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
    parser.add_argument(
        "--img_size", type=int, default=32, help="size of input images")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./tiny-imagenet-200",
        help="dir to tiny-imagenet",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--classes", type=int, default=10)

    ##################################### Architecture ############################################
    parser.add_argument(
        "--net", type=str, default="resnet18", help="model architecture"
    )
    parser.add_argument(
        "--imagenet_arch",
        action="store_true",
        help="architecture for imagenet size samples",
    )

    ##################################### General setting ############################################
    parser.add_argument("--seed", default=2, type=int, help="random seed")
    parser.add_argument(
        "--train_seed",
        default=1,
        type=int,
        help="seed for training (default value same as args.seed)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument(
        "--workers", type=int, default=4, help="number of workers in dataloader"
    )
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint file")
    parser.add_argument(
        "--save_dir",
        help="The directory used to save the trained models",
        default=None,
        type=str,
    )
    parser.add_argument("--mask", type=str, default=None, help="sparse model")

    ##################################### Training setting #################################################
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument(
        "--epochs", default=182, type=int, help="number of total epochs to run"
    )
    parser.add_argument("--warmup", default=0, type=int, help="warm up epochs")
    parser.add_argument("--print_freq", default=50, type=int, help="print frequency")
    parser.add_argument("--decreasing_lr", default="91,136", help="decreasing strategy")
    parser.add_argument(
        "--no-aug",
        action="store_true",
        default=False,
        help="No augmentation in training dataset (transformation).",
    )
    parser.add_argument("--no-l1-epochs", default=0, type=int, help="non l1 epochs")

    ##################################### Pruning setting #################################################
    parser.add_argument("--prune", type=str, default="omp", help="method to prune")
    parser.add_argument(
        "--pruning_times",
        default=1,
        type=int,
        help="overall times of pruning (only works for IMP)",
    )
    parser.add_argument(
        "--rate", default=0.95, type=float, help="pruning rate"
    )  # pruning rate is always 20%
    parser.add_argument(
        "--prune_type",
        default="rewind_lt",
        type=str,
        help="IMP type (lt, pt or rewind_lt)",
    )
    parser.add_argument(
        "--random_prune", action="store_true", help="whether using random prune"
    )
    parser.add_argument("--rewind_epoch", default=0, type=int, help="rewind checkpoint")
    parser.add_argument(
        "--rewind_pth", default=None, type=str, help="rewind checkpoint to load"
    )

    ##################################### Unlearn setting #################################################
    parser.add_argument(
        "--unlearn", type=str, default="retrain", help="method to unlearn"
    )
    parser.add_argument(
        "--unlearn_lr", default=0.01, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--unlearn_epochs",
        default=1,
        type=int,
        help="number of total epochs for unlearn to run",
    )

    parser.add_argument(
        "--class_to_replace", type=int, default=-1, help="Specific class to forget"
    )

    parser.add_argument(
        "--indexes_to_replace",
        type=list,
        default=None,
        help="Specific index data to forget",
    )
    parser.add_argument("--alpha", default=0.2, type=float, help="unlearn noise")

    parser.add_argument("--path", default=None, type=str, help="mask matrix")
    parser.add_argument(
        "--forget_class",
        type=str,
        default="4",  # 4
        nargs="?",
        help="class to forget",
        # choices=list(config.class_dict),
    )
    ##################################### Attack setting #################################################
    parser.add_argument(
        "--attack", type=str, default="backdoor", help="method to unlearn"
    )
    parser.add_argument(
        "--trigger_size",
        type=int,
        default=4,
        help="The size of trigger of backdoor attack",
    )
    args = parser.parse_args()

    main(args)
