import argparse
import copy
import random

import numpy as np
# from pynvml import *
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

import config
import models
import datasets
from lira_utils import *
from inferencemodel import *

def generate_class_dict(args):
    dataset_class_dict = [[] for _ in range(args.classes)]
    for i in range(len(args.aug_trainset)):
        _, _, tmp_class = args.aug_trainset[i]
        dataset_class_dict[tmp_class].append(i)

    return dataset_class_dict

@torch.no_grad()
def eval_training(target_model, testloader, device):
    start = time.time()
    target_model.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for images, _, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = target_model(images)
        loss = F.cross_entropy(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print("Evaluating Network.....")
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s".format(
            test_loss / len(testloader.dataset),
            correct.float() / len(testloader.dataset),
            finish - start,
        )
    )
    return correct.float() / len(testloader.dataset)


def main(args):
    checkpoint_path_folder = os.path.join(config.CHECKPOINT_PATH,
                                              "{unlearning_scenarios}".format(
                                                  unlearning_scenarios="forget_random_main"),
                                              "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset,
                                                                                 classes=args.classes),
                                              "{task}".format(task='unlearning'),#'unlearning' "relearning_retain"
                                              "{unlearning_method}_{para1}_{para2}".format(
                                                  unlearning_method=args.machine_unlearning,
                                                  para1=args.para1,
                                                  para2=args.para2))  # TODO unlearning

    jogging_checkpoint_path_folder = os.path.join(config.CHECKPOINT_PATH,
                                              "{unlearning_scenarios}".format(
                                                  unlearning_scenarios="forget_random_main"),
                                              "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset,
                                                                                 classes=args.classes),
                                              "{task}".format(task='relearning_retain'),  # 'unlearning' "relearning_retain"
                                              "{unlearning_method}_{para1}_{para2}".format(
                                                  unlearning_method=args.machine_unlearning,
                                                  para1=args.para1,
                                                  para2=args.para2))  # TODO unlearning

    unlearning_tpr = np.load(checkpoint_path_folder + f'/lira_unlearning_tpr.npy')
    unlearning_fpr = np.load(checkpoint_path_folder + f'/lira_unlearning_fpr.npy')

    up_tpr = np.load(checkpoint_path_folder + f'/up_tpr.npy')
    up_fpr = np.load(checkpoint_path_folder + f'/up_fpr.npy')

    relearning_tpr = np.load(jogging_checkpoint_path_folder + f'/lira_relearning_retain_tpr.npy')
    relearning_fpr = np.load(jogging_checkpoint_path_folder + f'/lira_relearning_retain_fpr.npy')

    privacy_methods = ['MIA-LiRa', 'MIA-UP', 'ReA']
    plt.figure(figsize=(2.8, 2.3))
    auc = np.trapz(unlearning_tpr, unlearning_fpr)
    plt.plot(unlearning_fpr,
                 unlearning_tpr, lw=2,
                 label=f'{privacy_methods[0]}:AUC = {auc:.2f}')

    auc = np.trapz(up_tpr, up_fpr)
    plt.plot(up_fpr,
             up_tpr, lw=2,
             label=f'{privacy_methods[1]}:AUC = {auc:.2f}')

    auc = np.trapz(relearning_tpr, relearning_fpr)
    plt.plot(relearning_fpr,
             relearning_tpr, lw=2,
             label=f'{privacy_methods[2]}:AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xscale('log')
    plt.yscale('log')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout(pad=0.0)
    print("saved figer to ", f'plot_our/sample_wise_mia/{args.machine_unlearning}_auc.pdf')
    # plt.savefig(f'plot_our/sample_wise_mia/{args.machine_unlearning}_auc.pdf', pad_inches=0)
    plt.savefig(f'plot_our/sample_wise_mia/{args.machine_unlearning}_auc.pdf', pad_inches=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Lira')
    parser.add_argument('--dataset', default='Cifar10')
    parser.add_argument('--net', default='ResNet18')
    parser.add_argument('--classes', default=10, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_aug', default=1, type=int)
    parser.add_argument('--logits_mul', default=1, type=int)
    parser.add_argument('--logits_strategy', default='log_logits')
    parser.add_argument('--in_model_loss_weight', default=1, type=float)
    parser.add_argument('--out_model_loss_weight', default=1, type=float)
    parser.add_argument('--num_val', default=None, type=int)
    parser.add_argument('--no_dataset_aug', action='store_true')
    parser.add_argument('--balance_shadow', action='store_true')

    parser.add_argument("--machine_unlearning", type=str, default=None)
    parser.add_argument("--para1", type=str, default=0)
    parser.add_argument("--para2", type=str, default=0)
    parser.add_argument(
        "-forget_perc", type=float, default=0.1, help="Percentage of trainset to forget"
    )

    args = parser.parse_args()

    main(args)