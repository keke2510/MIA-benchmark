import random
import os
# import wandb
# import optuna
from typing import Tuple, List
import sys
import argparse
import time

# import models
from datasets import CIFAR_MEAN, CIFAR_STD, TINYIMAGENET_MEAN, TINYIMAGENET_STD
from forget_random_strategies import get_metric_scores
from unlearn import *
from utils.overall_utils import *
import datasets
import models
import config
from utils.training_utils import *
import os.path as osp

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_seeds(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

"""
Get Args
"""
parser = argparse.ArgumentParser()
parser.add_argument("-net", type=str, default='ResNet18', help="net type")
parser.add_argument(
        "-weight_path",
        type=str,
        default=r".\log_files\model\pretrain\ResNet18-Cifar20-15\35-best.pth",
        help="Path to model weights. If you need to train a new model use pretrain_model.py",
)
parser.add_argument(
        "-dataset",
        type=str,
        default='Cifar20',
        nargs="?",
        choices=["Cifar19", "Cifar10", "Cifar20", "Cifar100", "Cinic10", "TinyImageNet", "PinsFaceRecognition"],
        help="dataset to train on",
)
parser.add_argument("-classes", type=int, default=15, help="number of classes")
parser.add_argument("-gpu", action="store_true", default=True, help="use gpu or not")
parser.add_argument("-b", type=int, default=256, help="batch size for dataloader")  # 128,32
parser.add_argument("-warm", type=int, default=1, help="warm up training phase")
parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
parser.add_argument(
        "-method",
        type=str,
        default="finetune",
)

parser.add_argument(
        "-forget_perc", type=float, default=0.1, help="Percentage of trainset to forget"
)

parser.add_argument(
    "-epochs", type=int, default=1, help="number of epochs of unlearning method to use"
)
parser.add_argument("-seed", type=int, default=0, help="seed for runs")

parser.add_argument("--para1", type=str, default=None)
parser.add_argument("--para2", type=str, default=None)

parser.add_argument("--blackbox", type=str, default='0')


args = parser.parse_args()


def resolve_img_size(dataset_name):
    if dataset_name == "TinyImageNet":
        return 64
    return 32

class CombinedImageDataset(Dataset):
    def __init__(self, images, class_list):
        self.images = images
        self.class_list = class_list
        #self.data = self._prepare_data()

    def _prepare_data(self):
        images = []
        for batch in self.original_dataloader:
            batch_images, _, _= batch
            for img in batch_images:
                images.append(img)
        return images

    def __len__(self):
        # 返回拼接图像的数量，等于原始图像数量的四分之一
        return 75#len(self.data) // 4

    def __getitem__(self, idx):
        label = self.class_list[0]
        return self.images[idx], label, label

class TransformableSubset(Subset):
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        img, _, label = super().__getitem__(idx)
        if self.transform:
            img = self.transform(img)
        return img, label, label

if __name__ == '__main__':
    set_seeds(args)
    batch_size = args.b

    # get network
    net = getattr(models, args.net)(num_classes=args.classes)
    unlearning_teacher = getattr(models, args.net)(num_classes=args.classes)
    if args.gpu and torch.cuda.is_available():
        net = net.cuda()
        unlearning_teacher = unlearning_teacher.cuda()

    root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"
    img_size = resolve_img_size(args.dataset)

    # TODO: cifar10 from retraining is augmented
    norm_mean = TINYIMAGENET_MEAN if args.dataset == "TinyImageNet" else CIFAR_MEAN
    norm_std = TINYIMAGENET_STD if args.dataset == "TinyImageNet" else CIFAR_STD
    mia_augmentation = [
        transforms.Resize((img_size, img_size)),
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ]

    trainset = getattr(datasets, args.dataset)(root=root, download=True, train=True,
                                               unlearning=True,
                                               img_size=img_size,
                                               data_aug=mia_augmentation)  # TODO if unlearning is true, the data aug is different

    validset = getattr(datasets, args.dataset)(root=root, download=True, train=False, unlearning=True,
                                               img_size=img_size)  # unlearning is set to True

    index_set_path_folder = os.path.join(config.CHECKPOINT_PATH,
                                         "{unlearning_scenarios}".format(unlearning_scenarios="forget_random_main"),
                                         "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset,
                                                                            classes=args.classes),
                                         "random_index_set")
    forget_set_idxs = np.load(index_set_path_folder + f'/forgetting_dataset_index_{args.forget_perc}.npy')

    remaining_set_idxs = list(set(np.arange(len(trainset))) - set(forget_set_idxs))
    forget_train_set = Subset(trainset, forget_set_idxs)
    retain_train_set = Subset(trainset, remaining_set_idxs)

    relearning_path = os.path.join(config.CHECKPOINT_PATH,
                            "{unlearning_scenarios}".format(unlearning_scenarios="forget_random_main"),
                            "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset, classes=args.classes),
                            "{task}".format(task="reminisence"),
                            "{unlearning_method}_{para1}_{para2}".format(unlearning_method=args.method,
                                                                         para1=args.para1, para2=args.para2))

    #where the unlearned model is
    checkpoint_path_folder = os.path.join(config.CHECKPOINT_PATH,
                                   "{unlearning_scenarios}".format(unlearning_scenarios="forget_random_main"),
                                   "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset, classes=args.classes),
                                   "{task}".format(task="unlearning"),
                                   "{unlearning_method}_{para1}_{para2}".format(unlearning_method=args.method, para1=args.para1, para2=args.para2))

    print("#####", relearning_path)
    os.makedirs(relearning_path, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_path_folder, "{epoch}-{type}.pth")
    weight_path = checkpoint_path.format(epoch=args.epochs, type="last")

    model_size_scaler = 1
    if args.net == "ViT":
        model_size_scaler = 1
    else:
        model_size_scaler = 1

    try:
        net.load_state_dict(torch.load(weight_path))
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            net = nn.DataParallel(net)
    except:
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            net = nn.DataParallel(net)
        net.load_state_dict(torch.load(weight_path))

    device = "cuda" if args.gpu else "cpu"
    forget_train_dataloader = DataLoader(list(forget_train_set), batch_size=batch_size)
    retain_train_dataloader = DataLoader(list(retain_train_set), batch_size=batch_size, shuffle=True)

    #here, perform lira



    validloader = DataLoader(validset, num_workers=0, batch_size=batch_size, shuffle=False)  # num_workers=4
    clean_acc, forgetting_acc, remaining_acc, zrf, mia = get_metric_scores(net, unlearning_teacher,
                                                                           retain_train_dataloader,
                                                                           forget_train_dataloader, validloader,
                                                                           device, fast=True)
    print("d_t = ", clean_acc, "| d_f = ", forgetting_acc, "| d_r = ", remaining_acc,
          "| zrf = ",
          zrf, "| mia = ", mia)
    start = time.time()

    if args.net == 'ViT':
        lr1 = 1e-4
        lr2 = 5e-5
    else:
        # ResNet18/ResNet50 and other CNN backbones share the same
        # conservative reminiscence fine-tuning schedule.
        lr1 = 0.01
        lr2 = 0.005

    _ = fit_one_cycle(
        epochs=3, model=net, train_loader=retain_train_dataloader, val_loader=validloader, lr=lr1,
        device=next(net.parameters()).device, model_name=args.net,
    )

    _ = fit_one_cycle(
        epochs=3, model=net, train_loader=retain_train_dataloader, val_loader=validloader, lr=lr2,
        device=next(net.parameters()).device, model_name=args.net,
    )

    clean_acc, forgetting_acc, remaining_acc, zrf, mia = get_metric_scores(net,
                                                                           unlearning_teacher,
                                                                           retain_train_dataloader,
                                                                           forget_train_dataloader,
                                                                           validloader,
                                                                           device,
                                                                           fast=True)
    end = time.time()
    time_elapsed = end - start

    state_dict = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
    torch.save(state_dict, os.path.join(relearning_path, "1-last.pth"))
    
    logname = osp.join(relearning_path, 'log_{}-{}-{}.tsv'.format(args.net, args.dataset, args.classes))
    with open(logname, 'w+') as f:
        columns = ['clean_acc',
                   'forgetting_acc',
                   'remaining_acc',
                   'zrf',
                   'mia',
                   'time'
                   ]
        f.write('\t'.join(columns) + '\n')

    print("d_t = ", clean_acc, "| d_f = ", forgetting_acc, "| d_r = ", remaining_acc,
          "| zrf = ",
          zrf, "| mia = ", mia, "| time = ", time_elapsed)

    with open(logname, 'a') as f:
        columns = [f"{clean_acc}",
                   f"{forgetting_acc}",
                   f"{remaining_acc}",
                   f"{zrf}",
                   f"{mia}",
                   f"{time_elapsed}"
                   ]
        f.write('\t'.join(columns) + '\n')
