import argparse
import copy
import json
import random

import numpy as np
# from pynvml import *
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

import config
import models
import datasets
from lira_utils import *
from models.inferencemodel import *

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
    # set random seed
    set_random_seed(args.seed)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # TODO load shadow and target models
    shadow_models = []
    for i in range(args.num_shadow):
        curr_model = InferenceModel(i, args).to(args.device)
        shadow_models.append(curr_model)

    # target_model = InferenceModel(7, args).to(args.device)
    # TODO target models
    if args.task == 'mia_lira':
        task_path = 'unlearning'
    elif args.task == 'rea':
        task_path = 'reminisence'
    target_model = getattr(models, args.net)(num_classes=args.classes)
    root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"
    trainset = getattr(datasets, args.dataset)(root=root, download=True, train=True,
                                                  unlearning=True)
    testset = getattr(datasets, args.dataset)(root=root, download=True, train=False,
                                                 unlearning=True)
    train_split_len = len(trainset)
    test_split_len = len(testset)

    if args.machine_unlearning is not None: #TDOO test the unlearned model
        print(f"Download forgetting_dataset_index_{args.forget_perc}.npy")
        checkpoint_path_folder = os.path.join(config.CHECKPOINT_PATH,
                                              "{unlearning_scenarios}".format(
                                                  unlearning_scenarios="forget_random_main"),
                                              "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset,
                                                                                 classes=args.classes),
                                              "{task}".format(task=task_path),#'unlearning' "reminisence"
                                              "{unlearning_method}_{para1}_{para2}".format(
                                                  unlearning_method=args.machine_unlearning,
                                                  para1=args.para1,
                                                  para2=args.para2))  # TODO unlearning

        checkpoint_path = os.path.join(checkpoint_path_folder, "{epoch}-{type}.pth")
        weight_path = checkpoint_path.format(epoch=1, type="last")
        index_set_path_folder = os.path.join(config.CHECKPOINT_PATH,
                                             "{unlearning_scenarios}".format(unlearning_scenarios="forget_random_main"),
                                             "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset,
                                                                                classes=args.classes),
                                             "random_index_set")
        target_in_data = np.load(index_set_path_folder + f'/forgetting_dataset_index_{args.forget_perc}.npy')
    else: #TODO test the original model
        weight_path = args.weight_path
        checkpoint_path_folder = os.path.join(config.CHECKPOINT_PATH,
                                              "{unlearning_scenarios}".format(
                                                  unlearning_scenarios="forget_random_main"),
                                              "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset,
                                                                                 classes=args.classes),
                                              "{task}".format(task=task_path))
        target_in_data = list(range(train_split_len))
    target_out_data = list(range(train_split_len, train_split_len + test_split_len))
    test_sample_number = min(500, len(target_out_data), len(target_in_data))

    print(f"load target model from {weight_path}")
    try:
        target_model.load_state_dict(torch.load(weight_path, weights_only=False))
    except:
        state_dict = torch.load(weight_path, weights_only=False)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '')  # 去除 'module.' 前缀
            new_state_dict[new_key] = value
        target_model.load_state_dict(new_state_dict)

    target_model = target_model.to(args.device)
    target_model.eval()

    # inferred set is the combination
    trainset = ConcatDataset([trainset, testset])

    args.aug_trainset = getattr(datasets, args.dataset)(root=root, download=True, train=True,
                                                           unlearning=False)
    args.aug_testset = getattr(datasets, args.dataset)(root=root, download=True, train=False,
                                                          unlearning=False,
                                                          data_augmentation=True)
    args.aug_trainset = ConcatDataset([args.aug_trainset,  args.aug_testset])

    args.img_shape = trainset[0][0].shape

    args.pred_logits = []  # N x (num of shadow + 1) x num_trials x num_class (target at -1)
    args.in_out_labels = []  # N x (num of shadow + 1)
    # args.canary_losses = []
    args.class_labels = []  # N
    args.img_id = []  # N

    #TODO evaluate
    testloader = DataLoader(testset, batch_size=256, shuffle=False)
    eval_training(target_model, testloader, args.device)

    negative_samples_idxes = list(np.random.choice(target_out_data, test_sample_number, replace=False))
    if test_sample_number < len(target_in_data):
        positive_samples_idxes = list(np.random.choice(target_in_data, test_sample_number, replace=False))
    else:
        positive_samples_idxes = list(target_in_data)
    sample_idxes = negative_samples_idxes + positive_samples_idxes
    for i in tqdm(range(len(sample_idxes))):  # the number of inferred dataset
    # for i in tqdm(range(1000)): #TODO i
        args.target_img_id = sample_idxes[i]
        # args.target_img_id = i #TODO i

        args.target_img, _, args.target_img_class = trainset[args.target_img_id]  # TODO inferred set
        args.target_img = args.target_img.unsqueeze(0).to(args.device)

        args.in_out_labels.append([])
        args.pred_logits.append([])

        if args.num_val:
            in_models, out_models = split_shadow_models(shadow_models, args.target_img_id)
            num_in = min(int(args.num_val / 2), len(in_models))
            num_out = args.num_val - num_in

            train_shadow_models = random.sample(in_models, num_in)
            train_shadow_models += random.sample(out_models, num_out)

            val_shadow_models = train_shadow_models
        else:
            train_shadow_models = shadow_models
            val_shadow_models = shadow_models

        curr_canaries = generate_aug_imgs(args)

        # get logits
        curr_canaries = torch.cat(curr_canaries, dim=0).to(args.device)
        for curr_model in val_shadow_models:
            args.pred_logits[-1].append(get_logits(curr_canaries, curr_model))
            args.in_out_labels[-1].append(int(args.target_img_id in curr_model.in_data))

        args.pred_logits[-1].append(get_logits(curr_canaries, target_model))
        args.in_out_labels[-1].append(int(args.target_img_id in target_in_data)) #TODO
        # args.in_out_labels[-1].append(int(args.target_img_id in target_model.in_data))

        args.img_id.append(args.target_img_id)
        args.class_labels.append(args.target_img_class)

    # accumulate results
    pred_logits = np.array(args.pred_logits)
    in_out_labels = np.array(args.in_out_labels)
    # print("in_out_labels", in_out_labels)
    # canary_losses = np.array(args.canary_losses)
    class_labels = np.array(args.class_labels)
    img_id = np.array(args.img_id)

    # save predictions
    os.makedirs(f'saved_predictions/{args.name}/', exist_ok=True)
    np.savez(f'saved_predictions/{args.name}/{args.save_name}.npz', pred_logits=pred_logits,
             in_out_labels=in_out_labels,  class_labels=class_labels,
             img_id=img_id) #canary_losses=canary_losses,

    ### dummy calculatiton of auc and acc
    ### to be simplified
    pred = np.load(f'saved_predictions/{args.name}/{args.save_name}.npz')

    pred_logits = pred['pred_logits']
    in_out_labels = pred['in_out_labels']
    class_labels = pred['class_labels']
    img_id = pred['img_id']

    in_out_labels = np.swapaxes(in_out_labels, 0, 1).astype(bool)
    pred_logits = np.swapaxes(pred_logits, 0, 1)

    scores = calibrate_logits(pred_logits, class_labels, args.logits_strategy)

    shadow_scores = scores[:-1]
    target_scores = scores[-1:]
    shadow_in_out_labels = in_out_labels[:-1]
    target_in_out_labels = in_out_labels[-1:]

    some_stats, tpr, fpr = cal_results(shadow_scores, shadow_in_out_labels, target_scores, target_in_out_labels,
                             logits_mul=args.logits_mul)
    print("save tpr to", checkpoint_path_folder + f'/lira_{args.task}_tpr.npy')
    np.save(checkpoint_path_folder + f'/lira_{args.task}_tpr.npy', np.array(tpr))
    np.save(checkpoint_path_folder + f'/lira_{args.task}_fpr.npy', np.array(fpr))
    summary_payload = {
        "attack_task": args.task,
        "dataset": args.dataset,
        "net": args.net,
        "classes": args.classes,
        "machine_unlearning": args.machine_unlearning,
        "para1": args.para1,
        "para2": args.para2,
        "num_shadow": args.num_shadow,
        "num_aug": args.num_aug,
        "seed": args.seed,
        "some_stats": some_stats,
    }
    with open(checkpoint_path_folder + f'/lira_{args.task}_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_payload, f, indent=2)

    print(some_stats)

    if not args.save_preds:
        os.remove(f'saved_predictions/{args.name}/{args.save_name}.npz')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Lira')
    parser.add_argument('--dataset', default='Cifar10')
    parser.add_argument('--net', default='ResNet18')
    parser.add_argument('--classes', default=10, type=int)
    parser.add_argument('--bs', default=512, type=int)
    parser.add_argument('--size', default=32, type=int)
    parser.add_argument('--name', default='test')
    parser.add_argument('--save_name', default='test')
    parser.add_argument('--num_shadow', default=None, type=int, required=True)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--patch', default=4, type=int, help="patch for ViT")
    parser.add_argument('--in_model_loss', default='ce', type=str)
    parser.add_argument('--out_model_loss', default='ce', type=str)
    parser.add_argument('--num_aug', default=1, type=int)
    parser.add_argument('--logits_mul', default=1, type=int)
    parser.add_argument('--logits_strategy', default='log_logits')
    parser.add_argument('--in_model_loss_weight', default=1, type=float)
    parser.add_argument('--out_model_loss_weight', default=1, type=float)
    parser.add_argument('--num_val', default=None, type=int)
    parser.add_argument('--no_dataset_aug', action='store_true')
    parser.add_argument('--balance_shadow', action='store_true')
    parser.add_argument('--target_logits', default=None, nargs='+', type=float)
    parser.add_argument('--save_preds', action='store_true')
    parser.add_argument('--offline', action='store_true')

    parser.add_argument('--weight_path', default=None, type=str)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)

    parser.add_argument("--machine_unlearning", type=str, default=None)
    parser.add_argument("--para1", type=str, default=0)
    parser.add_argument("--para2", type=str, default=0)
    parser.add_argument(
        "-forget_perc", type=float, default=0.1, help="Percentage of trainset to forget"
    )
    parser.add_argument(
        "-task", type=str, default='unlearning', help='unlearning, relearning_retain'
    )

    args = parser.parse_args()

    main(args)
