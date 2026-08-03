
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torchvision
from torch.utils.data import Subset
import torch
from torch.utils.data import Dataset
from collections import namedtuple
from torchvision import datasets
import random
import torch
from torch.utils.data import Dataset
try:
    import kagglehub
except ImportError:  # Optional dependency for datasets not used in this benchmark.
    kagglehub = None
try:
    from datasets import load_dataset, load_from_disk
except ImportError:  # Optional dependency for datasets not used in this benchmark.
    load_dataset = None
    load_from_disk = None
# from transformers import GPT2Tokenizer
import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import torchvision
from torchvision import transforms, datasets
# from transformers import GPT2Tokenizer, AutoTokenizer
import os
import shutil
import tarfile
import urllib.request
import zipfile
from PIL import Image
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))


CINIC10_MEAN = (0.47889522, 0.47227842, 0.43047404)
CINIC10_STD = (0.24205776, 0.23828046, 0.25874835)
TINYIMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINYIMAGENET_STD = (0.2302, 0.2265, 0.2262)


def resolve_torchvision_root(dataset: str) -> tuple[str, bool]:
    if dataset == 'cifar10':
        candidates = [
            os.path.join(BASE_DIR, 'data'),
            os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data')),
            os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'DHAttack-main', 'data')),
        ]
        marker = 'cifar-10-batches-py'
    elif dataset == 'cifar100':
        candidates = [
            os.path.join(BASE_DIR, 'data'),
            os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data')),
        ]
        marker = 'cifar-100-python'
    elif dataset == 'svhn':
        candidates = [
            os.path.join(BASE_DIR, 'data'),
            os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data')),
        ]
        marker = 'svhn'
    else:
        return os.path.join(BASE_DIR, 'data', dataset), True

    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, marker)):
            print(f"[RULI] Reusing local dataset root: {candidate}")
            return candidate, False
    print(f"[RULI] Local dataset root not found for {dataset}, fallback to download into: {candidates[0]}")
    return candidates[0], True


def resolve_cinic10_root() -> str:
    candidates = [
        os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'CINIC-10-download')),
        os.path.join(BASE_DIR, 'data', 'CINIC-10-download'),
        os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'DHAttack-main', 'data', 'CINIC-10-download')),
    ]
    for candidate in candidates:
        if all(os.path.isdir(os.path.join(candidate, split)) for split in ('train', 'valid', 'test')):
            print(f"[RULI] Reusing local CINIC-10 root: {candidate}")
            return candidate

    download_base = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data'))
    os.makedirs(download_base, exist_ok=True)
    archive_path = os.path.join(download_base, 'CINIC-10.tar.gz')
    target_root = os.path.join(download_base, 'CINIC-10-download')
    url = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz'
    print(f"[RULI] CINIC-10 not found, downloading from {url}")
    urllib.request.urlretrieve(url, archive_path)
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=download_base)
    extracted_root = os.path.join(download_base, 'CINIC-10')
    if os.path.isdir(extracted_root) and extracted_root != target_root:
        if os.path.isdir(target_root):
            shutil.rmtree(target_root)
        os.replace(extracted_root, target_root)
    return target_root


class Cinic10Dataset(Dataset):
    def __init__(self, root, split, transform):
        self.inner = datasets.ImageFolder(os.path.join(root, split), transform=transform)
        self.targets = list(self.inner.targets)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        return self.inner[index]


class Cinic10TrainDataset(Dataset):
    def __init__(self, root, transform):
        self.datasets = [
            datasets.ImageFolder(os.path.join(root, 'train'), transform=transform),
            datasets.ImageFolder(os.path.join(root, 'valid'), transform=transform),
        ]
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.cumulative = np.cumsum(self.lengths)
        self.targets = []
        for dataset in self.datasets:
            self.targets.extend(list(dataset.targets))

    def __len__(self):
        return int(self.cumulative[-1])

    def __getitem__(self, index):
        dataset_idx = int(np.searchsorted(self.cumulative, index, side='right'))
        offset = 0 if dataset_idx == 0 else int(self.cumulative[dataset_idx - 1])
        return self.datasets[dataset_idx][index - offset]


def resolve_tinyimagenet_root() -> str:
    repo_data_root = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data'))
    candidates = [
        os.path.join(repo_data_root, 'tiny-imagenet-200'),
        os.path.join(repo_data_root, 'TinyImageNet'),
    ]
    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, 'train')) and os.path.isdir(os.path.join(candidate, 'val')):
            print(f"[RULI] Reusing local TinyImageNet root: {candidate}")
            return candidate

    download_base = repo_data_root
    os.makedirs(download_base, exist_ok=True)
    archive_path = os.path.join(download_base, 'tiny-imagenet-200.zip')
    target_root = os.path.join(download_base, 'tiny-imagenet-200')
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    print(f"[RULI] TinyImageNet not found, downloading from {url}")
    urllib.request.urlretrieve(url, archive_path)
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(path=download_base)
    return target_root


def tinyimagenet_class_to_idx(root):
    train_dir = os.path.join(root, 'train')
    class_names = sorted(
        entry
        for entry in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, entry))
    )
    return {class_name: idx for idx, class_name in enumerate(class_names)}


class TinyImageNetValDataset(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        images_dir = os.path.join(root, 'val', 'images')
        annotations_path = os.path.join(root, 'val', 'val_annotations.txt')
        if os.path.isdir(images_dir) and os.path.isfile(annotations_path):
            class_to_idx = tinyimagenet_class_to_idx(root)
            samples = []
            with open(annotations_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        image_name, class_name = parts[0], parts[1]
                        image_path = os.path.join(images_dir, image_name)
                        if os.path.isfile(image_path) and class_name in class_to_idx:
                            samples.append((image_path, class_to_idx[class_name]))
            self.samples = samples
            self.targets = [target for _, target in samples]
            self.loader = lambda path: Image.open(path).convert('RGB')
        else:
            inner = datasets.ImageFolder(os.path.join(root, 'val'), transform=transform)
            self.samples = list(inner.samples)
            self.targets = list(inner.targets)
            self.loader = inner.loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target




class mul_loader:
    @staticmethod
    def _get_data_loader(dataset, batch_size, num_workers, shuffle=False):
        return data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    @staticmethod
    def load_data(dataset):

        if dataset == 'cifar10':
            root, download = resolve_torchvision_root('cifar10')
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            train_set = torchvision.datasets.CIFAR10(
                root=root, train=True, download=download, transform=transform_train)
            test_set = torchvision.datasets.CIFAR10(
                root=root, train=False, download=download, transform=transform_test)

            return train_set, test_set

        if dataset == 'cifar100':
            root, download = resolve_torchvision_root('cifar100')
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            train_set = torchvision.datasets.CIFAR100(
                root=root, train=True, download=download, transform=transform_train)
            test_set = torchvision.datasets.CIFAR100(
                root=root, train=False, download=download, transform=transform_test)

            return train_set, test_set

        if dataset == 'cinic10':
            root = resolve_cinic10_root()
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CINIC10_MEAN, CINIC10_STD),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CINIC10_MEAN, CINIC10_STD),
            ])

            train_set = Cinic10TrainDataset(root, transform_train)
            test_set = Cinic10Dataset(root, 'test', transform_test)
            return train_set, test_set

        if dataset == 'svhn':
            root, download = resolve_torchvision_root('svhn')
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ])

            train_set = torchvision.datasets.SVHN(
                root=root, split='train', download=download, transform=transform_train)
            test_set = torchvision.datasets.SVHN(
                root=root, split='test', download=download, transform=transform_test)

            return train_set, test_set
        #
        if dataset == 'TinyImageNet':
            dataset_dir = resolve_tinyimagenet_root()
            train_transform = transforms.Compose([
                transforms.Resize(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD),
            ])

            val_transform = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD),
            ])

            full_train_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform=train_transform)

            num_train = len(full_train_dataset)
            indices = list(range(num_train))
            split = int(np.floor(0.1 * num_train))  # Using 20% of the data for validation

            np.random.seed(42)
            np.random.shuffle(indices)
            train_idx = indices[split:]
            train_dataset = Subset(full_train_dataset, train_idx)
            train_dataset.dataset.transform = train_transform  # Apply training transforms
            val_dataset = TinyImageNetValDataset(dataset_dir, transform=val_transform)

            return train_dataset, val_dataset



    @staticmethod
    def load_mul_data(dataset, task, f_label=0, forget_size=4500, chunk_size=1000, start_index=0, forget_indices_path=None):
        split_data = {}

        def apply_shared_forget_indices(train_set):
            if not forget_indices_path:
                return None
            if not os.path.isfile(forget_indices_path):
                print(f"[RULI] Shared forget index not found, fallback to random selective split: {forget_indices_path}")
                return None
            forget_indices = np.load(forget_indices_path)
            forget_indices = np.asarray(forget_indices, dtype=int).reshape(-1)
            if len(forget_indices) == 0:
                raise ValueError(f"Shared forget index file is empty: {forget_indices_path}")
            max_index = len(train_set) - 1
            if forget_indices.min() < 0 or forget_indices.max() > max_index:
                raise ValueError(
                    f"Shared forget indices out of range for dataset of size {len(train_set)}: {forget_indices_path}"
                )
            all_indices = np.arange(len(train_set))
            remain_indices = np.setdiff1d(all_indices, forget_indices, assume_unique=False)
            return {
                'forget': Subset(train_set, forget_indices.tolist()),
                'remain': Subset(train_set, remain_indices.tolist()),
                'forget_index': forget_indices,
                'remain_index': remain_indices,
            }

        if dataset == 'cifar10':
            root, download = resolve_torchvision_root('cifar10')

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            if task == 'class-wise':
                train_set = torchvision.datasets.CIFAR10(
                    root=root, train=True, download=download, transform=transform_train)

                split_data['forget'], split_data['remain'] = LabelSplitter(train_set, label=f_label).split()
                if forget_size is not None:
                    result = RandomSplitter(split_data['forget'], num_samples=forget_size).split()
                    split_data['forget'] = result.selected_data
                    split_data['forget_index'] = result.selected_indices
                    split_data['remain_index'] = result.remaining_indices
                    split_data['remain'] = data.ConcatDataset([result.remaining_data, split_data['remain']])


            if task == 'selective':

                train_set = torchvision.datasets.CIFAR10(
                    root=root, train=True, download=download, transform=transform_train)
                shared_split = apply_shared_forget_indices(train_set)
                if shared_split is not None:
                    split_data.update(shared_split)
                    return split_data
                num_data = int(forget_size)
                result = RandomSplitter(train_set, num_samples=num_data).split()
                split_data['forget'] = result.selected_data
                split_data['remain'] = result.remaining_data
                split_data['forget_index'] = result.selected_indices
                split_data['remain_index'] = result.remaining_indices

            if task == 'identify':
                # use the vulnerable splitter
                train_set = torchvision.datasets.CIFAR10(
                    root=root, train=True, download=download, transform=transform_train)
                result = VulnerableSplitter(train_set, chunk_size, start_index).split()
                split_data['forget'] = result.selected_data
                split_data['remain'] = result.remaining_data
                split_data['forget_index'] = result.selected_indices
                split_data['remain_index'] = result.remaining_indices

            return split_data


        #######################################
        if dataset == 'cifar100':
            root, download = resolve_torchvision_root('cifar100')

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            if task == 'class-wise':
                train_set = torchvision.datasets.CIFAR100(
                    root=root, train=True, download=download, transform=transform_train)

                split_data['forget'], split_data['remain'] = LabelSplitter(train_set, label=f_label).split()

            if task == 'selective':

                train_set = torchvision.datasets.CIFAR100(
                    root=root, train=True, download=download, transform=transform_train)
                shared_split = apply_shared_forget_indices(train_set)
                if shared_split is not None:
                    split_data.update(shared_split)
                    return split_data
                num_data = int(forget_size)
                result = RandomSplitter(train_set, num_samples=num_data).split()
                split_data['forget'] = result.selected_data
                split_data['remain'] = result.remaining_data
                split_data['forget_index'] = result.selected_indices
                split_data['remain_index'] = result.remaining_indices

            return split_data

        if dataset == 'cinic10':
            root = resolve_cinic10_root()

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CINIC10_MEAN, CINIC10_STD),
            ])

            train_set = Cinic10TrainDataset(root, transform_train)

            if task == 'class-wise':
                split_data['forget'], split_data['remain'] = LabelSplitter(train_set, label=f_label).split()
                if forget_size is not None:
                    result = RandomSplitter(split_data['forget'], num_samples=forget_size).split()
                    split_data['forget'] = result.selected_data
                    split_data['forget_index'] = result.selected_indices
                    split_data['remain_index'] = result.remaining_indices
                    split_data['remain'] = data.ConcatDataset([result.remaining_data, split_data['remain']])

            if task == 'selective':
                shared_split = apply_shared_forget_indices(train_set)
                if shared_split is not None:
                    split_data.update(shared_split)
                    return split_data
                num_data = int(forget_size)
                result = RandomSplitter(train_set, num_samples=num_data).split()
                split_data['forget'] = result.selected_data
                split_data['remain'] = result.remaining_data
                split_data['forget_index'] = result.selected_indices
                split_data['remain_index'] = result.remaining_indices

            return split_data

        #######################################

        if dataset == 'svhn':
            root, download = resolve_torchvision_root('svhn')

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ])

            if task == 'class-wise':
                train_set = torchvision.datasets.SVHN(
                    root=root, split='train', download=download, transform=transform_train)

                split_data['forget'], split_data['remain'] = LabelSplitter(train_set, label=f_label).split()

            if task == 'selective':

                train_set = torchvision.datasets.SVHN(
                    root=root, split='train', download=download, transform=transform_train)
                print("train size is")
                print(len(train_set))
                shared_split = apply_shared_forget_indices(train_set)
                if shared_split is not None:
                    split_data.update(shared_split)
                    return split_data
                num_data = int(forget_size)
                result = RandomSplitter(train_set, num_samples=num_data).split()
                split_data['forget'] = result.selected_data
                split_data['remain'] = result.remaining_data
                split_data['forget_index'] = result.selected_indices
                split_data['remain_index'] = result.remaining_indices

            return split_data


        if dataset == 'TinyImageNet':
            dataset_dir = resolve_tinyimagenet_root()
            train_transform = transforms.Compose([
                transforms.Resize(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD),
            ])

            if task == 'class-wise':
                train_set = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'train'),
                                                             transform=train_transform)
                split_data['forget'], split_data['remain'] = LabelSplitter(train_set, label=f_label).split()

            if task == 'selective':
                print("dataset dir is", dataset_dir)
                train_set = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'train'),
                                                             transform=train_transform)
                shared_split = apply_shared_forget_indices(train_set)
                if shared_split is not None:
                    split_data.update(shared_split)
                    return split_data
                num_data = int(forget_size)
                result = RandomSplitter(train_set, num_samples=num_data).split()
                split_data['forget'] = result.selected_data
                split_data['remain'] = result.remaining_data
                split_data['forget_index'] = result.selected_indices
                split_data['remain_index'] = result.remaining_indices

            return split_data





    @staticmethod
    def load_selective_data(dataset, indices_file_path, forget_size):
        # Load the dataset using the dataset loader function
        split_data = {}
        train_set, test_set = mul_loader.load_data(dataset)
        if indices_file_path is not None:
            indices_data = torch.load(indices_file_path)
            vulnerable_indices = indices_data['vulnerable']
            print(vulnerable_indices)
            remaining_indices = indices_data['remaining']
            vulnerable_train_set = Subset(train_set, vulnerable_indices)
            remaining_train_set = Subset(train_set, remaining_indices)
            result = RandomSplitter(vulnerable_train_set, num_samples=forget_size).split()
            split_data['forget'] = result.selected_data
            split_data['remain'] = torch.utils.data.ConcatDataset([result.remaining_data, remaining_train_set])
            split_data['forget_index'] = result.selected_indices
            split_data['remain_index'] = result.remaining_indices
            print("forget size is", len(split_data['forget']))
            print("remain size is", len(split_data['remain']))
            print("forget index is", split_data['forget_index'])
            # print labels of the forget data
            subset_loader = torch.utils.data.DataLoader(split_data['forget'],
                                                        batch_size=len(split_data['forget']), shuffle=False)
            for data, labels in subset_loader:
                print("forget labels are", labels)
        return split_data

    @staticmethod
    def load_mixed_data(dataset, vulnerable_file_path, privacy_file_path):
        # Load the dataset using the dataset loader function
        split_data = {}
        train_set, test_set = mul_loader.load_data(dataset)
        all_indices = list(range(len(train_set)))

        if vulnerable_file_path is not None:
            vulnerable_indices = torch.load(vulnerable_file_path)
            privacy_indices = torch.load(privacy_file_path)
            privacy_indices = privacy_indices['vulnerable']
            try:
                vulnerable_indices = vulnerable_indices['vulnerable']
            except:
                vulnerable_indices = vulnerable_indices

            num_vulnerable = min(len(vulnerable_indices), 600)
            num_privacy = min(len(privacy_indices), 600)
            sampled_vulnerable = np.random.choice(vulnerable_indices, num_vulnerable, replace=False).tolist()
            sampled_privacy = np.random.choice(privacy_indices, num_privacy, replace=False).tolist()
            forget_indices = sampled_vulnerable + sampled_privacy
            remained_indices = list(set(all_indices) - set(forget_indices))
            forget_train_set = Subset(train_set, forget_indices)
            remaining_train_set = Subset(train_set, remained_indices)
            split_data['forget'] = forget_train_set
            split_data['vulnerable'] = Subset(train_set, sampled_vulnerable)
            split_data['privacy'] = Subset(train_set, sampled_privacy)
            split_data['remain'] = remaining_train_set
            split_data['privacy_index'] = sampled_privacy
            split_data['vulnerable_index'] = sampled_vulnerable
            split_data['forget_index'] = forget_indices
            split_data['remain_index'] = remained_indices


        return split_data

    @staticmethod
    def load_mixed_vulnerable_data(dataset, vulnerable_file_path, privacy_file_path):
        split_data = {}
        train_set, test_set = mul_loader.load_data(dataset)
        all_indices = list(range(len(train_set)))
        vulnerable_indices = torch.load(vulnerable_file_path)
        privacy_indices = torch.load(privacy_file_path)
        privacy_indices = privacy_indices['vulnerable']
        try:
            vulnerable_indices = vulnerable_indices['vulnerable']
        except:
            vulnerable_indices = vulnerable_indices

        num_vulnerable = min(len(vulnerable_indices), 600)
        sampled_vulnerable = np.random.choice(vulnerable_indices, num_vulnerable, replace=False).tolist()
        #print("vulnerable indices", len(sampled_vulnerable))
        forget_indices = sampled_vulnerable
        random_indices = list(set(all_indices) - set(forget_indices))
        sample_random = np.random.choice(random_indices, 600, replace=False).tolist()
        #print("random indices", sample_random)
        random_train_set = Subset(train_set, random_indices)
        forget_train_set_vulnerable = Subset(train_set, forget_indices)
        forget_train_set_random = Subset(train_set, sample_random)
        split_data['forget'] = data.ConcatDataset([forget_train_set_vulnerable, forget_train_set_random])
        remaining_indices = list(set(all_indices) - set(forget_indices))
        remain_train_set = Subset(train_set, remaining_indices)
        #split_data['forget'] = forget_train_set
        split_data['random'] = random_train_set
        split_data['remain'] = Subset(train_set, remaining_indices)
        split_data['forget_index'] = forget_indices
        split_data['random_index'] = sample_random
        split_data['remain_index'] = remaining_indices
        return split_data



    @staticmethod
    def load_test_data(dataset, task='selective', f_label=0):

        split_data_test = {}
        if dataset == 'cifar10':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            if task == 'class-wise':
                test_set = torchvision.datasets.CIFAR10(
                    root='./data/cifar10', train=False, download=True, transform=transform_test)
                split_data_test['forget'], split_data_test['remain'] = LabelSplitter(test_set, label=f_label).split()
                return split_data_test['remain']

            if task == 'selective' or task == 'vulnerable':
                test_set = torchvision.datasets.CIFAR10(
                    root='./data/cifar10', train=False, download=True, transform=transform_test)
                return test_set

        if dataset == 'cifar100':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            if task == 'class-wise':
                test_set = torchvision.datasets.CIFAR100(
                    root='./data/cifar100', train=False, download=True, transform=transform_test)
                split_data_test['forget'], split_data_test['remain'] = LabelSplitter(test_set, label=f_label).split()
                return split_data_test['remain']

            if task == 'selective' or task == 'vulnerable':
                test_set = torchvision.datasets.CIFAR100(
                    root='./data/cifar100', train=False, download=True, transform=transform_test)
                return test_set

        if dataset == 'cinic10':
            root = resolve_cinic10_root()
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CINIC10_MEAN, CINIC10_STD),
            ])
            test_set = Cinic10Dataset(root, 'test', transform_test)
            if task == 'class-wise':
                split_data_test['forget'], split_data_test['remain'] = LabelSplitter(test_set, label=f_label).split()
                return split_data_test['remain']

            if task == 'selective' or task == 'vulnerable':
                return test_set

        if dataset == 'svhn':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ])
            if task == 'class-wise':
                test_set = torchvision.datasets.SVHN(
                    root='./data/svhn', split='test', download=True, transform=transform_test)
                split_data_test['forget'], split_data_test['remain'] = LabelSplitter(test_set, label=f_label).split()
                return split_data_test['remain']

            if task == 'selective' or task == 'vulnerable':
                test_set = torchvision.datasets.SVHN(
                    root='./data/svhn', split='test', download=True, transform=transform_test)
                return test_set

        if dataset == 'TinyImageNet':
            dataset_dir = resolve_tinyimagenet_root()
            val_transform = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD),
            ])

            if task == 'class-wise':
                val_dataset = TinyImageNetValDataset(dataset_dir, transform=val_transform)
                split_data_test['forget'], split_data_test['remain'] = LabelSplitter(val_dataset, label=f_label).split()
                return split_data_test['remain']

            if task == 'selective' or task == 'vulnerable':
                val_dataset = TinyImageNetValDataset(dataset_dir, transform=val_transform)
                return val_dataset

    @staticmethod
    def poisson_subsample(dataset, lam):
        """ Subsample the dataset based on Poisson distribution. """
        num_samples = len(dataset)
        keep_indices = np.where(np.random.poisson(lam, num_samples) > 0)[0]
        return Subset(dataset, keep_indices)
    @staticmethod
    def uniform_subsample(dataset, size):
        """ Uniformly subsample the dataset. """
        num_samples = len(dataset)
        keep_indices = torch.randperm(num_samples)[:size]
        return Subset(dataset, keep_indices)


class RandomSplitter:
    def __init__(self,
                  data: Dataset, num_samples: int = 0):

        self.data = data
        self.num_samples = num_samples
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                    else "cpu")

    def split(self) -> [Dataset, Dataset]:
        SplitResult = namedtuple('SplitResult',
                                 ['selected_data', 'remaining_data', 'selected_indices',
                                  'remaining_indices'])
        if self.num_samples > 0:
            random_indices = torch.randperm(len(self.data))[:self.num_samples]
            remaining_indices = torch.tensor([i for i in range(len(self.data))
                                             if i not in random_indices])
            selected_data = torch.utils.data.Subset(self.data, random_indices)
            remaining_data = torch.utils.data.Subset(self.data,
                                                     remaining_indices)
        else:
            selected_data = None
            remaining_data = self.data
            random_indices = []
            remaining_indices = list(range(len(self.data)))

        return SplitResult(selected_data, remaining_data, random_indices, remaining_indices)


class LabelSplitter:

    def __init__(self, data: Dataset, label: int):
        self.data = data
        self.label = label
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def split(self) -> Dataset:
        if self.label is None:
            return self.data, None
        else:
            mask = torch.tensor([target == self.label for target in self.data.targets])
            selected_indices = torch.where(mask)[0]
            remaining_indices = torch.where(~mask)[0]
            selected_data = torch.utils.data.Subset(
                self.data, selected_indices)
            remaining_data = torch.utils.data.Subset(
                self.data, remaining_indices)

        return selected_data, remaining_data


class VulnerableSplitter:

    def __init__(self, data: Dataset, chunk_size: int = 1000, start_index: int = 0, device=None):
        self.data = data
        self.chunk_size = chunk_size
        self.current_index = start_index  # Counter to track chunk processing
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.all_indices = list(range(len(data)))
        random.shuffle(self.all_indices)  # Shuffle the indices for randomness

    def split(self):
        # Define the SplitResult namedtuple similar to other splitters
        SplitResult = namedtuple('SplitResult',
                                 ['selected_data', 'remaining_data', 'selected_indices', 'remaining_indices'])

        # Check if there are enough samples left for a split
        if self.current_index >= len(self.all_indices):
            return SplitResult(None, self.data, [], list(range(len(self.data))))  # No more data to split

        # Get the next chunk of indices
        end_index = min(self.current_index + self.chunk_size, len(self.all_indices))
        selected_indices = self.all_indices[self.current_index:end_index]

        # Update current index for the next iteration
        self.current_index = end_index

        # Create subsets for the selected data and the remaining data
        selected_data = torch.utils.data.Subset(self.data, selected_indices)
        remaining_indices = self.all_indices[end_index:]
        remaining_data = torch.utils.data.Subset(self.data, remaining_indices)

        # Return the selected chunk and the remaining data in the same format as the other splitters
        return SplitResult(selected_data, remaining_data, selected_indices, remaining_indices)

    def reset(self):
        # Reshuffle the data and reset the counter for another pass
        random.shuffle(self.all_indices)
        self.current_index = 0


class ChunkSplitter:
    def __init__(self, data: Dataset, chunk_size: int = 1000, device=None):
        self.data = data
        self.chunk_size = chunk_size
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.all_indices = list(range(len(data)))  # All available indices
        random.shuffle(self.all_indices)  # Shuffle for randomness
        self.current_index = 0  # Keep track of the current global index

    def split(self):
        # Check if there are enough samples left for a split
        if self.current_index >= len(self.all_indices):
            return {
                'forget': None,
                'remain': None,
                'forget_indices': [],
                'remaining_indices': []
            }

        end_index = min(self.current_index + self.chunk_size, len(self.all_indices))
        forget_indices = self.all_indices[self.current_index:end_index]
        self.current_index = end_index
        remain_indices = [idx for idx in range(len(self.data)) if idx not in forget_indices]
        forget_data = torch.utils.data.Subset(self.data, forget_indices)
        remain_data = torch.utils.data.Subset(self.data, remain_indices)
        result = {
            'forget': forget_data,
            'remain': remain_data,
            'forget_indices': forget_indices,
            'remaining_indices': remain_indices
        }
        return result
