import os
import sys
import random
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
import torchvision
from pathlib import Path
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from PIL import ImageFilter

# Ensure we can import from two levels up
sys.path.append(str(Path(__file__).resolve().parents[2]))

from RAHFL.Dataset.cifar import CIFAR10, CIFAR100
from RAHFL.Dataset.dataaug import AugMixDataset
from RAHFL.Dataset.sampling import iid_sampling, non_iid_dirichlet_sampling
from RAHFL.Network.Models_Def.resnet import ResNet10, ResNet12
from RAHFL.Network.Models_Def.shufflenet import ShuffleNetG2
from RAHFL.Network.Models_Def.mobilnet_v2 import MobileNetV2


# ──────────────────────────────────────────────────────────────────────────────
# Set reproducible seed
Seed = 0
random.seed(Seed)
np.random.seed(Seed)
torch.manual_seed(Seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Seed)
# ──────────────────────────────────────────────────────────────────────────────


class TwoCropTransform:
    """Create two crops of the same image."""
    def __init__(self, transform, transform_weak):
        self.transform = transform
        self.transform_weak = transform_weak

    def __call__(self, x):
        return [self.transform(x), self.transform_weak(x)]


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


def init_logs(log_level=logging.INFO, log_path: str = None, sub_name=None):
    """
    Initialize a logger that writes to PROJECT_PATH/Logs/.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Determine base directory for logs
    if log_path is None:
        log_dir = Path.cwd() / "Logs"
    else:
        log_dir = Path(log_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    script_name = Path(sys.argv[0]).stem
    if sub_name is None:
        log_filename = f"{script_name}.log"
    else:
        log_filename = f"{script_name}_{sub_name}.log"

    log_file_path = log_dir / log_filename
    fh = logging.FileHandler(str(log_file_path), mode="w")
    fh.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    )
    fh.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fh)

    return logger


def mkdirs(dirpath: str):
    """
    Create directories if they do not exist.
    Accepts both string paths and Path-like.
    """
    Path(dirpath).mkdir(parents=True, exist_ok=True)


class CIFAR10_C_origin(Dataset):
    def __init__(self, path: str, c_type=None, transform=None, corrupt_rate=0):
        base_path = Path(path)
        images = np.load(base_path / f"{c_type}.npy")
        labels = np.load(base_path / "labels.npy")

        concat_images = np.empty(shape=(1, 32, 32, 3), dtype=np.uint8)
        concat_labels = np.empty(shape=(1,), dtype=labels.dtype)
        for i in range(1, 6):
            start = (i - 1) * 10000
            end = start + 500
            concat_images = np.concatenate(
                [concat_images, images[start:end, ...]],
                axis=0
            )
            concat_labels = np.concatenate(
                [concat_labels, labels[start:end]],
                axis=0
            )

        # Drop the initial dummy entry
        self.images = concat_images[1:, ...].astype(np.uint8)
        self.labels = concat_labels[1:, ...]
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        img = self.images[index, ...]
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label


class CIFAR_C(Dataset):
    def __init__(self, root: str, dataidxs=None, train=True, transform=None, corrupt_rate=0):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.corrupt_rate = corrupt_rate
        self.data, self.target = self.Construct_Participant_Dataset()

    def Construct_Participant_Dataset(self):
        root_path = Path(self.root)
        if self.train:
            data_path = root_path / "train"
        else:
            data_path = root_path / "test"

        if self.corrupt_rate == 0:
            data = np.load(data_path / "clean.npy")
        else:
            data = np.load(data_path / f"random_corrupt_{self.corrupt_rate}.npy")
        target = np.load(data_path / "labels.npy")

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = self.data[index, ...]
        label = self.target[index]
        if self.transform:
            img = self.transform(img)
        return img, label


class CIFAR100_RandomC(Dataset):
    def __init__(
        self,
        path: str,
        dataidxs=None,
        c_type=None,
        transform=None,
        corrupt_rate=0,
        index=0
    ):
        base_path = Path(path)
        if c_type == "random_noise":
            images = np.load(base_path / f"{c_type}_{index}.npy")
        else:
            images = np.load(base_path / f"{c_type}_{corrupt_rate}.npy")
        labels = np.load(base_path / "labels.npy")

        if dataidxs is not None:
            images = images[dataidxs]
            labels = labels[dataidxs]

        self.images = images.astype(np.uint8)
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        img = self.images[index, ...]
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label


class Cifar10FL(Dataset):
    def __init__(
        self,
        root: str,
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        noise_type=None,
        noise_rate=0
    ):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.data, self.target = self.Construct_Participant_Dataset()

    def Construct_Participant_Dataset(self):
        cifar_dataobj = CIFAR10(
            self.root,
            self.train,
            self.transform,
            self.target_transform,
            self.download,
            self.noise_type,
            self.noise_rate
        )

        if torchvision.__version__ == "0.2.1":
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_noisy_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            if self.train:
                data = cifar_dataobj.train_data
                target = np.array(cifar_dataobj.train_noisy_labels)
            else:
                data = cifar_dataobj.test_data
                target = np.array(cifar_dataobj.test_labels)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)


class Cifar100FL(Dataset):
    def __init__(
        self,
        root: str,
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        noise_type=None,
        noise_rate=0
    ):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.data, self.target = self.Construct_Participant_Dataset()

    def Construct_Participant_Dataset(self):
        cifar_dataobj = CIFAR100(
            self.root,
            self.train,
            self.transform,
            self.target_transform,
            self.download,
            self.noise_type,
            self.noise_rate
        )

        if torchvision.__version__ == "0.2.1":
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            if self.train:
                data = cifar_dataobj.train_data
                target = np.array(cifar_dataobj.train_noisy_labels)
            else:
                data = cifar_dataobj.test_data
                target = np.array(cifar_dataobj.test_labels)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)


def partition_data(
    dataset: str,
    datadir: str,
    partition: str,
    num_classes: int,
    num_users: int,
    corrupt_rate: float,
    test_corrupt_rate: float,
    dirichlet_beta: float = 1.0,
    noise_type=None,
    noise_rate=0
):
    """
    Partition the dataset indices into IID or non-IID splits.
    """
    data_path = Path(datadir)
    if dataset == "cifar10":
        X_train, y_train, X_test, y_test = load_cifar10c_data(data_path, corrupt_rate, test_corrupt_rate)
    elif dataset == "cifar100":
        X_train, y_train, X_test, y_test = load_cifar100_data(data_path, noise_type, noise_rate)
    else:
        raise ValueError(f"Unsupported dataset for partitioning: {dataset}")

    if partition == "iid":
        net_dataidx_map = iid_sampling(y_train, num_users)
    elif partition == "noniid":
        net_dataidx_map = non_iid_dirichlet_sampling(
            y_train, num_classes, num_users, dirichlet_beta=dirichlet_beta
        )
    else:
        raise ValueError(f"Unsupported partition type: {partition}")

    return y_train, net_dataidx_map


def generate_public_data_indexs(dataset: str, datadir: str, size: int):
    """
    Randomly select `size` indices from the training set of the given dataset.
    """
    data_path = Path(datadir)
    if dataset == "cifar100":
        X_train, y_train, X_test, y_test = load_cifar100_data(data_path)
    else:
        raise ValueError(f"Unsupported dataset for public indexing: {dataset}")

    n_train = y_train.shape[0]
    all_idxs = list(range(n_train))
    return np.random.choice(all_idxs, size, replace=False)


def load_cifar10c_data(datadir: Path, corrupt_rate: float, test_corrupt_rate: float):
    """
    Load CIFAR-10-C train and test using CIFAR_C class.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    cifar10_train_ds = CIFAR_C(str(datadir), train=True, transform=transform, corrupt_rate=corrupt_rate)
    cifar10_test_ds = CIFAR_C(str(datadir), train=False, transform=transform, corrupt_rate=test_corrupt_rate)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test   = cifar10_test_ds.data, cifar10_test_ds.target
    return X_train, y_train, X_test, y_test


def load_cifar10_data(datadir: Path, noise_type=None, noise_rate=0):
    """
    Load CIFAR-10 FL dataset (clean or noisy) using Cifar10FL class.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    cifar10_train_ds = Cifar10FL(
        str(datadir), train=True, download=True,
        transform=transform, noise_type=noise_type, noise_rate=noise_rate
    )
    cifar10_test_ds = Cifar10FL(
        str(datadir), train=False, download=True, transform=transform
    )

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test   = cifar10_test_ds.data, cifar10_test_ds.target
    return X_train, y_train, X_test, y_test


def load_cifar100_data(datadir: Path, noise_type=None, noise_rate=0):
    """
    Load CIFAR-100 FL dataset (clean or noisy) using Cifar100FL class.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    cifar100_train_ds = Cifar100FL(
        str(datadir), train=True, download=True,
        transform=transform, noise_type=noise_type, noise_rate=noise_rate
    )
    cifar100_test_ds = Cifar100FL(
        str(datadir), train=False, download=True, transform=transform
    )

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test   = cifar100_test_ds.data, cifar100_test_ds.target
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Example usage with Linux-style paths
    dataset_name = "cifar_10/CIFAR-10_styled_train"
    dataset_root = Path.cwd() / dataset_name

    # Use forward slashes via pathlib; no backslashes anywhere
    cifar10styled = CIFAR_C(root=str(dataset_root), train=True, transform=None, corrupt_rate=1)
    print(f"Number of samples in CIFAR-10 styled train: {len(cifar10styled)}")

    data = cifar10styled.data
    label = cifar10styled.target

    # Display a few samples (assuming data shape is (N, 32, 32, 3))
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

    for j in range(min(5, len(data))):
        imgdata = data[j, :, :, :]
        plt.cla()
        plt.imshow(np.transpose(imgdata, (0, 1, 2)))
        plt.pause(0.1)
        print(label_names[label[j]])

    # Example for CIFAR100FL
    cifar100_test = Cifar100FL(root=str(dataset_root), train=False)
    print(f"Number of samples in CIFAR-100 FL test: {len(cifar100_test)}")
