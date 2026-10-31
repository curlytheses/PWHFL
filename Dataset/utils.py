import sys
import torch
import random
import numpy as np

from pathlib import Path
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

# Locate the checkout root for direct-script execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from RAHFL.Dataset.helpers import TwoCropTransform, GaussianBlur, init_logs as init_logs, mkdirs as mkdirs
from RAHFL.Dataset.init_dataset import Cifar10FL, Cifar100FL, CIFAR_C
from RAHFL.Network.registry import create_model
from RAHFL.Dataset.dataaug import AugMixDataset
from RAHFL.Dataset.sampling import iid_sampling, non_iid_dirichlet_sampling


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
Seed = 0
random.seed(Seed)
np.random.seed(Seed)
torch.manual_seed(Seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Seed)
# ──────────────────────────────────────────────────────────────────────────────


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
        _, y_train, _, _ = load_cifar10c_data(data_path, corrupt_rate, test_corrupt_rate)
    elif dataset == "cifar10_clean":
        _, y_train, _, _ = load_cifar10_data(data_path, noise_type, noise_rate)
    elif dataset == "cifar100":
        _, y_train, _, _ = load_cifar100_data(data_path, noise_type, noise_rate)
    else:
        raise ValueError(f"Unsupported partition dataset: {dataset}")

    if partition == "iid":
        net_dataidx_map = iid_sampling(y_train, num_users)
    elif partition == "noniid":
        net_dataidx_map = non_iid_dirichlet_sampling(
            y_train, num_classes, num_users, dirichlet_beta=dirichlet_beta
        )

    else:
        raise ValueError(f"Unsupported partition: {partition}")

    return y_train, net_dataidx_map


def generate_public_data_indexs(dataset: str, datadir: str, size: int):
    """
    Randomly select `size` indices from the training set of the given dataset.
    """
    data_path = Path(datadir)
    if dataset == "cifar100":
        _, y_train, _, _ = load_cifar100_data(data_path)
    elif dataset == "cifar10_clean":
        _, y_train, _, _ = load_cifar10_data(data_path)
    else:
        raise ValueError(f"Unsupported public dataset: {dataset}")
    n_train = y_train.shape[0]
    all_idxs = list(range(n_train))
    return np.random.choice(all_idxs, size, replace=False)


def get_dataloader(
    dataset: str,
    datadir: str,
    train_bs: int,
    test_bs: int,
    corrupt_rate,
    test_corrupt_rate,
    dataidxs=None,
    augmix_module=None,
    augmix_all_ops=False
):
    """
    Create PyTorch DataLoaders for CIFAR-10, CIFAR-100, or TinyImageNet.

    - `datadir` should be a string or Path pointing to the base data folder.
    - Always uses Linux-style forward slashes via pathlib.Path internally.
    """
    data_path = Path(datadir)

    if dataset == "cifar10_clean" and (corrupt_rate != 0 or test_corrupt_rate != 0):
        raise ValueError("cifar10_clean requires both corruption rates to be zero")

    def cifar10_dataset(root, **kwargs):
        if dataset == "cifar10_clean":
            kwargs.pop("corrupt_rate", None)
            return Cifar10FL(root, download=True, **kwargs)
        return CIFAR_C(root, **kwargs)

    # ─── CIFAR-10 Branch ───────────────────────────────────────────────────────
    if dataset in ("cifar10", "cifar10_clean"):
        if augmix_module is None:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4),
                    mode="reflect"
                ).data.squeeze()),
                transforms.ToPILImage(),
                transforms.ColorJitter(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
                )
            ])
            train_ds = cifar10_dataset(
                str(data_path),
                dataidxs=dataidxs,
                train=True,
                transform=transform_train,
                corrupt_rate=corrupt_rate
            )
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4),
                    mode="reflect"
                ).data.squeeze()),
                transforms.ToPILImage(),
                transforms.ColorJitter(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
            ])
            transform_train_weak = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4),
                    mode="reflect"
                ).data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
            ])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
                )
            ])
            train_ds = cifar10_dataset(
                str(data_path),
                dataidxs=dataidxs,
                train=True,
                transform=TwoCropTransform(transform_train, transform_train_weak),
                corrupt_rate=corrupt_rate
            )
            train_ds = AugMixDataset(train_ds, preprocess, jsd_or_nojsd=augmix_module, all_ops=augmix_all_ops)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            )
        ])
        test_ds = cifar10_dataset(
            str(data_path),
            train=False,
            transform=transform_test,
            corrupt_rate=test_corrupt_rate
        )

    # ─── CIFAR-100 Branch ───────────────────────────────────────────────────────
    elif dataset == "cifar100":
        if augmix_module is None:
            transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                    std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
                )
            ])
            train_ds = CIFAR_C(
                str(data_path),
                dataidxs=dataidxs,
                train=True,
                transform=transform_train,
                corrupt_rate=corrupt_rate
            )
        else:
            transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
            ])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                    std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
                )
            ])
            train_ds = CIFAR_C(
                str(data_path),
                dataidxs=dataidxs,
                train=True,
                transform=transform_train,
                corrupt_rate=corrupt_rate
            )
            train_ds = AugMixDataset(train_ds, preprocess, jsd_or_nojsd=augmix_module, all_ops=augmix_all_ops)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
            )
        ])
        test_ds = CIFAR_C(
            str(data_path),
            train=False,
            transform=transform_test,
            corrupt_rate=test_corrupt_rate
        )

    # ─── TinyImageNet Branch ────────────────────────────────────────────────────
    elif dataset == "tinyimagenet":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        train_folder = data_path / "train"
        train_ds = datasets.ImageFolder(str(train_folder), transform=transform_train)
        if dataidxs is not None:
            train_ds = torch.utils.data.Subset(train_ds, dataidxs)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        test_folder = data_path / "val"
        test_ds = datasets.ImageFolder(str(test_folder), transform=transform_test)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_dl = data.DataLoader(
        dataset=train_ds,
        batch_size=train_bs,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    test_dl = data.DataLoader(
        dataset=test_ds,
        batch_size=test_bs,
        drop_last=True,
        shuffle=False,
        num_workers=0
    )

    return train_dl, test_dl, train_ds, test_ds


def init_nets(n_parties: int, nets_name_list, num_classes=10):
    """
    Initialize a dictionary of networks for each party.
    """
    if n_parties != len(nets_name_list):
        raise ValueError("n_parties must match nets_name_list length")
    return {i: create_model(name, num_classes) for i, name in enumerate(nets_name_list)}
