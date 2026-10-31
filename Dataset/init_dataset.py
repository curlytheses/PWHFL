import sys
import random
import numpy as np
import torch
import torchvision
from pathlib import Path
from torch.utils.data import Dataset

# Locate the checkout root for direct-script execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from RAHFL.Dataset.helpers import TwoCropTransform, GaussianBlur, init_logs, mkdirs
from RAHFL.Dataset.cifar import CIFAR10, CIFAR100
from RAHFL.Dataset import dataaug as _dataaug_seed_compat  # noqa: F401 -- preserve legacy import-time RNG seeding


# ──────────────────────────────────────────────────────────────────────────────
# Set reproducible seed
Seed = 0
random.seed(Seed)
np.random.seed(Seed)
torch.manual_seed(Seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Seed)
# ──────────────────────────────────────────────────────────────────────────────


class CIFAR10_C_origin(Dataset):
    def __init__(self, path: str, c_type=None, transform=None, corrupt_rate=0):
        # Legacy argument retained for external callers; this reader selects all severities.
        if corrupt_rate != 0:
            import warnings
            warnings.warn("CIFAR10_C_origin ignores corrupt_rate; use CIFAR_C for rate selection.", DeprecationWarning, stacklevel=2)
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
            filename = data_path / f"random_corrupt_{self.corrupt_rate}.npy"
            if not filename.exists():
                # Accept canonical CLI filenames (1.npy) as well as legacy 1.0.npy.
                filename = data_path / f"random_corrupt_{float(self.corrupt_rate):g}.npy"
            data = np.load(filename)
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
    """Compatibility entry point; implementation lives in Dataset.utils."""
    from RAHFL.Dataset.utils import partition_data as implementation
    return implementation(dataset, datadir, partition, num_classes, num_users, corrupt_rate, test_corrupt_rate, dirichlet_beta, noise_type, noise_rate)


def generate_public_data_indexs(dataset: str, datadir: str, size: int):
    """Compatibility entry point; implementation lives in Dataset.utils."""
    from RAHFL.Dataset.utils import generate_public_data_indexs as implementation
    return implementation(dataset, datadir, size)


def load_cifar10c_data(datadir: Path, corrupt_rate: float, test_corrupt_rate: float):
    """Compatibility entry point; implementation lives in Dataset.utils."""
    from RAHFL.Dataset.utils import load_cifar10c_data as implementation
    return implementation(datadir, corrupt_rate, test_corrupt_rate)


def load_cifar10_data(datadir: Path, noise_type=None, noise_rate=0):
    """Compatibility entry point; implementation lives in Dataset.utils."""
    from RAHFL.Dataset.utils import load_cifar10_data as implementation
    return implementation(datadir, noise_type, noise_rate)


def load_cifar100_data(datadir: Path, noise_type=None, noise_rate=0):
    """Compatibility entry point; implementation lives in Dataset.utils."""
    from RAHFL.Dataset.utils import load_cifar100_data as implementation
    return implementation(datadir, noise_type, noise_rate)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
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

# Public compatibility APIs: optional dataset readers remain available.
__all__ = ["TwoCropTransform", "GaussianBlur", "init_logs", "mkdirs",
           "CIFAR10_C_origin", "CIFAR_C", "CIFAR100_RandomC", "Cifar10FL", "Cifar100FL",
           "partition_data", "generate_public_data_indexs", "load_cifar10c_data",
           "load_cifar10_data", "load_cifar100_data"]
