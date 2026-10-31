"""Deterministic validation datasets with the existing CIFAR-10 normalization.

There is deliberately no train=True option or dataset-partitioning function.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from .download import ASSETS, ensure_dataset, file_sha256

CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
MEAN = tuple(x / 255.0 for x in (125.3, 123.0, 113.9))
STD = tuple(x / 255.0 for x in (63.0, 62.1, 66.7))


def normalize(image):
    array = np.asarray(image)
    if array.shape != (32, 32, 3) or array.dtype != np.uint8:
        raise ValueError("Validation images must be uint8 RGB with shape (32, 32, 3)")
    tensor = torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1).float().div(255)
    return (tensor - tensor.new_tensor(MEAN)[:, None, None]) / tensor.new_tensor(STD)[:, None, None]


class ArrayValidationDataset(Dataset):
    role = "validation-only"

    def __init__(self, images, labels):
        labels = np.asarray(labels)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if images.ndim != 4 or images.shape[1:] != (32, 32, 3) or images.dtype != np.uint8:
            raise ValueError("Expected uint8 images with shape (N, 32, 32, 3)")
        if labels.ndim != 1 or len(labels) != len(images) or not len(labels):
            raise ValueError("Images and nonempty one-dimensional labels must have equal lengths")
        if not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or labels.max() >= len(CLASS_NAMES):
            raise ValueError("Labels must be integer CIFAR-10 indices in [0, 9]")
        self.images = images
        self.labels = labels.astype(np.int64, copy=False)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return normalize(self.images[index]), int(self.labels[index])


class FolderValidationDataset(Dataset):
    role = "validation-only"

    def __init__(self, root, resize=False):
        self.root = Path(root)
        self.resize = resize
        self.samples = []
        unknown = {p.name for p in self.root.iterdir() if p.is_dir()} - set(CLASS_NAMES)
        if unknown:
            raise ValueError(f"Unknown class folders {sorted(unknown)}; use CIFAR-10 class names")
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
        for label, name in enumerate(CLASS_NAMES):
            for path in sorted((self.root / name).rglob("*")):
                if path.is_file() and path.suffix.lower() in extensions:
                    self.samples.append((path, label))
        if not self.samples:
            raise ValueError(f"No validation images found in {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.resize:
                image = image.resize((32, 32), Image.Resampling.BILINEAR)
            return normalize(image), label


def make_loader(dataset, batch_size=512, workers=0):
    if getattr(dataset, "role", None) != "validation-only":
        raise ValueError("Only validation-only datasets may use this loader")
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and workers nonnegative")
    generator = torch.Generator().manual_seed(0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                      num_workers=workers, generator=generator)


def read_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if set(config) != {"datasets"} or not isinstance(config["datasets"], list) or not config["datasets"]:
        raise ValueError("Config must contain only a nonempty 'datasets' list")
    seen = set()
    for entry in config["datasets"]:
        allowed = {"name", "kind", "root", "split", "images", "labels", "resize", "corruption", "severity"}
        if not isinstance(entry, dict) or set(entry) - allowed:
            raise ValueError("Unknown validation dataset configuration fields")
        if not isinstance(entry.get("name"), str) or not entry["name"] or entry["name"] in seen:
            raise ValueError("Every dataset needs a unique nonempty name")
        seen.add(entry["name"])
        if entry.get("split") not in ("test", "validation"):
            raise ValueError("Validation-only config requires split='test' or split='validation'; training is forbidden")
        if entry.get("kind") not in (*ASSETS, "numpy", "folder", "cifar10-c"):
            raise ValueError(f"Unsupported validation kind: {entry.get('kind')}")
        specific = {"numpy": {"images", "labels"}, "folder": {"resize"},
                    "cifar10-c": {"corruption", "severity"}}.get(entry["kind"], set())
        if set(entry) - ({"name", "kind", "root", "split"} | specific):
            raise ValueError(f"Configuration fields do not apply to {entry['kind']}")
        if not isinstance(entry.get("root"), str) or not entry["root"]:
            raise ValueError("Every dataset requires a root path")
        if "resize" in entry and (entry["kind"] != "folder" or not isinstance(entry["resize"], bool)):
            raise ValueError("resize must be a boolean and is only valid for folder datasets")
        resolved = dict(entry)
        resolved["root"] = str((path.parent / entry["root"]).resolve())
        yield resolved


def load_dataset(entry, download=False):
    if entry.get("split") not in ("test", "validation"):
        raise ValueError("Training splits are forbidden in the validation-only loader")
    root = Path(entry["root"])
    kind = entry["kind"]
    metadata = dict(entry, role="validation-only", class_names=list(CLASS_NAMES))
    files = []
    if kind in ASSETS:
        files = ensure_dataset(kind, root, download=download)
        if kind == "cifar10.1":
            images = np.load(files[0], mmap_mode="r", allow_pickle=False)
            labels = np.load(files[1], allow_pickle=False)
        else:
            with np.load(files[0], allow_pickle=False) as archive:
                # Object metadata in the authors' archive is not needed; never unpickle it.
                images, labels = archive["images"], archive["labels"]
        dataset = ArrayValidationDataset(images, labels)
        if len(dataset) != 2000:
            raise ValueError(f"Unexpected sample count for the pinned {kind} test release")
        metadata["sources"] = [dict(url=url, git_blob_sha=sha) for url, _, _, sha in ASSETS[kind]]
        metadata["source_split"] = "test"
        metadata["release"] = "v6" if kind == "cifar10.1" else "pinned test archive"
    elif kind == "numpy":
        if not all(isinstance(entry.get(k), str) for k in ("images", "labels")):
            raise ValueError("numpy datasets require explicit images and labels filenames")
        files = [root / entry["images"], root / entry["labels"]]
        dataset = ArrayValidationDataset(np.load(files[0], mmap_mode="r", allow_pickle=False),
                                         np.load(files[1], allow_pickle=False))
    elif kind == "cifar10-c":
        # Official CIFAR-10-C stacks five 10,000-image severity blocks.
        severity = entry.get("severity")
        corruption = entry.get("corruption", "")
        if type(severity) is not int or severity not in range(1, 6):
            raise ValueError("cifar10-c requires integer severity in [1, 5]")
        if not corruption or Path(corruption).name != corruption:
            raise ValueError("cifar10-c requires a corruption filename stem")
        files = [root / (corruption + ".npy"), root / "labels.npy"]
        images = np.load(files[0], mmap_mode="r", allow_pickle=False)
        labels = np.load(files[1], allow_pickle=False)
        if len(images) != 50000 or len(labels) not in (10000, 50000):
            raise ValueError("Expected official CIFAR-10-C arrays, not generated private training data")
        start, end = (severity - 1) * 10000, severity * 10000
        dataset = ArrayValidationDataset(images[start:end], labels if len(labels) == 10000 else labels[start:end])
    elif kind == "folder":
        dataset = FolderValidationDataset(root, resize=entry.get("resize", False))
        # Record content hashes and labels, rather than trusting alphabetical class indices.
        metadata["files"] = [dict(path=str(p.relative_to(root)), label=y, sha256=file_sha256(p))
                             for p, y in dataset.samples]
    else:
        raise ValueError(f"Unsupported validation kind: {kind}")
    if files:
        metadata["files"] = [dict(path=str(p), sha256=file_sha256(p)) for p in files]
    metadata["sample_count"] = len(dataset)
    return dataset, metadata
