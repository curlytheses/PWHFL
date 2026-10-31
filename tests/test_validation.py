import json

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import transforms

from RAHFL.Network.registry import create_model
from Validation.data import (ArrayValidationDataset, FolderValidationDataset,
                             CLASS_NAMES, MEAN, STD, make_loader, normalize,
                             read_config, load_dataset)
from Validation.download import ASSETS, ensure_dataset, file_sha256
from Validation.evaluate import evaluate_model, load_checkpoint, main


def array_dataset(n=5):
    return ArrayValidationDataset(np.zeros((n, 32, 32, 3), dtype=np.uint8), np.arange(n) % 10)


def test_normalization_matches_training_test_transform():
    image = np.random.default_rng(12).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    original = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    torch.testing.assert_close(normalize(image), original(image), rtol=0, atol=0)


def test_evaluation_is_read_only_and_keeps_last_batch():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3072, 10),
                                torch.nn.BatchNorm1d(10), torch.nn.Dropout(.5))
    model.train()
    model[3].eval()  # Mixed initial modes must be restored individually.
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    gradients = [p.grad.clone() for p in model.parameters()]
    modes = [m.training for m in model.modules()]
    rng = torch.random.get_rng_state().clone()
    loader = make_loader(array_dataset(), batch_size=2)
    metrics = evaluate_model(model, loader, torch.device("cpu"))
    assert metrics["sample_count"] == 5
    assert sum(map(sum, metrics["confusion_matrix"])) == 5
    assert metrics == evaluate_model(model, loader, torch.device("cpu"))
    assert [m.training for m in model.modules()] == modes
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    for parameter, gradient in zip(model.parameters(), gradients):
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_accuracy_and_ce_are_sample_weighted():
    class ConstantModel(torch.nn.Module):
        def forward(self, x):
            logits = torch.zeros((len(x), 10))
            logits[:, 0] = 2
            return logits
    dataset = ArrayValidationDataset(np.zeros((5, 32, 32, 3), np.uint8), np.array([0, 0, 0, 1, 1]))
    result = evaluate_model(ConstantModel(), make_loader(dataset, 2), torch.device("cpu"))
    expected_ce = torch.nn.functional.cross_entropy(ConstantModel()(torch.zeros(5, 1)), torch.tensor([0, 0, 0, 1, 1])).item()
    assert result["correct"] == 3
    assert result["accuracy_percent"] == 60.0
    assert result["cross_entropy"] == pytest.approx(expected_ce)


@pytest.mark.parametrize("labels", [np.array([0.2, 1.0]), np.array([0, 10]), np.array([-1, 0]), np.array([0])])
def test_invalid_labels_rejected(labels):
    with pytest.raises(ValueError):
        ArrayValidationDataset(np.zeros((2, 32, 32, 3), np.uint8), labels)


def test_folder_labels_follow_semantics_not_subset_sorting(tmp_path):
    for name in ["cat", "truck"]:
        (tmp_path / name).mkdir()
        Image.new("RGB", (32, 32)).save(tmp_path / name / "sample.png")
    dataset = FolderValidationDataset(tmp_path)
    assert [dataset[i][1] for i in range(len(dataset))] == [CLASS_NAMES.index("cat"), CLASS_NAMES.index("truck")]


def test_train_split_and_unknown_config_rejected(tmp_path):
    config = tmp_path / "validation.json"
    entry = {"name": "bad", "kind": "cifar10.2", "root": "data", "split": "train"}
    config.write_text(json.dumps({"datasets": [entry]}))
    with pytest.raises(ValueError, match="training is forbidden"):
        list(read_config(config))
    with pytest.raises(ValueError, match="Training splits"):
        load_dataset(entry)
    config.write_text(json.dumps({"datasets": [], "train": True}))
    with pytest.raises(ValueError):
        list(read_config(config))


def test_download_registry_contains_only_test_files_and_checks_integrity(tmp_path):
    assert all("train" not in filename for entries in ASSETS.values() for _, filename, _, _ in entries)
    (tmp_path / "cifar102_test.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="file size"):
        ensure_dataset("cifar10.2", tmp_path)
    with pytest.raises(ValueError, match="Unsupported"):
        ensure_dataset("cifar10.2-train", tmp_path)


@pytest.mark.parametrize("wrapped", [False, True])
def test_strict_checkpoint_loading(tmp_path, wrapped):
    model = create_model("ResNet10")
    state = model.state_dict()
    path = tmp_path / "model.ckpt"
    torch.save({"state_dict": {"module." + key: value for key, value in state.items()}} if wrapped else state, path)
    other = create_model("ResNet10")
    assert load_checkpoint(other, path) == file_sha256(path)
    for key, value in other.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    partial = dict(state)
    del partial["linear.weight"]
    torch.save(partial, path)
    with pytest.raises(RuntimeError):
        load_checkpoint(other, path)


@pytest.mark.parametrize("directory_mode", [False, True])
def test_cli_evaluates_multiple_datasets_without_touching_checkpoint(tmp_path, directory_mode):
    images = np.zeros((5, 32, 32, 3), np.uint8)
    np.save(tmp_path / "images.npy", images)
    np.save(tmp_path / "labels.npy", np.arange(5))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"datasets": [
        dict(name=name, kind="numpy", root=".", split="validation", images="images.npy", labels="labels.npy")
        for name in ["heldout-a", "heldout-b"]]}))
    checkpoint = tmp_path / "ResNet10_0.ckpt"
    torch.save(create_model("ResNet10").state_dict(), checkpoint)
    before = file_sha256(checkpoint)
    output = tmp_path / "metrics.json"
    if directory_mode:
        second = tmp_path / "ResNet12_1.ckpt"
        torch.save(create_model("ResNet12").state_dict(), second)
        second_before = file_sha256(second)
        source_args = ["--checkpoint-dir", str(tmp_path), "--models", "ResNet10", "ResNet12"]
    else:
        source_args = ["--checkpoint", str(checkpoint), "--architecture", "ResNet10"]
    report = main(["--config", str(config), *source_args,
                   "--batch-size", "2", "--output", str(output)])
    assert len(report["results"]) == (4 if directory_mode else 2)
    assert all(row["sample_count"] == 5 for row in report["results"])
    assert file_sha256(checkpoint) == before
    assert output.is_file() and output.with_suffix(".csv").is_file()
    assert {row["dataset"] for row in report["results"]} == {"heldout-a", "heldout-b"}
    if directory_mode:
        assert {row["model"] for row in report["results"]} == {"ResNet10_0", "ResNet12_1"}
        assert file_sha256(second) == second_before


def test_evaluation_restores_modes_after_invalid_model_output():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3072, 100), torch.nn.Dropout())
    model.train()
    model[-1].eval()
    modes = [module.training for module in model.modules()]
    with pytest.raises(ValueError, match="CIFAR-10 logits"):
        evaluate_model(model, make_loader(array_dataset(), 2), torch.device("cpu"))
    assert [module.training for module in model.modules()] == modes


@pytest.mark.parametrize("kind,extra", [("cifar10.2", {"images": "train.npy"}),
                                       ("folder", {"resize": "false"})])
def test_config_does_not_silently_ignore_inapplicable_options(tmp_path, kind, extra):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"datasets": [dict(name="heldout", kind=kind, root=".",
                                                    split="validation", **extra)]}))
    with pytest.raises(ValueError):
        list(read_config(config))


def test_official_cifar_c_severity_slicing(tmp_path, monkeypatch):
    from Validation import data
    images = np.broadcast_to(np.zeros((1, 32, 32, 3), np.uint8), (50000, 32, 32, 3))
    labels = np.repeat(np.arange(5), 10000)
    for name in ["fog.npy", "labels.npy"]:
        (tmp_path / name).write_bytes(b"fixture")
    monkeypatch.setattr(data.np, "load", lambda path, **kwargs: labels if path.name == "labels.npy" else images)
    dataset, _ = load_dataset(dict(name="fog-3", kind="cifar10-c", root=str(tmp_path), split="test", corruption="fog", severity=3))
    assert len(dataset) == 10000
    assert np.all(dataset.labels == 2)
