import json
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import transforms

from RAHFL.Dataset import dataaug, init_dataset, utils
from RAHFL.Dataset.cifar import noisify, noisify_pairflip, noisify_multiclass_symmetric
from RAHFL.HHF import ablation, pwhfl
from RAHFL.HHF.reporting import save_history
from RAHFL.Network.registry import DEFAULT_MODELS, MODEL_FACTORIES, create_model


@pytest.mark.parametrize("mode,views", [("jsd", 4), ("onejsd", 4), ("nojsd", 1)])
@pytest.mark.parametrize("two_crops", [False, True])
def test_augmix_modes(mode, views, two_crops):
    image = Image.fromarray(np.full((32, 32, 3), 100, dtype=np.uint8))
    source = [([image, image] if two_crops else image, 3)]
    dataset = dataaug.AugMixDataset(source, transforms.ToTensor(), mode)
    result, label = dataset[0]
    assert label == 3
    if views == 1:
        assert result.shape == (3, 32, 32)
    else:
        assert len(result) == views
        assert all(x.shape == (3, 32, 32) for x in result)


def test_augmix_extended_operators_are_selectable(monkeypatch):
    calls = []
    def operation(image, severity):
        calls.append(severity)
        return image
    monkeypatch.setattr(dataaug.augmentations, "augmentations_all", [operation])
    image = Image.fromarray(np.zeros((32, 32, 3), np.uint8))
    dataaug.aug(image, transforms.ToTensor(), all_ops=True)
    assert calls


@pytest.mark.parametrize("kind,function", [("pairflip", noisify_pairflip), ("symmetric", noisify_multiclass_symmetric)])
def test_noise_seed_and_zero_noise(kind, function):
    labels = np.tile(np.arange(10), 100)[:, None]
    actual, rate = noisify(dataset="cifar10", train_labels=labels, noise_type=kind,
                          noise_rate=0.4, random_state=37)
    expected, expected_rate = function(labels, 0.4, random_state=37)
    np.testing.assert_array_equal(actual, expected)
    assert rate == expected_rate
    other, _ = noisify(train_labels=labels, noise_type=kind, noise_rate=0.4, random_state=38)
    assert not np.array_equal(actual, other)
    clean, rate = noisify(train_labels=labels, noise_type=kind, noise_rate=0)
    np.testing.assert_array_equal(clean, labels)
    assert rate == 0


def test_compatibility_exports_delegate(monkeypatch):
    assert init_dataset.TwoCropTransform is utils.TwoCropTransform
    assert init_dataset.GaussianBlur is utils.GaussianBlur
    for name, args in [
        ("load_cifar10_data", ("data", None, 0)),
        ("load_cifar100_data", ("data", None, 0)),
        ("load_cifar10c_data", ("data", .5, 0)),
        ("generate_public_data_indexs", ("cifar100", "data", 10)),
        ("partition_data", ("cifar10", "data", "iid", 10, 4, .5, 0, 1.0, None, 0)),
    ]:
        implementation = Mock(return_value=object())
        monkeypatch.setattr(utils, name, implementation)
        assert getattr(init_dataset, name)(*args) is implementation.return_value
        implementation.assert_called_once_with(*args)


@pytest.mark.parametrize("name", DEFAULT_MODELS)
def test_models_preserve_embedding_and_expose_identical_logits(name):
    model = create_model(name).eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        logits, embedding = model(images)
        fast = model.forward_logits(images)
    torch.testing.assert_close(logits, fast, rtol=0, atol=0)
    assert embedding.shape == (2, 1024)
    assert any(key.startswith("bn_projector.") for key in model.state_dict())
    assert {"ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "ShuffleNetG3"} <= set(MODEL_FACTORIES)


def test_component_runner_and_reference_rows(monkeypatch):
    configs = []
    def experiment(config, **kwargs):
        configs.append(config)
        return [1.0] * 4, [2.0] * 4
    monkeypatch.setattr(ablation, "run_experiment", experiment)
    assert len(ablation.run_component_ablation()) == 7
    assert configs[0]["do_collab"] is False
    assert configs[4]["use_jsd"] is True
    assert configs[-1]["use_arc"] is True
    for runner, default, reference in [(ablation.run_pwkt_ablation, 4, 5),
                                       (ablation.run_adajsd_ablation, 3, 4),
                                       (ablation.run_mce_ablation, 4, 5),
                                       (ablation.run_dcl_ablation, 2, 3)]:
        assert len(runner()) == default
        assert len(runner(include_reference=True)) == reference


def test_histories_are_saved(tmp_path):
    save_history(tmp_path, [[10.0]], [[0.2]], [[1.1]], [20.0])
    result = json.loads((tmp_path / "training_metrics.json").read_text())
    assert result["accuracy_percent_by_round"] == [[10.0]]
    assert result["local_loss_by_round"] == [[1.1]]
    assert result["final_accuracy_percent"] == [20.0]


def test_unsupported_loss_fails_instead_of_being_ignored():
    with pytest.raises(ValueError, match="loss_function"):
        pwhfl.update_model_via_private_data("cpu", None, 1, None, "invalid", "Adam", .001, None)


def test_training_loader_rejects_external_validation_dataset():
    with pytest.raises(ValueError, match="Unsupported dataset"):
        utils.get_dataloader("cifar10.1", "unused", 2, 2, 0, 0)


def test_pretraining_retains_effective_legacy_rng_binding():
    from RAHFL.Network import pretrain
    # The previous `from numpy import *` overwrote Python's random module.
    # Import cleanup must not change which RNG the existing seed call targets.
    assert pretrain.random is np.random
