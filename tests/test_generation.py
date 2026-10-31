from pathlib import Path

import numpy as np
from PIL import Image
import pytest

pytest.importorskip("cv2")
pytest.importorskip("skimage")
pytest.importorskip("wand.image")
from RAHFL.Dataset import make_cifar_c as generation  # noqa: E402


def test_sixth_frost_texture_is_available_explicitly(monkeypatch):
    seen = []
    original = generation.cv2.imread
    def read(path):
        seen.append(Path(path).name)
        return original(path)
    monkeypatch.setattr(generation.cv2, "imread", read)
    monkeypatch.setattr(generation.np.random, "randint", lambda *args, **kwargs: 5 if args == (6,) else 0)
    result = generation.frost(Image.new("RGB", (32, 32)), texture_count=6)
    assert seen == ["frost6.jpg"]
    assert result.shape == (32, 32, 3)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("max_severity", [4, 5])
def test_generation_uses_requested_severity_and_portable_paths(tmp_path, monkeypatch, max_severity):
    class FakeCIFAR:
        def __init__(self, **kwargs):
            self.data = np.zeros((20, 32, 32, 3), np.uint8)
            self.targets = np.arange(20) % 10
    seen = []
    def corruption(image, severity):
        seen.append(severity)
        return np.asarray(image)
    monkeypatch.setattr(generation, "__file__", str(tmp_path / "make_cifar_c.py"))
    monkeypatch.setattr(generation.dset, "CIFAR100", FakeCIFAR)
    monkeypatch.setattr(generation, "d", {"test": corruption})
    generation.generate(apply_corruption=True, max_severity=max_severity)
    assert min(seen) == 1 and max(seen) == max_severity
    for split in ["train", "test"]:
        assert np.load(tmp_path / "cifar_100_c" / split / "random_corrupt_1.npy").shape == (20, 32, 32, 3)


def test_legacy_float_corruption_rate_finds_cli_filename(tmp_path):
    from RAHFL.Dataset.init_dataset import CIFAR_C
    (tmp_path / "train").mkdir()
    np.save(tmp_path / "train/random_corrupt_1.npy", np.zeros((3, 32, 32, 3), np.uint8))
    np.save(tmp_path / "train/labels.npy", np.arange(3))
    assert len(CIFAR_C(tmp_path, corrupt_rate=1.0)) == 3
