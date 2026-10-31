"""Shared augmentation and filesystem helpers; no seeding at import time."""
import logging
import random
import sys
from pathlib import Path

from PIL import ImageFilter


class TwoCropTransform:
    def __init__(self, transform, transform_weak):
        self.transform = transform
        self.transform_weak = transform_weak

    def __call__(self, x):
        return [self.transform(x), self.transform_weak(x)]


class GaussianBlur:
    def __init__(self, sigma=(.1, 2.)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


def init_logs(log_level=logging.INFO, log_path=None, sub_name=None):
    logger = logging.getLogger("RAHFL")
    logger.setLevel(log_level)
    log_dir = Path(log_path) if log_path is not None else Path(__file__).resolve().parents[1] / "Logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    script_name = Path(sys.argv[0]).stem
    filename = f"{script_name}.log" if sub_name is None else f"{script_name}_{sub_name}.log"
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(str(log_dir / filename), mode="w")
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    ))
    logger.addHandler(handler)
    return logger


def mkdirs(dirpath):
    Path(dirpath).mkdir(parents=True, exist_ok=True)
