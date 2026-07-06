from __future__ import print_function

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from RAHFL.Dataset import augmentations
import numpy as np

import torch
from torch.utils.data import Dataset
import random

Seed = 0
seed = Seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def aug(image, preprocess):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  mixture_width = 3
  mixture_depth = -1
  aug_severity = 3
  all_ops = False

  aug_list = augmentations.augmentations
  if all_ops:
    aug_list = augmentations.augmentations_all

  ws = np.float32(np.random.dirichlet([1] * mixture_width))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(preprocess(image))
  for i in range(mixture_width):
    image_aug = image.copy()
    depth = mixture_depth if mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, aug_severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed


class AugMixDataset(Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, jsd_or_nojsd='jsd'):
    self.dataset = dataset
    self.preprocess = preprocess
    self.jsd_or_nojsd = jsd_or_nojsd

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.jsd_or_nojsd == 'jsd' or 'onejsd':
      # Only apply aug() to the first image crop
      im_tuple = (self.preprocess(x[0]), aug(x[0], self.preprocess), aug(x[0], self.preprocess), self.preprocess(x[1]))
      return im_tuple, y
    else:
      return aug(x, self.preprocess), y

  def __len__(self):
    return len(self.dataset)
