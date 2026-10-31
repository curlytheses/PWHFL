# Validation of frozen checkpoints

Run commands from the repository root. This pipeline is separate from private training and public-data collaboration. It only reads checkpoints, evaluates explicitly declared held-out datasets, and writes reports. It does not use validation scores to select a checkpoint or modify the original training algorithm.

## Registered datasets

| Dataset | Selected release | Images | Upstream source |
| --- | --- | ---: | --- |
| CIFAR-10.1 | v6 | 2,000 | [Authors' repository](https://github.com/modestyachts/CIFAR-10.1) |
| CIFAR-10.2 | Test archive only | 2,000 | [Authors' repository](https://github.com/modestyachts/cifar-10.2) |

These datasets were selected because the existing models output the ten CIFAR-10 classes. No classification-head or label-space adaptation is performed. URLs are pinned to upstream commits in `Validation/download.py`; downloaded and existing assets must match their recorded byte size and Git blob hash. Reports additionally record SHA-256 hashes. No training archive is registered, and train splits are rejected by the config loader.

```bash
python -m Validation.download --root validation_data
python -m Validation.evaluate --config validation.example.json \
  --checkpoint-dir APEX_Storage_hetero_4client/cifar10_iid_0/random_corrupt_0.5 \
  --output validation_results/pwhfl.json
```

The default client order is `ResNet10 ResNet12 ShuffleNet Mobilenetv2`. Directory mode loads `<architecture>_<position>.ckpt` in the order given by `--models`. For a subset whose original indices differ, evaluate its files individually:

```bash
python -m Validation.evaluate --config validation.example.json \
  --checkpoint /path/to/ShuffleNet_2.ckpt --architecture ShuffleNet \
  --output validation_results/shufflenet.json
```

Bare state dictionaries, DataParallel `module.` prefixes, and `state_dict`/`model_state_dict` wrappers are accepted. Loading is strict: missing or incompatible parameters fail. Projection-head parameters are retained for old checkpoints even though validation can skip computing their unused embeddings.

## Preprocessing and metrics

- Defaults: CPU, batch size 512, zero workers, seed 0 for model construction; checkpoint weights replace initialization. Pass `--device cuda`, `--batch-size`, or `--workers` explicitly to change execution settings.
- Input images are RGB `uint8` of shape `(32, 32, 3)`. Normalization exactly matches the original CIFAR-10 test transform: mean `(125.3, 123.0, 113.9) / 255`, standard deviation `(63.0, 62.1, 66.7) / 255`.
- No augmentation, shuffle, or dropped final batch. Inference uses float32, `eval()` and `torch.inference_mode()`; no optimizer is constructed. Existing module modes are restored when calling the evaluator programmatically.
- Accuracy and cross-entropy are weighted by the number of samples. Confusion-matrix rows are true labels and columns are predictions. A class with no examples has a null per-class accuracy.
- The aggregate is the arithmetic mean of client accuracies for each dataset; it is not an ensemble score or normalized corruption error (mCE).
- The JSON report includes dataset/checkpoint hashes, class names, preprocessing, software versions, and all metrics. The sibling CSV contains one row per model and dataset.

The evaluator does not overwrite input checkpoints. Identical CPU inputs and checkpoints are covered by repeatability tests; numerical equivalence across different devices or library versions is not assumed.

## Other held-out datasets

Config paths are relative to the JSON file. Entries require unique names, a root, a supported kind, and `split` equal to `test` or `validation`. Unknown or inapplicable fields are rejected. Custom data must use the same CIFAR-10 semantics and class index order:

`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`.

For example, save this config next to the repo's example config and replace the paths:

```json
{
  "datasets": [
    {
      "name": "additional-heldout-arrays",
      "kind": "numpy",
      "root": "validation_data/custom",
      "split": "validation",
      "images": "images.npy",
      "labels": "labels.npy"
    },
    {
      "name": "additional-heldout-images",
      "kind": "folder",
      "root": "validation_data/custom-folders",
      "split": "validation",
      "resize": false
    },
    {
      "name": "CIFAR-10-C-fog-severity-3",
      "kind": "cifar10-c",
      "root": "validation_data/CIFAR-10-C",
      "split": "test",
      "corruption": "fog",
      "severity": 3
    }
  ]
}
```

`numpy` accepts an `(N,32,32,3)` uint8 image array and integer labels of shape `(N,)` or `(N,1)` in `[0,9]`; pickle loading is disabled. `folder` expects semantic class subdirectories such as `cat/` and `truck/`. Missing classes are allowed and retain their original indices. Folder images must already be 32×32 unless `resize: true` explicitly enables bilinear resizing. RGB conversion is always applied to folder images.

For `cifar10-c`, obtain the official test archive via the [original corruption benchmark repository](https://github.com/hendrycks/robustness). This adapter requires the official 50,000-image corruption stacks and selects one 10,000-image severity block, with severity in `[1,5]`. Add separate named entries for more corruptions or severities. It does not automatically download that archive or treat the repository's generated private-training data as the official benchmark.

CIFAR-100 and SVHN labels do not match these ten-class checkpoints and are not extra validation choices for this pipeline. For custom arrays/folders, the user must ensure the examples were held out and labels have the required meanings; file shape and split declarations cannot prove the absence of training overlap. Keep the registered validation datasets out of training, public collaboration, and tuning when reporting held-out generalization.

## Verification scope

Automated tests cover input validation, semantic label mapping, checkpoint compatibility, complete-batch accounting, and preservation of model parameters, buffers, gradients, RNG state, and module modes during evaluation. A separate end-to-end smoke run successfully evaluated all 4,000 registered examples using an untrained ResNet10 checkpoint. That verifies the download/loader/report path, not research accuracy. No trained checkpoints were supplied for a performance comparison, and a full training experiment was not run.
