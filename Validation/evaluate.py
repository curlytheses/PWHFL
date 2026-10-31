"""Evaluate frozen saved models on extra datasets, never on a training path."""
import argparse
import csv
import json
import os
import platform
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import PIL
import torch
import torch.nn.functional as F

from RAHFL.Network.registry import DEFAULT_MODELS, MODEL_FACTORIES, create_model
from .data import CLASS_NAMES, MEAN, STD, load_dataset, make_loader, read_config
from .download import file_sha256


def load_checkpoint(model, path):
    """Load bare or DataParallel state dictionaries, retaining strict checks."""
    path = Path(path)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("Validation requires a PyTorch version supporting weights_only=True; tested with 2.5.1") from exc
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, Mapping) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Expected a nonempty model state dictionary in {path}")
    keys = list(state)
    if not all(isinstance(key, str) for key in keys):
        raise ValueError("Checkpoint keys must be strings")
    if any(key.startswith("module.") for key in keys):
        if not all(key.startswith("module.") for key in keys):
            raise ValueError("Checkpoint mixes DataParallel and unwrapped keys")
        state = {key[len("module."):]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return file_sha256(path)


def evaluate_model(model, loader, device):
    """Read-only inference: restore each module's mode and never touch gradients."""
    if getattr(loader.dataset, "role", None) != "validation-only":
        raise ValueError("Expected a validation-only dataset")
    modes = {module: module.training for module in model.modules()}
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    nll_sum, count = 0.0, 0
    try:
        model.eval()
        with torch.inference_mode():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                # The shipped models can avoid computing an unused embedding in eval.
                if hasattr(model, "forward_logits"):
                    logits = model.forward_logits(images)
                else:
                    output = model(images)
                    logits = output[0] if isinstance(output, (tuple, list)) else output
                if logits.shape != (len(labels), 10) or not torch.isfinite(logits).all():
                    raise ValueError("Expected finite [batch, 10] CIFAR-10 logits")
                nll_sum += F.cross_entropy(logits, labels, reduction="sum").item()
                predictions = logits.argmax(dim=1)
                bins = torch.bincount((labels * 10 + predictions).cpu(), minlength=100)
                confusion += bins.reshape(10, 10)
                count += len(labels)
    finally:
        for module, training in modes.items():
            module.training = training
    if count == 0 or count != len(loader.dataset):
        raise ValueError("Validation must visit every sample exactly once, including the final partial batch")
    support = confusion.sum(dim=1)
    per_class = [100.0 * confusion[i, i].item() / support[i].item() if support[i] else None for i in range(10)]
    return {
        "sample_count": count,
        "correct": int(confusion.diag().sum().item()),
        "accuracy_percent": 100.0 * confusion.diag().sum().item() / count,
        "cross_entropy": nll_sum / count,
        "per_class_accuracy_percent": dict(zip(CLASS_NAMES, per_class)),
        "confusion_matrix": confusion.tolist(),
    }


def write_results(report, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    with output.with_suffix(".csv").open("w", newline="") as handle:
        columns = ["model", "dataset", "sample_count", "correct", "accuracy_percent", "cross_entropy"]
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["results"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("validation.example.json"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint-dir", type=Path, help="Contains architecture_client-index.ckpt for each client")
    source.add_argument("--checkpoint", type=Path, help="Evaluate a single checkpoint")
    parser.add_argument("--architecture", choices=sorted(MODEL_FACTORIES), help="Required with --checkpoint")
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_FACTORIES), default=list(DEFAULT_MODELS))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu (default), cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=0, help="Model construction seed; checkpoint weights replace initialization")
    parser.add_argument("--download", action="store_true", help="Fetch only missing registered test assets")
    parser.add_argument("--output", type=Path, default=Path("validation_results/results.json"))
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0:
        parser.error("--batch-size must be positive and --workers nonnegative")
    if args.output.suffix.lower() != ".json":
        parser.error("--output must end in .json (a sibling .csv is also written)")
    if args.checkpoint:
        if not args.architecture:
            parser.error("--architecture is required with --checkpoint")
        checkpoints = [(args.architecture, args.architecture, args.checkpoint)]
    else:
        if args.architecture:
            parser.error("--architecture is only valid with --checkpoint")
        checkpoints = [(f"{name}_{i}", name, args.checkpoint_dir / f"{name}_{i}.ckpt")
                       for i, name in enumerate(args.models)]
    for _, _, path in checkpoints:
        if not path.is_file():
            parser.error(f"Checkpoint not found: {path}")
    entries = list(read_config(args.config))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; use --device cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    loaded = [(entry["name"], *load_dataset(entry, download=args.download)) for entry in entries]
    report = {
        "role": "validation-only", "checkpoint_selection": "user-supplied; no selection or training performed",
        "config": str(args.config.resolve()), "seed": args.seed, "device": str(device),
        "batch_size": args.batch_size, "workers": args.workers,
        "preprocessing": {"rgb_size": [32, 32], "mean": MEAN, "std": STD,
                          "augmentation": False, "shuffle": False, "drop_last": False},
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pillow": PIL.__version__},
        "inference_dtype": "float32",
        "datasets": {name: metadata for name, _, metadata in loaded},
        "checkpoints": {}, "results": [],
    }
    # Prevent accidental output replacement of any input checkpoint or dataset array.
    protected = {path.resolve() for _, _, path in checkpoints} | {args.config.resolve()}
    for _, _, metadata in loaded:
        for item in metadata.get("files", []):
            path = Path(item["path"])
            protected.add((Path(metadata["root"]) / path).resolve())
    if args.output.resolve() in protected or args.output.with_suffix(".csv").resolve() in protected:
        parser.error("Output must not replace an input file")
    for identifier, architecture, path in checkpoints:
        model = create_model(architecture, num_classes=10)
        digest = load_checkpoint(model, path)
        model.to(device)
        report["checkpoints"][identifier] = {"path": str(path.resolve()), "architecture": architecture, "sha256": digest}
        for name, dataset, _ in loaded:
            metrics = evaluate_model(model, make_loader(dataset, args.batch_size, args.workers), device)
            report["results"].append(dict(model=identifier, dataset=name, **metrics))
            print(f"{identifier} | {name} | n={metrics['sample_count']} | accuracy={metrics['accuracy_percent']:.4f}% | CE={metrics['cross_entropy']:.6f}")
        del model
    report["client_mean_accuracy_percent"] = {
        name: float(np.mean([row["accuracy_percent"] for row in report["results"] if row["dataset"] == name]))
        for name, _, _ in loaded
    }
    write_results(report, args.output)
    print(f"Saved {args.output} and {args.output.with_suffix('.csv')}")
    return report


if __name__ == "__main__":
    main()
