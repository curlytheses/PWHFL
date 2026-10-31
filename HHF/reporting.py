"""Persist existing history lists without feeding them back into training."""
import json
import os
import tempfile
from pathlib import Path


def save_history(directory, accuracy, collaboration_loss, local_loss, final_accuracy):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "accuracy_percent_by_round": accuracy,
        "collaboration_loss_by_round_or_batch": collaboration_loss,
        "local_loss_by_round": local_loss,
        "final_accuracy_percent": final_accuracy,
    }
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, suffix=".tmp", delete=False) as handle:
        temp = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, directory / "training_metrics.json")
    finally:
        temp.unlink(missing_ok=True)
