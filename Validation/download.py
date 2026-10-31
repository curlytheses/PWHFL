"""Download ONLY held-out CIFAR-10.1 v6 / CIFAR-10.2 test assets.

URLs and Git blob hashes are pinned to the authors' repositories. No train
archive is registered. Existing files are verified rather than overwritten.
"""
import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

ASSETS = {
    "cifar10.1": (
        ("https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/d9982abb0bfc4846b8d13a11e66b887d946205d0/datasets/cifar10.1_v6_data.npy",
         "cifar10.1_v6_data.npy", 6144128, "24349cc5f3fb59648d75cd8015b5a184b1f5c250"),
        ("https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/d9982abb0bfc4846b8d13a11e66b887d946205d0/datasets/cifar10.1_v6_labels.npy",
         "cifar10.1_v6_labels.npy", 8128, "e713b09cecf2ecd933970001909fcdec7c907ab0"),
    ),
    "cifar10.2": (
        ("https://raw.githubusercontent.com/modestyachts/cifar-10.2/521467a52ad494b78ff9c70051099c0d9ba74d40/cifar102_test.npz",
         "cifar102_test.npz", 6393664, "7a3829681be56af70ea80ec99cf3822e2f5edff3"),
    ),
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset(path, size, blob_sha):
    path = Path(path)
    if path.stat().st_size != size:
        raise ValueError(f"Unexpected file size: {path}; expected {size} bytes")
    digest = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != blob_sha:
        raise ValueError(f"Dataset integrity check failed: {path}")


def ensure_dataset(name, root, download=False):
    if name not in ASSETS:
        raise ValueError(f"Unsupported downloadable dataset: {name}")
    root = Path(root)
    for url, filename, size, sha in ASSETS[name]:
        destination = root / filename
        if destination.exists():
            verify_asset(destination, size, sha)
            continue
        if not download:
            raise FileNotFoundError(f"Missing {destination}; run python -m Validation.download or pass --download")
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=root, suffix=".part", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "PWHFL-validation/1"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    received = 0
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        received += len(chunk)
                        if received > size:
                            raise ValueError(f"Download exceeds expected size: {filename}")
                        handle.write(chunk)
                handle.flush()
                verify_asset(temporary, size, sha)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        print(f"Downloaded and verified: {destination}")
    return [root / item[1] for item in ASSETS[name]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(ASSETS), default=sorted(ASSETS))
    parser.add_argument("--root", type=Path, default=Path("validation_data"))
    args = parser.parse_args(argv)
    for name in dict.fromkeys(args.datasets):
        ensure_dataset(name, args.root / name, download=True)


if __name__ == "__main__":
    main()
