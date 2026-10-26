"""Download and verify the trained YOLO models for this version of the application."""

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import urllib.error
from urllib.parse import urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, target, expected_hash):
    target = Path(target)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("Model manifest contains an invalid SHA-256 checksum.")
    if urlparse(url).scheme not in {"https", "file"}:
        raise ValueError("Model downloads require HTTPS.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) == expected_hash:
            return
        raise ValueError(f"{target} differs from this release. Move it aside before downloading.")
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".partial", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output, urlopen(url, timeout=60) as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                output.write(chunk)
        if sha256(temporary) != expected_hash:
            raise ValueError(f"SHA-256 verification failed for {target.name}.")
        if target.exists():
            raise ValueError(f"{target} appeared during the download. Existing model preserved.")
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
    for model in manifest["models"]:
        name = model["filename"]
        if name not in {"upper.pt", "lower.pt"}:
            raise ValueError(f"Unexpected model filename: {name}")
        target = ROOT / "models" / name
        download(manifest["release_base_url"] + "/" + name, target, model["sha256"])
        print(f"Verified {target.name}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, urllib.error.URLError) as exc:
        print(f"Model download failed: {exc}", file=sys.stderr)
        sys.exit(1)
