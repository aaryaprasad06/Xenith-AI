"""Embedding model files: where they live, where they come from, and a checked download.

The files are too large for git (`models/*.onnx` other than YuNet is ignored),
so each one is pinned by sha256 and fetched on demand:

    uv run python -m face.models            # every model
    uv run python -m face.models sface      # one model

Sources and licenses are in docs/CITATIONS.md.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass(frozen=True)
class ModelFile:
    name: str
    filename: str
    sha256: str
    url: str
    member: str | None = None  # file inside the archive at `url`, if it is a zip


MODELS = {
    "sface": ModelFile(
        name="sface",
        filename="face_recognition_sface_2021dec.onnx",
        sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        url="https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
    ),
    "arcface_r50": ModelFile(
        name="arcface_r50",
        filename="arcface_w600k_r50.onnx",
        sha256="4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        member="w600k_r50.onnx",
    ),
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_file(name: str, models_dir: Path = MODELS_DIR) -> Path:
    """Path to a downloaded model, or FileNotFoundError saying how to fetch it."""
    spec = MODELS[name]
    path = models_dir / spec.filename
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Run: uv run python -m face.models {name}")
    return path


def fetch(name: str, models_dir: Path = MODELS_DIR) -> Path:
    """Download one model if it is missing, and refuse a file whose sha256 does not match."""
    spec = MODELS[name]
    path = models_dir / spec.filename
    if path.is_file() and sha256_of(path) == spec.sha256:
        return path
    models_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        download = Path(tmp) / "download"
        print(f"downloading {spec.url}")
        with urllib.request.urlopen(spec.url) as response, download.open("wb") as fh:
            shutil.copyfileobj(response, fh)
        candidate = download
        if spec.member is not None:
            with zipfile.ZipFile(download) as archive:
                candidate = Path(archive.extract(spec.member, tmp))
        actual = sha256_of(candidate)
        if actual != spec.sha256:
            raise ValueError(f"{name}: sha256 {actual} does not match the pinned {spec.sha256}")
        shutil.move(candidate, path)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help=f"models to fetch: {', '.join(MODELS)} (default: all)")
    args = parser.parse_args(argv)
    unknown = [name for name in args.names if name not in MODELS]
    if unknown:
        parser.error(f"unknown model(s): {', '.join(unknown)}")
    for name in args.names or MODELS:
        print(f"{name}: {fetch(name)}")


if __name__ == "__main__":
    main()
