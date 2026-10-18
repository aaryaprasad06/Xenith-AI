import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from eval.far_frr import EMBEDDERS
from face import models
from face.embed import ArcFaceEmbedder, RawPixelEmbedder, SFaceEmbedder
from face.models import MODELS, ModelFile, fetch, model_file, sha256_of


def _have(name: str) -> bool:
    try:
        model_file(name)
    except FileNotFoundError:
        return False
    return True


def _crop(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (112, 112), dtype=np.uint8)


def test_every_registered_embedder_has_its_name():
    assert set(EMBEDDERS) == {"raw_pixels", "sface", "arcface_r50"}
    assert RawPixelEmbedder.name == "raw_pixels"
    assert SFaceEmbedder.name == "sface"
    assert ArcFaceEmbedder.name == "arcface_r50"


@pytest.mark.parametrize(("cls", "dim"), [(SFaceEmbedder, 128), (ArcFaceEmbedder, 512)])
def test_model_embedders_give_stable_unit_vectors(cls, dim):
    if not _have(cls.name):
        pytest.skip(f"run: uv run python -m face.models {cls.name}")
    embedder = cls()
    a = embedder.embed(_crop(0))
    again = embedder.embed(_crop(0))
    other = embedder.embed(_crop(1))
    assert a.vector.shape == (dim,)
    assert a.vector.dtype == np.float32
    assert float(np.linalg.norm(a.vector)) == pytest.approx(1.0, abs=1e-5)
    assert float(a.vector @ again.vector) == pytest.approx(1.0, abs=1e-5)
    assert float(a.vector @ other.vector) < 0.999


def test_missing_model_says_how_to_fetch_it(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="python -m face.models sface"):
        model_file("sface", models_dir=tmp_path)


def _serve(monkeypatch, payload: bytes) -> None:
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda url: io.BytesIO(payload))


def test_fetch_checks_the_sha256(monkeypatch, tmp_path: Path):
    good = b"model bytes"
    spec = ModelFile("toy", "toy.onnx", sha256="0" * 64, url="https://example.invalid/toy.onnx")
    monkeypatch.setitem(MODELS, "toy", spec)
    _serve(monkeypatch, good)
    with pytest.raises(ValueError, match="does not match"):
        fetch("toy", models_dir=tmp_path)
    assert not (tmp_path / "toy.onnx").exists()

    import hashlib

    monkeypatch.setitem(MODELS, "toy", ModelFile("toy", "toy.onnx", hashlib.sha256(good).hexdigest(), spec.url))
    path = fetch("toy", models_dir=tmp_path)
    assert path.read_bytes() == good
    assert sha256_of(path) == hashlib.sha256(good).hexdigest()


def test_fetch_extracts_one_member_from_a_zip(monkeypatch, tmp_path: Path):
    import hashlib

    inner = b"weights"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("other.onnx", b"not this one")
        archive.writestr("w.onnx", inner)
    monkeypatch.setitem(
        MODELS,
        "zipped",
        ModelFile("zipped", "zipped.onnx", hashlib.sha256(inner).hexdigest(), "https://example.invalid/a.zip", "w.onnx"),
    )
    _serve(monkeypatch, buffer.getvalue())
    assert fetch("zipped", models_dir=tmp_path).read_bytes() == inner
