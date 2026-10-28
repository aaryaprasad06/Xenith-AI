"""Export validation tests; run with the optional export dependency group."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from face import adaface_export


@pytest.mark.parametrize("backend", ["opencv", "pytorch"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_export_rejects_non_finite_output(monkeypatch, tmp_path, backend, value):
    def forward(_):
        return torch.full((1, 512), value if backend == "pytorch" else 0.0)

    check = SimpleNamespace(
        setInput=lambda _: None,
        forward=lambda: np.full((1, 512), value if backend == "opencv" else 0.0),
    )
    monkeypatch.setattr(adaface_export, "load_ir101", lambda _: forward)
    monkeypatch.setattr(torch.onnx, "export", lambda *args, **kwargs: None)
    monkeypatch.setattr(adaface_export.cv2.dnn, "readNetFromONNX", lambda _: check)

    with pytest.raises(ValueError, match="non-finite"):
        adaface_export.export(tmp_path / "weights", tmp_path / "model.onnx")


@pytest.mark.parametrize("result", [np.nan, np.inf, 0.01, "error", 1e-5])
def test_main_only_replaces_output_after_successful_validation(
    monkeypatch, tmp_path, result
):
    out = tmp_path / "model.onnx"
    out.write_bytes(b"existing model")
    temporary = out.with_suffix(".tmp.onnx")

    def export(weights, path):
        path.write_bytes(b"new model")
        if result == "error":
            raise RuntimeError("export failed")
        return result

    monkeypatch.setattr(adaface_export, "model_file", lambda _: Path("weights"))
    monkeypatch.setattr(adaface_export, "export", export)

    if result == "error":
        with pytest.raises(RuntimeError, match="export failed"):
            adaface_export.main(["--out", str(out)])
    elif not np.isfinite(result) or result > 1e-3:
        with pytest.raises(SystemExit, match="export not kept"):
            adaface_export.main(["--out", str(out)])
    else:
        adaface_export.main(["--out", str(out)])

    assert out.read_bytes() == (b"new model" if result == 1e-5 else b"existing model")
    assert not temporary.exists()
