"""One interface for every face-embedding model, so candidates can be swapped.

An embedder takes the 112×112 crop from `face.align.align_crop` and returns an
L2-normalised vector plus a quality number. Higher quality means the model
trusts the vector more; models without a quality signal return 1.0.

`RawPixelEmbedder` is not a face model. It is the floor any real model must
beat, and it lets the evaluation harness run before a model is chosen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Embedding:
    vector: np.ndarray  # float32 (D,), unit length
    quality: float


class Embedder(Protocol):
    name: str

    def embed(self, ir_crop: np.ndarray, depth_crop: np.ndarray | None = None) -> Embedding:
        """`ir_crop` is uint8 (112, 112); `depth_crop` is float32 metres or None."""
        ...


def l2_normalise(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("cannot normalise a zero vector")
    return vector / norm


class RawPixelEmbedder:
    """Downscaled, mean-centred IR pixels. A baseline, not a recogniser."""

    name = "raw_pixels"

    def __init__(self, size: int = 32) -> None:
        self.size = size

    def embed(self, ir_crop: np.ndarray, depth_crop: np.ndarray | None = None) -> Embedding:
        small = cv2.resize(ir_crop, (self.size, self.size), interpolation=cv2.INTER_AREA)
        pixels = small.astype(np.float32).ravel()
        pixels -= pixels.mean()
        return Embedding(vector=l2_normalise(pixels), quality=1.0)


def _bgr(ir_crop: np.ndarray) -> np.ndarray:
    return ir_crop if ir_crop.ndim == 3 else cv2.cvtColor(ir_crop, cv2.COLOR_GRAY2BGR)


class SFaceEmbedder:
    """SFace (OpenCV Zoo), 128-D. Small and fast; trained on visible-light faces."""

    name = "sface"

    def __init__(self, path: str | None = None) -> None:
        from face.models import model_file

        self._model = cv2.FaceRecognizerSF.create(path or str(model_file("sface")), "")

    def embed(self, ir_crop: np.ndarray, depth_crop: np.ndarray | None = None) -> Embedding:
        # The crop from `align_crop` already uses the template SFace was trained on.
        return Embedding(vector=l2_normalise(self._model.feature(_bgr(ir_crop))), quality=1.0)


class ArcFaceEmbedder:
    """ArcFace ResNet-50 trained on WebFace600K (InsightFace buffalo_l), 512-D.

    Large ArcFace models are reported to carry over from colour to near-infrared
    faces with little loss, which small models do not. Non-commercial weights.
    """

    name = "arcface_r50"

    def __init__(self, path: str | None = None) -> None:
        from face.models import model_file

        self._net = cv2.dnn.readNetFromONNX(path or str(model_file("arcface_r50")))

    def embed(self, ir_crop: np.ndarray, depth_crop: np.ndarray | None = None) -> Embedding:
        blob = cv2.dnn.blobFromImage(
            _bgr(ir_crop), 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
        )
        self._net.setInput(blob)
        return Embedding(vector=l2_normalise(self._net.forward()), quality=1.0)


class AdaFaceEmbedder:
    """AdaFace IR-101 trained on WebFace12M (CVLFace release), 512-D.

    AdaFace trains with a margin that adapts to image quality, so blurred, dark
    or turned faces pull the model less. The length of the raw output before
    normalising is the model's own quality estimate and is returned as `quality`.
    """

    name = "adaface_ir101"

    def __init__(self, path: str | None = None) -> None:
        from face.models import model_file

        self._net = cv2.dnn.readNetFromONNX(path or str(model_file("adaface_ir101")))

    def embed(self, ir_crop: np.ndarray, depth_crop: np.ndarray | None = None) -> Embedding:
        # The CVLFace weights take RGB scaled to [-1, 1], aligned to the ArcFace template.
        blob = cv2.dnn.blobFromImage(
            _bgr(ir_crop), 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
        )
        self._net.setInput(blob)
        raw = self._net.forward().ravel()
        return Embedding(vector=l2_normalise(raw), quality=float(np.linalg.norm(raw)))
