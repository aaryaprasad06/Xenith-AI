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
