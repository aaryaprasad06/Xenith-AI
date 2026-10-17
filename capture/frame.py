from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

EmitterState = Literal["on", "off"]


@dataclass(frozen=True)
class Intrinsics:
    """Plain copy of the left-IR stream intrinsics (no live SDK objects)."""

    fx: float
    fy: float
    ppx: float
    ppy: float
    width: int
    height: int
    model: str
    coeffs: tuple[float, ...]


@dataclass(frozen=True)
class Frame:
    """One synchronised left-IR + depth sample, with RGB when it is enabled.

    ir:    uint8 (H, W)
    depth: float32 (H, W) metres, 0 means invalid
    rgb:   uint8 (H, W, 3) BGR, mapped into the IR viewport through depth.
           Black where depth was missing. None when the colour stream is off.
    rgb_timestamp: the colour frame's own time; the colour camera is a
           separate rolling-shutter sensor, so it is not exactly `timestamp`.
    """

    ir: np.ndarray
    depth: np.ndarray
    intrinsics: Intrinsics
    timestamp: float
    emitter_state: EmitterState
    rgb: np.ndarray | None = None
    rgb_timestamp: float | None = None


def center_roi_median_m(depth: np.ndarray, frac: float) -> float:
    """Median of non-zero depth in a centred box spanning `frac` of each side.

    Returns NaN when the box has no valid depth.
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    h, w = depth.shape[:2]
    bh = max(1, round(h * frac))
    bw = max(1, round(w * frac))
    y0 = (h - bh) // 2
    x0 = (w - bw) // 2
    roi = depth[y0 : y0 + bh, x0 : x0 + bw]
    valid = roi[roi > 0]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))


def center_roi_box(width: int, height: int, frac: float) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the same centred box used by `center_roi_median_m`."""
    bh = max(1, round(height * frac))
    bw = max(1, round(width * frac))
    y0 = (height - bh) // 2
    x0 = (width - bw) // 2
    return x0, y0, x0 + bw, y0 + bh
