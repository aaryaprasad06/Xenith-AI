"""Reject a frame before liveness when the face is not usable.

Blur is not checked here yet. Depth-island size is measured by the caller
and is not a hard fail: a real face against a white wall is an island of depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Face centre must sit inside this fraction of the frame, on both axes.
CENTRE_MARGIN = 0.15


@dataclass(frozen=True)
class QualityReport:
    ok: bool
    reason: str
    hole_frac: float
    distance_m: float | None


def _roi(
    depth: np.ndarray, x: float, y: float, width: float, height: float
) -> tuple[int, int, int, int] | None:
    image_h, image_w = depth.shape
    x0 = max(0, int(np.floor(x)))
    y0 = max(0, int(np.floor(y)))
    x1 = min(image_w, int(np.ceil(x + width)))
    y1 = min(image_h, int(np.ceil(y + height)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def assess_quality(
    depth: np.ndarray,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    quality: dict[str, Any],
) -> QualityReport:
    """One face box against the quality section of the config."""
    min_face_px = float(quality["min_face_px"])
    if min(width, height) < min_face_px:
        return QualityReport(
            ok=False,
            reason=f"face is too small ({min(width, height):.0f} px, need {min_face_px:.0f})",
            hole_frac=1.0,
            distance_m=None,
        )

    image_h, image_w = depth.shape
    cx = x + width / 2
    cy = y + height / 2
    if not (
        CENTRE_MARGIN * image_w <= cx <= (1 - CENTRE_MARGIN) * image_w
        and CENTRE_MARGIN * image_h <= cy <= (1 - CENTRE_MARGIN) * image_h
    ):
        return QualityReport(
            ok=False,
            reason="move to the centre of the camera",
            hole_frac=1.0,
            distance_m=None,
        )

    roi = _roi(depth, x, y, width, height)
    if roi is None:
        return QualityReport(
            ok=False,
            reason="face box is outside the image",
            hole_frac=1.0,
            distance_m=None,
        )
    x0, y0, x1, y1 = roi
    patch = depth[y0:y1, x0:x1]
    valid = patch[patch > 0]
    hole_frac = 1.0 - (valid.size / patch.size)
    distance = float(np.median(valid)) if valid.size else None

    max_hole = float(quality["max_hole_frac"])
    if hole_frac > max_hole:
        return QualityReport(
            ok=False,
            reason=f"too much of the face has no depth ({hole_frac:.0%})",
            hole_frac=hole_frac,
            distance_m=distance,
        )
    if distance is None:
        return QualityReport(
            ok=False,
            reason="no depth on the face",
            hole_frac=hole_frac,
            distance_m=None,
        )

    near = float(quality["min_dist_m"])
    far = float(quality["max_dist_m"])
    if distance < near:
        return QualityReport(
            ok=False,
            reason=f"too close ({distance:.2f} m); stand between {near:.2f} and {far:.2f} m",
            hole_frac=hole_frac,
            distance_m=distance,
        )
    if distance > far:
        return QualityReport(
            ok=False,
            reason=f"too far ({distance:.2f} m); stand between {near:.2f} and {far:.2f} m",
            hole_frac=hole_frac,
            distance_m=distance,
        )
    return QualityReport(ok=True, reason="ok", hole_frac=hole_frac, distance_m=distance)
