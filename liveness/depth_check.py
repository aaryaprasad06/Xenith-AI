"""Face-shape check on laser-on depth.

Deproject the face box, fit a plane, and measure how far the cloud sticks
out from that plane (5th to 95th percentile). A print or a screen is a plane.
The nose-to-cheek gap is a second measurement at the nose landmark.

A learned score is not consulted. Failing this check cannot be overridden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from capture.frame import Intrinsics
from capture.quality_gate import QualityReport, assess_quality

MIN_PLANE_POINTS = 80
# Points farther than this from the fitted plane are stereo spikes, not a face.
GROSS_OUTLIER_M = 0.05
RANSAC_INLIER_M = 0.01
RANSAC_ITERS = 40
CHEEK_OFFSET_FRAC = 0.32
PATCH_RADIUS = 2


@dataclass(frozen=True)
class GeometryReport:
    ok: bool
    reason: str
    relief_m: float | None
    nose_cheek_m: float | None
    hole_frac: float
    distance_m: float | None
    # True only when the face was measured and its shape is wrong (a likely
    # spoof), not when the quality gate or missing depth stopped the check.
    shape_failed: bool = False


def deproject(
    us: np.ndarray, vs: np.ndarray, zs: np.ndarray, intrinsics: Intrinsics
) -> np.ndarray:
    """Pinhole deprojection. Distortion is ignored; D435 coeffs are small at the face."""
    x = (us - intrinsics.ppx) / intrinsics.fx * zs
    y = (vs - intrinsics.ppy) / intrinsics.fy * zs
    return np.stack([x, y, zs], axis=1)


def _svd_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    _u, _s, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    return centroid, normal


def _plane_from_three(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> tuple[np.ndarray, float] | None:
    normal = np.cross(p1 - p0, p2 - p0)
    length = float(np.linalg.norm(normal))
    if length < 1e-9:
        return None
    normal = normal / length
    return normal, float(-np.dot(normal, p0))


def fit_plane(points: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC plane, refit with least squares on the inliers."""
    count = len(points)
    best: np.ndarray | None = None
    best_count = 0
    for _ in range(RANSAC_ITERS):
        choice = rng.choice(count, size=3, replace=False)
        plane = _plane_from_three(points[choice[0]], points[choice[1]], points[choice[2]])
        if plane is None:
            continue
        normal, offset = plane
        inliers = np.abs(points @ normal + offset) < RANSAC_INLIER_M
        inlier_count = int(inliers.sum())
        if inlier_count > best_count:
            best_count = inlier_count
            best = inliers
    if best is None or best_count < 3:
        return _svd_plane(points)
    return _svd_plane(points[best])


def _face_points(
    depth: np.ndarray, x: float, y: float, width: float, height: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    image_h, image_w = depth.shape
    x0 = max(0, int(np.floor(x)))
    y0 = max(0, int(np.floor(y)))
    x1 = min(image_w, int(np.ceil(x + width)))
    y1 = min(image_h, int(np.ceil(y + height)))
    if x1 <= x0 or y1 <= y0:
        return None
    area = (x1 - x0) * (y1 - y0)
    stride = max(1, int(np.sqrt(area / 4000)))
    roi = depth[y0:y1:stride, x0:x1:stride]
    rows, cols = np.nonzero(roi > 0)
    if rows.size < MIN_PLANE_POINTS:
        return None
    us = (x0 + cols * stride).astype(np.float64)
    vs = (y0 + rows * stride).astype(np.float64)
    zs = roi[rows, cols].astype(np.float64)
    return us, vs, zs


def _patch_median(depth: np.ndarray, u: float, v: float) -> float:
    image_h, image_w = depth.shape
    x = round(u)
    y = round(v)
    x0 = max(0, x - PATCH_RADIUS)
    x1 = min(image_w, x + PATCH_RADIUS + 1)
    y0 = max(0, y - PATCH_RADIUS)
    y1 = min(image_h, y + PATCH_RADIUS + 1)
    if x1 <= x0 or y1 <= y0:
        return float("nan")
    patch = depth[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))


def nose_cheek_gap(
    depth: np.ndarray,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    landmarks: np.ndarray,
) -> float:
    """Cheek depth minus nose depth. Positive when the nose is closer to the camera."""
    nose_u, nose_v = float(landmarks[2, 0]), float(landmarks[2, 1])
    offset = CHEEK_OFFSET_FRAC * width
    inset = PATCH_RADIUS + 1
    left_u = min(max(nose_u - offset, x + inset), x + width - inset)
    right_u = min(max(nose_u + offset, x + inset), x + width - inset)
    nose_z = _patch_median(depth, nose_u, nose_v)
    left_z = _patch_median(depth, left_u, nose_v)
    right_z = _patch_median(depth, right_u, nose_v)
    if not np.isfinite([nose_z, left_z, right_z]).all():
        return float("nan")
    return float((left_z + right_z) / 2.0 - nose_z)


def _cm(metres: float) -> str:
    return f"{metres * 100:.1f} cm"


def _measure(
    depth: np.ndarray,
    intrinsics: Intrinsics,
    x: float,
    y: float,
    width: float,
    height: float,
    landmarks: np.ndarray,
) -> tuple[float | None, float | None]:
    pixels = _face_points(depth, x, y, width, height)
    gap = nose_cheek_gap(depth, x=x, y=y, width=width, height=height, landmarks=landmarks)
    if pixels is None:
        return None, None if np.isnan(gap) else gap
    us, vs, zs = pixels
    points = deproject(us, vs, zs, intrinsics)
    centroid, normal = fit_plane(points, np.random.default_rng(0))
    signed = (points - centroid) @ normal
    keep = np.abs(signed) <= GROSS_OUTLIER_M
    if int(keep.sum()) < MIN_PLANE_POINTS:
        keep = np.ones(len(points), dtype=bool)
    signed = signed[keep]
    relief = float(np.percentile(signed, 95) - np.percentile(signed, 5))
    return relief, None if np.isnan(gap) else gap


def check_geometry(
    depth: np.ndarray,
    intrinsics: Intrinsics,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    landmarks: np.ndarray,
    quality: dict[str, Any],
    liveness: dict[str, Any],
    ir: np.ndarray | None = None,
) -> GeometryReport:
    gate: QualityReport = assess_quality(
        depth, x=x, y=y, width=width, height=height, quality=quality, ir=ir
    )
    relief, gap = (None, None)
    if gate.reason != "face box is outside the image":
        relief, gap = _measure(depth, intrinsics, x, y, width, height, landmarks)
    if not gate.ok:
        return GeometryReport(
            ok=False,
            reason=gate.reason,
            relief_m=relief,
            nose_cheek_m=gap,
            hole_frac=gate.hole_frac,
            distance_m=gate.distance_m,
        )
    if relief is None:
        return GeometryReport(
            ok=False,
            reason="not enough depth points on the face",
            relief_m=None,
            nose_cheek_m=gap,
            hole_frac=gate.hole_frac,
            distance_m=gate.distance_m,
        )

    flat_max = float(liveness["flat_max_m"])
    relief_min = float(liveness["relief_min_m"])
    relief_max = float(liveness.get("relief_max_m", 0.08))
    nose_min = float(liveness.get("nose_cheek_min_m", 0.008))

    if relief < flat_max:
        reason = (
            f"flat surface (relief {_cm(relief)}; a face needs at least {_cm(relief_min)})"
        )
    elif relief < relief_min:
        reason = (
            f"not enough face shape (relief {_cm(relief)}; "
            f"a print is under {_cm(flat_max)}, a face is at least {_cm(relief_min)})"
        )
    elif relief > relief_max:
        reason = f"depth shape is too deep to be a face (relief {_cm(relief)})"
    elif gap is None:
        reason = "no depth on the nose or cheeks"
    elif gap < nose_min:
        reason = (
            f"nose does not stick out from the cheeks (gap {_cm(gap)}, need {_cm(nose_min)})"
        )
    else:
        return GeometryReport(
            ok=True,
            reason="face shape is live",
            relief_m=relief,
            nose_cheek_m=gap,
            hole_frac=gate.hole_frac,
            distance_m=gate.distance_m,
        )
    return GeometryReport(
        ok=False,
        reason=reason,
        relief_m=relief,
        nose_cheek_m=gap,
        hole_frac=gate.hole_frac,
        distance_m=gate.distance_m,
        shape_failed=True,
    )
