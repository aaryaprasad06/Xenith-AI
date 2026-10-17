import cv2
import numpy as np
import pytest

from capture.frame import Intrinsics
from capture.quality_gate import assess_quality
from liveness.depth_check import deproject

QUALITY = {
    "min_dist_m": 0.10,
    "max_dist_m": 0.60,
    "min_face_px": 120,
    "max_hole_frac": 0.2,
}


def _depth(value: float, holes: bool = False) -> np.ndarray:
    depth = np.full((480, 848), value, dtype=np.float32)
    if holes:
        depth[200:280, 400:500] = 0
    return depth


def test_distance_and_size_and_centre():
    ok = assess_quality(_depth(0.45), x=364, y=180, width=120, height=120, quality=QUALITY)
    assert ok.ok is True
    assert ok.distance_m == pytest.approx(0.45)

    small = assess_quality(_depth(0.45), x=364, y=180, width=80, height=120, quality=QUALITY)
    assert small.ok is False
    assert "too small" in small.reason

    close = assess_quality(_depth(0.05), x=364, y=180, width=120, height=120, quality=QUALITY)
    assert "too close" in close.reason

    far = assess_quality(_depth(0.90), x=364, y=180, width=120, height=120, quality=QUALITY)
    assert "too far" in far.reason

    side = assess_quality(_depth(0.45), x=0, y=180, width=120, height=120, quality=QUALITY)
    assert side.reason == "move to the centre of the camera"


def _ir(blurred: bool) -> np.ndarray:
    rng = np.random.default_rng(0)
    ir = rng.integers(0, 256, size=(480, 848), dtype=np.uint8)
    return cv2.GaussianBlur(ir, (0, 0), 6) if blurred else ir


def test_blur_check_rejects_a_blurred_face_and_passes_a_sharp_one():
    quality = {**QUALITY, "blur_min": 20.0}
    box = {"x": 364, "y": 180, "width": 120, "height": 120, "quality": quality}
    sharp = assess_quality(_depth(0.45), ir=_ir(False), **box)
    assert sharp.ok is True
    assert sharp.sharpness is not None and sharp.sharpness >= 20.0

    blurry = assess_quality(_depth(0.45), ir=_ir(True), **box)
    assert blurry.ok is False
    assert "blurry" in blurry.reason
    assert blurry.sharpness is not None and blurry.sharpness < sharp.sharpness


def test_blur_check_is_skipped_without_ir_or_threshold():
    box = {"x": 364, "y": 180, "width": 120, "height": 120}
    no_ir = assess_quality(_depth(0.45), quality={**QUALITY, "blur_min": 20.0}, **box)
    assert no_ir.ok is True and no_ir.sharpness is None
    no_threshold = assess_quality(_depth(0.45), quality=QUALITY, ir=_ir(True), **box)
    assert no_threshold.ok is True


def test_principal_point_deprojects_to_the_optical_axis():
    intr = Intrinsics(400, 400, 100, 80, 200, 160, "brown_conrady", (0, 0, 0, 0, 0))
    points = deproject(
        np.array([100.0]),
        np.array([80.0]),
        np.array([0.5]),
        intr,
    )
    assert points.shape == (1, 3)
    assert abs(points[0, 0]) < 1e-9
    assert abs(points[0, 1]) < 1e-9
    assert points[0, 2] == 0.5
