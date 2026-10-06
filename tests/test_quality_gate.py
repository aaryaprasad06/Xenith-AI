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
