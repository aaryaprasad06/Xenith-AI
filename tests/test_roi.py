import math

import numpy as np
import pytest

from capture.frame import center_roi_box, center_roi_median_m


def test_median_ignores_invalid_pixels():
    depth = np.zeros((100, 200), dtype=np.float32)
    x0, y0, x1, y1 = center_roi_box(200, 100, 0.1)
    depth[y0:y1, x0:x1] = 0.40
    depth[y0, x0:x1] = 0.0  # one row of holes must not drag the median down
    assert center_roi_median_m(depth, 0.1) == pytest.approx(0.40)


def test_median_of_known_values():
    depth = np.zeros((100, 100), dtype=np.float32)
    x0, y0, x1, y1 = center_roi_box(100, 100, 0.2)
    roi = depth[y0:y1, x0:x1]
    roi[:] = 0.5
    roi[0, :] = 0.3
    roi[1, :] = 0.9
    assert center_roi_median_m(depth, 0.2) == pytest.approx(0.5)


def test_pixels_outside_box_are_ignored():
    depth = np.full((100, 100), 2.0, dtype=np.float32)
    x0, y0, x1, y1 = center_roi_box(100, 100, 0.1)
    depth[y0:y1, x0:x1] = 0.45
    assert center_roi_median_m(depth, 0.1) == pytest.approx(0.45)


def test_all_zero_roi_is_nan():
    assert math.isnan(center_roi_median_m(np.zeros((50, 50), dtype=np.float32), 0.1))


@pytest.mark.parametrize("frac", [0.0, -0.1, 1.5])
def test_bad_fraction_rejected(frac: float):
    with pytest.raises(ValueError):
        center_roi_median_m(np.ones((10, 10), dtype=np.float32), frac)


def test_box_is_centered_and_sized():
    assert center_roi_box(848, 480, 0.1) == (381, 216, 466, 264)
