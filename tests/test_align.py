import numpy as np
import pytest

from face.align import TEMPLATE_112, align_crop


def test_align_keeps_nose_depth_on_the_template():
    ir = np.zeros((200, 200), dtype=np.uint8)
    depth = np.zeros((200, 200), dtype=np.float32)
    landmarks = TEMPLATE_112 + np.array([20, 30], dtype=np.float32)
    nose_u, nose_v = landmarks[2]
    depth[round(float(nose_v)), round(float(nose_u))] = np.float32(0.42)
    ir_out, depth_out = align_crop(ir, depth, landmarks)
    assert ir_out.shape == (112, 112)
    assert depth_out.shape == (112, 112)
    tx, ty = TEMPLATE_112[2]
    patch = depth_out[int(ty) - 2 : int(ty) + 3, int(tx) - 2 : int(tx) + 3]
    assert float(patch.max()) == pytest.approx(0.42)
