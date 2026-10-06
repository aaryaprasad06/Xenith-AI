import numpy as np

from face.detect import (
    Detection,
    FaceDetector,
    merge_detections,
    model_path,
    prefer_larger_face,
)


def _box(x: float, y: float, side: float, score: float = 0.9) -> Detection:
    return Detection(x, y, side, side, score, np.zeros((5, 2)))


def test_blank_ir_has_no_face():
    detector = FaceDetector(model_path())
    blank = np.zeros((480, 848), dtype=np.uint8)
    assert detector.detect(blank) == []


def test_speck_and_duplicate_boxes_are_dropped():
    real = _box(100, 80, 220, score=0.7)
    speck = _box(10, 10, 32, score=0.99)
    same_face = _box(110, 90, 200, score=0.95)
    other = _box(500, 80, 180, score=0.8)
    merged = merge_detections([speck, same_face, real, other])
    assert sorted(face.width for face in merged) == [180.0, 200.0]


def test_laser_off_is_used_when_it_sees_a_bigger_face():
    speck = [_box(10, 10, 32)]
    close = [_box(200, 100, 280, score=0.8)]
    assert prefer_larger_face(speck, close) is close
    assert prefer_larger_face(close, [_box(200, 100, 300)]) is close
