import numpy as np

from face.detect import FaceDetector, model_path


def test_blank_ir_has_no_face():
    detector = FaceDetector(model_path())
    blank = np.zeros((480, 848), dtype=np.uint8)
    assert detector.detect(blank) == []
