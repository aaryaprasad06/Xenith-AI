"""YuNet face detector. Box plus five landmarks, on the left IR image.

The model file is `models/face_detection_yunet_2023mar.onnx`. See docs/CITATIONS.md.
Landmarks, in order: right eye, left eye, nose tip, right mouth corner, left mouth corner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / MODEL_FILENAME


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    width: float
    height: float
    score: float
    landmarks: np.ndarray  # (5, 2) float, pixels


def model_path() -> Path:
    if not DEFAULT_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"YuNet model missing at {DEFAULT_MODEL_PATH}. See docs/CITATIONS.md."
        )
    return DEFAULT_MODEL_PATH


class FaceDetector:
    def __init__(
        self,
        path: str | Path,
        *,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
    ) -> None:
        self._path = str(path)
        self._score = score_threshold
        self._nms = nms_threshold
        self._detector: cv2.FaceDetectorYN | None = None
        self._size: tuple[int, int] | None = None

    def detect(self, ir: np.ndarray) -> list[Detection]:
        if ir.ndim == 2:
            image = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        else:
            image = ir
        height, width = image.shape[:2]
        if self._detector is None or self._size != (width, height):
            self._detector = cv2.FaceDetectorYN.create(
                self._path,
                "",
                (width, height),
                self._score,
                self._nms,
                5000,
            )
            self._size = (width, height)
        self._detector.setInputSize((width, height))
        _retval, faces = self._detector.detect(image)
        if faces is None:
            return []
        found: list[Detection] = []
        for row in faces:
            found.append(
                Detection(
                    x=float(row[0]),
                    y=float(row[1]),
                    width=float(row[2]),
                    height=float(row[3]),
                    score=float(row[14]),
                    landmarks=np.asarray(row[4:14], dtype=np.float64).reshape(5, 2),
                )
            )
        return found


_cached: FaceDetector | None = None


def get_detector() -> FaceDetector:
    global _cached
    if _cached is None:
        _cached = FaceDetector(model_path())
    return _cached
