"""YuNet face detector. Box plus five landmarks, on the left IR image.

The model file is `models/face_detection_yunet_2023mar.onnx`. See docs/CITATIONS.md.
Landmarks, in order: right eye, left eye, nose tip, right mouth corner, left mouth corner.

YuNet was trained on faces of about 10–300 px. On this camera a face closer than
~30 cm is larger than that, so detection also runs on half-size and quarter-size
copies and the boxes are scaled back. Boxes under SPECK_MIN_PX are dropped:
they are background hits, not a face.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / MODEL_FILENAME

# Full frame, then smaller copies so a close face falls inside YuNet's range.
SCALES = (1.0, 0.5, 0.25)
SPECK_MIN_PX = 48
IOU_SAME_FACE = 0.4


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


def _intersection(a: Detection, b: Detection) -> float:
    ax1, ay1 = a.x + a.width, a.y + a.height
    bx1, by1 = b.x + b.width, b.y + b.height
    iw = max(0.0, min(ax1, bx1) - max(a.x, b.x))
    ih = max(0.0, min(ay1, by1) - max(a.y, b.y))
    return iw * ih


def _iou(a: Detection, b: Detection) -> float:
    inter = _intersection(a, b)
    union = a.width * a.height + b.width * b.height - inter
    if union <= 0:
        return 0.0
    return inter / union


def merge_detections(faces: list[Detection]) -> list[Detection]:
    """Drop specks, then duplicate boxes of the same face from another scale."""
    faces = [face for face in faces if min(face.width, face.height) >= SPECK_MIN_PX]
    by_score = sorted(faces, key=lambda face: face.score, reverse=True)
    kept: list[Detection] = []
    for face in by_score:
        if any(_iou(face, prev) >= IOU_SAME_FACE for prev in kept):
            continue
        kept.append(face)
    by_area = sorted(kept, key=lambda face: face.width * face.height, reverse=True)
    visible: list[Detection] = []
    for face in by_area:
        area = face.width * face.height
        if area <= 0:
            continue
        if any(_intersection(face, big) / area >= 0.5 for big in visible):
            continue
        visible.append(face)
    return visible


def prefer_larger_face(primary: list[Detection], alternate: list[Detection]) -> list[Detection]:
    """Use the other image when it found a clearly bigger face.

    The laser-on frame is preferred when both see a similar face, because that
    is the frame the depth belongs to.
    """

    def biggest(faces: list[Detection]) -> float:
        if not faces:
            return 0.0
        return max(min(face.width, face.height) for face in faces)

    if biggest(alternate) > max(biggest(primary) * 1.2, float(SPECK_MIN_PX)):
        return alternate
    return primary


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
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._by_size: dict[tuple[int, int], cv2.FaceDetectorYN] = {}

    def _detector_for(self, width: int, height: int) -> cv2.FaceDetectorYN:
        key = (width, height)
        detector = self._by_size.get(key)
        if detector is None:
            detector = cv2.FaceDetectorYN.create(
                self._path,
                "",
                (width, height),
                self._score,
                self._nms,
                5000,
            )
            self._by_size[key] = detector
        detector.setInputSize((width, height))
        return detector

    def _detect_bgr(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        _retval, faces = self._detector_for(width, height).detect(image)
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

    def detect(self, ir: np.ndarray) -> list[Detection]:
        gray = ir if ir.ndim == 2 else cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(self._clahe.apply(gray), cv2.COLOR_GRAY2BGR)
        height, width = image.shape[:2]
        found: list[Detection] = []
        for scale in SCALES:
            if scale == 1.0:
                view = image
                sx = sy = 1.0
            else:
                scaled_w = max(1, round(width * scale))
                scaled_h = max(1, round(height * scale))
                view = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                sx = width / scaled_w
                sy = height / scaled_h
            for face in self._detect_bgr(view):
                found.append(
                    Detection(
                        x=face.x * sx,
                        y=face.y * sy,
                        width=face.width * sx,
                        height=face.height * sy,
                        score=face.score,
                        landmarks=face.landmarks * np.array([sx, sy]),
                    )
                )
        return merge_detections(found)


def faces_for_depth_frame(
    detector: FaceDetector,
    on_ir: np.ndarray,
    off_ir: np.ndarray | None,
    min_face_px: float,
) -> list[Detection]:
    """Detect on the laser-on image, and on the laser-off image when that miss is small.

    Up close the laser washes the face out. The laser-off image often still
    shows it. Depth is still taken from the laser-on frame.
    """
    primary = detector.detect(on_ir)
    if off_ir is None:
        return primary

    def biggest(faces: list[Detection]) -> float:
        if not faces:
            return 0.0
        return max(min(face.width, face.height) for face in faces)

    if biggest(primary) >= min_face_px:
        return primary
    return prefer_larger_face(primary, detector.detect(off_ir))


_cached: FaceDetector | None = None


def get_detector() -> FaceDetector:
    global _cached
    if _cached is None:
        _cached = FaceDetector(model_path())
    return _cached
