"""Similarity-warp a face to the 112×112 crop used by later embedding models.

The same transform is applied to depth with nearest-neighbour sampling so
invalid depth (0) is not blended into a neighbouring pixel.
"""

from __future__ import annotations

import cv2
import numpy as np

# InsightFace / ArcFace 5-point template for a 112×112 crop.
TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def align_crop(
    ir: np.ndarray,
    depth: np.ndarray,
    landmarks: np.ndarray,
    size: int = 112,
) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    transform, _inliers = cv2.estimateAffinePartial2D(src, TEMPLATE_112, method=cv2.LMEDS)
    if transform is None:
        raise ValueError("could not estimate a face alignment")
    ir_out = cv2.warpAffine(
        ir, transform, (size, size), flags=cv2.INTER_LINEAR, borderValue=0
    )
    depth_out = cv2.warpAffine(
        depth, transform, (size, size), flags=cv2.INTER_NEAREST, borderValue=0
    )
    return ir_out, depth_out
