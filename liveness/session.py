"""Decide liveness from a short window of frames.

Depth is taken from laser-on frames. The window must also contain a laser-off
frame, so the check is not a single laser-on snapshot. Detection runs on the
laser-on IR image: that is the frame the exposure cap keeps usable. A dark
laser-off image is not required to contain a detectable face.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from capture.frame import Frame
from face.detect import Detection
from liveness.depth_check import GeometryReport, check_geometry

Status = Literal["no_frame", "no_face", "rejected", "live"]
# How far the face box centre may travel across the window, as a fraction of image width.
MAX_MOTION_FRAC = 0.15


@dataclass(frozen=True)
class LivenessDecision:
    status: Status
    reason: str
    relief_m: float | None
    nose_cheek_m: float | None

    @property
    def live(self) -> bool:
        return self.status == "live"


def judge_window(
    frames: Sequence[Frame],
    detections: Sequence[Sequence[Detection]],
    *,
    saw_emitter_off: bool,
    cfg: dict[str, Any],
) -> LivenessDecision:
    """`frames` are laser-on only, paired 1:1 with `detections`."""
    if len(frames) != len(detections):
        raise ValueError("frames and detections must be the same length")
    required = int(cfg["liveness"]["frames_required"])
    if len(frames) < required:
        return LivenessDecision(
            status="no_frame",
            reason=f"need {required} laser-on frames, have {len(frames)}",
            relief_m=None,
            nose_cheek_m=None,
        )
    if not saw_emitter_off:
        return LivenessDecision(
            status="rejected",
            reason="the laser did not turn off during this check",
            relief_m=None,
            nose_cheek_m=None,
        )

    frames = frames[-required:]
    detections = detections[-required:]
    reports: list[GeometryReport] = []
    centres: list[tuple[float, float]] = []
    for frame, faces in zip(frames, detections, strict=True):
        if len(faces) == 0:
            return LivenessDecision(
                status="no_face",
                reason="no face in the infrared image",
                relief_m=None,
                nose_cheek_m=None,
            )
        if len(faces) > 1:
            return LivenessDecision(
                status="rejected",
                reason=(
                    f"{len(faces)} faces in frame; stand alone in front of the camera"
                ),
                relief_m=None,
                nose_cheek_m=None,
            )
        face = faces[0]
        centres.append((face.x + face.width / 2, face.y + face.height / 2))
        reports.append(
            check_geometry(
                frame.depth,
                frame.intrinsics,
                x=face.x,
                y=face.y,
                width=face.width,
                height=face.height,
                landmarks=face.landmarks,
                quality=cfg["quality"],
                liveness=cfg["liveness"],
            )
        )

    centre_arr = np.asarray(centres, dtype=np.float64)
    jump = float(np.linalg.norm(centre_arr.max(axis=0) - centre_arr.min(axis=0)))
    limit = MAX_MOTION_FRAC * frames[0].ir.shape[1]
    if jump > limit:
        last = reports[-1]
        return LivenessDecision(
            status="rejected",
            reason="face moved too much during the check; hold still",
            relief_m=last.relief_m,
            nose_cheek_m=last.nose_cheek_m,
        )

    for report in reports:
        if not report.ok:
            return LivenessDecision(
                status="rejected",
                reason=report.reason,
                relief_m=report.relief_m,
                nose_cheek_m=report.nose_cheek_m,
            )

    last = reports[-1]
    relief = last.relief_m or 0.0
    gap = last.nose_cheek_m or 0.0
    return LivenessDecision(
        status="live",
        reason=(
            f"{required} laser-on frames show a face shape "
            f"(relief {relief * 100:.1f} cm, nose-to-cheek {gap * 100:.1f} cm), "
            "and the laser turned off during the check. "
            "Identity matching is not built, so the door stays locked."
        ),
        relief_m=last.relief_m,
        nose_cheek_m=last.nose_cheek_m,
    )
