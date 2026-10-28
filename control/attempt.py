"""One door attempt over a window of recent frames: liveness, then identity.

Liveness runs first and on laser-on frames (depth). Only a live window reaches
the matcher, which uses the laser-off frames: the face is found in every
laser-off frame, while the dot pattern breaks it up on laser-on (journal,
2026-10-17). The laser-off face must sit where the live face was, so the
matcher cannot be fed a different face than the one whose depth was checked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np

from capture.frame import Frame
from control.state_machine import Evidence
from face.align import align_crop
from face.detect import Detection, faces_for_depth_frame
from face.embed import Embedder
from face.match import Gallery, WindowMatch, match_window
from liveness.session import MAX_MOTION_FRAC, judge_window

Status = Literal[
    "camera_unavailable",
    "locked_out",
    "no_frame",
    "no_face",
    "rejected",
    "no_gallery",
    "no_match",
    "matched",
]


class Detector(Protocol):
    def detect(self, ir: np.ndarray) -> list[Detection]: ...


@dataclass(frozen=True)
class Attempt:
    status: Status
    evidence: Evidence
    relief_m: float | None = None
    nose_cheek_m: float | None = None
    match_score: float | None = None


def _centre(face: Detection) -> np.ndarray:
    return np.array([face.x + face.width / 2, face.y + face.height / 2])


def _nearest_off_ir(timestamp: float, off_frames: Sequence[Frame]) -> np.ndarray | None:
    if not off_frames:
        return None
    return min(off_frames, key=lambda frame: abs(frame.timestamp - timestamp)).ir


class Identifier:
    """Embeds the laser-off faces of a window and matches them to the gallery."""

    def __init__(self, gallery: Gallery, embedder: Embedder, match: dict[str, Any]) -> None:
        if embedder.name != gallery.embedder:
            raise ValueError(f"gallery was built with {gallery.embedder}, not {embedder.name}")
        self.gallery = gallery
        self.embedder = embedder
        self.threshold = float(match["threshold"])
        self.frames_required = int(match["frames_required"])
        self.agree_frac = float(match["agree_frac"])

    def identify(
        self,
        off_frames: Sequence[Frame],
        detector: Detector,
        *,
        live_centre: np.ndarray,
        min_face_px: float,
    ) -> WindowMatch:
        vectors = []
        for frame in off_frames:
            faces = [f for f in detector.detect(frame.ir) if min(f.width, f.height) >= min_face_px]
            if len(faces) != 1:
                continue
            limit = MAX_MOTION_FRAC * frame.ir.shape[1]
            if float(np.linalg.norm(_centre(faces[0]) - live_centre)) > limit:
                continue
            ir_crop, depth_crop = align_crop(frame.ir, frame.depth, faces[0].landmarks)
            vectors.append(self.embedder.embed(ir_crop, depth_crop).vector)
        return match_window(
            self.gallery,
            vectors,
            threshold=self.threshold,
            frames_required=self.frames_required,
            agree_frac=self.agree_frac,
        )


def evaluate_window(
    frames: Sequence[Frame],
    detector: Detector,
    cfg: dict[str, Any],
    identifier: Identifier | None,
    *,
    no_identifier_reason: str = "no one is enrolled",
) -> Attempt:
    """`frames` is the recent window, oldest first, laser-on and laser-off both."""
    required = int(cfg["liveness"]["frames_required"])
    min_face_px = float(cfg["quality"]["min_face_px"])
    on_frames = [f for f in frames if f.emitter_state == "on"][-required:]
    off_frames = [f for f in frames if f.emitter_state == "off"]
    detections = [
        faces_for_depth_frame(detector, f.ir, _nearest_off_ir(f.timestamp, off_frames), min_face_px)
        for f in on_frames
    ]
    live = judge_window(on_frames, detections, saw_emitter_off=bool(off_frames), cfg=cfg)
    geometry = {"relief_m": live.relief_m, "nose_cheek_m": live.nose_cheek_m}

    if live.status in ("no_frame", "no_face"):
        evidence = Evidence(False, False, False, False, None, live.reason)
        return Attempt(status=live.status, evidence=evidence, **geometry)
    if live.status == "rejected":
        # Only a wrong face shape got past the quality checks; anything else
        # (two faces, motion, the gate) stopped before the shape was judged.
        evidence = Evidence(True, live.shape_failed, False, False, None, live.reason)
        return Attempt(status="rejected", evidence=evidence, **geometry)

    if identifier is None:
        evidence = Evidence(True, True, True, False, None, f"live, but {no_identifier_reason}")
        return Attempt(status="no_gallery", evidence=evidence, **geometry)

    live_centre = _centre(detections[-1][0])
    found = identifier.identify(off_frames, detector, live_centre=live_centre, min_face_px=min_face_px)
    checked = found.score is not None
    if found.person_id is None:
        evidence = Evidence(True, True, True, checked, None, f"live, but {found.reason}")
        return Attempt(status="no_match", evidence=evidence, match_score=found.score, **geometry)
    evidence = Evidence(True, True, True, True, found.person_id, f"live and {found.reason}")
    return Attempt(status="matched", evidence=evidence, match_score=found.score, **geometry)
