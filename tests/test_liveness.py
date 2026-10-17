"""Synthetic flat and curved depth. No camera and no YuNet model."""

from __future__ import annotations

import numpy as np

from capture.config import load_config
from capture.frame import Frame, Intrinsics
from face.detect import Detection
from liveness.depth_check import check_geometry
from liveness.session import judge_window

H, W = 480, 848
BOX = (364.0, 180.0, 120.0, 120.0)  # centred: (424, 240)


def intrinsics() -> Intrinsics:
    return Intrinsics(
        fx=425.0,
        fy=425.0,
        ppx=424.0,
        ppy=240.0,
        width=W,
        height=H,
        model="brown_conrady",
        coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
    )


def landmarks() -> np.ndarray:
    return np.array(
        [
            [400.0, 210.0],
            [448.0, 210.0],
            [424.0, 240.0],
            [410.0, 270.0],
            [438.0, 270.0],
        ],
        dtype=np.float64,
    )


def face_depth(bulge: float, z0: float = 0.50) -> np.ndarray:
    depth = np.zeros((H, W), dtype=np.float32)
    x, y, bw, bh = (int(v) for v in BOX)
    ys, xs = np.mgrid[y : y + bh, x : x + bw]
    cx = x + bw / 2
    cy = y + bh / 2
    rr = np.clip(((xs - cx) / (bw / 2)) ** 2 + ((ys - cy) / (bh / 2)) ** 2, 0, 1)
    depth[y : y + bh, x : x + bw] = np.float32(z0 - bulge * (1 - rr))
    return depth


def detection_at(x: float = BOX[0], y: float = BOX[1]) -> Detection:
    shift = np.array([x - BOX[0], y - BOX[1]], dtype=np.float64)
    return Detection(
        x=x,
        y=y,
        width=BOX[2],
        height=BOX[3],
        score=0.99,
        landmarks=landmarks() + shift,
    )


def frame(depth: np.ndarray, emitter: str = "on") -> Frame:
    return Frame(
        ir=np.random.default_rng(0).integers(0, 256, size=(H, W), dtype=np.uint8),
        depth=depth,
        intrinsics=intrinsics(),
        timestamp=0.0,
        emitter_state=emitter,  # type: ignore[arg-type]
    )


def test_flat_print_is_rejected():
    cfg = load_config()
    x, y, bw, bh = BOX
    report = check_geometry(
        face_depth(0.0),
        intrinsics(),
        x=x,
        y=y,
        width=bw,
        height=bh,
        landmarks=landmarks(),
        quality=cfg["quality"],
        liveness=cfg["liveness"],
    )
    assert report.ok is False
    assert report.relief_m is not None
    assert report.relief_m < cfg["liveness"]["flat_max_m"]
    assert report.reason.startswith("flat surface")


def test_curved_face_passes():
    cfg = load_config()
    x, y, bw, bh = BOX
    report = check_geometry(
        face_depth(0.035),
        intrinsics(),
        x=x,
        y=y,
        width=bw,
        height=bh,
        landmarks=landmarks(),
        quality=cfg["quality"],
        liveness=cfg["liveness"],
    )
    assert report.relief_m is not None and report.relief_m >= cfg["liveness"]["relief_min_m"]
    assert report.nose_cheek_m is not None and report.nose_cheek_m >= cfg["liveness"]["nose_cheek_min_m"]
    assert report.ok is True


def test_holes_reject_before_shape():
    cfg = load_config()
    depth = face_depth(0.035)
    x, y, bw, bh = (int(v) for v in BOX)
    depth[y : y + bh : 2, x : x + bw] = 0
    report = check_geometry(
        depth,
        intrinsics(),
        x=BOX[0],
        y=BOX[1],
        width=BOX[2],
        height=BOX[3],
        landmarks=landmarks(),
        quality=cfg["quality"],
        liveness=cfg["liveness"],
    )
    assert report.ok is False
    assert "no depth" in report.reason


def test_window_accepts_five_curved_frames_with_laser_off():
    cfg = load_config()
    depth = face_depth(0.035)
    frames = [frame(depth, "on") for _ in range(5)]
    faces = [[detection_at()] for _ in range(5)]
    decision = judge_window(frames, faces, saw_emitter_off=True, cfg=cfg)
    assert decision.status == "live"
    assert decision.live is True
    assert "door stays locked" in decision.reason


def test_window_rejects_a_flat_print():
    cfg = load_config()
    frames = [frame(face_depth(0.0), "on") for _ in range(5)]
    faces = [[detection_at()] for _ in range(5)]
    decision = judge_window(frames, faces, saw_emitter_off=True, cfg=cfg)
    assert decision.status == "rejected"
    assert decision.reason.startswith("flat surface")


def test_window_requires_a_laser_off_frame():
    cfg = load_config()
    frames = [frame(face_depth(0.035), "on") for _ in range(5)]
    faces = [[detection_at()] for _ in range(5)]
    decision = judge_window(frames, faces, saw_emitter_off=False, cfg=cfg)
    assert decision.status == "rejected"
    assert "laser did not turn off" in decision.reason


def test_window_waits_for_enough_frames():
    cfg = load_config()
    decision = judge_window([], [], saw_emitter_off=True, cfg=cfg)
    assert decision.status == "no_frame"


def test_two_faces_are_rejected():
    cfg = load_config()
    frames = [frame(face_depth(0.035), "on")]
    cfg = {**cfg, "liveness": {**cfg["liveness"], "frames_required": 1}}
    faces = [[detection_at(), detection_at(x=300.0)]]
    decision = judge_window(frames, faces, saw_emitter_off=True, cfg=cfg)
    assert decision.status == "rejected"
    assert "2 faces" in decision.reason
