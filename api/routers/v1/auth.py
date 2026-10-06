from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends

from api.deps import get_camera
from api.schemas import AuthResponse
from api.services.camera import CameraHub
from capture.config import load_config
from capture.frame import Frame
from face.detect import faces_for_depth_frame, get_detector
from liveness.session import judge_window

router = APIRouter(prefix="/auth", tags=["auth"])


def _nearest_off_ir(timestamp: float, off_frames: list[Frame]) -> np.ndarray | None:
    if not off_frames:
        return None
    nearest = min(off_frames, key=lambda frame: abs(frame.timestamp - timestamp))
    return nearest.ir


def authenticate(camera: Annotated[CameraHub, Depends(get_camera)]) -> AuthResponse:
    """Run geometry liveness. The door stays locked until identity matching exists."""
    if camera.camera_state in {"disabled", "error"}:
        return AuthResponse(
            status="camera_unavailable",
            reason=camera.error or "camera is not running",
        )
    frames = camera.recent_frames()
    cfg = load_config(camera.config_path)
    required = int(cfg["liveness"]["frames_required"])
    on_frames = [frame for frame in frames if frame.emitter_state == "on"][-required:]
    off_frames = [frame for frame in frames if frame.emitter_state == "off"]
    detector = get_detector()
    min_face_px = float(cfg["quality"]["min_face_px"])
    detections = [
        faces_for_depth_frame(
            detector,
            frame.ir,
            _nearest_off_ir(frame.timestamp, off_frames),
            min_face_px,
        )
        for frame in on_frames
    ]
    decision = judge_window(
        on_frames,
        detections,
        saw_emitter_off=any(frame.emitter_state == "off" for frame in frames),
        cfg=cfg,
    )
    return AuthResponse(
        status=decision.status,
        reason=decision.reason,
        relief_m=decision.relief_m,
        nose_cheek_m=decision.nose_cheek_m,
    )


router.post("", name="v1:auth")(authenticate)
router.post("/", name="v1:auth-slash", include_in_schema=False)(authenticate)
