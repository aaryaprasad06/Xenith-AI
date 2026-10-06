from typing import Annotated

from fastapi import APIRouter, Depends

from api.deps import get_camera
from api.schemas import AuthResponse
from api.services.camera import CameraHub
from capture.config import load_config
from face.detect import get_detector
from liveness.session import judge_window

router = APIRouter(prefix="/auth", tags=["auth"])


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
    detector = get_detector()
    detections = [detector.detect(frame.ir) for frame in on_frames]
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
