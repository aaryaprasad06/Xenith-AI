from typing import Annotated

from fastapi import APIRouter, Depends

from api.deps import get_camera
from api.schemas import AuthResponse
from api.services.camera import CameraHub

router = APIRouter(prefix="/auth", tags=["auth"])


def authenticate(camera: Annotated[CameraHub, Depends(get_camera)]) -> AuthResponse:
    """Capture is live, but face matching is not built yet. Always deny."""
    if camera.camera_state in {"disabled", "error"}:
        return AuthResponse(
            status="camera_unavailable",
            reason=camera.error or "camera is not running",
        )
    return AuthResponse(
        status="pipeline_not_ready",
        reason="liveness and identity matching are not implemented yet",
    )


router.post("", name="v1:auth")(authenticate)
router.post("/", name="v1:auth-slash", include_in_schema=False)(authenticate)
