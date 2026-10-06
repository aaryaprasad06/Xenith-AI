from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.deps import get_camera
from api.schemas import DeviceInfo
from api.services.camera import CameraHub

router = APIRouter(prefix="/device", tags=["device"])


@router.get("/info", name="v1:device:info")
def device_info(camera: Annotated[CameraHub, Depends(get_camera)]) -> DeviceInfo:
    if camera.info is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=camera.error or "camera is not running",
        )
    return camera.info
