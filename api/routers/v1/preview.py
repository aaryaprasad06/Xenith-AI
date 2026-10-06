import time
from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from api.deps import get_camera
from api.services.camera import CameraHub

router = APIRouter(prefix="/preview", tags=["preview"])


def _mjpeg(camera: CameraHub) -> Iterator[bytes]:
    last: bytes | None = None
    while camera.error is None:
        frame = camera.latest_jpeg()
        if frame is None or frame is last:
            time.sleep(0.03)
            continue
        last = frame
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


@router.get("/stream", name="v1:preview:stream")
def stream(camera: Annotated[CameraHub, Depends(get_camera)]) -> StreamingResponse:
    if camera.camera_state in {"disabled", "error"}:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=camera.error or "camera is not running",
        )
    return StreamingResponse(
        _mjpeg(camera),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
