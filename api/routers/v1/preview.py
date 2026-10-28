import time
from collections.abc import Iterator
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from api.deps import get_camera
from api.services.camera import CameraHub

router = APIRouter(prefix="/preview", tags=["preview"])


def _mjpeg(camera: CameraHub, channel: Literal["ir", "rgb"] = "ir") -> Iterator[bytes]:
    last: bytes | None = None
    while camera.error is None:
        frame = camera.latest_jpeg(channel)
        if frame is None or frame is last:
            time.sleep(0.03)
            continue
        last = frame
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


def _stream(camera: CameraHub, channel: Literal["ir", "rgb"]) -> StreamingResponse:
    if camera.camera_state in {"disabled", "error"}:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=camera.error or "camera is not running",
        )
    if channel == "rgb" and camera.latest_jpeg("rgb") is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RGB preview is not ready; the camera must have colour capture enabled",
        )
    return StreamingResponse(
        _mjpeg(camera, channel),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/stream", name="v1:preview:stream")
def stream(camera: Annotated[CameraHub, Depends(get_camera)]) -> StreamingResponse:
    return _stream(camera, "ir")


@router.get("/rgb", name="v1:preview:rgb")
def rgb_stream(camera: Annotated[CameraHub, Depends(get_camera)]) -> StreamingResponse:
    """Colour preview from the same pipeline, with no second camera connection."""
    return _stream(camera, "rgb")
