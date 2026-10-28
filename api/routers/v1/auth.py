from typing import Annotated

from fastapi import APIRouter, Depends, Query

from api.deps import get_camera, get_door, require_admin
from api.schemas import (
    AccessEventRecord,
    AccessLogResponse,
    AuthResponse,
    DoorStateResponse,
)
from api.services.camera import CameraHub
from api.services.door import DoorService
from control.relay import MockRelay

router = APIRouter(prefix="/auth", tags=["auth"])


def authenticate(
    camera: Annotated[CameraHub, Depends(get_camera)],
    door: Annotated[DoorService, Depends(get_door)],
) -> AuthResponse:
    """Liveness on the recent frames, then 1:N matching; unlock on a live enrolled face."""
    return door.authenticate(camera)


router.post("", name="v1:auth")(authenticate)
router.post("/", name="v1:auth-slash", include_in_schema=False)(authenticate)


@router.get("/state", name="v1:auth:state")
def door_state(door: Annotated[DoorService, Depends(get_door)]) -> DoorStateResponse:
    """Current relay state, including automatic re-lock and cooldown expiry."""
    state, locked, failures = door.controller.snapshot()
    return DoorStateResponse(
        state=state.value,
        locked=locked,
        failures=failures,
        mock_relay=isinstance(door.relay, MockRelay),
    )


@router.get("/log", name="v1:auth:log", dependencies=[Depends(require_admin)])
def access_log(
    door: Annotated[DoorService, Depends(get_door)],
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> AccessLogResponse:
    """Recent door attempts, newest first, spoofs included."""
    events = [AccessEventRecord(**vars(e)) for e in door.access_log.recent(limit)]
    return AccessLogResponse(events=events)
