from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from api.core.config import Settings
from api.services.camera import CameraHub
from api.services.door import DoorService
from api.services.guided import GuidedRecorder
from api.services.users import UserStore


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_raw_root(request: Request) -> Path:
    return request.app.state.settings.raw_root


def get_camera(request: Request) -> CameraHub:
    return request.app.state.camera


def get_door(request: Request) -> DoorService:
    return request.app.state.door


def get_guided(request: Request) -> GuidedRecorder:
    return request.app.state.guided


def get_users(request: Request) -> UserStore:
    return request.app.state.users


def require_admin(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    if not settings.admin_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FACE_ID_ADMIN_TOKEN is not set",
        )
    if authorization != f"Bearer {settings.admin_token}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="admin token required",
        )
