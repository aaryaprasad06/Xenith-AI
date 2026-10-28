"""FastAPI app. Same shape as RealSenseID-REST: versioned routers over one device wrapper.

Run: uv run uvicorn api.main:app --host 127.0.0.1 --port 8000
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from api.core.config import Settings, load_settings
from api.deps import get_camera, get_door, get_users, require_admin
from api.routers.v1 import auth, device, preview, recordings, users
from api.schemas import AuthResponse, DeleteResponse, EnrollResponse, HealthResponse
from api.services.camera import CameraHub
from api.services.door import DoorService
from api.services.guided import GuidedRecorder
from api.services.users import UserStore

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.users = UserStore(settings.db_path)
        application.state.camera = CameraHub(
            enabled=settings.camera_enabled,
            config_path=str(settings.config_path) if settings.config_path else None,
        )
        application.state.camera.start()
        application.state.door = DoorService(
            config_path=settings.config_path,
            gallery_path=settings.gallery_path,
            access_log_path=settings.access_log_path,
        )
        application.state.door.start()
        application.state.guided = GuidedRecorder(
            application.state.camera, settings.raw_root
        )
        log.info("camera state: %s", application.state.camera.camera_state)
        yield
        application.state.guided.cancel()
        application.state.door.stop()
        application.state.camera.stop()

    app = FastAPI(
        title="RealSense Face-ID",
        version="0.1.0",
        summary="D435 face authentication API",
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["device"])
    def health(camera: Annotated[CameraHub, Depends(get_camera)]) -> HealthResponse:
        return HealthResponse(camera=camera.camera_state)

    # PRD names, same handlers as the /v1 routers.
    @app.post("/authenticate", tags=["auth"])
    def authenticate_alias(
        camera: Annotated[CameraHub, Depends(get_camera)],
        door: Annotated[DoorService, Depends(get_door)],
    ) -> AuthResponse:
        return auth.authenticate(camera, door)

    @app.post(
        "/admin/enroll",
        tags=["users"],
        status_code=201,
        dependencies=[Depends(require_admin)],
    )
    def enroll_alias(
        users_store: Annotated[UserStore, Depends(get_users)],
        user_id: Annotated[str, Query(min_length=1, max_length=100)],
        door: Annotated[DoorService, Depends(get_door)],
    ) -> EnrollResponse:
        return users.enroll(users_store, user_id, door)

    @app.delete(
        "/admin/person/{user_id}",
        tags=["users"],
        dependencies=[Depends(require_admin)],
    )
    def delete_alias(
        user_id: str,
        users_store: Annotated[UserStore, Depends(get_users)],
        door: Annotated[DoorService, Depends(get_door)],
    ) -> DeleteResponse:
        return users.delete_user(user_id, users_store, door)

    prefix = settings.api_v1_prefix
    app.include_router(auth.router, prefix=prefix)
    app.include_router(users.router, prefix=prefix)
    app.include_router(preview.router, prefix=prefix)
    app.include_router(device.router, prefix=prefix)
    app.include_router(recordings.router, prefix=prefix)
    return app


app = create_app()
