from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    camera: Literal["streaming", "starting", "disabled", "error"]


class DeviceInfo(BaseModel):
    name: str
    serial: str
    firmware: str
    product_line: str | None = None


class UserRecord(BaseModel):
    user_id: str
    created_at: str


class UsersResponse(BaseModel):
    users: list[UserRecord]


class EnrollResponse(BaseModel):
    status: Literal["recorded"]
    user_id: str
    message: str


class DeleteResponse(BaseModel):
    status: Literal["ok"] = "ok"
    user_id: str


class AuthResponse(BaseModel):
    status: Literal[
        "camera_unavailable",
        "no_frame",
        "no_face",
        "rejected",
        "live",
    ]
    user_id: str | None = None
    decision: Literal["denied"] = "denied"
    reason: str
    relief_m: float | None = None
    nose_cheek_m: float | None = None


class ErrorBody(BaseModel):
    detail: str = Field(examples=["admin token required"])
