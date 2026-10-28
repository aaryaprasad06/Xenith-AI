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
    created_at: str | None = None
    face_enrolled: bool = False


class UsersResponse(BaseModel):
    users: list[UserRecord]


class EnrollResponse(BaseModel):
    status: Literal["recorded"]
    user_id: str
    message: str


class DeleteResponse(BaseModel):
    status: Literal["ok"] = "ok"
    user_id: str


class EnrollTakesRequest(BaseModel):
    session: str
    consent: bool = False


class EnrollTakesResponse(BaseModel):
    user_id: str
    face_enrolled: bool = True
    takes_used: int


class AuthResponse(BaseModel):
    # How far the attempt got. `no_gallery`, `no_match` and `matched` all passed liveness.
    status: Literal[
        "camera_unavailable",
        "locked_out",
        "no_frame",
        "no_face",
        "rejected",
        "no_gallery",
        "no_match",
        "matched",
    ]
    decision: Literal["granted", "denied"] = "denied"
    # Only set when granted, so a denied caller learns nothing about who is enrolled.
    user_id: str | None = None
    reason: str
    state: str = "idle"
    failures: int = 0
    relief_m: float | None = None
    nose_cheek_m: float | None = None
    match_score: float | None = None


class DoorStateResponse(BaseModel):
    state: str
    locked: bool
    failures: int
    mock_relay: bool


class AccessEventRecord(BaseModel):
    id: int
    ts: str
    decision: str
    status: str
    reason: str
    state: str
    path: str
    liveness_geom: float | None
    liveness_model: float | None
    match_score: float | None
    person_id: str | None
    latency_ms: float


class AccessLogResponse(BaseModel):
    events: list[AccessEventRecord]


class RecordingOptions(BaseModel):
    poses: list[str]
    lighting: list[str]
    attacks: list[str]
    default_frames: int


class RecordRequest(BaseModel):
    person: str
    session: str
    pose: str = "center"
    lighting: str = "lit"
    glasses: bool = False
    attack: str = "none"
    frames: int = Field(default=30, ge=2, le=300)
    consent: bool = False
    overwrite: bool = False


class TakeSummary(BaseModel):
    person: str
    session: str
    take: str
    pose: str
    lighting: str
    glasses: bool
    attack: str
    frames: int
    laser_on: int
    laser_off: int
    recorded_at: str


class TakesResponse(BaseModel):
    takes: list[TakeSummary]


class RecordResponse(BaseModel):
    take: TakeSummary
    # Frames with exactly one face at least `quality.min_face_px` wide, as `eval.far_frr` counts them.
    faces_on: int
    faces_off: int


class DeleteTakeResponse(BaseModel):
    status: Literal["ok"] = "ok"
    take: str


class GuidedStart(BaseModel):
    person: str
    session: str
    step: Literal["circle", "dark", "print", "screen"] = "circle"
    glasses: bool = False
    consent: bool = False
    overwrite: bool = False


class GuidedState(BaseModel):
    status: Literal["idle", "running", "saving", "done", "failed", "cancelled"]
    step: str | None = None
    phase: str | None = None
    prompt: str
    progress: float
    # Ring ticks counter-clockwise from screen right, on a mirrored preview.
    ticks: list[bool]
    counts: dict[str, int]
    saved: list[str]
    pairs_per_s: float
    pointer: tuple[float, float] | None = None


class ErrorBody(BaseModel):
    detail: str = Field(examples=["admin token required"])
