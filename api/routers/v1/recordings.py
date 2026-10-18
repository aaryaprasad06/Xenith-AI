"""Record labelled takes from the console. Same files as `capture.recorder`, through the shared camera."""

import json
import re
import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.deps import get_camera, get_guided, get_raw_root, require_admin
from api.schemas import (
    DeleteTakeResponse,
    GuidedStart,
    GuidedState,
    RecordingOptions,
    RecordRequest,
    RecordResponse,
    TakesResponse,
    TakeSummary,
)
from api.services.camera import CameraHub
from api.services.guided import GuidedRecorder
from capture.config import load_config
from capture.frame import Frame
from capture.recorder import (
    ATTACKS,
    DEFAULT_FRAMES,
    LIGHTING,
    POSES,
    TakeLabels,
    iter_takes,
    save_take,
)
from face.detect import FaceDetector, get_detector

router = APIRouter(prefix="/recordings", tags=["recordings"], dependencies=[Depends(require_admin)])

_PART = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _summary(take_dir: Path) -> TakeSummary:
    meta = json.loads((take_dir / "meta.json").read_text(encoding="utf-8"))
    labels = meta["labels"]
    states = [entry["emitter_state"] for entry in meta["frames"]]
    return TakeSummary(
        person=labels["person"],
        session=labels["session"],
        take=take_dir.name,
        pose=labels["pose"],
        lighting=labels["lighting"],
        glasses=labels["glasses"],
        attack=labels["attack"],
        frames=len(states),
        laser_on=states.count("on"),
        laser_off=states.count("off"),
        recorded_at=meta["recorded_at"],
    )


def _one_face(detector: FaceDetector, frames: list[Frame], min_face_px: float) -> int:
    count = 0
    for frame in frames:
        faces = [face for face in detector.detect(frame.ir) if min(face.width, face.height) >= min_face_px]
        count += len(faces) == 1
    return count


@router.get("/options", name="v1:recordings:options")
def options() -> RecordingOptions:
    return RecordingOptions(
        poses=list(POSES),
        lighting=list(LIGHTING),
        attacks=list(ATTACKS),
        default_frames=DEFAULT_FRAMES,
    )


@router.get("", name="v1:recordings")
def list_takes(
    root: Annotated[Path, Depends(get_raw_root)],
    person: Annotated[str | None, Query(max_length=64)] = None,
    session: Annotated[str | None, Query(max_length=64)] = None,
) -> TakesResponse:
    takes = [_summary(take_dir) for take_dir in iter_takes(root)]
    takes = [
        take
        for take in takes
        if (person is None or take.person == person) and (session is None or take.session == session)
    ]
    return TakesResponse(takes=takes)


@router.post("", name="v1:recordings:record", status_code=status.HTTP_201_CREATED)
def record(
    request: RecordRequest,
    camera: Annotated[CameraHub, Depends(get_camera)],
    root: Annotated[Path, Depends(get_raw_root)],
) -> RecordResponse:
    if not request.consent:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="record only people whose consent is on file",
        )
    try:
        labels = TakeLabels(
            person=request.person,
            session=request.session,
            pose=request.pose,
            lighting=request.lighting,
            glasses=request.glasses,
            attack=request.attack,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    if labels.take_dir(root).exists() and not request.overwrite:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="this take is already recorded")

    try:
        # 30 fps, with room for the stream to stall briefly.
        frames = camera.capture(request.frames, timeout_s=request.frames / 30 * 3 + 5)
    except RuntimeError as exc:
        busy = "already recording" in str(exc)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT if busy else status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)) from exc

    out = save_take(root, labels, frames, overwrite=request.overwrite)
    min_face_px = float(load_config(camera.config_path)["quality"]["min_face_px"])
    detector = get_detector()
    return RecordResponse(
        take=_summary(out),
        faces_on=_one_face(detector, [f for f in frames if f.emitter_state == "on"], min_face_px),
        faces_off=_one_face(detector, [f for f in frames if f.emitter_state == "off"], min_face_px),
    )


def _state(guided: GuidedRecorder) -> GuidedState:
    return GuidedState(**vars(guided.snapshot()))


@router.post("/guided", name="v1:recordings:guided:start", status_code=status.HTTP_201_CREATED)
def start_guided(
    request: GuidedStart,
    guided: Annotated[GuidedRecorder, Depends(get_guided)],
) -> GuidedState:
    if not request.consent:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="record only people whose consent is on file",
        )
    try:
        guided.start(
            step=request.step,
            person=request.person,
            session=request.session,
            glasses=request.glasses,
            overwrite=request.overwrite,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except RuntimeError as exc:
        busy = "already running" in str(exc)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT if busy else status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return _state(guided)


@router.get("/guided", name="v1:recordings:guided")
def guided_state(guided: Annotated[GuidedRecorder, Depends(get_guided)]) -> GuidedState:
    return _state(guided)


@router.delete("/guided", name="v1:recordings:guided:cancel")
def cancel_guided(guided: Annotated[GuidedRecorder, Depends(get_guided)]) -> GuidedState:
    guided.cancel()
    return _state(guided)


@router.delete("/{person}/{session}/{take}", name="v1:recordings:delete")
def delete_take(
    person: str,
    session: str,
    take: str,
    root: Annotated[Path, Depends(get_raw_root)],
) -> DeleteTakeResponse:
    if not all(_PART.fullmatch(part) for part in (person, session, take)):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="bad take path")
    take_dir = root / person / session / take
    if not (take_dir / "meta.json").is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="take not found")
    shutil.rmtree(take_dir)
    # Drop the session and person folders once they are empty.
    for parent in (take_dir.parent, take_dir.parent.parent):
        if not any(parent.iterdir()):
            parent.rmdir()
    return DeleteTakeResponse(take=f"{person}/{session}/{take}")
