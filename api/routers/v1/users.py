import re
import sqlite3
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.deps import get_door, get_raw_root, get_users, require_admin
from api.schemas import (
    DeleteResponse,
    EnrollResponse,
    EnrollTakesRequest,
    EnrollTakesResponse,
    UserRecord,
    UsersResponse,
)
from api.services.door import DoorService
from api.services.users import UserStore

router = APIRouter(
    prefix="/users", tags=["users"], dependencies=[Depends(require_admin)]
)

_USER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")

_RECORDED = "ID added. Face embeddings are not enrolled yet. Record consented face samples to continue."


def _check_user_id(user_id: str) -> str:
    if _USER_ID.fullmatch(user_id) is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="user_id must be 1-100 letters, digits, dot, underscore, or hyphen",
        )
    return user_id


@router.get("", name="v1:users")
def list_users(
    store: Annotated[UserStore, Depends(get_users)],
    door: Annotated[DoorService, Depends(get_door)],
) -> UsersResponse:
    try:
        enrolled = set(door.gallery_people())
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Face gallery is unavailable; check the gallery file.",
        ) from exc
    people = {person.user_id: person for person in store.list_users()}
    for person_id in enrolled:
        people.setdefault(person_id, UserRecord(user_id=person_id))
    for person in people.values():
        person.face_enrolled = person.user_id in enrolled
    return UsersResponse(
        users=sorted(people.values(), key=lambda person: person.user_id)
    )


@router.post("/enroll", name="v1:users:enroll", status_code=status.HTTP_201_CREATED)
def enroll(
    store: Annotated[UserStore, Depends(get_users)],
    user_id: Annotated[str, Query(min_length=1, max_length=100)],
    door: Annotated[DoorService, Depends(get_door)],
) -> EnrollResponse:
    user_id = _check_user_id(user_id)
    try:
        if user_id in door.gallery_people():
            raise HTTPException(status_code=409, detail="user already enrolled")
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Face gallery is unavailable; check the gallery file.",
        ) from exc
    try:
        record = store.enroll(user_id)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="user already enrolled"
        ) from exc
    return EnrollResponse(status="recorded", user_id=record.user_id, message=_RECORDED)


@router.post("/{user_id}/enroll-takes", name="v1:users:enroll-takes")
def enroll_takes(
    user_id: str,
    body: EnrollTakesRequest,
    root: Annotated[Path, Depends(get_raw_root)],
    door: Annotated[DoorService, Depends(get_door)],
) -> EnrollTakesResponse:
    if not body.consent:
        raise HTTPException(
            status_code=422, detail="Confirm this person's consent is on file."
        )
    try:
        count = door.enroll_from_session(root, user_id, body.session)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OSError, KeyError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not enroll face samples; check recordings, model files and gallery storage.",
        ) from exc
    return EnrollTakesResponse(user_id=user_id, takes_used=count)


@router.delete("/{user_id}", name="v1:users:delete")
def delete_user(
    user_id: str,
    store: Annotated[UserStore, Depends(get_users)],
    door: Annotated[DoorService, Depends(get_door)],
) -> DeleteResponse:
    _check_user_id(user_id)
    try:
        revoked = door.revoke_person(user_id)
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Face gallery could not be updated; access was not removed.",
        ) from exc
    removed = store.delete(user_id)
    if not removed and not revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="user not found"
        )
    return DeleteResponse(user_id=user_id)
