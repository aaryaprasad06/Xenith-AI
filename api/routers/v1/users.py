import re
import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.deps import get_users, require_admin
from api.schemas import DeleteResponse, EnrollResponse, UsersResponse
from api.services.users import UserStore

router = APIRouter(prefix="/users", tags=["users"], dependencies=[Depends(require_admin)])

_USER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")

_RECORDED = (
    "Id recorded. Face embeddings are not computed yet, so this person cannot be matched."
)


def _check_user_id(user_id: str) -> str:
    if _USER_ID.fullmatch(user_id) is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="user_id must be 1-100 letters, digits, dot, underscore, or hyphen",
        )
    return user_id


@router.get("", name="v1:users")
def list_users(store: Annotated[UserStore, Depends(get_users)]) -> UsersResponse:
    return UsersResponse(users=store.list_users())


@router.post("/enroll", name="v1:users:enroll", status_code=status.HTTP_201_CREATED)
def enroll(
    store: Annotated[UserStore, Depends(get_users)],
    user_id: Annotated[str, Query(min_length=1, max_length=100)],
) -> EnrollResponse:
    user_id = _check_user_id(user_id)
    try:
        record = store.enroll(user_id)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="user already enrolled") from exc
    return EnrollResponse(status="recorded", user_id=record.user_id, message=_RECORDED)


@router.delete("/{user_id}", name="v1:users:delete")
def delete_user(user_id: str, store: Annotated[UserStore, Depends(get_users)]) -> DeleteResponse:
    _check_user_id(user_id)
    if not store.delete(user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user not found")
    return DeleteResponse(user_id=user_id)
