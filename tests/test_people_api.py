"""Directory and gallery identities must agree about who has face access."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.deps import get_door, get_raw_root
from capture.recorder import TakeLabels, save_take
from face.match import Gallery
from tests import test_api
from tests.test_attempt import ADA, BOB, gallery, hub, service, window

AUTH = {"Authorization": "Bearer test-token"}
client = test_api.client


def test_lists_gallery_people_without_metadata(client: TestClient):
    gallery({"ada": ADA}).save(client.app.state.settings.gallery_path)
    response = client.get("/v1/users", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["users"] == [
        {"user_id": "ada", "created_at": None, "face_enrolled": True}
    ]
    assert client.get("/v1/users").status_code == 401


def test_lists_metadata_and_faces_once_with_correct_status(client: TestClient):
    gallery({"ada": ADA, "bob": BOB}).save(client.app.state.settings.gallery_path)
    client.app.state.users.enroll("ada")
    client.app.state.users.enroll("carol")
    rows = client.get("/v1/users", headers=AUTH).json()["users"]
    assert [(r["user_id"], r["face_enrolled"]) for r in rows] == [
        ("ada", True),
        ("bob", True),
        ("carol", False),
    ]
    assert rows[0]["created_at"] is not None
    assert rows[1]["created_at"] is None


@pytest.mark.parametrize("route", ["/v1/users/enroll", "/admin/enroll"])
def test_adding_id_does_not_grant_face_access_and_existing_face_is_duplicate(
    client: TestClient, route: str
):
    gallery({"ada": ADA}).save(client.app.state.settings.gallery_path)
    assert (
        client.post(route, params={"user_id": "ada"}, headers=AUTH).status_code == 409
    )
    added = client.post(route, params={"user_id": "carol"}, headers=AUTH)
    assert added.status_code == 201
    rows = client.get("/v1/users", headers=AUTH).json()["users"]
    assert rows[-1]["user_id"] == "carol"
    assert rows[-1]["face_enrolled"] is False


@pytest.mark.parametrize("route", ["/v1/users/ada", "/admin/person/ada"])
def test_delete_revokes_cached_face_access_and_preserves_other_people(
    client: TestClient, tmp_path: Path, route: str
):
    door = service(tmp_path, {"ada": ADA, "bob": BOB})
    client.app.dependency_overrides[get_door] = lambda: door
    client.app.state.camera = hub(window())
    client.app.state.users.enroll("ada")
    assert client.post("/authenticate").json()["decision"] == "granted"

    assert client.delete(route).status_code == 401
    deleted = client.delete(route, headers=AUTH)
    assert deleted.status_code == 200
    assert Gallery.load(door.gallery_path).people == ["bob"]
    assert client.app.state.users.list_users() == []
    assert client.post("/authenticate").json()["decision"] == "denied"
    rows = client.get("/v1/users", headers=AUTH).json()["users"]
    assert [r["user_id"] for r in rows] == ["bob"]


def test_deletes_last_gallery_identity_without_metadata(client: TestClient):
    path = client.app.state.settings.gallery_path
    gallery({"ada": ADA}).save(path)
    assert client.delete("/v1/users/ada", headers=AUTH).status_code == 200
    assert not path.exists()
    assert client.get("/v1/users", headers=AUTH).json()["users"] == []
    assert client.delete("/v1/users/ada", headers=AUTH).status_code == 404


def test_missing_person_does_not_change_the_gallery(client: TestClient):
    path = client.app.state.settings.gallery_path
    gallery({"ada": ADA}).save(path)
    before = path.read_bytes()
    assert client.delete("/v1/users/nobody", headers=AUTH).status_code == 404
    assert path.read_bytes() == before


def test_broken_gallery_does_not_claim_empty_or_delete_metadata(client: TestClient):
    path = client.app.state.settings.gallery_path
    path.write_bytes(b"broken gallery")
    client.app.state.users.enroll("ada")
    assert client.get("/v1/users", headers=AUTH).status_code == 503
    assert client.delete("/v1/users/ada", headers=AUTH).status_code == 503
    assert [r.user_id for r in client.app.state.users.list_users()] == ["ada"]
    assert path.read_bytes() == b"broken gallery"


def test_failed_gallery_write_preserves_access_data(client: TestClient, monkeypatch):
    path = client.app.state.settings.gallery_path
    gallery({"ada": ADA, "bob": BOB}).save(path)
    before = path.read_bytes()
    client.app.state.users.enroll("ada")

    def fail_save(self, destination):
        raise OSError("disk full")

    monkeypatch.setattr(Gallery, "save", fail_save)
    assert client.delete("/v1/users/ada", headers=AUTH).status_code == 503
    assert path.read_bytes() == before
    assert [r.user_id for r in client.app.state.users.list_users()] == ["ada"]
    assert not list(path.parent.glob(".gallery-*.npz"))


def session_enrollment(client, tmp_path):
    door = service(tmp_path, {"bob": BOB})
    client.app.dependency_overrides[get_door] = lambda: door
    root = tmp_path / "raw"
    client.app.dependency_overrides[get_raw_root] = lambda: root
    save_take(root, TakeLabels("ada", "s1", "center", "lit"), window())
    return door


def test_enrolls_saved_session_preserves_others_and_refreshes_matcher(client, tmp_path):
    door = session_enrollment(client, tmp_path)
    client.app.state.camera = hub(window())
    assert client.post("/authenticate").json()["decision"] == "denied"
    url = "/v1/users/ada/enroll-takes"
    body = {"session": "s1", "consent": True}
    assert client.post(url, json=body).status_code == 401
    for _ in range(2):
        response = client.post(url, json=body, headers=AUTH)
        assert response.status_code == 200
        assert response.json() == {
            "user_id": "ada",
            "face_enrolled": True,
            "takes_used": 1,
        }
        updated = Gallery.load(door.gallery_path)
        assert updated.people == ["ada", "bob"]
        assert len(updated.owners) == 4
        rows = client.get("/v1/users", headers=AUTH).json()["users"]
        assert all(row["face_enrolled"] for row in rows)
        assert client.post("/authenticate").json()["user_id"] == "ada"


@pytest.mark.parametrize(
    "session,consent", [("s1", False), ("../s1", True), ("missing", True)]
)
def test_bad_session_enrollment_preserves_gallery(client, tmp_path, session, consent):
    door = session_enrollment(client, tmp_path)
    before = door.gallery_path.read_bytes()
    response = client.post(
        "/v1/users/ada/enroll-takes",
        json={"session": session, "consent": consent},
        headers=AUTH,
    )
    assert response.status_code == 422
    assert door.gallery_path.read_bytes() == before


def test_attack_takes_cannot_enroll_a_face(client, tmp_path):
    door = session_enrollment(client, tmp_path)
    root = tmp_path / "raw"
    save_take(root, TakeLabels("ada", "s2", "center", "lit", attack="print"), window())
    before = door.gallery_path.read_bytes()
    response = client.post(
        "/v1/users/ada/enroll-takes",
        json={"session": "s2", "consent": True},
        headers=AUTH,
    )
    assert response.status_code == 422
    assert door.gallery_path.read_bytes() == before


def test_failed_enrollment_write_preserves_gallery(client, tmp_path, monkeypatch):
    door = session_enrollment(client, tmp_path)
    before = door.gallery_path.read_bytes()

    def fail_save(self, destination):
        raise OSError("disk full")

    monkeypatch.setattr(Gallery, "save", fail_save)
    response = client.post(
        "/v1/users/ada/enroll-takes",
        json={"session": "s1", "consent": True},
        headers=AUTH,
    )
    assert response.status_code == 503
    assert door.gallery_path.read_bytes() == before
    assert not list(door.gallery_path.parent.glob(".gallery-*.npz"))
