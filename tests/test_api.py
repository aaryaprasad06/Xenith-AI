from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.core.config import Settings
from api.main import create_app
from api.routers.v1.auth import authenticate
from api.services.camera import CameraHub
from capture.frame import Frame, Intrinsics


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings(
        admin_token="test-token",
        camera_enabled=False,
        db_path=tmp_path / "users.db",
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_camera_disabled(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "camera": "disabled"}


def test_auth_denies_until_the_matcher_exists(client: TestClient):
    response = client.post("/v1/auth")
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "denied"
    assert body["user_id"] is None
    assert body["status"] == "camera_unavailable"

    alias = client.post("/authenticate")
    assert alias.status_code == 200
    assert alias.json()["decision"] == "denied"


def test_auth_waits_until_frames_exist():
    hub = CameraHub(enabled=True)
    body = authenticate(hub)
    assert body.decision == "denied"
    assert body.status == "no_frame"


def test_auth_says_no_face_on_a_blank_window():
    hub = CameraHub(enabled=True)
    intr = Intrinsics(425, 425, 424, 240, 848, 480, "brown_conrady", (0, 0, 0, 0, 0))
    blank = np.zeros((480, 848), dtype=np.uint8)
    depth = np.full((480, 848), 0.5, dtype=np.float32)
    for index in range(5):
        hub._recent.append(
            Frame(
                ir=blank,
                depth=depth,
                intrinsics=intr,
                timestamp=float(index),
                emitter_state="on",
            )
        )
    hub._recent.append(
        Frame(ir=blank, depth=depth, intrinsics=intr, timestamp=5.0, emitter_state="off")
    )
    body = authenticate(hub)
    assert body.decision == "denied"
    assert body.status == "no_face"


def test_device_and_preview_need_a_camera(client: TestClient):
    assert client.get("/v1/device/info").status_code == 503
    assert client.get("/v1/preview/stream").status_code == 503


def test_user_routes_require_the_admin_token(client: TestClient):
    assert client.get("/v1/users").status_code == 401
    created = client.post(
        "/v1/users/enroll",
        params={"user_id": "ada"},
        headers={"Authorization": "Bearer test-token"},
    )
    assert created.status_code == 201
    assert created.json()["status"] == "recorded"
    assert "embeddings" in created.json()["message"]

    listed = client.get("/v1/users", headers={"Authorization": "Bearer test-token"})
    assert [row["user_id"] for row in listed.json()["users"]] == ["ada"]

    again = client.post(
        "/admin/enroll",
        params={"user_id": "ada"},
        headers={"Authorization": "Bearer test-token"},
    )
    assert again.status_code == 409

    removed = client.delete("/admin/person/ada", headers={"Authorization": "Bearer test-token"})
    assert removed.status_code == 200
    missing = client.delete("/v1/users/ada", headers={"Authorization": "Bearer test-token"})
    assert missing.status_code == 404


def test_bad_user_id_is_rejected(client: TestClient):
    response = client.post(
        "/v1/users/enroll",
        params={"user_id": "../nope"},
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 422
