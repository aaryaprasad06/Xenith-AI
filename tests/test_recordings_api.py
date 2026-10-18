import threading
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.core.config import Settings
from api.main import create_app
from api.services.camera import CameraHub
from capture.frame import Frame, Intrinsics
from capture.recorder import load_take

INTR = Intrinsics(425, 425, 424, 240, 848, 480, "brown_conrady", (0, 0, 0, 0, 0))
AUTH = {"Authorization": "Bearer test-token"}


def _frame(index: int) -> Frame:
    return Frame(
        ir=np.zeros((480, 848), dtype=np.uint8),
        depth=np.full((480, 848), 0.4, dtype=np.float32),
        intrinsics=INTR,
        timestamp=float(index),
        emitter_state="on" if index % 2 == 0 else "off",
    )


class FakeHub(CameraHub):
    """A streaming hub whose capture returns synthetic frames without a thread."""

    def __init__(self) -> None:
        super().__init__(enabled=True)
        self._jpeg = b"jpeg"

    def capture(self, count: int, timeout_s: float) -> list[Frame]:
        return [_frame(i) for i in range(count)]


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings(
        admin_token="test-token",
        camera_enabled=False,
        db_path=tmp_path / "users.db",
        raw_root=tmp_path / "raw",
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def _take(**overrides) -> dict:
    return {"person": "p001", "session": "s1", "pose": "center", "lighting": "lit", "consent": True, **overrides}


def test_recording_routes_require_the_admin_token(client: TestClient):
    assert client.get("/v1/recordings").status_code == 401
    assert client.post("/v1/recordings", json=_take()).status_code == 401


def test_record_needs_consent(client: TestClient):
    response = client.post("/v1/recordings", json=_take(consent=False), headers=AUTH)
    assert response.status_code == 422
    assert "consent" in response.json()["detail"]


def test_record_without_a_camera_is_unavailable(client: TestClient):
    response = client.post("/v1/recordings", json=_take(), headers=AUTH)
    assert response.status_code == 503


def test_record_saves_a_take_the_recorder_can_load(client: TestClient, tmp_path: Path):
    client.app.state.camera = FakeHub()
    response = client.post("/v1/recordings", json=_take(frames=6, attack="print"), headers=AUTH)
    assert response.status_code == 201
    body = response.json()
    assert body["take"]["take"] == "center_lit_print"
    assert body["take"]["laser_on"] == 3
    assert body["take"]["laser_off"] == 3
    assert body["faces_on"] == 0
    assert body["faces_off"] == 0

    take = load_take(tmp_path / "raw" / "p001" / "s1" / "center_lit_print")
    assert len(take.frames) == 6
    assert take.labels.attack == "print"

    listed = client.get("/v1/recordings", params={"person": "p001"}, headers=AUTH).json()
    assert [t["take"] for t in listed["takes"]] == ["center_lit_print"]
    assert client.get("/v1/recordings", params={"person": "p002"}, headers=AUTH).json()["takes"] == []


def test_record_refuses_to_overwrite_unless_asked(client: TestClient):
    client.app.state.camera = FakeHub()
    assert client.post("/v1/recordings", json=_take(frames=2), headers=AUTH).status_code == 201
    assert client.post("/v1/recordings", json=_take(frames=2), headers=AUTH).status_code == 409
    again = client.post("/v1/recordings", json=_take(frames=4, overwrite=True), headers=AUTH)
    assert again.status_code == 201
    assert again.json()["take"]["frames"] == 4


def test_record_rejects_bad_labels(client: TestClient):
    client.app.state.camera = FakeHub()
    assert client.post("/v1/recordings", json=_take(pose="sideways"), headers=AUTH).status_code == 422
    assert client.post("/v1/recordings", json=_take(person="../x"), headers=AUTH).status_code == 422


def test_delete_removes_the_take_and_empty_folders(client: TestClient, tmp_path: Path):
    client.app.state.camera = FakeHub()
    client.post("/v1/recordings", json=_take(frames=2), headers=AUTH)
    response = client.delete("/v1/recordings/p001/s1/center_lit", headers=AUTH)
    assert response.status_code == 200
    assert not (tmp_path / "raw" / "p001").exists()
    assert client.delete("/v1/recordings/p001/s1/center_lit", headers=AUTH).status_code == 404
    assert client.delete("/v1/recordings/p001/s1/..", headers=AUTH).status_code in {404, 422}


def test_options_list_the_recorder_labels(client: TestClient):
    body = client.get("/v1/recordings/options", headers=AUTH).json()
    assert "center" in body["poses"]
    assert "screen" in body["attacks"]
    assert body["default_frames"] == 30


def test_hub_capture_collects_the_next_frames_in_order():
    hub = CameraHub(enabled=True)
    hub._jpeg = b"jpeg"
    hub._push(_frame(0), b"jpeg")  # before the take: not included
    result: list[Frame] = []
    worker = threading.Thread(target=lambda: result.extend(hub.capture(3, timeout_s=5)))
    worker.start()
    while hub._take is None:
        pass
    for index in range(1, 6):
        hub._push(_frame(index), b"jpeg")
    worker.join(timeout=5)
    assert [frame.timestamp for frame in result] == [1.0, 2.0, 3.0]
    assert hub._take is None


def test_hub_capture_times_out_and_resets():
    hub = CameraHub(enabled=True)
    hub._jpeg = b"jpeg"
    with pytest.raises(TimeoutError):
        hub.capture(3, timeout_s=0.05)
    assert hub._take is None
