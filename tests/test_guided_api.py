from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.core.config import Settings
from api.main import create_app
from api.services.camera import CameraHub

AUTH = {"Authorization": "Bearer test-token"}


class StreamingHub(CameraHub):
    """Reports streaming, but no frames arrive unless a test pushes them."""

    def __init__(self) -> None:
        super().__init__(enabled=True)
        self._jpeg = b"jpeg"


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


def _streaming(client: TestClient) -> StreamingHub:
    hub = StreamingHub()
    client.app.state.camera = hub
    client.app.state.guided.camera = hub
    return hub


def _start(**overrides) -> dict:
    return {"person": "p001", "session": "s1", "step": "circle", "consent": True, **overrides}


def test_guided_needs_the_admin_token_and_consent(client: TestClient):
    assert client.get("/v1/recordings/guided").status_code == 401
    response = client.post("/v1/recordings/guided", json=_start(consent=False), headers=AUTH)
    assert response.status_code == 422


def test_guided_is_idle_at_first(client: TestClient):
    body = client.get("/v1/recordings/guided", headers=AUTH).json()
    assert body["status"] == "idle"


def test_guided_without_a_camera_is_unavailable(client: TestClient):
    assert client.post("/v1/recordings/guided", json=_start(), headers=AUTH).status_code == 503


def test_guided_starts_listens_and_cancels(client: TestClient):
    hub = _streaming(client)
    response = client.post("/v1/recordings/guided", json=_start(), headers=AUTH)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "running"
    assert body["prompt"] == "Put your face in the circle."
    assert len(body["ticks"]) == 48
    assert hub._listeners
    assert client.post("/v1/recordings/guided", json=_start(), headers=AUTH).status_code == 409

    cancelled = client.delete("/v1/recordings/guided", headers=AUTH).json()
    assert cancelled["status"] == "cancelled"
    assert not hub._listeners


def test_guided_refuses_to_overwrite_existing_takes(client: TestClient, tmp_path: Path):
    _streaming(client)
    (tmp_path / "raw" / "p001" / "s1" / "left_lit").mkdir(parents=True)
    response = client.post("/v1/recordings/guided", json=_start(), headers=AUTH)
    assert response.status_code == 409
    assert "left_lit" in response.json()["detail"]
    # The dark step writes a different take, so it is allowed.
    assert client.post("/v1/recordings/guided", json=_start(step="dark"), headers=AUTH).status_code == 201
    client.delete("/v1/recordings/guided", headers=AUTH)


def test_guided_rejects_bad_labels(client: TestClient):
    _streaming(client)
    assert client.post("/v1/recordings/guided", json=_start(person="../x"), headers=AUTH).status_code == 422
    assert client.post("/v1/recordings/guided", json=_start(step="wave"), headers=AUTH).status_code == 422


def test_listener_queues_whole_pairs_and_keeps_the_newest(tmp_path: Path):
    import numpy as np

    from api.services.guided import GuidedRecorder
    from capture.frame import Frame, Intrinsics

    intr = Intrinsics(425, 425, 424, 240, 848, 480, "brown_conrady", (0, 0, 0, 0, 0))
    recorder = GuidedRecorder(StreamingHub(), tmp_path)
    blank = np.zeros((480, 848), dtype=np.uint8)
    depth = np.zeros((480, 848), dtype=np.float32)
    for index in range(10):
        state = "on" if index % 2 == 0 else "off"
        recorder._on_frame(Frame(ir=blank, depth=depth, intrinsics=intr, timestamp=index / 30, emitter_state=state))
    pairs = [recorder._pairs.get_nowait() for _ in range(recorder._pairs.qsize())]
    assert len(pairs) == 2
    assert [(p.on.timestamp, p.off.timestamp) for p in pairs] == [(6 / 30, 7 / 30), (8 / 30, 9 / 30)]
