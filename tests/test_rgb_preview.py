"""The colour preview shares capture without substituting IR or changing takes."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from api.routers.v1.preview import _mjpeg
from api.services import camera as camera_module
from api.services.camera import CameraHub
from capture.recorder import load_take
from tests import test_recordings_api

client = test_recordings_api.client
AUTH = test_recordings_api.AUTH


def test_rgb_preview_does_not_fall_back_to_ir():
    camera = CameraHub(enabled=True)
    camera._push(test_recordings_api._frame(0), b"infrared")
    assert camera.latest_jpeg() == b"infrared"
    assert camera.latest_jpeg("rgb") is None
    camera._push(test_recordings_api._frame(1), b"new infrared", b"colour")
    assert camera.latest_jpeg() == b"new infrared"
    assert camera.latest_jpeg("rgb") == b"colour"
    generator = _mjpeg(camera, "rgb")
    assert next(generator) == b"--frame\r\nContent-Type: image/jpeg\r\n\r\ncolour\r\n"
    generator.close()


def test_rgb_route_reports_missing_camera_or_colour(client):
    assert client.get("/v1/preview/rgb").status_code == 503
    client.app.state.camera = test_recordings_api.FakeHub()
    response = client.get("/v1/preview/rgb")
    assert response.status_code == 503
    assert "RGB" in response.json()["detail"]


def test_capture_encodes_colour_from_laser_on_frames(monkeypatch):
    rgb = np.full((480, 848, 3), (20, 70, 200), dtype=np.uint8)
    frames = [
        replace(test_recordings_api._frame(0), rgb=rgb),
        replace(test_recordings_api._frame(1), rgb=np.zeros_like(rgb)),
    ]

    class Stream:
        def __init__(self, cfg):
            pass

        def __enter__(self):
            return iter(frames)

        def __exit__(self, *exc):
            pass

    monkeypatch.setattr(camera_module, "RealSenseStream", Stream)
    camera = CameraHub(enabled=True)
    camera._run()
    image = cv2.imdecode(
        np.frombuffer(camera.latest_jpeg("rgb"), np.uint8), cv2.IMREAD_COLOR
    )
    assert image.shape == rgb.shape
    assert np.mean(image, axis=(0, 1)) == pytest.approx((20, 70, 200), abs=3)
    assert len(camera.recent_frames()) == 2
    assert camera.latest_jpeg() != camera.latest_jpeg("rgb")


def test_normal_recording_preserves_genuine_ir_depth_and_rgb(client, tmp_path):
    class ColourCamera(test_recordings_api.FakeHub):
        def capture(self, count, timeout_s):
            return [
                replace(
                    frame,
                    rgb=np.full((480, 848, 3), (20, 70, 200), dtype=np.uint8),
                    rgb_timestamp=frame.timestamp,
                )
                for frame in super().capture(count, timeout_s)
            ]

    client.app.state.camera = ColourCamera()
    response = client.post(
        "/v1/recordings",
        json={"person": "p_new", "session": "capture_1", "consent": True, "frames": 6},
        headers=AUTH,
    )
    assert response.status_code == 201
    take = load_take(tmp_path / "raw" / "p_new" / "capture_1" / "center_lit")
    assert take.labels.attack == "none"
    assert take.labels.pose == "center"
    assert len(take.frames) == 6
    assert all(
        frame.rgb is not None and frame.rgb.shape == (480, 848, 3)
        for frame in take.frames
    )
    assert all(
        frame.ir.shape == frame.depth.shape == (480, 848) for frame in take.frames
    )
