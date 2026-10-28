"""One D435 pipeline shared by the preview stream, authentication and recording.

Mirrors the RealSenseID-REST idea of a single device wrapper behind the routers.
Matching and the door live in `api/services/door.py`, which reads frames from here.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from typing import Literal

import cv2
import pyrealsense2 as rs

from api.schemas import DeviceInfo
from capture.config import load_config
from capture.frame import Frame
from capture.realsense_stream import RealSenseStream

log = logging.getLogger(__name__)


def read_device_info() -> DeviceInfo:
    ctx = rs.context()
    devices = ctx.query_devices()
    if devices.size() == 0:
        raise RuntimeError("no RealSense device connected")
    dev = devices[0]
    product_line = None
    if dev.supports(rs.camera_info.product_line):
        product_line = dev.get_info(rs.camera_info.product_line)
    return DeviceInfo(
        name=dev.get_info(rs.camera_info.name),
        serial=dev.get_info(rs.camera_info.serial_number),
        firmware=dev.get_info(rs.camera_info.firmware_version),
        product_line=product_line,
    )


class CameraHub:
    def __init__(self, *, enabled: bool, config_path: str | None = None) -> None:
        self.enabled = enabled
        self.config_path = config_path
        self.info: DeviceInfo | None = None
        self.error: str | None = None
        self._jpeg: bytes | None = None
        self._rgb_jpeg: bytes | None = None
        self._recent: deque[Frame] = deque(maxlen=16)
        # A take in progress: the stream thread appends here until `_take_count`.
        self._take: list[Frame] | None = None
        self._take_count = 0
        self._take_done = threading.Event()
        # Called from the stream thread with every frame; keep them cheap.
        self._listeners: list[Callable[[Frame], None]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def camera_state(self) -> Literal["streaming", "starting", "disabled", "error"]:
        if not self.enabled:
            return "disabled"
        if self.error:
            return "error"
        if self._jpeg is None:
            return "starting"
        return "streaming"

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            self.info = read_device_info()
        except (RuntimeError, OSError) as exc:
            self.error = str(exc)
            log.warning("camera unavailable: %s", exc)
            return
        self._thread = threading.Thread(
            target=self._run, name="d435-preview", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6)
            self._thread = None

    def latest_jpeg(self, channel: Literal["ir", "rgb"] = "ir") -> bytes | None:
        with self._lock:
            return self._rgb_jpeg if channel == "rgb" else self._jpeg

    def recent_frames(self) -> list[Frame]:
        """Newest frames, oldest first. Includes both laser-on and laser-off."""
        with self._lock:
            return list(self._recent)

    def capture(self, count: int, timeout_s: float) -> list[Frame]:
        """The next `count` frames from the stream, in order, laser-on and laser-off both.

        Raises RuntimeError when the camera is not streaming or a take is already
        recording, and TimeoutError when the frames do not arrive in time.
        """
        if self.camera_state != "streaming":
            raise RuntimeError(self.error or "camera is not streaming")
        with self._lock:
            if self._take is not None:
                raise RuntimeError("a take is already recording")
            self._take = []
            self._take_count = count
            self._take_done.clear()
        done = self._take_done.wait(timeout_s)
        with self._lock:
            frames, self._take = self._take, None
        if not done:
            raise TimeoutError(
                f"got {len(frames)} of {count} frames in {timeout_s:.0f} s"
            )
        return frames

    def add_listener(self, listener: Callable[[Frame], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[Frame], None]) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def _push(self, frame: Frame, jpeg: bytes, rgb_jpeg: bytes | None = None) -> None:
        with self._lock:
            listeners = list(self._listeners)
            self._jpeg = jpeg
            if rgb_jpeg is not None:
                self._rgb_jpeg = rgb_jpeg
            self._recent.append(frame)
            if self._take is not None and len(self._take) < self._take_count:
                self._take.append(frame)
                if len(self._take) == self._take_count:
                    self._take_done.set()
        for listener in listeners:
            listener(frame)

    def _run(self) -> None:
        try:
            cfg = load_config(self.config_path)
            with RealSenseStream(cfg) as stream:
                for frame in stream:
                    if self._stop.is_set():
                        break
                    ok, encoded = cv2.imencode(
                        ".jpg", frame.ir, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                    )
                    if not ok:
                        continue
                    rgb_jpeg = None
                    # RGB is aligned through depth. Laser-on frames have reliable
                    # depth, so using them avoids a flickering colour preview.
                    if frame.rgb is not None and frame.emitter_state == "on":
                        rgb_ok, rgb_encoded = cv2.imencode(
                            ".jpg", frame.rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                        )
                        if rgb_ok:
                            rgb_jpeg = rgb_encoded.tobytes()
                    self._push(frame, encoded.tobytes(), rgb_jpeg)
        except (RuntimeError, OSError) as exc:
            self.error = str(exc)
            log.exception("preview stream stopped")
