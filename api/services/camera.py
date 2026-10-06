"""One D435 pipeline shared by the preview stream.

Mirrors the RealSenseID-REST idea of a single device wrapper behind the routers.
Face matching is not in this wrapper yet.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
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
        self._recent: deque[Frame] = deque(maxlen=16)
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
        self._thread = threading.Thread(target=self._run, name="d435-preview", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6)
            self._thread = None

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def recent_frames(self) -> list[Frame]:
        """Newest frames, oldest first. Includes both laser-on and laser-off."""
        with self._lock:
            return list(self._recent)

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
                    with self._lock:
                        self._jpeg = encoded.tobytes()
                        self._recent.append(frame)
        except (RuntimeError, OSError) as exc:
            self.error = str(exc)
            log.exception("preview stream stopped")
