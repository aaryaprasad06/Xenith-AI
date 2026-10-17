"""D435 capture: aligned left-IR + depth frames with per-frame emitter tags.

Run the live preview with:  uv run python -m capture.realsense_stream
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self

import cv2
import numpy as np
import pyrealsense2 as rs

from capture.config import load_config
from capture.filters import DepthFilterChain
from capture.frame import (
    EmitterState,
    Frame,
    Intrinsics,
    center_roi_box,
    center_roi_median_m,
)

FRAME_TIMEOUT_MS = 5000
PREVIEW_DEPTH_MIN_M = 0.2
PREVIEW_DEPTH_MAX_M = 1.0


def emitter_state_from_metadata(value: int) -> EmitterState:
    """Map `frame_laser_power_mode` metadata (1 = laser on, 0 = off) to a tag."""
    if value == 1:
        return "on"
    if value == 0:
        return "off"
    raise ValueError(f"unexpected frame_laser_power_mode value: {value}")


def _read_emitter_state(depth_frame: Any) -> EmitterState:
    meta = rs.frame_metadata_value.frame_laser_power_mode
    if not depth_frame.supports_frame_metadata(meta):
        raise RuntimeError(
            "depth frame has no frame_laser_power_mode metadata; cannot tag "
            "emitter state (is the UVC metadata patch / udev setup installed?)"
        )
    return emitter_state_from_metadata(int(depth_frame.get_frame_metadata(meta)))


def _intrinsics_from_profile(profile: Any) -> Intrinsics:
    intr = profile.as_video_stream_profile().get_intrinsics()
    return Intrinsics(
        fx=float(intr.fx),
        fy=float(intr.fy),
        ppx=float(intr.ppx),
        ppy=float(intr.ppy),
        width=int(intr.width),
        height=int(intr.height),
        model=str(intr.model),
        coeffs=tuple(float(c) for c in intr.coeffs),
    )


class RealSenseStream:
    """Context manager that yields `Frame`s from a D435."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._pipeline: rs.pipeline | None = None
        self._align = rs.align(rs.stream.infrared)
        self._filters = DepthFilterChain(cfg["filters"])
        self._rgb_enabled = bool(cfg["camera"].get("rgb", False))
        self.depth_scale: float = 0.0
        self.intrinsics: Intrinsics | None = None

    def __enter__(self) -> Self:
        cam = self._cfg["camera"]
        w, h, fps = int(cam["width"]), int(cam["height"]), int(cam["fps"])

        config = rs.config()
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        config.enable_stream(rs.stream.infrared, 1, w, h, rs.format.y8, fps)
        if self._rgb_enabled:
            config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)

        pipeline = rs.pipeline()
        resolved = config.resolve(rs.pipeline_wrapper(pipeline))
        depth_sensor = resolved.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.depth_units, float(cam["depth_units"]))
        limit_us = cam.get("auto_exposure_limit_us")
        if limit_us is not None:
            # Alternating the laser confuses auto-exposure: it lengthens the
            # shutter on the dark frame, then the laser-on frame clips to white.
            depth_sensor.set_option(rs.option.auto_exposure_limit_toggle, 1)
            depth_sensor.set_option(rs.option.auto_exposure_limit, float(limit_us))
        if self._cfg["emitter"].get("alternate", False):
            self._enable_alternating_emitter(depth_sensor)

        profile = pipeline.start(config)
        self._pipeline = pipeline
        self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        self.intrinsics = _intrinsics_from_profile(profile.get_stream(rs.stream.infrared, 1))
        return self

    def __exit__(self, *exc: object) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

    @staticmethod
    def _enable_alternating_emitter(depth_sensor: Any) -> None:
        depth_sensor.set_option(rs.option.emitter_enabled, 1)
        try:
            depth_sensor.set_option(rs.option.emitter_on_off, 1)
        except RuntimeError:
            # librealsense issue 11908: emitter_on_off can be refused while
            # auto-exposure is on. Freeze exposure at its current value, retry.
            exposure = depth_sensor.get_option(rs.option.exposure)
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
            depth_sensor.set_option(rs.option.exposure, exposure)
            depth_sensor.set_option(rs.option.emitter_on_off, 1)

    def __iter__(self) -> Iterator[Frame]:
        if self._pipeline is None or self.intrinsics is None:
            raise RuntimeError("RealSenseStream must be used as a context manager")
        while True:
            yield self._next_frame()

    def _next_frame(self) -> Frame:
        assert self._pipeline is not None and self.intrinsics is not None
        frames = self._pipeline.wait_for_frames(FRAME_TIMEOUT_MS)

        # Metadata must be read from the raw frame, before align/filters.
        emitter = _read_emitter_state(frames.get_depth_frame())

        aligned = self._align.process(frames)
        depth_frame = aligned.get_depth_frame()
        ir_frame = aligned.get_infrared_frame(1)
        if not depth_frame or not ir_frame:
            raise RuntimeError("frameset missing depth or infrared after align")

        if emitter == "on":
            depth_frame = self._filters.process(depth_frame)

        ir = np.asanyarray(ir_frame.get_data()).copy()
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth = depth_raw.astype(np.float32) * np.float32(self.depth_scale)
        # z16 65535 is the out-of-range sentinel (disparity ~0, and what the
        # farthest-from-around hole fill emits); it is not a measurement.
        depth[depth_raw == np.iinfo(np.uint16).max] = 0.0
        if depth.shape != ir.shape:
            raise RuntimeError(f"depth {depth.shape} and IR {ir.shape} shapes differ")

        rgb = None
        rgb_timestamp = None
        if self._rgb_enabled:
            # align maps colour into the IR viewport using this frame's depth,
            # so pixels with no depth come out black.
            color_frame = aligned.get_color_frame()
            if not color_frame:
                raise RuntimeError("frameset missing colour after align")
            rgb = np.asanyarray(color_frame.get_data()).copy()
            rgb_timestamp = float(color_frame.get_timestamp()) / 1000.0
            if rgb.shape[:2] != ir.shape:
                raise RuntimeError(f"RGB {rgb.shape} and IR {ir.shape} shapes differ")

        return Frame(
            ir=ir,
            depth=depth,
            intrinsics=self.intrinsics,
            timestamp=float(ir_frame.get_timestamp()) / 1000.0,
            emitter_state=emitter,
            rgb=rgb,
            rgb_timestamp=rgb_timestamp,
        )


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    span = PREVIEW_DEPTH_MAX_M - PREVIEW_DEPTH_MIN_M
    norm = np.clip((depth - PREVIEW_DEPTH_MIN_M) / span, 0.0, 1.0)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img[depth <= 0] = 0
    return img


def _put_text(img: np.ndarray, text: str, row: int) -> None:
    cv2.putText(img, text, (10, 24 + 26 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, 24 + 26 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="path to YAML config")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    frac = float(cfg["preview"]["center_box_frac"])

    last_median = float("nan")
    fps = 0.0
    prev_t = time.monotonic()

    with RealSenseStream(cfg) as stream:
        print(f"depth scale: {stream.depth_scale} m/unit")
        for frame in stream:
            now = time.monotonic()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            if frame.emitter_state == "on":
                last_median = center_roi_median_m(frame.depth, frac)

            h, w = frame.ir.shape
            x0, y0, x1, y1 = center_roi_box(w, h, frac)

            ir_view = cv2.cvtColor(frame.ir, cv2.COLOR_GRAY2BGR)
            depth_view = _colorize_depth(frame.depth)
            for view in (ir_view, depth_view):
                cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0), 2)

            median_txt = "n/a" if np.isnan(last_median) else f"{last_median:.3f} m"
            center_ir = frame.ir[y0:y1, x0:x1]
            too_bright = float((center_ir >= 250).mean()) > 0.3
            _put_text(ir_view, f"emitter: {frame.emitter_state}", 0)
            _put_text(ir_view, f"{fps:5.1f} fps", 1)
            _put_text(depth_view, f"center median: {median_txt}", 0)
            if np.isnan(last_median):
                hint = "white wall — IR too bright" if too_bright else "no depth in box"
                _put_text(depth_view, hint, 1)

            cv2.imshow("IR (left)", ir_view)
            cv2.imshow("Depth", depth_view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
