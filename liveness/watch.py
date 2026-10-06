"""Live geometry liveness on the D435.

Run: uv run python -m liveness.watch

Draws the face box on the left IR image and prints LIVE or the rejection
reason. Press q to quit. This does not unlock anything.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from capture.config import load_config
from capture.frame import Frame
from capture.realsense_stream import RealSenseStream, _colorize_depth, _put_text
from face.detect import Detection, faces_for_depth_frame, get_detector
from liveness.session import judge_window


def _draw_face(view: np.ndarray, faces: list[Detection]) -> None:
    for face in faces:
        x0, y0 = int(face.x), int(face.y)
        x1, y1 = int(face.x + face.width), int(face.y + face.height)
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0), 2)
        for lx, ly in face.landmarks:
            cv2.circle(view, (int(lx), int(ly)), 3, (0, 255, 255), -1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="path to YAML config")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    required = int(cfg["liveness"]["frames_required"])
    min_face_px = float(cfg["quality"]["min_face_px"])
    detector = get_detector()

    recent: deque[Frame] = deque(maxlen=max(12, required * 2))
    on_frames: deque[Frame] = deque(maxlen=required)
    on_faces: deque[list[Detection]] = deque(maxlen=required)
    last_status = "…"
    last_reason = "waiting for a laser-on frame"
    last_relief: float | None = None
    last_faces: list[Detection] = []
    printed = 0

    with RealSenseStream(cfg) as stream:
        for frame in stream:
            recent.append(frame)
            ir_view = cv2.cvtColor(frame.ir, cv2.COLOR_GRAY2BGR)
            depth_view = _colorize_depth(frame.depth)
            _put_text(ir_view, f"emitter: {frame.emitter_state}", 0)

            if frame.emitter_state == "on":
                off_ir = next(
                    (item.ir for item in reversed(recent) if item.emitter_state == "off"),
                    None,
                )
                faces = faces_for_depth_frame(detector, frame.ir, off_ir, min_face_px)
                on_frames.append(frame)
                on_faces.append(faces)
                last_faces = faces
                decision = judge_window(
                    list(on_frames),
                    list(on_faces),
                    saw_emitter_off=any(item.emitter_state == "off" for item in recent),
                    cfg=cfg,
                )
                last_status = decision.status
                last_reason = decision.reason
                last_relief = decision.relief_m
                printed += 1
                if printed % 10 == 0:
                    print(f"{decision.status}: {decision.reason}")

            _draw_face(ir_view, last_faces)
            _draw_face(depth_view, last_faces)
            colour = (0, 255, 0) if last_status == "live" else (0, 0, 255)
            if last_status in {"no_face", "no_frame", "…"}:
                colour = (0, 255, 255)
            cv2.putText(
                ir_view,
                last_status.upper(),
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                colour,
                2,
                cv2.LINE_AA,
            )
            _put_text(depth_view, last_reason[:90], 0)
            if last_relief is not None:
                _put_text(depth_view, f"relief {last_relief * 100:.1f} cm", 1)

            cv2.imshow("Liveness IR", ir_view)
            cv2.imshow("Liveness depth", depth_view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
