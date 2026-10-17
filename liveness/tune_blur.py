"""Read the blur score live, to choose `quality.blur_min`.

Run: uv run python -m liveness.tune_blur

Shows the sharpness of the face crop on the laser-on IR image, which is what
the quality gate scores, and on the latest laser-off image for comparison.
Hold the face in one state and press a key to tag the readings:

    1 = sharp (still, in focus)    2 = moving    3 = defocused / too close
    q = quit and print the summary

The summary gives min / p5 / median / max per tag, for laser-on and
laser-off. Set `blur_min` above the "moving" and "defocused" maxima and below
the "sharp" minimum.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from capture.config import load_config
from capture.frame import Frame
from capture.quality_gate import sharpness
from capture.realsense_stream import RealSenseStream, _put_text
from face.detect import faces_for_depth_frame, get_detector

TAGS = {ord("1"): "sharp", ord("2"): "moving", ord("3"): "defocused"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="path to YAML config")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    min_face_px = float(cfg["quality"]["min_face_px"])
    blur_min = cfg["quality"].get("blur_min")
    detector = get_detector()

    recent: deque[Frame] = deque(maxlen=12)
    readings: dict[str, list[float]] = {name: [] for name in TAGS.values()}
    off_readings: dict[str, list[float]] = {name: [] for name in TAGS.values()}
    tag = "sharp"
    on_score: float | None = None
    off_score: float | None = None

    with RealSenseStream(cfg) as stream:
        for frame in stream:
            recent.append(frame)
            view = cv2.cvtColor(frame.ir, cv2.COLOR_GRAY2BGR)
            if frame.emitter_state == "on":
                off_ir = next(
                    (f.ir for f in reversed(recent) if f.emitter_state == "off"), None
                )
                faces = faces_for_depth_frame(detector, frame.ir, off_ir, min_face_px)
                on_score = off_score = None
                if len(faces) == 1:
                    f = faces[0]
                    box = (f.x, f.y, f.width, f.height)
                    on_score = sharpness(frame.ir, *box)
                    if off_ir is not None:
                        off_score = sharpness(off_ir, *box)
                        if off_score is not None:
                            off_readings[tag].append(off_score)
                    if on_score is not None:
                        readings[tag].append(on_score)
                    cv2.rectangle(
                        view,
                        (int(f.x), int(f.y)),
                        (int(f.x + f.width), int(f.y + f.height)),
                        (0, 255, 0),
                        2,
                    )
            _put_text(view, f"tag: {tag}  (1 sharp, 2 moving, 3 defocused, q quit)", 0)
            if on_score is None:
                _put_text(view, "need exactly one face", 1)
            else:
                off_text = "n/a" if off_score is None else f"{off_score:.1f}"
                _put_text(view, f"laser-on {on_score:.1f}   laser-off {off_text}", 1)
                if blur_min is not None:
                    verdict = "pass" if on_score >= float(blur_min) else "BLURRY"
                    _put_text(view, f"blur_min {float(blur_min):.1f}: {verdict}", 2)
            cv2.imshow("Blur tuning", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in TAGS:
                tag = TAGS[key]
    cv2.destroyAllWindows()

    for title, table in (("laser-on", readings), ("laser-off", off_readings)):
        print(f"\n{title}")
        print("tag        n      min      p5   median      max")
        for name, values in table.items():
            if values:
                arr = np.asarray(values)
                print(
                    f"{name:<9} {len(arr):4d} {arr.min():8.1f} {np.percentile(arr, 5):8.1f}"
                    f" {np.median(arr):8.1f} {arr.max():8.1f}"
                )


if __name__ == "__main__":
    main()
