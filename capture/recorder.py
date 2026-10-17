"""Record labelled IR + depth + RGB takes for identity and spoof evaluation.

One take is one person, one session, one pose and one lighting condition:

    data/raw/<person>/<session>/<pose>_<lighting>[_glasses]/
        meta.json        labels, intrinsics, and one entry per frame
        0000.npz ...     ir (uint8), depth (float32 m), rgb (uint8 BGR, if on)

Both laser-on and laser-off frames are kept, in capture order, so the
liveness window can be replayed exactly. Record only people whose consent
is on file.

    uv run python -m capture.recorder --person p001 --session s1 \\
        --pose center --lighting lit --consent

Space records a take, q quits.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np

from capture.frame import Frame, Intrinsics

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
POSES = ("center", "left", "right", "up", "down", "up_left", "up_right", "down_left", "down_right")
LIGHTING = ("lit", "dim", "dark")
ATTACKS = ("none", "print", "screen", "ir_print", "projector", "mask")
DEFAULT_FRAMES = 30
_LABEL_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True)
class TakeLabels:
    person: str
    session: str
    pose: str
    lighting: str
    glasses: bool = False
    # A spoof take records the person being impersonated as `person`.
    attack: str = "none"

    def __post_init__(self) -> None:
        for name in ("person", "session"):
            if not _LABEL_RE.match(getattr(self, name)):
                raise ValueError(f"{name} must be 1-64 letters, digits, '_' or '-'")
        if self.pose not in POSES:
            raise ValueError(f"pose must be one of {POSES}")
        if self.lighting not in LIGHTING:
            raise ValueError(f"lighting must be one of {LIGHTING}")
        if self.attack not in ATTACKS:
            raise ValueError(f"attack must be one of {ATTACKS}")

    def take_dir(self, root: Path) -> Path:
        name = f"{self.pose}_{self.lighting}"
        if self.glasses:
            name += "_glasses"
        if self.attack != "none":
            name += f"_{self.attack}"
        return root / self.person / self.session / name


@dataclass(frozen=True)
class Take:
    labels: TakeLabels
    frames: list[Frame]
    recorded_at: str


def save_take(
    root: Path,
    labels: TakeLabels,
    frames: Sequence[Frame],
    *,
    overwrite: bool = False,
) -> Path:
    if not frames:
        raise ValueError("a take needs at least one frame")
    out = labels.take_dir(root)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} already exists")
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, frame in enumerate(frames):
        name = f"{i:04d}.npz"
        arrays = {"ir": frame.ir, "depth": frame.depth}
        if frame.rgb is not None:
            arrays["rgb"] = frame.rgb
        np.savez_compressed(out / name, **arrays)
        entries.append(
            {
                "file": name,
                "timestamp": frame.timestamp,
                "rgb_timestamp": frame.rgb_timestamp,
                "emitter_state": frame.emitter_state,
            }
        )
    meta = {
        "labels": asdict(labels),
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "intrinsics": asdict(frames[0].intrinsics),
        "frames": entries,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def load_take(take_dir: Path) -> Take:
    meta = json.loads((take_dir / "meta.json").read_text(encoding="utf-8"))
    intr = meta["intrinsics"]
    intrinsics = Intrinsics(**{**intr, "coeffs": tuple(intr["coeffs"])})
    frames = []
    for entry in meta["frames"]:
        with np.load(take_dir / entry["file"]) as data:
            frames.append(
                Frame(
                    ir=data["ir"],
                    depth=data["depth"],
                    intrinsics=intrinsics,
                    timestamp=float(entry["timestamp"]),
                    emitter_state=entry["emitter_state"],
                    rgb=data.get("rgb"),
                    rgb_timestamp=entry["rgb_timestamp"],
                )
            )
    return Take(labels=TakeLabels(**meta["labels"]), frames=frames, recorded_at=meta["recorded_at"])


def iter_takes(root: Path) -> list[Path]:
    """Every take directory under `root`, sorted."""
    return sorted(path.parent for path in root.glob("*/*/*/meta.json"))


def main(argv: list[str] | None = None) -> None:
    import cv2

    from capture.config import load_config
    from capture.realsense_stream import RealSenseStream

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--person", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--pose", choices=POSES, default="center")
    parser.add_argument("--lighting", choices=LIGHTING, default="lit")
    parser.add_argument("--glasses", action="store_true")
    parser.add_argument("--attack", choices=ATTACKS, default="none")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--consent",
        action="store_true",
        help="confirm this person's signed consent is on file (required)",
    )
    args = parser.parse_args(argv)
    if not args.consent:
        parser.error("--consent is required: record only people whose consent is on file")

    labels = TakeLabels(
        person=args.person,
        session=args.session,
        pose=args.pose,
        lighting=args.lighting,
        glasses=args.glasses,
        attack=args.attack,
    )
    if labels.take_dir(args.root).exists() and not args.overwrite:
        parser.error(f"{labels.take_dir(args.root)} already exists (use --overwrite)")

    cfg = load_config(args.config)
    mode: Literal["preview", "recording"] = "preview"
    buffer: list[Frame] = []
    with RealSenseStream(cfg) as stream:
        for frame in stream:
            if mode == "recording":
                buffer.append(frame)
                if len(buffer) >= args.frames:
                    out = save_take(args.root, labels, buffer, overwrite=args.overwrite)
                    print(f"saved {len(buffer)} frames to {out}")
                    break
            view = cv2.cvtColor(frame.ir, cv2.COLOR_GRAY2BGR)
            status = f"recording {len(buffer)}/{args.frames}" if mode == "recording" else "space: record"
            cv2.putText(view, f"{labels.pose} {labels.lighting}  {status}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("recorder (IR)", view)
            if frame.rgb is not None:
                cv2.imshow("recorder (RGB)", frame.rgb)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and mode == "preview":
                mode = "recording"
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
