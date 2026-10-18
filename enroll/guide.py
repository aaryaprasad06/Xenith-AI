"""Guided recording, like a phone's face enrolment: the system says what to do next.

A `circle` step first learns the person's own straight-ahead pose, then asks
them to roll their head round until each of the eight directions in
`face.pose.DIRECTIONS` has enough frames. The other steps hold one pose:
`dark` (room light off) and the spoofs `print` and `screen`.

Frames arrive as laser-on / laser-off pairs, so every saved take still has
both kinds next to each other, as `liveness.session.judge_window` needs.
Each direction becomes one ordinary take that `capture.recorder.load_take`
and `eval.far_frr` read unchanged. No camera or threads here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from capture.frame import Frame
from capture.quality_gate import assess_quality
from capture.recorder import DEFAULT_FRAMES, TakeLabels
from face.detect import Detection
from face.pose import DIRECTIONS, HeadTurn, direction, head_turn

Step = Literal["circle", "dark", "print", "screen"]
Phase = Literal["center", "circle", "still", "done"]
STEPS: tuple[Step, ...] = ("circle", "dark", "print", "screen")

_SPOOF_SUBJECT = {"print": "the printed photo", "screen": "the phone"}
_HINT = {
    "right": "to your right",
    "up_right": "up and to your right",
    "up": "up",
    "up_left": "up and to your left",
    "left": "to your left",
    "down_left": "down and to your left",
    "down": "down",
    "down_right": "down and to your right",
}


@dataclass(frozen=True)
class GuideSettings:
    center_max_deg: float  # how far from neutral still counts as looking straight
    # How far a turn must go to count. Up/down is smaller: necks nod less than they turn.
    reach_yaw_deg: float
    reach_pitch_deg: float
    pairs_per_pose: int
    ticks: int

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> GuideSettings:
        guide = cfg["guide"]
        return cls(
            center_max_deg=float(guide["center_max_deg"]),
            reach_yaw_deg=float(guide["reach_yaw_deg"]),
            reach_pitch_deg=float(guide["reach_pitch_deg"]),
            pairs_per_pose=int(guide["pairs_per_pose"]),
            ticks=int(guide["ticks"]),
        )


@dataclass(frozen=True)
class Pair:
    on: Frame
    off: Frame | None = None

    @property
    def frames(self) -> list[Frame]:
        return [self.on] if self.off is None else [self.on, self.off]


class Pairer:
    """Turns the frame stream into laser-on / laser-off pairs.

    A laser-off frame pairs with the laser-on frame just before it. Once any
    laser-off frame has been seen, a laser-on frame without a partner (one was
    dropped) is skipped; with the emitter not alternating, each laser-on frame
    is its own pair.
    """

    MAX_GAP_S = 0.06  # about two frames at 30 fps

    def __init__(self) -> None:
        self._on: Frame | None = None
        self._alternating = False

    def push(self, frame: Frame) -> Pair | None:
        if frame.emitter_state == "off":
            self._alternating = True
            on, self._on = self._on, None
            if on is not None and 0 < frame.timestamp - on.timestamp <= self.MAX_GAP_S:
                return Pair(on, frame)
            return None
        previous, self._on = self._on, frame
        if previous is not None and not self._alternating:
            return Pair(previous)
        return None


def _sentence(text: str) -> str:
    text = text.strip()
    return text[:1].upper() + text[1:] + ("" if text.endswith(".") else ".")


@dataclass
class Guide:
    step: Step
    person: str
    session: str
    settings: GuideSettings
    quality: dict[str, Any]
    glasses: bool = False
    phase: Phase = field(init=False)
    prompt: str = field(init=False)
    neutral: HeadTurn | None = field(init=False, default=None)
    # The last turn relative to neutral, in full-turn units (1.0 reaches the ring).
    pointer: HeadTurn | None = field(init=False, default=None)
    _baseline: list[tuple[Pair, HeadTurn]] = field(init=False, default_factory=list)
    _buckets: dict[str, list[Pair]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.step not in STEPS:
            raise ValueError(f"step must be one of {STEPS}")
        self.phase = "center" if self.step == "circle" else "still"
        self.prompt = self._no_face_prompt()

    # What will be written, known before recording so clashes can be refused up front.
    def planned_labels(self) -> list[TakeLabels]:
        if self.step == "circle":
            poses = ("center", *DIRECTIONS)
            return [self._labels(pose) for pose in poses]
        return [self._labels("center")]

    def _labels(self, pose: str) -> TakeLabels:
        spoof = self.step in _SPOOF_SUBJECT
        return TakeLabels(
            person=self.person,
            session=self.session,
            pose=pose,
            lighting="dark" if self.step == "dark" else "lit",
            glasses=self.glasses and not spoof,
            attack=self.step if spoof else "none",
        )

    @property
    def still_pairs(self) -> int:
        return DEFAULT_FRAMES // 2

    @property
    def done(self) -> bool:
        return self.phase == "done"

    @property
    def ticks(self) -> list[bool]:
        """Ring ticks, counter-clockwise from screen right on a mirrored preview.

        Each direction owns the ticks in its 45° sector, and they light from the
        sector's middle outwards as that direction's frames are collected, so a
        lit ring means the frames are really there.
        """
        n = self.settings.ticks
        if self.phase == "done":
            return [True] * n
        if self.phase != "circle":
            return [False] * n
        need = self.settings.pairs_per_pose
        lit = [False] * n
        sector = 2 * math.pi / len(DIRECTIONS)
        for k, name in enumerate(DIRECTIONS):
            mine = []
            for i in range(n):
                angle = (i + 0.5) * 2 * math.pi / n
                if round(angle / sector) % len(DIRECTIONS) == k:
                    gap = abs((angle - k * sector + math.pi) % (2 * math.pi) - math.pi)
                    mine.append((gap, i))
            mine.sort()
            fill = min(len(self._buckets.get(name, [])), need) / need
            for _, i in mine[: math.ceil(fill * len(mine))]:
                lit[i] = True
        return lit

    def counts(self) -> dict[str, int]:
        return {name: len(pairs) for name, pairs in self._buckets.items()}

    @property
    def progress(self) -> float:
        need = self.settings.pairs_per_pose
        if self.phase == "done":
            return 1.0
        if self.phase == "center":
            return len(self._baseline) / need
        if self.phase == "still":
            return len(self._buckets.get("center", [])) / self.still_pairs
        have = sum(min(len(self._buckets.get(name, [])), need) for name in DIRECTIONS)
        return have / (need * len(DIRECTIONS))

    def _no_face_prompt(self) -> str:
        if self.step in _SPOOF_SUBJECT:
            return f"Hold {_SPOOF_SUBJECT[self.step]} up to the camera."
        if self.step == "dark":
            return "Turn the room light off and look at the camera."
        return "Put your face in the circle."

    def feed(self, pair: Pair, faces: list[Detection]) -> None:
        """One laser-on frame (with its laser-off partner) and the faces found on it."""
        if self.done:
            return
        if not faces:
            self.prompt = self._no_face_prompt()
            return
        if len(faces) > 1:
            self.prompt = "Only one face, please."
            return
        face = faces[0]
        if self.step in _SPOOF_SUBJECT:
            # A screen or print often has holes in depth, which is the point; only size matters.
            if min(face.width, face.height) < float(self.quality["min_face_px"]):
                self.prompt = f"Move {_SPOOF_SUBJECT[self.step]} closer."
                return
        else:
            report = assess_quality(
                pair.on.depth, x=face.x, y=face.y, width=face.width, height=face.height, quality=self.quality
            )
            if not report.ok:
                self.prompt = "Move closer." if report.reason.startswith("face is too small") else _sentence(report.reason)
                return
        if self.phase == "still":
            self._feed_still(pair)
            return
        turn = head_turn(face.landmarks, pair.on.depth, pair.on.intrinsics)
        if turn is None:
            self.prompt = "Keep your whole face in view of the camera."
            return
        if self.phase == "center":
            self._feed_center(pair, turn)
        else:
            self._feed_circle(pair, turn)

    def _feed_still(self, pair: Pair) -> None:
        held = self._buckets.setdefault("center", [])
        held.append(pair)
        self.prompt = "Hold still."
        if len(held) >= self.still_pairs:
            self.phase = "done"
            self.prompt = "Done."

    def _feed_center(self, pair: Pair, turn: HeadTurn) -> None:
        if self._baseline:
            yaw = float(np.median([t.yaw for _, t in self._baseline]))
            pitch = float(np.median([t.pitch for _, t in self._baseline]))
            if turn.minus(HeadTurn(yaw, pitch)).magnitude > self.settings.center_max_deg:
                self._baseline.clear()
                self.prompt = "Hold still and look straight at the camera."
        elif not self.prompt.startswith("Hold still"):
            self.prompt = "Look straight at the camera and hold still."
        self._baseline.append((pair, turn))
        if len(self._baseline) >= self.settings.pairs_per_pose:
            self.neutral = HeadTurn(
                float(np.median([t.yaw for _, t in self._baseline])),
                float(np.median([t.pitch for _, t in self._baseline])),
            )
            self._buckets["center"] = [p for p, _ in self._baseline]
            self.phase = "circle"
            self.prompt = "Move your head slowly to complete the circle."

    def _feed_circle(self, pair: Pair, turn: HeadTurn) -> None:
        assert self.neutral is not None
        need = self.settings.pairs_per_pose
        rel = turn.minus(self.neutral).scaled(self.settings.reach_yaw_deg, self.settings.reach_pitch_deg)
        self.pointer = rel
        here = direction(rel) if rel.magnitude >= 1.0 else None
        if here is not None:
            bucket = self._buckets.setdefault(here, [])
            if len(bucket) < need:
                bucket.append(pair)
        missing = [name for name in DIRECTIONS if len(self._buckets.get(name, [])) < need]
        if not missing:
            self.phase = "done"
            self.prompt = "Done."
            return
        if here is None:
            # Already turning but short of the ring: say so, rather than repeating the start prompt.
            self.prompt = "A little further." if rel.magnitude >= 0.5 else "Move your head slowly to complete the circle."
            return
        if here in missing:
            self.prompt = "Good. Hold it there a moment." if len(missing) > 2 else "Almost there. Hold it there."
            return
        # This direction is full: point to the nearest one still missing, so the head keeps circling.
        k = DIRECTIONS.index(here)
        steps = len(DIRECTIONS)
        nearest = min(missing, key=lambda name: min((DIRECTIONS.index(name) - k) % steps, (k - DIRECTIONS.index(name)) % steps))
        self.prompt = f"Now turn {_HINT[nearest]}."

    def takes(self) -> list[tuple[TakeLabels, list[Frame]]]:
        """Each filled pose as a take, frames in capture order."""
        out = []
        for labels in self.planned_labels():
            pairs = self._buckets.get(labels.pose, [])
            if pairs:
                out.append((labels, [frame for pair in pairs for frame in pair.frames]))
        return out
