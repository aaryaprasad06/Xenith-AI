import math
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from capture.config import load_config
from capture.frame import Frame, Intrinsics
from capture.recorder import load_take, save_take
from enroll.guide import Guide, GuideSettings, Pair, Pairer
from face.detect import Detection
from face.pose import DIRECTIONS, HeadTurn, direction, head_turn

INTR = Intrinsics(425, 425, 424, 240, 848, 480, "brown_conrady", (0, 0, 0, 0, 0))
SETTINGS = GuideSettings(center_max_deg=6, reach_yaw_deg=18, reach_pitch_deg=12, pairs_per_pose=4, ticks=48)
QUALITY = load_config()["quality"]
# Right eye, left eye, nose, right mouth, left mouth, around the image centre.
LANDMARKS = np.array([[394, 225], [454, 225], [424, 250], [400, 275], [448, 275]], dtype=np.float64)
_clock = iter(range(100_000))


@cache
def _plane_depth(yaw: float, pitch: float, distance: float = 0.4) -> np.ndarray:
    """Depth of a flat face through (0, 0, distance) turned by yaw / pitch degrees."""
    normal = np.array([math.tan(math.radians(yaw)), -math.tan(math.radians(pitch)), -1.0])
    us, vs = np.meshgrid(np.arange(848, dtype=np.float64), np.arange(480, dtype=np.float64))
    rays = np.stack([(us - INTR.ppx) / INTR.fx, (vs - INTR.ppy) / INTR.fy, np.ones_like(us)], axis=-1)
    depth = ((normal[2] * distance) / (rays @ normal)).astype(np.float32)
    depth.setflags(write=False)
    return depth


def _face(size: float = 150.0, landmarks: np.ndarray = LANDMARKS) -> Detection:
    return Detection(x=424 - size / 2, y=240 - size / 2, width=size, height=size, score=0.9, landmarks=landmarks)


def _frame(state: str, depth: np.ndarray, t: float | None = None) -> Frame:
    return Frame(
        ir=np.zeros((480, 848), dtype=np.uint8),
        depth=depth,
        intrinsics=INTR,
        timestamp=float(next(_clock)) / 30 if t is None else t,
        emitter_state=state,
    )


def _pair(yaw: float = 0.0, pitch: float = 0.0, distance: float = 0.4, depth: np.ndarray | None = None) -> Pair:
    depth = _plane_depth(yaw, pitch, distance) if depth is None else depth
    on = _frame("on", depth)
    return Pair(on, _frame("off", depth, t=on.timestamp + 1 / 30))


def _guide(step: str = "circle", **kw) -> Guide:
    return Guide(step=step, person="p001", session="s1", settings=SETTINGS, quality=QUALITY, **kw)


def _towards(name: str, reach: float = 1.4) -> tuple[float, float]:
    """Yaw / pitch degrees that land in `name`, `reach` times the way to the ring."""
    angle = DIRECTIONS.index(name) * math.pi / 4
    return -math.cos(angle) * reach * SETTINGS.reach_yaw_deg, math.sin(angle) * reach * SETTINGS.reach_pitch_deg


def _learn_neutral(guide: Guide) -> None:
    for _ in range(SETTINGS.pairs_per_pose):
        guide.feed(_pair(), [_face()])
    assert guide.phase == "circle"


def _holes() -> np.ndarray:
    depth = np.array(_plane_depth(0, 0))
    depth[215:235, 384:404] = 0  # no depth round the right eye
    return depth


def test_head_turn_reads_degrees_from_depth():
    straight = head_turn(LANDMARKS, _plane_depth(0, 0), INTR)
    assert straight.yaw == pytest.approx(0, abs=0.5)
    assert straight.pitch == pytest.approx(0, abs=0.5)
    left = head_turn(LANDMARKS, _plane_depth(20, 0), INTR)
    assert left.yaw == pytest.approx(20, abs=1)
    up = head_turn(LANDMARKS, _plane_depth(0, 15), INTR)
    assert up.pitch == pytest.approx(15, abs=1)
    assert up.yaw == pytest.approx(0, abs=1)


def test_head_turn_ignores_roll_and_needs_depth_at_the_points():
    centre = LANDMARKS.mean(axis=0)
    angle = math.radians(25)
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    rolled = head_turn((LANDMARKS - centre) @ rot.T + centre, _plane_depth(0, 0), INTR)
    assert rolled.yaw == pytest.approx(0, abs=0.5)
    assert rolled.pitch == pytest.approx(0, abs=0.5)
    assert head_turn(LANDMARKS, _holes(), INTR) is None


def test_directions_follow_the_mirrored_screen():
    assert direction(HeadTurn(0.0, 1.0)) == "up"
    assert direction(HeadTurn(1.0, 0.0)) == "left"  # person's left is screen left
    assert direction(HeadTurn(-1.0, -1.0)) == "down_right"


def test_pairer_pairs_each_laser_off_with_the_laser_on_before_it():
    depth = _plane_depth(0, 0)
    pairer = Pairer()
    on = _frame("on", depth, t=1.0)
    assert pairer.push(on) is None
    pair = pairer.push(_frame("off", depth, t=1.033))
    assert pair is not None and pair.on is on and pair.off is not None
    # A dropped laser-off: the lone laser-on is skipped once alternation is known.
    assert pairer.push(_frame("on", depth, t=1.066)) is None
    assert pairer.push(_frame("on", depth, t=1.132)) is None
    assert pairer.push(_frame("off", depth, t=1.3)) is None  # too long after its laser-on


def test_pairer_without_alternation_uses_every_laser_on_frame():
    depth = _plane_depth(0, 0)
    pairer = Pairer()
    assert pairer.push(_frame("on", depth, t=1.0)) is None
    pair = pairer.push(_frame("on", depth, t=1.033))
    assert pair is not None and pair.off is None


def test_circle_learns_neutral_then_fills_every_direction():
    guide = _guide()
    assert guide.phase == "center"
    _learn_neutral(guide)
    for name in DIRECTIONS:
        for _ in range(SETTINGS.pairs_per_pose + 2):
            guide.feed(_pair(*_towards(name)), [_face()])
    assert guide.done
    assert all(guide.ticks)
    takes = guide.takes()
    assert [labels.pose for labels, _ in takes] == ["center", *DIRECTIONS]
    assert all(len(frames) == 2 * SETTINGS.pairs_per_pose for _, frames in takes)
    assert all(frames[0].emitter_state == "on" and frames[1].emitter_state == "off" for _, frames in takes)


def test_a_plain_nod_fills_up_and_down():
    guide = _guide()
    _learn_neutral(guide)
    for _ in range(SETTINGS.pairs_per_pose):
        guide.feed(_pair(0, 15), [_face()])
        guide.feed(_pair(0, -15), [_face()])
    assert guide.counts()["up"] == SETTINGS.pairs_per_pose
    assert guide.counts()["down"] == SETTINGS.pairs_per_pose


def test_ticks_only_light_for_frames_collected():
    guide = _guide()
    _learn_neutral(guide)
    assert not any(guide.ticks)
    guide.feed(_pair(*_towards("right")), [_face()])
    lit = [i for i, on in enumerate(guide.ticks) if on]
    # One of four pairs for "right": a quarter of its six ticks, nearest screen right.
    assert lit in ([0, 47], [0], [47])
    for _ in range(SETTINGS.pairs_per_pose):
        guide.feed(_pair(*_towards("right")), [_face()])
    assert sorted(i for i, on in enumerate(guide.ticks) if on) == [0, 1, 2, 45, 46, 47]


def test_pointer_and_prompt_when_short_of_the_ring():
    guide = _guide()
    _learn_neutral(guide)
    guide.feed(_pair(*_towards("up", reach=0.7)), [_face()])
    assert guide.prompt == "A little further."
    assert guide.pointer.pitch == pytest.approx(0.7, abs=0.1)
    assert guide.counts().get("up", 0) == 0


def test_circle_holds_then_points_to_the_nearest_missing_direction():
    guide = _guide()
    _learn_neutral(guide)
    guide.feed(_pair(*_towards("right")), [_face()])
    assert guide.prompt == "Good. Hold it there a moment."
    for _ in range(SETTINGS.pairs_per_pose):
        guide.feed(_pair(*_towards("right")), [_face()])
    assert guide.prompt in ("Now turn up and to your right.", "Now turn down and to your right.")
    for name in DIRECTIONS[1:-1]:
        for _ in range(SETTINGS.pairs_per_pose):
            guide.feed(_pair(*_towards(name)), [_face()])
    assert guide.prompt == "Now turn down and to your right."
    assert not guide.done


def test_moving_while_learning_neutral_starts_again():
    guide = _guide()
    for _ in range(SETTINGS.pairs_per_pose - 1):
        guide.feed(_pair(), [_face()])
    guide.feed(_pair(20, 0), [_face()])
    assert guide.phase == "center"
    assert "Hold still" in guide.prompt
    assert guide.progress == pytest.approx(1 / SETTINGS.pairs_per_pose)


def test_guide_prompts_for_no_face_two_faces_small_far_and_missing_depth():
    guide = _guide()
    guide.feed(_pair(), [])
    assert guide.prompt == "Put your face in the circle."
    guide.feed(_pair(), [_face(), _face()])
    assert guide.prompt == "Only one face, please."
    guide.feed(_pair(), [_face(size=40)])
    assert guide.prompt == "Move closer."
    guide.feed(_pair(distance=0.9), [_face()])
    assert guide.prompt.startswith("Too far")
    guide.feed(_pair(depth=_holes()), [_face()])
    assert guide.prompt == "Keep your whole face in view of the camera."
    assert guide.progress == 0


def test_spoof_step_ignores_missing_depth_and_labels_the_attack():
    guide = _guide("screen", glasses=True)
    for _ in range(guide.still_pairs):
        guide.feed(_pair(depth=np.zeros((480, 848), dtype=np.float32)), [_face()])
    assert guide.done
    [(labels, frames)] = guide.takes()
    assert labels.attack == "screen"
    assert labels.glasses is False
    assert labels.take_dir(Path("raw")).name == "center_lit_screen"
    assert len(frames) == 2 * guide.still_pairs


def test_dark_step_saves_a_take_the_recorder_loads(tmp_path):
    guide = _guide("dark")
    for _ in range(guide.still_pairs):
        guide.feed(_pair(), [_face()])
    [(labels, frames)] = guide.takes()
    assert labels.lighting == "dark"
    take = load_take(save_take(tmp_path, labels, frames))
    assert len(take.frames) == 2 * guide.still_pairs


def test_unknown_step_is_refused():
    with pytest.raises(ValueError, match="step"):
        _guide("wave")
