"""Head turn in degrees from the D435 depth at four YuNet landmarks.

The eyes and mouth corners are deprojected to 3D with the laser-on depth. The
face points along the normal of the plane they span. Unlike a 2D estimate from
the image alone, up/down comes out as well as left/right, because depth sees
the forehead tilt towards or away from the camera directly.

Always compare a turn with the person's own straight-ahead turn: the eye
sockets sit behind the mouth corners, so a face looking straight at the camera
does not read exactly zero.

Directions are the person's own: `left` is the person turning to their left,
which points the face towards image +x on the unmirrored camera image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from capture.frame import Intrinsics

# Person-facing directions, counter-clockwise from the person's right, as seen
# on a mirrored preview (screen right is the person's right).
DIRECTIONS = ("right", "up_right", "up", "up_left", "left", "down_left", "down", "down_right")
PATCH_RADIUS = 3  # 7×7 median: small holes at the eyes do not lose the point


@dataclass(frozen=True)
class HeadTurn:
    yaw: float  # degrees, + is the person's left
    pitch: float  # degrees, + is chin up

    def minus(self, neutral: HeadTurn) -> HeadTurn:
        return HeadTurn(self.yaw - neutral.yaw, self.pitch - neutral.pitch)

    def scaled(self, yaw_full: float, pitch_full: float) -> HeadTurn:
        """In units of a full turn per axis, so 1.0 means far enough in that direction."""
        return HeadTurn(self.yaw / yaw_full, self.pitch / pitch_full)

    @property
    def magnitude(self) -> float:
        return math.hypot(self.yaw, self.pitch)

    @property
    def screen_angle(self) -> float:
        """Angle on a mirrored preview, radians in [0, 2π): 0 is screen right, π/2 is up."""
        return math.atan2(self.pitch, -self.yaw) % (2 * math.pi)


def _point(depth: np.ndarray, u: float, v: float, intrinsics: Intrinsics) -> np.ndarray | None:
    image_h, image_w = depth.shape
    x, y = round(u), round(v)
    patch = depth[max(0, y - PATCH_RADIUS) : y + PATCH_RADIUS + 1, max(0, x - PATCH_RADIUS) : x + PATCH_RADIUS + 1]
    valid = patch[patch > 0]
    if not (0 <= x < image_w and 0 <= y < image_h) or valid.size == 0:
        return None
    z = float(np.median(valid))
    return np.array([(u - intrinsics.ppx) / intrinsics.fx * z, (v - intrinsics.ppy) / intrinsics.fy * z, z])


def head_turn(landmarks: np.ndarray, depth: np.ndarray, intrinsics: Intrinsics) -> HeadTurn | None:
    """`landmarks` is YuNet's (5, 2): right eye, left eye, nose, right mouth, left mouth.

    None when depth is missing at any of the four points.
    """
    right_eye, left_eye, _nose, right_mouth, left_mouth = np.asarray(landmarks, dtype=np.float64)
    points = [_point(depth, u, v, intrinsics) for u, v in (right_eye, left_eye, right_mouth, left_mouth)]
    if any(p is None for p in points):
        return None
    eye_r, eye_l, mouth_r, mouth_l = points
    across = eye_l - eye_r  # image +x on a frontal face
    down = (mouth_r + mouth_l) / 2 - (eye_r + eye_l) / 2  # image +y
    normal = np.cross(down, across)  # towards the camera (camera z is forward)
    length = float(np.linalg.norm(normal))
    if length < 1e-9:
        return None
    nx, ny, nz = normal / length
    if nz >= 0:  # facing away from the camera: the landmarks are wrong
        return None
    # Camera axes: x right, y down, z forward. A face turned to its left points to +x.
    yaw = math.degrees(math.atan2(nx, -nz))
    pitch = math.degrees(math.atan2(-ny, -nz))
    return HeadTurn(yaw=yaw, pitch=pitch)


def direction(turn: HeadTurn) -> str:
    """The nearest of the eight `DIRECTIONS` for a turn already relative to neutral."""
    sector = round(turn.screen_angle / (2 * math.pi / len(DIRECTIONS))) % len(DIRECTIONS)
    return DIRECTIONS[sector]
