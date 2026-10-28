"""Liveness then matching over a fake window: synthetic depth, a fake detector and embedder."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from api.services.camera import CameraHub
from api.services.door import DoorService
from capture.config import load_config
from capture.frame import Frame
from control.attempt import Identifier, evaluate_window
from control.relay import MockRelay
from face.detect import Detection
from face.embed import Embedding, l2_normalise
from face.match import Gallery
from tests.test_liveness import detection_at, face_depth, intrinsics

H, W = 480, 848
ADA = l2_normalise(np.arange(1, 9, dtype=np.float32))
BOB = l2_normalise(np.arange(8, 0, -1, dtype=np.float32) ** 2)


class FakeDetector:
    """One face at the synthetic box, or wherever `at` says on laser-off images."""

    def __init__(self, off_at: tuple[float, float] | None = None) -> None:
        self.off_at = off_at

    def detect(self, ir: np.ndarray) -> list[Detection]:
        if self.off_at is not None and int(ir[0, 0]) == 1:
            return [detection_at(*self.off_at)]
        return [detection_at()]


class FakeEmbedder:
    name = "fake"

    def __init__(self, vector: np.ndarray = ADA) -> None:
        self.vector = vector

    def embed(
        self, ir_crop: np.ndarray, depth_crop: np.ndarray | None = None
    ) -> Embedding:
        return Embedding(vector=self.vector, quality=1.0)


def window(bulge: float = 0.035) -> list[Frame]:
    """8 laser-on and 8 laser-off frames, alternating, like the camera ring."""
    frames = []
    on_depth = face_depth(bulge)
    for i in range(16):
        on = i % 2 == 0
        ir = np.full((H, W), 0 if on else 1, dtype=np.uint8)
        frames.append(
            Frame(
                ir=ir,
                depth=on_depth if on else np.zeros((H, W), np.float32),
                intrinsics=intrinsics(),
                timestamp=i / 30,
                emitter_state="on" if on else "off",
            )
        )
    return frames


def gallery(people: dict[str, np.ndarray] | None = None) -> Gallery:
    people = people or {"ada": ADA}
    return Gallery.from_embeddings("fake", {p: [v] for p, v in people.items()})


def identifier(g: Gallery, embedder: FakeEmbedder | None = None) -> Identifier:
    return Identifier(g, embedder or FakeEmbedder(), load_config()["match"])


def test_live_enrolled_face_matches():
    attempt = evaluate_window(
        window(), FakeDetector(), load_config(), identifier(gallery())
    )
    assert attempt.status == "matched"
    assert attempt.evidence.person_id == "ada"
    assert attempt.match_score == pytest.approx(1.0)
    assert attempt.relief_m is not None


def test_live_stranger_is_no_match_and_counts():
    g = gallery({"bob": BOB})
    attempt = evaluate_window(window(), FakeDetector(), load_config(), identifier(g))
    assert attempt.status == "no_match"
    assert attempt.evidence.identity_checked
    assert attempt.evidence.person_id is None


def test_flat_print_never_reaches_the_matcher():
    class Exploding(FakeEmbedder):
        def embed(self, ir_crop, depth_crop=None):
            raise AssertionError("a spoof must not be embedded")

    attempt = evaluate_window(
        window(bulge=0.0),
        FakeDetector(),
        load_config(),
        identifier(gallery(), Exploding()),
    )
    assert attempt.status == "rejected"
    assert attempt.evidence.quality_ok
    assert not attempt.evidence.live


def test_no_gallery_is_live_but_denied():
    attempt = evaluate_window(window(), FakeDetector(), load_config(), None)
    assert attempt.status == "no_gallery"
    assert attempt.evidence.live
    assert not attempt.evidence.identity_checked


def test_laser_off_face_elsewhere_is_not_used():
    # The off-frame face is 300 px from the face whose depth was checked.
    detector = FakeDetector(off_at=(64.0, 180.0))
    attempt = evaluate_window(window(), detector, load_config(), identifier(gallery()))
    assert attempt.status == "no_match"
    assert not attempt.evidence.identity_checked
    assert "need 3" in attempt.evidence.reason


def test_gallery_from_another_embedder_is_refused():
    g = Gallery.from_embeddings("arcface_r50", {"ada": [ADA]})
    with pytest.raises(ValueError, match="arcface_r50"):
        identifier(g)


def service(tmp_path: Path, people: dict[str, np.ndarray] | None = None) -> DoorService:
    if people is not None:
        gallery(people).save(tmp_path / "gallery.npz")
    return DoorService(
        config_path=None,
        gallery_path=tmp_path / "gallery.npz",
        access_log_path=tmp_path / "access.db",
        relay=MockRelay(),
        detector_factory=FakeDetector,
        embedders={"fake": FakeEmbedder},
    )


def hub(frames: list[Frame]) -> CameraHub:
    camera = CameraHub(enabled=True)
    camera._recent.extend(frames)
    return camera


def test_door_unlocks_for_a_live_enrolled_face_and_logs_it(tmp_path: Path):
    door = service(tmp_path, {"ada": ADA})
    body = door.authenticate(hub(window()))
    assert body.decision == "granted"
    assert body.user_id == "ada"
    assert body.state == "unlocked"
    assert not door.relay.locked
    [row] = door.access_log.recent()
    assert row.decision == "granted"
    assert row.person_id == "ada"
    assert row.path.endswith("matched,unlocked")
    assert row.liveness_geom is not None


def test_door_denies_repeated_spoofs_without_development_lockout(tmp_path: Path):
    door = service(tmp_path, {"ada": ADA})
    camera = hub(window(bulge=0.0))
    bodies = [door.authenticate(camera) for _ in range(10)]
    assert all(b.decision == "denied" and b.user_id is None for b in bodies)
    assert all(b.status == "rejected" and b.state == "idle" for b in bodies)
    assert door.relay.locked
    assert len(door.access_log.recent()) == len(bodies)
    assert door.authenticate(hub(window())).decision == "granted"


def test_door_without_a_gallery_denies_a_live_face(tmp_path: Path):
    door = service(tmp_path)
    body = door.authenticate(hub(window()))
    assert body.status == "no_gallery"
    assert body.decision == "denied"
    assert "no one is enrolled" in body.reason


def test_door_picks_up_a_rebuilt_gallery(tmp_path: Path):
    door = service(tmp_path, {"bob": BOB})
    assert door.authenticate(hub(window())).status == "no_match"
    gallery({"ada": ADA}).save(tmp_path / "gallery.npz")
    # Same-second rewrites can keep the mtime; force a different one.
    stat = (tmp_path / "gallery.npz").stat()
    os.utime(tmp_path / "gallery.npz", (stat.st_atime, stat.st_mtime + 1))
    assert door.authenticate(hub(window())).decision == "granted"
