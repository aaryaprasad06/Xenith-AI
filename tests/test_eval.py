from pathlib import Path

import numpy as np
import pytest

from capture.frame import Frame, Intrinsics
from capture.recorder import Take, TakeLabels
from eval.far_frr import Sample, embed_take, evaluate, split
from eval.metrics import (
    eer,
    fmr,
    fnmr,
    identification_rates,
    rule_of_three,
    threshold_at_fmr,
)
from face.align import TEMPLATE_112
from face.detect import Detection
from face.embed import RawPixelEmbedder, l2_normalise

GENUINE = np.array([0.9, 0.8, 0.7, 0.4])
IMPOSTOR = np.array([0.1, 0.2, 0.3, 0.5, 0.6])


def test_fmr_and_fnmr_count_scores_at_the_threshold_as_accepted():
    assert fmr(IMPOSTOR, 0.5) == pytest.approx(2 / 5)
    assert fnmr(GENUINE, 0.7) == pytest.approx(1 / 4)


def test_threshold_at_fmr_meets_the_target():
    t = threshold_at_fmr(IMPOSTOR, 0.2)
    assert fmr(IMPOSTOR, t) <= 0.2
    assert fmr(IMPOSTOR, np.nextafter(0.5, -np.inf)) > 0.2
    # A target finer than 1/n can only clear every impostor.
    assert fmr(IMPOSTOR, threshold_at_fmr(IMPOSTOR, 1e-6)) == 0.0


def test_eer_on_separable_and_overlapping_scores():
    rate, _ = eer(np.array([0.9, 0.8]), np.array([0.1, 0.2]))
    assert rate == 0.0
    rate, _ = eer(GENUINE, IMPOSTOR)
    assert 0.0 < rate < 0.5


def test_rule_of_three():
    assert rule_of_three(1225) == pytest.approx(3 / 1225)
    assert np.isnan(rule_of_three(0))


def test_identification_rates():
    fnir, fpir = identification_rates(
        np.array([0.9, 0.9, 0.3]), np.array([True, False, True]), np.array([0.2, 0.6]), 0.5
    )
    assert fnir == pytest.approx(2 / 3)  # one wrong person, one below threshold
    assert fpir == pytest.approx(1 / 2)


def _sample(person: str, session: str, take: str, vector: np.ndarray) -> Sample:
    return Sample(person, session, take, l2_normalise(vector), frames_used=3, frames_seen=3)


def _people(n: int, rng: np.random.Generator, noise: float) -> list[Sample]:
    centres = rng.normal(size=(n, 64))
    samples = []
    for i, centre in enumerate(centres):
        for session in ("s1", "s2"):
            for take in ("center_lit", "left_lit"):
                samples.append(
                    _sample(f"p{i:03d}", session, take, centre + noise * rng.normal(size=64))
                )
    return samples


def test_cross_session_split_enrols_on_the_first_session():
    samples = _people(3, np.random.default_rng(0), 0.1)
    enrol, probe = split(samples, "cross-session")
    assert {s.session for s in enrol} == {"s1"}
    assert {s.session for s in probe} == {"s2"}
    enrol, probe = split(samples, "cross-take")
    assert len(enrol) == len(probe) == 6


def test_evaluate_separates_distinct_people():
    samples = _people(5, np.random.default_rng(1), 0.1)
    report = evaluate(*split(samples, "cross-session"))
    assert report["people"] == 5
    assert report["probes"] == 10
    assert report["genuine_scores"] == 10
    assert report["impostor_scores"] == 40
    assert report["impostor_person_pairs"] == 10
    assert report["fmr_bound_95"] == pytest.approx(0.3)
    assert report["eer"] == 0.0
    assert report["rank1_accuracy"] == 1.0
    assert report["verification"][0]["fnmr"] == 0.0


def test_evaluate_on_noise_is_near_chance():
    samples = _people(6, np.random.default_rng(2), 50.0)
    report = evaluate(*split(samples, "cross-session"))
    assert report["eer"] > 0.2


def test_evaluate_needs_two_people_and_a_probe():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="at least 2"):
        evaluate(*split(_people(1, rng, 0.1), "cross-session"))
    one_session = [s for s in _people(2, rng, 0.1) if s.session == "s1"]
    with pytest.raises(ValueError, match="second session"):
        evaluate(*split(one_session, "cross-session"))


class _OneFace:
    def __init__(self, landmarks: np.ndarray) -> None:
        self.landmarks = landmarks

    def detect(self, ir: np.ndarray) -> list[Detection]:
        return [Detection(300, 140, 200, 200, 0.9, self.landmarks)]


def _take(n: int) -> Take:
    intr = Intrinsics(420.0, 420.0, 424.0, 240.0, 848, 480, "brown_conrady", (0, 0, 0, 0, 0))
    rng = np.random.default_rng(4)
    frames = [
        Frame(
            ir=rng.integers(0, 256, (480, 848), dtype=np.uint8),
            depth=np.full((480, 848), 0.4, dtype=np.float32),
            intrinsics=intr,
            timestamp=i / 30,
            emitter_state="on" if i % 2 == 0 else "off",
        )
        for i in range(n)
    ]
    labels = TakeLabels(person="p001", session="s1", pose="center", lighting="lit")
    return Take(labels=labels, frames=frames, recorded_at="2026-10-18T00:00:00+00:00")


def test_embed_take_uses_only_the_chosen_emitter_frames(tmp_path: Path):
    detector = _OneFace(TEMPLATE_112.astype(np.float64) * 2 + [300, 140])
    sample = embed_take(_take(6), "center_lit", detector, RawPixelEmbedder(), emitter="off", min_face_px=80)
    assert sample is not None
    assert (sample.frames_used, sample.frames_seen) == (3, 3)
    assert np.linalg.norm(sample.vector) == pytest.approx(1.0)
    small = embed_take(_take(6), "center_lit", detector, RawPixelEmbedder(), emitter="any", min_face_px=250)
    assert small is None
