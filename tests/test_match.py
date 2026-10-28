"""Matching with known embeddings. No camera and no model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from face.embed import l2_normalise
from face.match import Gallery, match_window


def axis(i: int, dim: int = 8) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


def near(i: int, noise: float = 0.1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalise(axis(i) + noise * rng.standard_normal(8).astype(np.float32))


def gallery() -> Gallery:
    return Gallery.from_embeddings(
        "raw_pixels",
        {"ada": [near(0, seed=1), near(0, seed=2)], "bob": [near(1, seed=3)]},
    )


def test_enrolled_person_matches():
    found = gallery().match(near(0, seed=9), threshold=0.45)
    assert found.person_id == "ada"
    assert found.score > 0.9
    assert set(found.scores) == {"ada", "bob"}


def test_stranger_is_unknown():
    found = gallery().match(axis(5), threshold=0.45)
    assert found.person_id is None
    assert found.score < 0.45


def test_person_score_is_their_best_row():
    # bob is enrolled from two very different poses; a probe like either one matches.
    g = Gallery.from_embeddings("raw_pixels", {"bob": [axis(1), axis(2)]})
    assert g.match(axis(2), threshold=0.9).person_id == "bob"
    # 1 mean template + 2 individual rows
    assert len(g.owners) == 3


def test_gallery_round_trips(tmp_path: Path):
    path = tmp_path / "g.npz"
    gallery().save(path)
    loaded = Gallery.load(path)
    assert loaded.embedder == "raw_pixels"
    assert loaded.people == ["ada", "bob"]
    assert loaded.match(near(1, seed=4), 0.45).person_id == "bob"


def test_query_of_the_wrong_size_is_refused():
    with pytest.raises(ValueError, match="dimensions"):
        gallery().match(np.ones(4, dtype=np.float32), 0.45)


def test_rows_must_be_unit_vectors():
    with pytest.raises(ValueError, match="unit"):
        Gallery("raw_pixels", ["ada"], np.full((1, 8), 2.0, dtype=np.float32))


def test_window_matches_when_frames_agree():
    vectors = [near(0, seed=s) for s in range(4)]
    found = match_window(gallery(), vectors, threshold=0.45, frames_required=3, agree_frac=0.6)
    assert found.person_id == "ada"
    assert found.agreeing == 4


def test_window_needs_enough_frames():
    found = match_window(gallery(), [near(0)], threshold=0.45, frames_required=3, agree_frac=0.6)
    assert found.person_id is None
    assert found.score is None
    assert "need 3" in found.reason


def test_window_refuses_disagreeing_frames():
    # Two frames look like ada, two like bob: the mean picks one, but only half agree.
    vectors = [axis(0), axis(0), axis(1), axis(1)]
    found = match_window(gallery(), vectors, threshold=0.45, frames_required=3, agree_frac=0.6)
    assert found.person_id is None
    assert "disagree" in found.reason


def test_window_says_unknown_for_a_stranger():
    vectors = [axis(6), axis(6), axis(6)]
    found = match_window(gallery(), vectors, threshold=0.45, frames_required=3, agree_frac=0.6)
    assert found.person_id is None
    assert "not an enrolled person" in found.reason
