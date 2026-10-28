"""1:N matching of a face embedding against the enrolled gallery.

The gallery keeps several unit vectors per person: their mean template and the
individual enrolment vectors (PRD 4.5, the individual ones help recall when a
pose was only seen once). A person's score is their best row, so
`scores = gallery @ query` followed by a max per person.

`match_window` decides over the laser-off frames of one attempt. The frames are
averaged into one query, and the per-frame top match must also agree with it,
so a single odd frame cannot carry the decision.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from face.embed import l2_normalise


@dataclass(frozen=True)
class Match:
    person_id: str | None  # None when the best score is below the threshold
    best_id: str  # top person even when below the threshold
    score: float
    scores: dict[str, float]


class Gallery:
    def __init__(self, embedder: str, owners: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(owners) != len(vectors):
            raise ValueError("need one owner per gallery row")
        if len(owners) == 0:
            raise ValueError("the gallery is empty")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("gallery rows must be unit vectors")
        self.embedder = embedder
        self.owners = np.asarray(owners, dtype=str)
        self.vectors = vectors
        self.people = sorted(set(self.owners.tolist()))

    @classmethod
    def from_embeddings(cls, embedder: str, by_person: Mapping[str, Sequence[np.ndarray]]) -> Gallery:
        """Mean template plus every individual vector, for each person."""
        owners: list[str] = []
        rows: list[np.ndarray] = []
        for person in sorted(by_person):
            own = [l2_normalise(v) for v in by_person[person]]
            if not own:
                continue
            owners.append(person)
            rows.append(l2_normalise(np.mean(own, axis=0)))
            owners += [person] * len(own)
            rows += own
        if not rows:
            raise ValueError("the gallery is empty")
        return cls(embedder, owners, np.stack(rows))

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            np.savez(fh, embedder=np.asarray(self.embedder), owners=self.owners, vectors=self.vectors)

    @classmethod
    def load(cls, path: Path) -> Gallery:
        with np.load(path, allow_pickle=False) as data:
            return cls(str(data["embedder"]), data["owners"].tolist(), data["vectors"])

    def scores(self, query: np.ndarray) -> dict[str, float]:
        query = l2_normalise(query)
        if query.shape[0] != self.dim:
            raise ValueError(f"query has {query.shape[0]} dimensions, gallery has {self.dim}")
        row_scores = self.vectors @ query
        return {p: float(row_scores[self.owners == p].max()) for p in self.people}

    def match(self, query: np.ndarray, threshold: float) -> Match:
        scores = self.scores(query)
        best_id = max(scores, key=scores.__getitem__)
        best = scores[best_id]
        return Match(
            person_id=best_id if best >= threshold else None,
            best_id=best_id,
            score=best,
            scores=scores,
        )


@dataclass(frozen=True)
class WindowMatch:
    person_id: str | None
    score: float | None
    frames_used: int
    agreeing: int
    reason: str


def match_window(
    gallery: Gallery,
    vectors: Sequence[np.ndarray],
    *,
    threshold: float,
    frames_required: int,
    agree_frac: float,
) -> WindowMatch:
    """Match the mean of `vectors`; the per-frame top matches must mostly agree."""
    used = len(vectors)
    if used < frames_required:
        return WindowMatch(
            person_id=None,
            score=None,
            frames_used=used,
            agreeing=0,
            reason=f"need {frames_required} clear laser-off faces to match, have {used}",
        )
    overall = gallery.match(l2_normalise(np.mean([l2_normalise(v) for v in vectors], axis=0)), threshold)
    agreeing = sum(gallery.match(v, threshold).best_id == overall.best_id for v in vectors)
    if overall.person_id is None:
        return WindowMatch(
            person_id=None,
            score=overall.score,
            frames_used=used,
            agreeing=agreeing,
            reason=f"not an enrolled person (best score {overall.score:.2f}, needs {threshold:.2f})",
        )
    needed = math.ceil(agree_frac * used)
    if agreeing < needed:
        return WindowMatch(
            person_id=None,
            score=overall.score,
            frames_used=used,
            agreeing=agreeing,
            reason=f"frames disagree on who this is ({agreeing} of {used} agree, needs {needed})",
        )
    return WindowMatch(
        person_id=overall.person_id,
        score=overall.score,
        frames_used=used,
        agreeing=agreeing,
        reason=f"matched {overall.person_id} (score {overall.score:.2f}, {agreeing} of {used} frames agree)",
    )
