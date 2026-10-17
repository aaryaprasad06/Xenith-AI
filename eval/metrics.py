"""Verification and identification error rates, from similarity scores.

Scores are cosine similarities: higher means "same person". A comparison is
accepted when its score is at or above the threshold.

Verification (1:1): FMR is the share of impostor scores accepted, FNMR the
share of genuine scores rejected.

Identification (1:N): a mated search has the person in the gallery, a
non-mated search does not. FNIR is the share of mated searches whose top match
is wrong or below the threshold. FPIR is the share of non-mated searches whose
top match is at or above it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def fmr(impostor: np.ndarray, threshold: float) -> float:
    impostor = np.asarray(impostor, dtype=np.float64)
    return float(np.mean(impostor >= threshold)) if impostor.size else float("nan")


def fnmr(genuine: np.ndarray, threshold: float) -> float:
    genuine = np.asarray(genuine, dtype=np.float64)
    return float(np.mean(genuine < threshold)) if genuine.size else float("nan")


def threshold_at_fmr(impostor: np.ndarray, target: float) -> float:
    """Lowest threshold whose FMR is at most `target`.

    With few impostor scores a small target cannot be resolved; the threshold
    then sits just above the highest impostor score (FMR 0), which is all the
    data can show.
    """
    scores = np.sort(np.asarray(impostor, dtype=np.float64))[::-1]
    if scores.size == 0:
        raise ValueError("no impostor scores")
    allowed = int(np.floor(target * scores.size))
    if allowed >= scores.size:
        return float(scores[-1])
    return float(np.nextafter(scores[allowed], np.inf))


@dataclass(frozen=True)
class RocPoint:
    threshold: float
    fmr: float
    fnmr: float


def roc(genuine: np.ndarray, impostor: np.ndarray) -> list[RocPoint]:
    """One point per distinct score, from the strictest threshold down."""
    thresholds = np.unique(np.concatenate([genuine, impostor]))[::-1]
    return [RocPoint(float(t), fmr(impostor, t), fnmr(genuine, t)) for t in thresholds]


def eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """(equal error rate, threshold), taken where |FMR - FNMR| is smallest."""
    points = roc(np.asarray(genuine), np.asarray(impostor))
    best = min(points, key=lambda p: (abs(p.fmr - p.fnmr), p.fmr + p.fnmr))
    return (best.fmr + best.fnmr) / 2, best.threshold


def rule_of_three(n_trials: int) -> float:
    """95% upper bound on an error rate after 0 errors in `n_trials` independent trials."""
    if n_trials <= 0:
        return float("nan")
    return min(1.0, 3.0 / n_trials)


def identification_rates(
    mated_top_score: np.ndarray,
    mated_top_correct: np.ndarray,
    non_mated_top_score: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    """(FNIR, FPIR) at one threshold."""
    mated_top_score = np.asarray(mated_top_score, dtype=np.float64)
    mated_top_correct = np.asarray(mated_top_correct, dtype=bool)
    non_mated = np.asarray(non_mated_top_score, dtype=np.float64)
    hit = mated_top_correct & (mated_top_score >= threshold)
    fnir = float(1.0 - hit.mean()) if hit.size else float("nan")
    fpir = float(np.mean(non_mated >= threshold)) if non_mated.size else float("nan")
    return fnir, fpir
