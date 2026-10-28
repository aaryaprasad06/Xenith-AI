"""Identity error rates on recorded D435 takes, for any `Embedder`.

Run: uv run python -m eval.far_frr --embedder raw_pixels

Each genuine take (attack "none") becomes one sample: the face is detected on
every frame with the chosen emitter state, aligned to 112×112, embedded, and
the vectors averaged. One take is one attempt at the door, so a take, not a
frame, is the unit of a trial: frames of one take are not independent.

Protocols:
    cross-session  enrol each person on their first session, probe with the
                   others. The honest test; needs 2+ sessions per person.
    cross-take     enrol on every other take of a person, probe with the rest.
                   For dry runs before second sessions exist. Optimistic.

Every probe is searched against the full gallery (a mated search) and against
the gallery without its own person (a non-mated search), so 1:N rates come
from the same data. Results go to `data/eval/<embedder>_<protocol>/`.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from capture.config import load_config
from capture.recorder import DEFAULT_ROOT, Take, iter_takes, load_take
from eval.metrics import (
    eer,
    fnmr,
    identification_rates,
    roc,
    rule_of_three,
    threshold_at_fmr,
)
from face.align import align_crop
from face.detect import Detection
from face.embed import EMBEDDERS, Embedder, l2_normalise

Split = Literal["cross-session", "cross-take"]
Emitter = Literal["on", "off", "any"]
FMR_TARGETS = (1e-2, 1e-3, 1e-4)
FPIR_TARGETS = (1e-2, 1e-3)
DEFAULT_OUT = DEFAULT_ROOT.parent / "eval"


class Detector(Protocol):
    """What `embed_take` needs from `face.detect.FaceDetector`."""

    def detect(self, ir: np.ndarray) -> list[Detection]: ...


@dataclass(frozen=True)
class Sample:
    person: str
    session: str
    take: str
    vector: np.ndarray
    frames_used: int
    frames_seen: int


def embed_take(
    take: Take,
    take_name: str,
    detector: Detector,
    embedder: Embedder,
    *,
    emitter: Emitter,
    min_face_px: float,
) -> Sample | None:
    """Average embedding of the frames with exactly one usable face, or None."""
    vectors = []
    seen = 0
    for frame in take.frames:
        if emitter != "any" and frame.emitter_state != emitter:
            continue
        seen += 1
        faces = [
            face for face in detector.detect(frame.ir)
            if min(face.width, face.height) >= min_face_px
        ]
        if len(faces) != 1:
            continue
        ir_crop, depth_crop = align_crop(frame.ir, frame.depth, faces[0].landmarks)
        vectors.append(embedder.embed(ir_crop, depth_crop).vector)
    if not vectors:
        return None
    return Sample(
        person=take.labels.person,
        session=take.labels.session,
        take=take_name,
        vector=l2_normalise(np.mean(vectors, axis=0)),
        frames_used=len(vectors),
        frames_seen=seen,
    )


def split(samples: Sequence[Sample], protocol: Split) -> tuple[list[Sample], list[Sample]]:
    """(enrol, probe) samples."""
    by_person: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_person[sample.person].append(sample)
    enrol: list[Sample] = []
    probe: list[Sample] = []
    for person in sorted(by_person):
        own = sorted(by_person[person], key=lambda s: (s.session, s.take))
        if protocol == "cross-session":
            first = own[0].session
            enrol += [s for s in own if s.session == first]
            probe += [s for s in own if s.session != first]
        else:
            enrol += own[0::2]
            probe += own[1::2]
    return enrol, probe


def _gallery(enrol: Sequence[Sample]) -> tuple[list[str], np.ndarray]:
    """One template per person: the normalised mean of their enrol samples."""
    by_person: dict[str, list[np.ndarray]] = defaultdict(list)
    for sample in enrol:
        by_person[sample.person].append(sample.vector)
    people = sorted(by_person)
    matrix = np.stack([l2_normalise(np.mean(by_person[p], axis=0)) for p in people])
    return people, matrix


def evaluate(enrol: Sequence[Sample], probe: Sequence[Sample]) -> dict[str, Any]:
    people, gallery = _gallery(enrol)
    probes = [s for s in probe if s.person in people]
    if len(people) < 2:
        raise ValueError(f"need at least 2 enrolled people for impostor scores, have {len(people)}")
    if not probes:
        raise ValueError("no probe takes from an enrolled person (record a second session)")

    index = {p: i for i, p in enumerate(people)}
    scores = np.stack([s.vector for s in probes]) @ gallery.T  # (probes, people)
    own = np.array([index[s.person] for s in probes])
    rows = np.arange(len(probes))

    genuine = scores[rows, own]
    impostor_mask = np.ones_like(scores, dtype=bool)
    impostor_mask[rows, own] = False
    impostor = scores[impostor_mask]
    pairs = {
        tuple(sorted((s.person, other)))
        for s in probes for other in people if other != s.person
    }

    top = scores.argmax(axis=1)
    mated_top_score = scores[rows, top]
    mated_correct = top == own
    non_mated = np.where(impostor_mask, scores, -np.inf).max(axis=1)

    rate, eer_threshold = eer(genuine, impostor)
    verification = []
    for target in FMR_TARGETS:
        threshold = threshold_at_fmr(impostor, target)
        verification.append({
            "fmr_target": target,
            # Below 1/n the data cannot show the target; the threshold then only
            # clears every impostor score seen.
            "resolvable": target * impostor.size >= 1,
            "threshold": threshold,
            "fnmr": fnmr(genuine, threshold),
        })
    identification = []
    for target in FPIR_TARGETS:
        threshold = threshold_at_fmr(non_mated, target)
        fnir, fpir = identification_rates(mated_top_score, mated_correct, non_mated, threshold)
        identification.append({
            "fpir_target": target,
            "resolvable": target * non_mated.size >= 1,
            "threshold": threshold,
            "fnir": fnir,
            "fpir": fpir,
        })

    return {
        "people": len(people),
        "probes": len(probes),
        "genuine_scores": int(genuine.size),
        "impostor_scores": int(impostor.size),
        "impostor_person_pairs": len(pairs),
        "eer": rate,
        "eer_threshold": eer_threshold,
        "rank1_accuracy": float(mated_correct.mean()),
        # 0 false accepts in n independent trials bounds FMR by 3/n at 95%.
        # Pairs of people are the independent unit, not individual scores.
        "fmr_bound_95": rule_of_three(len(pairs)),
        "verification": verification,
        "identification": identification,
        "genuine": genuine.tolist(),
        "impostor": impostor.tolist(),
    }


def _write(out: Path, report: dict[str, Any], samples: Sequence[Sample]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in report.items() if k not in ("genuine", "impostor")}
    (out / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / "roc.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["threshold", "fmr", "fnmr"])
        for point in roc(np.asarray(report["genuine"]), np.asarray(report["impostor"])):
            writer.writerow([f"{point.threshold:.6f}", point.fmr, point.fnmr])
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["kind", "score"])
        writer.writerows(("genuine", s) for s in report["genuine"])
        writer.writerows(("impostor", s) for s in report["impostor"])
    with (out / "takes.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["person", "session", "take", "frames_used", "frames_seen"])
        writer.writerows((s.person, s.session, s.take, s.frames_used, s.frames_seen) for s in samples)


def _print(report: dict[str, Any]) -> None:
    print(
        f"{report['people']} people, {report['probes']} probe takes, "
        f"{report['genuine_scores']} genuine / {report['impostor_scores']} impostor scores, "
        f"{report['impostor_person_pairs']} impostor person pairs"
    )
    print(f"EER {report['eer']:.2%} at {report['eer_threshold']:.3f}   rank-1 {report['rank1_accuracy']:.2%}")
    print(f"FMR upper bound (95%, rule of three): {report['fmr_bound_95']:.2e}")
    for row in report["verification"]:
        note = "" if row["resolvable"] else "  (too few impostors to resolve)"
        print(f"  FMR {row['fmr_target']:.0e}: FNMR {row['fnmr']:.2%} at {row['threshold']:.3f}{note}")
    for row in report["identification"]:
        note = "" if row["resolvable"] else "  (too few searches to resolve)"
        print(
            f"  FPIR {row['fpir_target']:.0e}: FNIR {row['fnir']:.2%} "
            f"(FPIR {row['fpir']:.2%}) at {row['threshold']:.3f}{note}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--embedder", choices=sorted(EMBEDDERS), default="raw_pixels")
    parser.add_argument("--protocol", choices=("cross-session", "cross-take"), default="cross-session")
    parser.add_argument(
        "--emitter",
        choices=("off", "on", "any"),
        default="off",
        help="frames to embed: laser-off has no dot pattern on the face",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    from face.detect import get_detector

    cfg = load_config(args.config)
    min_face_px = float(cfg["quality"]["min_face_px"])
    detector = get_detector()
    embedder = EMBEDDERS[args.embedder]()

    samples: list[Sample] = []
    for take_dir in iter_takes(args.root):
        take = load_take(take_dir)
        if take.labels.attack != "none":
            continue
        name = take_dir.name
        sample = embed_take(take, name, detector, embedder, emitter=args.emitter, min_face_px=min_face_px)
        if sample is None:
            print(f"skipped {take_dir.relative_to(args.root)}: no frame with exactly one face")
            continue
        samples.append(sample)
    if not samples:
        parser.error(f"no usable genuine takes under {args.root}")

    enrol, probe = split(samples, args.protocol)
    try:
        report = evaluate(enrol, probe)
    except ValueError as err:
        parser.error(str(err))
    out = args.out or DEFAULT_OUT / f"{embedder.name}_{args.protocol}_{args.emitter}"
    _write(out, report, samples)
    _print(report)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
