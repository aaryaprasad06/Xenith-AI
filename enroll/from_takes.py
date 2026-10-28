"""Build the matching gallery from recorded takes. A stand-in until `enroll.py` exists.

Run: uv run python -m enroll.from_takes --embedder arcface_r50

Each genuine take (attack "none") is embedded the way `eval.far_frr` does it
(laser-off frames averaged), and every take becomes one gallery vector of its
person. By default each person is enrolled from their first session only, so
later sessions can still be used to test the door.

Takes exist only for people whose consent is on file (`capture.recorder`
requires `--consent`). The gallery is a plain, unencrypted `.npz` under
`data/`, which git ignores; the encrypted store is `enroll/store.py`, not built.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from capture.config import load_config
from capture.recorder import DEFAULT_ROOT, TakeLabels, iter_takes, load_take
from eval.far_frr import embed_take
from face.embed import EMBEDDERS
from face.match import Gallery

DEFAULT_OUT = DEFAULT_ROOT.parent / "gallery.npz"


def _labels(take_dir: Path) -> TakeLabels:
    meta = json.loads((take_dir / "meta.json").read_text(encoding="utf-8"))
    return TakeLabels(**meta["labels"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--embedder", choices=sorted(EMBEDDERS), default="arcface_r50")
    parser.add_argument("--person", action="append", help="enrol only these people (repeatable)")
    parser.add_argument(
        "--session",
        default=None,
        help="enrol from this session; default is each person's first session",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    from face.detect import get_detector

    cfg = load_config(args.config)
    min_face_px = float(cfg["quality"]["min_face_px"])
    detector = get_detector()
    embedder = EMBEDDERS[args.embedder]()

    # Labels first, from meta.json alone: loading every take's frames twice is slow.
    labelled = [(d, _labels(d)) for d in iter_takes(args.root)]
    labelled = [(d, lab) for d, lab in labelled if lab.attack == "none"]
    if args.person:
        labelled = [(d, lab) for d, lab in labelled if lab.person in args.person]
    first: dict[str, str] = {}
    for _d, lab in labelled:
        first[lab.person] = min(first.get(lab.person, lab.session), lab.session)

    by_person: dict[str, list[np.ndarray]] = defaultdict(list)
    for take_dir, labels in labelled:
        if labels.session != (args.session or first[labels.person]):
            continue
        take = load_take(take_dir)
        sample = embed_take(take, take_dir.name, detector, embedder, emitter="off", min_face_px=min_face_px)
        if sample is None:
            print(f"skipped {take_dir.relative_to(args.root)}: no frame with exactly one face")
            continue
        by_person[labels.person].append(sample.vector)
        print(f"{labels.person} {labels.session} {take_dir.name}: {sample.frames_used} frames")
    if not by_person:
        parser.error(f"no usable genuine takes under {args.root}")

    gallery = Gallery.from_embeddings(embedder.name, by_person)
    gallery.save(args.out)
    counts = ", ".join(f"{p} ({len(by_person[p])} takes)" for p in gallery.people)
    print(f"wrote {args.out}: {embedder.name}, {counts}")


if __name__ == "__main__":
    main()
