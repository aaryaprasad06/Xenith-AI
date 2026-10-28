"""The door behind `/authenticate`: state machine, mock relay, access log, gallery.

The gallery file is re-read when it changes on disk, so rebuilding it with
`python -m enroll.from_takes` takes effect without restarting the API. The
embedder named in the gallery is loaded on first use (ArcFace takes a second).
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from api.schemas import AuthResponse
from api.services.camera import CameraHub
from capture.config import load_config
from capture.recorder import TakeLabels, load_take
from control.attempt import Attempt, Detector, Identifier, evaluate_window
from control.relay import MockRelay, Relay
from control.state_machine import DoorController, Evidence, Outcome, State
from eval.far_frr import embed_take
from face.detect import get_detector
from face.embed import EMBEDDERS, Embedder
from face.match import Gallery
from logging_.access_log import AccessEvent, AccessLog

log = logging.getLogger(__name__)
POLL_S = 0.2


class DoorService:
    def __init__(
        self,
        *,
        config_path: Path | None,
        gallery_path: Path,
        access_log_path: Path,
        relay: Relay | None = None,
        detector_factory: Callable[[], Detector] = get_detector,
        embedders: dict[str, Callable[[], Embedder]] = EMBEDDERS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config_path = config_path
        self.gallery_path = gallery_path
        self.access_log = AccessLog(access_log_path)
        self.relay = relay or MockRelay()
        cfg = load_config(config_path)
        self.controller = DoorController.from_config(cfg["door"], self.relay, clock)
        self._detector_factory = detector_factory
        self._embedders = embedders
        self._embedder: Embedder | None = None
        self._identifier: Identifier | None = None
        self._gallery_mtime: float | None = None
        self._attempt_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._poll, name="door-relock", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        # Shutting down leaves the door locked.
        if not self.relay.locked:
            self.relay.lock()

    def _poll(self) -> None:
        while not self._stop.wait(POLL_S):
            self.controller.poll()

    def gallery_people(self) -> list[str]:
        """Enrolled identities, without loading their embedding model."""
        with self._attempt_lock:
            try:
                return Gallery.load(self.gallery_path).people
            except FileNotFoundError:
                return []

    def revoke_person(self, person_id: str) -> bool:
        """Remove every gallery row for one identity before any new attempt."""
        with self._attempt_lock:
            try:
                gallery = Gallery.load(self.gallery_path)
            except FileNotFoundError:
                return False
            keep = gallery.owners != person_id
            if keep.all():
                return False
            if keep.any():
                remaining = Gallery(
                    gallery.embedder,
                    gallery.owners[keep].tolist(),
                    gallery.vectors[keep],
                )
                with tempfile.NamedTemporaryFile(
                    dir=self.gallery_path.parent,
                    prefix=".gallery-",
                    suffix=".npz",
                    delete=False,
                ) as temporary:
                    path = Path(temporary.name)
                try:
                    remaining.save(path)
                    path.replace(self.gallery_path)
                finally:
                    path.unlink(missing_ok=True)
            else:
                self.gallery_path.unlink(missing_ok=True)
            self._identifier = None
            self._gallery_mtime = None
            return True

    def enroll_from_session(self, root: Path, person: str, session: str) -> int:
        """Enroll genuine saved takes, preserving every other gallery identity."""
        TakeLabels(person=person, session=session, pose="center", lighting="lit")
        with self._attempt_lock:
            try:
                current = Gallery.load(self.gallery_path)
            except FileNotFoundError:
                current = None
            name = current.embedder if current is not None else "arcface_r50"
            if name not in self._embedders:
                raise ValueError(f"gallery uses unknown embedder {name}")
            if self._embedder is None or self._embedder.name != name:
                self._embedder = self._embedders[name]()
            detector = self._detector_factory()
            min_face_px = float(load_config(self.config_path)["quality"]["min_face_px"])
            vectors = []
            for meta in sorted((root / person / session).glob("*/meta.json")):
                take = load_take(meta.parent)
                if take.labels.person != person or take.labels.session != session:
                    raise ValueError(
                        "recording labels do not match the selected person/session"
                    )
                if take.labels.attack != "none":
                    continue
                sample = embed_take(
                    take,
                    meta.parent.name,
                    detector,
                    self._embedder,
                    emitter="off",
                    min_face_px=min_face_px,
                )
                if sample is not None:
                    vectors.append(sample.vector)
            if not vectors:
                raise ValueError(
                    "No usable face samples in this session. Record clear genuine takes first."
                )
            updated = Gallery.from_embeddings(name, {person: vectors})
            if current is not None:
                keep = current.owners != person
                updated = Gallery(
                    name,
                    current.owners[keep].tolist() + updated.owners.tolist(),
                    np.concatenate([current.vectors[keep], updated.vectors]),
                )
            self.gallery_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=self.gallery_path.parent,
                prefix=".gallery-",
                suffix=".npz",
                delete=False,
            ) as temporary:
                path = Path(temporary.name)
            try:
                updated.save(path)
                path.replace(self.gallery_path)
            finally:
                path.unlink(missing_ok=True)
            self._identifier = None
            self._gallery_mtime = None
            return len(vectors)

    def identifier(self, match_cfg: dict) -> tuple[Identifier | None, str]:
        """The matcher for the current gallery file, or None and why not."""
        try:
            mtime = self.gallery_path.stat().st_mtime
        except FileNotFoundError:
            self._identifier, self._gallery_mtime = None, None
            return None, "no one is enrolled (no gallery file)"
        if self._identifier is None or mtime != self._gallery_mtime:
            gallery = Gallery.load(self.gallery_path)
            if gallery.embedder not in self._embedders:
                return None, f"gallery uses unknown embedder {gallery.embedder}"
            if self._embedder is None or self._embedder.name != gallery.embedder:
                self._embedder = self._embedders[gallery.embedder]()
            self._identifier = Identifier(gallery, self._embedder, match_cfg)
            self._gallery_mtime = mtime
            log.info(
                "gallery loaded: %d people, %s", len(gallery.people), gallery.embedder
            )
        return self._identifier, ""

    def authenticate(self, camera: CameraHub) -> AuthResponse:
        # One attempt at a time: the counter and the relay must see them in order.
        with self._attempt_lock:
            started = time.perf_counter()
            attempt, outcome = self._attempt(camera)
            response = AuthResponse(
                status="locked_out"
                if outcome.path == (State.COOLDOWN,)
                else attempt.status,
                decision=outcome.decision,
                user_id=attempt.evidence.person_id
                if outcome.decision == "granted"
                else None,
                reason=outcome.reason,
                state=outcome.state.value,
                failures=outcome.failures,
                relief_m=attempt.relief_m,
                nose_cheek_m=attempt.nose_cheek_m,
                match_score=attempt.match_score,
            )
            self.access_log.record(
                AccessEvent(
                    ts=AccessLog.now(),
                    decision=response.decision,
                    status=response.status,
                    reason=response.reason,
                    state=response.state,
                    path=",".join(s.value for s in outcome.path),
                    liveness_geom=attempt.relief_m,
                    liveness_model=None,
                    match_score=attempt.match_score,
                    person_id=attempt.evidence.person_id,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            )
            return response

    def _attempt(self, camera: CameraHub) -> tuple[Attempt, Outcome]:
        refused = self.controller.refusal()
        if refused is not None:
            return Attempt(
                status="locked_out", evidence=_nothing(refused.reason)
            ), refused
        if camera.camera_state in {"disabled", "error"}:
            reason = camera.error or "camera is not running"
            attempt = Attempt(status="camera_unavailable", evidence=_nothing(reason))
            return attempt, self.controller.submit(attempt.evidence)
        cfg = load_config(self.config_path or camera.config_path)
        identifier, why_not = self.identifier(cfg["match"])
        attempt = evaluate_window(
            camera.recent_frames(),
            self._detector_factory(),
            cfg,
            identifier,
            no_identifier_reason=why_not,
        )
        return attempt, self.controller.submit(attempt.evidence)


def _nothing(reason: str) -> Evidence:
    return Evidence(False, False, False, False, None, reason)
