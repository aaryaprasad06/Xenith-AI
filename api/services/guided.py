"""Runs one `enroll.guide.Guide` against the shared camera in a worker thread.

The stream thread pairs each laser-off frame with the laser-on frame before it
and drops the pair into a short queue; detection runs here. When detection
falls behind, whole old pairs are dropped, never one frame of a pair, so the
preview never stalls and every pair that arrives is usable.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from api.services.camera import CameraHub
from capture.config import load_config
from capture.frame import Frame
from capture.recorder import save_take
from enroll.guide import Guide, GuideSettings, Pair, Pairer, Step
from face.detect import FaceDetector, faces_for_depth_frame, model_path

log = logging.getLogger(__name__)

Status = Literal["idle", "running", "saving", "done", "failed", "cancelled"]


@dataclass
class GuidedSnapshot:
    status: Status = "idle"
    step: Step | None = None
    phase: str | None = None
    prompt: str = ""
    progress: float = 0.0
    ticks: list[bool] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    saved: list[str] = field(default_factory=list)
    # Pairs the worker judged per second; about 15 when it keeps up with 30 fps.
    pairs_per_s: float = 0.0
    # Where the head points on the mirrored ring: x right, y up, 1.0 at the ring.
    pointer: tuple[float, float] | None = None


class GuidedRecorder:
    def __init__(self, camera: CameraHub, root: Path) -> None:
        self.camera = camera
        self.root = root
        self._lock = threading.Lock()
        self._pairs: queue.Queue[Pair] = queue.Queue(maxsize=2)
        self._pairer = Pairer()
        self._rate = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._snapshot = GuidedSnapshot()
        self._detector: FaceDetector | None = None

    def snapshot(self) -> GuidedSnapshot:
        with self._lock:
            return GuidedSnapshot(**vars(self._snapshot))

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self, *, step: Step, person: str, session: str, glasses: bool, overwrite: bool
    ) -> GuidedSnapshot:
        """Raises ValueError for bad labels, FileExistsError on a clash, RuntimeError when busy or no camera."""
        if self.active:
            raise RuntimeError("a guided recording is already running")
        if self.camera.camera_state != "streaming":
            raise RuntimeError(self.camera.error or "camera is not streaming")
        cfg = load_config(self.camera.config_path)
        guide = Guide(
            step=step,
            person=person,
            session=session,
            settings=GuideSettings.from_config(cfg),
            quality=cfg["quality"],
            glasses=glasses,
        )
        clashes = [labels.take_dir(self.root) for labels in guide.planned_labels()]
        clashes = [path.name for path in clashes if path.exists()]
        if clashes and not overwrite:
            raise FileExistsError(f"already recorded: {', '.join(clashes)}")
        if self._detector is None:
            # Its own detector: the shared one is not safe to use from two threads.
            self._detector = FaceDetector(model_path())
        self._stop.clear()
        with self._pairs.mutex:
            self._pairs.queue.clear()
        self._pairer = Pairer()
        self._rate = 0.0
        self._publish(guide, "running")
        self._thread = threading.Thread(
            target=self._run,
            args=(guide, float(cfg["guide"]["timeout_s"]), overwrite),
            name="guided-recording",
            daemon=True,
        )
        self.camera.add_listener(self._on_frame)
        self._thread.start()
        return self.snapshot()

    def cancel(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def _on_frame(self, frame: Frame) -> None:
        pair = self._pairer.push(frame)
        if pair is None:
            return
        while True:
            try:
                self._pairs.put_nowait(pair)
                return
            except queue.Full:
                try:
                    self._pairs.get_nowait()  # drop the oldest pair, keep the newest
                except queue.Empty:
                    pass

    def _publish(self, guide: Guide, status: Status, saved: list[str] | None = None, prompt: str | None = None) -> None:
        with self._lock:
            self._snapshot = GuidedSnapshot(
                status=status,
                step=guide.step,
                phase=guide.phase,
                prompt=prompt or guide.prompt,
                progress=guide.progress,
                ticks=guide.ticks,
                counts=guide.counts(),
                saved=saved or [],
                pairs_per_s=round(self._rate, 1),
                pointer=None if guide.pointer is None else (-guide.pointer.yaw, guide.pointer.pitch),
            )

    def _run(self, guide: Guide, timeout_s: float, overwrite: bool) -> None:
        assert self._detector is not None
        cfg_quality = guide.quality
        min_face_px = float(cfg_quality["min_face_px"])
        deadline = time.monotonic() + timeout_s
        last = 0.0
        try:
            while not guide.done:
                if self._stop.is_set():
                    self._publish(guide, "cancelled", prompt="Cancelled.")
                    return
                if time.monotonic() > deadline or self.camera.error:
                    reason = self.camera.error or "Took too long. Start again when you are ready."
                    self._publish(guide, "failed", prompt=reason)
                    return
                try:
                    pair = self._pairs.get(timeout=0.25)
                except queue.Empty:
                    continue
                started = time.monotonic()
                off_ir = pair.off.ir if pair.off is not None else None
                faces = faces_for_depth_frame(self._detector, pair.on.ir, off_ir, min_face_px)
                guide.feed(pair, faces)
                # Smoothed rate, limited by both the detector and the 15 pairs/s the camera gives.
                if last:
                    self._rate = 0.8 * self._rate + 0.2 / max(started - last, 1e-3)
                last = started
                self._publish(guide, "running")
            self.camera.remove_listener(self._on_frame)
            self._publish(guide, "saving", prompt="Saving.")
            saved = [
                save_take(self.root, labels, frames, overwrite=overwrite).name for labels, frames in guide.takes()
            ]
            self._publish(guide, "done", saved=saved, prompt=f"Done. Saved {len(saved)} take(s).")
        except Exception as exc:  # the page must see a failure, not a silent stop
            log.exception("guided recording failed")
            self._publish(guide, "failed", prompt=f"Recording failed: {exc}")
        finally:
            self.camera.remove_listener(self._on_frame)
