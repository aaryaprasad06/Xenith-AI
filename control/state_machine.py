"""The door decision as a state machine (PRD section 3).

    IDLE -> FACE_DETECTED -> QUALITY_OK -> LIVENESS_PASS -> MATCHED -> UNLOCKED -> IDLE
                 |             |               |              |
                 +-------------+---------------+--------------+--> DENIED -> IDLE

One attempt is one `submit` with the evidence the pipeline gathered over a
window of frames (`control.attempt`). The machine walks the states that
evidence reaches and records the path.

COOLDOWN is the lockout (disabled when `max_failures` is 0): after
`max_failures` failed attempts in a row it refuses everything for
`cooldown_seconds`. Only failures that say something
about the person count: a face of the wrong shape, an unknown face, or frames
that disagree on who it is. A face too far away, two faces, or a missing gallery
do not count, so a real user standing badly is not locked out. Those attempts
never reach matching, so not counting them gives an attacker nothing to probe.

UNLOCKED re-locks on its own after `unlock_seconds`; `poll` does that and is
called on every access and by the API's background thread.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from control.relay import Relay


class State(StrEnum):
    IDLE = "idle"
    FACE_DETECTED = "face_detected"
    QUALITY_OK = "quality_ok"
    LIVENESS_PASS = "liveness_pass"
    MATCHED = "matched"
    UNLOCKED = "unlocked"
    COOLDOWN = "cooldown"
    DENIED = "denied"


@dataclass(frozen=True)
class Evidence:
    """How far one attempt got. Each flag assumes the ones before it are True."""

    face: bool  # exactly one face over the window
    quality_ok: bool  # framing and depth good enough to judge the shape
    live: bool  # geometry liveness passed
    identity_checked: bool  # the matcher ran (a gallery exists, enough clear frames)
    person_id: str | None  # the matched person, None when unknown
    reason: str


@dataclass(frozen=True)
class Outcome:
    decision: Literal["granted", "denied"]
    state: State  # after the attempt
    path: tuple[State, ...]
    reason: str
    failures: int  # failed attempts in a row that count toward the lockout


class DoorController:
    def __init__(
        self,
        relay: Relay,
        *,
        unlock_seconds: float,
        max_failures: int,
        cooldown_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_failures < 0:
            raise ValueError("max_failures must be non-negative (0 disables lockout)")
        self.relay = relay
        self.unlock_seconds = float(unlock_seconds)
        self.max_failures = int(max_failures)
        self.cooldown_seconds = float(cooldown_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._state = State.IDLE
        self._until = 0.0  # end of UNLOCKED or COOLDOWN
        self._failures = 0
        if not relay.locked:
            relay.lock()

    @classmethod
    def from_config(
        cls,
        door: dict[str, Any],
        relay: Relay,
        clock: Callable[[], float] = time.monotonic,
    ) -> DoorController:
        return cls(
            relay,
            unlock_seconds=door["unlock_seconds"],
            max_failures=door["max_failures"],
            cooldown_seconds=door["cooldown_seconds"],
            clock=clock,
        )

    @property
    def state(self) -> State:
        with self._lock:
            self._expire()
            return self._state

    @property
    def failures(self) -> int:
        with self._lock:
            self._expire()
            return self._failures

    def poll(self) -> None:
        """Re-lock after `unlock_seconds` and end a finished cooldown."""
        with self._lock:
            self._expire()

    def snapshot(self) -> tuple[State, bool, int]:
        """Read state, relay lock and failure count together after expiring timers."""
        with self._lock:
            self._expire()
            return self._state, self.relay.locked, self._failures

    def refusal(self) -> Outcome | None:
        """The answer to any attempt during a lockout, or None when attempts are allowed."""
        with self._lock:
            self._expire()
            return self._refusal()

    def submit(self, evidence: Evidence) -> Outcome:
        with self._lock:
            self._expire()
            refused = self._refusal()
            if refused is not None:
                return refused

            if not evidence.face:
                return self._finish("denied", (), evidence.reason, counts=False)
            path = [State.FACE_DETECTED]
            if not evidence.quality_ok:
                return self._finish("denied", path, evidence.reason, counts=False)
            path.append(State.QUALITY_OK)
            if not evidence.live:
                return self._finish("denied", path, evidence.reason, counts=True)
            path.append(State.LIVENESS_PASS)
            if evidence.person_id is None:
                return self._finish(
                    "denied", path, evidence.reason, counts=evidence.identity_checked
                )
            path.append(State.MATCHED)
            return self._finish("granted", path, evidence.reason, counts=False)

    def _refusal(self) -> Outcome | None:
        if self._state != State.COOLDOWN:
            return None
        left = max(0.0, self._until - self._clock())
        return Outcome(
            decision="denied",
            state=State.COOLDOWN,
            path=(State.COOLDOWN,),
            reason=f"too many failed attempts; try again in {left:.0f} s",
            failures=self._failures,
        )

    def _finish(
        self,
        decision: Literal["granted", "denied"],
        path: list[State] | tuple[State, ...],
        reason: str,
        *,
        counts: bool,
    ) -> Outcome:
        path = list(path)
        now = self._clock()
        if decision == "granted":
            self._failures = 0
            if self.relay.locked:
                self.relay.unlock()
            self._state = State.UNLOCKED
            self._until = now + self.unlock_seconds
            path.append(State.UNLOCKED)
        else:
            if path:
                path.append(State.DENIED)
            if counts:
                self._failures += 1
            if self.max_failures > 0 and self._failures >= self.max_failures:
                # A lockout also closes a door that is still open.
                if not self.relay.locked:
                    self.relay.lock()
                self._state = State.COOLDOWN
                self._until = now + self.cooldown_seconds
                path.append(State.COOLDOWN)
                reason = f"{reason}; {self._failures} failed attempts, locked out for {self.cooldown_seconds:.0f} s"
            elif self._state != State.UNLOCKED:
                self._state = State.IDLE
            if path and path[-1] != self._state:
                path.append(self._state)
        return Outcome(
            decision=decision,
            state=self._state,
            path=tuple(path),
            reason=reason,
            failures=self._failures,
        )

    def _expire(self) -> None:
        if self._state == State.UNLOCKED and self._clock() >= self._until:
            self.relay.lock()
            self._state = State.IDLE
        elif self._state == State.COOLDOWN and self._clock() >= self._until:
            self._failures = 0
            self._state = State.IDLE
