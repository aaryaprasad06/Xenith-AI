"""The door strike behind one interface.

On this dev machine `MockRelay` only logs. The GPIO relay for the Pi goes
behind the same interface later. Locked is the safe state: a relay starts
locked, and anything that fails leaves it locked.
"""

from __future__ import annotations

import logging
from typing import Protocol

log = logging.getLogger(__name__)


class Relay(Protocol):
    @property
    def locked(self) -> bool: ...

    def unlock(self) -> None: ...

    def lock(self) -> None: ...


class MockRelay:
    """Logs lock and unlock and keeps the events for tests."""

    def __init__(self) -> None:
        self._locked = True
        self.events: list[str] = []

    @property
    def locked(self) -> bool:
        return self._locked

    def unlock(self) -> None:
        self._locked = False
        self.events.append("unlock")
        log.info("mock relay: UNLOCK")

    def lock(self) -> None:
        self._locked = True
        self.events.append("lock")
        log.info("mock relay: LOCK")
