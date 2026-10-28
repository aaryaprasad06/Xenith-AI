"""State-machine transitions with a fake clock and the mock relay."""

from __future__ import annotations

from capture.config import load_config
from control.relay import MockRelay
from control.state_machine import DoorController, Evidence, State


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def door(max_failures: int = 3) -> tuple[DoorController, MockRelay, Clock]:
    relay, clock = MockRelay(), Clock()
    controller = DoorController(
        relay,
        unlock_seconds=5,
        max_failures=max_failures,
        cooldown_seconds=60,
        clock=clock,
    )
    return controller, relay, clock


NO_FACE = Evidence(False, False, False, False, None, "no face in the infrared image")
TOO_FAR = Evidence(True, False, False, False, None, "face is too far")
SPOOF = Evidence(True, True, False, False, None, "flat surface")
UNKNOWN = Evidence(True, True, True, True, None, "not an enrolled person")
NO_GALLERY = Evidence(True, True, True, False, None, "no one is enrolled")
ADA = Evidence(True, True, True, True, "ada", "matched ada")


def test_starts_idle_and_locked():
    controller, relay, _ = door()
    assert controller.state == State.IDLE
    assert relay.locked


def test_live_enrolled_person_walks_every_state_and_unlocks():
    controller, relay, _ = door()
    out = controller.submit(ADA)
    assert out.decision == "granted"
    assert out.path == (
        State.FACE_DETECTED,
        State.QUALITY_OK,
        State.LIVENESS_PASS,
        State.MATCHED,
        State.UNLOCKED,
    )
    assert controller.state == State.UNLOCKED
    assert not relay.locked


def test_spoof_is_denied_after_quality():
    controller, relay, _ = door()
    out = controller.submit(SPOOF)
    assert out.decision == "denied"
    assert out.path == (State.FACE_DETECTED, State.QUALITY_OK, State.DENIED, State.IDLE)
    assert out.failures == 1
    assert relay.locked
    assert relay.events == []


def test_unknown_face_is_denied_after_liveness():
    controller, _, _ = door()
    out = controller.submit(UNKNOWN)
    assert out.path[-3:] == (State.LIVENESS_PASS, State.DENIED, State.IDLE)
    assert out.failures == 1


def test_no_face_stays_idle_and_does_not_count():
    controller, _, _ = door()
    out = controller.submit(NO_FACE)
    assert out.decision == "denied"
    assert out.path == ()
    assert out.failures == 0


def test_bad_framing_and_missing_gallery_do_not_count():
    controller, _, _ = door(max_failures=1)
    assert controller.submit(TOO_FAR).failures == 0
    assert controller.submit(NO_GALLERY).failures == 0
    assert controller.state == State.IDLE


def test_relocks_after_unlock_seconds():
    controller, relay, clock = door()
    controller.submit(ADA)
    clock.t += 4.9
    controller.poll()
    assert not relay.locked
    clock.t += 0.2
    controller.poll()
    assert relay.locked
    assert controller.state == State.IDLE
    assert relay.events == ["unlock", "lock"]


def test_a_second_grant_keeps_the_door_open_longer():
    controller, relay, clock = door()
    controller.submit(ADA)
    clock.t += 4
    controller.submit(ADA)
    clock.t += 4
    controller.poll()
    assert not relay.locked
    assert relay.events == ["unlock"]


def test_lockout_after_max_failures_then_cooldown_ends():
    controller, relay, clock = door(max_failures=3)
    controller.submit(SPOOF)
    controller.submit(UNKNOWN)
    out = controller.submit(SPOOF)
    assert out.state == State.COOLDOWN
    assert out.path[-1] == State.COOLDOWN
    assert "locked out for 60 s" in out.reason

    # Even an enrolled person is refused during the cooldown.
    refused = controller.submit(ADA)
    assert refused.decision == "denied"
    assert refused.path == (State.COOLDOWN,)
    assert relay.locked
    assert controller.refusal() is not None

    clock.t += 60
    assert controller.state == State.IDLE
    assert controller.failures == 0
    assert controller.refusal() is None
    assert controller.submit(ADA).decision == "granted"


def test_development_config_allows_retries_after_repeated_failures():
    relay = MockRelay()
    controller = DoorController.from_config(load_config()["door"], relay, clock=Clock())
    for evidence in [SPOOF, UNKNOWN] * 10:
        out = controller.submit(evidence)
        assert out.decision == "denied"
        assert out.state == State.IDLE
        assert relay.locked
        assert controller.refusal() is None
    assert controller.submit(ADA).decision == "granted"
    assert controller.failures == 0


def test_success_resets_the_failure_count():
    controller, _, clock = door(max_failures=2)
    controller.submit(SPOOF)
    controller.submit(ADA)
    clock.t += 10
    out = controller.submit(SPOOF)
    assert out.failures == 1
    assert out.state == State.IDLE


def test_lockout_closes_an_open_door():
    controller, relay, _ = door(max_failures=1)
    controller.submit(ADA)
    out = controller.submit(SPOOF)
    assert out.state == State.COOLDOWN
    assert relay.locked


def test_denial_while_open_leaves_the_door_open():
    controller, relay, _ = door(max_failures=3)
    controller.submit(ADA)
    out = controller.submit(UNKNOWN)
    assert out.state == State.UNLOCKED
    assert out.path[-2:] == (State.DENIED, State.UNLOCKED)
    assert not relay.locked
