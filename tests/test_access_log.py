from pathlib import Path

from logging_.access_log import AccessEvent, AccessLog


def event(decision: str, status: str, person: str | None = None) -> AccessEvent:
    return AccessEvent(
        ts=AccessLog.now(),
        decision=decision,
        status=status,
        reason="test",
        state="idle",
        path="face_detected,quality_ok,denied,idle",
        liveness_geom=0.002,
        liveness_model=None,
        match_score=None,
        person_id=person,
        latency_ms=12.5,
    )


def test_every_attempt_is_kept_newest_first(tmp_path: Path):
    log = AccessLog(tmp_path / "access.db")
    first = log.record(event("denied", "rejected"))
    second = log.record(event("granted", "matched", "ada"))
    rows = log.recent()
    assert [r.id for r in rows] == [second, first]
    assert rows[0].person_id == "ada"
    assert rows[1].status == "rejected"
    assert rows[1].liveness_model is None
    assert len(log.recent(limit=1)) == 1


def test_log_survives_reopening(tmp_path: Path):
    AccessLog(tmp_path / "access.db").record(event("denied", "no_face"))
    assert len(AccessLog(tmp_path / "access.db").recent()) == 1
