# Demo runbook

The current demo uses the D435 and a mock relay on this Linux machine.
The relay logs lock/unlock; it does not drive a physical door.

## Before presenting

1. Connect the D435 over USB 3. Close other camera previews before starting
   the API, which needs exclusive ownership of the camera.
2. Set a local admin token in `.env` instead of the example `change-me`.
3. The gallery was built from the consented `p001/s1` recordings on
   2026-10-28. To rebuild it from the available genuine takes:

   ```bash
   uv run python -m enroll.from_takes --embedder arcface_r50
   ```

4. Start both services from the repository root:

   ```bash
   ./dev.sh
   ```

5. Open <http://127.0.0.1:4321>. Confirm the preview is moving and the health
   message says the camera is streaming. Start with the enrolled person
   facing the camera, centred and still, about 30–50 cm away. Use a dark
   background; a white wall has caused clipped IR and missing depth here.

Click **Authenticate**. The result identifies a granted person or explains a
denial. The **Mock door** indicator polls the actual controller and shows the
automatic re-lock after five seconds.

The same live check can be demonstrated from a terminal:

```bash
curl -sS -X POST http://127.0.0.1:8000/authenticate
curl -sS http://127.0.0.1:8000/v1/auth/state
```

`POST /v1/auth` is the versioned equivalent. `GET /v1/auth/state` returns
`state`, `locked`, `failures`, and `mock_relay`; reading state never unlocks.

## Rehearsal checklist

- Authenticate the enrolled person five times while facing forward. Expect
  a grant with the person ID. Check the mock relay unlocks and re-locks.
- Try a consenting person absent from the gallery. Expect denial. Only
  one identity is currently recorded, so stranger rejection remains unverified.
- Try a phone-screen photo and a printed photo, centred in the same position.
  Expect denial. Check the reason: a framing rejection does not establish
  that geometry rejected the spoof.
- Check lockout after the configured number of counted failures and recovery
  after cooldown. Framing failures do not increment this counter.
- Inspect `GET /v1/auth/log` through the API docs at
  <http://127.0.0.1:8000/docs> using the configured admin bearer token.

Do this before presenting. Passing unit tests and recorded-frame replay do
not establish live-camera reliability.

## What has been verified

On 2026-10-28, the web build passed. Recorded frames were replayed through
`DoorService`, including matching, controller, mock relay and a temporary
access log. Centre lit/dark, right and upward poses granted access. Left and
downward poses were denied by the nose-to-cheek check; keep a forward-facing
pose for the initial demonstration. One downward take contains a severe
depth outlier and needs investigation.

The recorded screen attempt was denied for framing. It does not demonstrate
shape-based spoof rejection in this replay. The gallery and genuine replay
use the same session, so the high match scores check wiring, not recognition
accuracy. No D435 was connected during these checks.

A later live check on 2026-10-28 used D435 `236422073835`, firmware `5.17.3.10`.
`POST /authenticate` granted `p001` with score 0.7235 and 8/8 agreeing frames.
Depth relief was 0.0469 m and nose-to-cheek gap was 0.0233 m. The state endpoint
subsequently reported the relay locked and controller idle. This confirms one
live success; repeated accepts and live spoof denials still need rehearsal.

If the USB camera disconnects, authentication denies. Reconnect it, restart
`./dev.sh`, and refresh the page. Health polling reports camera failure, and
the preview reconnects when the service streams again. Cleanup now preserves
the useful disconnect error instead of replacing it with a pipeline-stop error.

## If the camera is unavailable

```bash
FACE_ID_CAMERA=0 ./dev.sh
```

This demonstrates the console and denial when the camera is unavailable.
It cannot demonstrate live recognition. The camera-free pipeline and door
tests can demonstrate the implemented decision behavior separately:

```bash
uv run pytest -q tests/test_attempt.py tests/test_state_machine.py tests/test_access_log.py
```

Live enrolment, encrypted identity storage, full user deletion, threshold
calibration and measured recognition/spoof error rates remain unfinished.
The People page lists IDs and gallery identities with their face-access status.
Adding an ID does not enroll a face. Delete removes the ID and its gallery
embeddings, revoking future matches; it leaves recorded takes intact. Rebuilding
the gallery from those takes can enroll the person again.
