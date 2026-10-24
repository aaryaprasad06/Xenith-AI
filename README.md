# RealSense Face-ID

Door-side face authentication with anti-spoofing on an Intel RealSense D435
(IR + depth). The full spec is in [docs/PRD.md](docs/PRD.md).

**Target for now:** this development machine (x86_64 Linux), not the Raspberry
Pi 4 the PRD names. Pi-specific work is deferred; see the status table.

## Status

| PRD step | State |
| --- | --- |
| 1. Environment + capture | Done: live aligned IR + depth, filter chain, emitter tagging |
| 2. Pi build | Deferred: running on this machine for now |
| 3. Detect + align + quality gate | YuNet, 112×112 align, and the quality checks except blur |
| 4. Baseline identity (embed + match) | Not started |
| 5. Geometry liveness | Depth shape gate is in. A printed photo and a phone screen are rejected |
| 6. Door control (API, relay, state machine, log) | API and console are in. No state machine, relay, or log |
| 7-10. Learned liveness, tuning, evaluation, hardening | Not started (Pi-specific parts of step 10 deferred) |

The full task-by-task list is in [docs/CHECKLIST.md](docs/CHECKLIST.md).
The decision log is in [docs/journal.md](docs/journal.md).

Capture details: [docs/capture.md](docs/capture.md). Liveness is below.

## Requirements

- Linux with a RealSense D435 on a USB 3 port
- [uv](https://docs.astral.sh/uv/) (Python 3.12 is pinned in `.python-version`)
- librealsense udev rules, so the camera works without root
  (see the [librealsense Linux guide](https://github.com/realsenseai/librealsense/blob/master/doc/distribution_linux.md))

`pyrealsense2` ships as a wheel for x86_64 Linux, so `uv sync` is all that is
needed here. A Raspberry Pi 4 (aarch64) has no wheel and needs a source build;
that is deferred (PRD, section 7).

## Setup

```bash
uv sync
uv run python -m face.models   # embedding models (SFace, ArcFace, AdaFace weights), sha256-checked
uv run --group export python -m face.adaface_export   # AdaFace to ONNX, once
uv run pytest        # no camera needed
```

## Run the live preview

```bash
uv run python -m capture.realsense_stream
# or with another config file:
uv run python -m capture.realsense_stream --config path/to/other.yaml
```

Two windows open:

- **IR (left)**: the left infrared image, with a green center box, the current
  emitter state (`on` / `off`) and the fps.
- **Depth**: the depth map colourised over a fixed 0.2-1.0 m range, with the
  same center box and `center median: X.XXX m`.

Press `q` in either window to quit.

## Run the liveness check

```bash
uv run python -m liveness.watch
```

Two windows open: the left IR image with the face box and five landmarks, and
the depth image with the decision. Press `q` to quit.

The check uses laser-on depth only. It needs five laser-on frames and at least
one laser-off frame in the same short window. A flat photo or screen should
read `rejected` (relief under 0.5 cm). A face should read `live` when the
relief is at least 1.5 cm and the nose is closer than the cheeks. Those
numbers are the starting thresholds in `configs/default.yaml`, not yet tuned
on this camera.

`live` does not unlock the door. Identity matching is not built.

The same check runs when the console's Authenticate button calls `POST /v1/auth`.
The decision is still `denied`, so a live stranger cannot open the door.

Stand between 0.10 m and 0.60 m, on a black or green wall. A white wall has no
depth, so the check says the face has no depth.

## API and web console

The HTTP API follows the RealSenseID-REST layout: a FastAPI app, `/v1` routers
for auth, users, preview, and device, and one camera wrapper behind them.
The console is an Astro app in `web/`, installed with pnpm. It is not the
RealSenseID NiceGUI, and it does not talk to an F4xx module.

```bash
cp .env.example .env   # set FACE_ID_ADMIN_TOKEN
./dev.sh
```

That starts the API on http://127.0.0.1:8000 and the console on http://127.0.0.1:4321. Ctrl+C stops both. OpenAPI docs: http://127.0.0.1:8000/docs

The first time, install the console packages: `cd web && pnpm install`.

The console proxies `/v1`, `/health`, `/authenticate`, and `/admin` to the API.

The console pairs Bricolage Grotesque headings with Instrument Sans controls,
graphite camera previews, and a yellow primary check action. Guided recording
has an animated face ring driven by capture state and
actual progress. It supports reduced motion, keyboard step selection, camera
fallbacks, and responsive setup forms. Identity matching and door control remain
unavailable.

| Page | What it calls |
| --- | --- |
| Preview | `GET /v1/preview/stream` (MJPEG of the left IR image), `POST /v1/auth` |
| Record | Guided capture: `POST/GET/DELETE /v1/recordings/guided` (head-turn circle, dark, print, screen), and `GET/POST/DELETE /v1/recordings` for single takes. Admin token. Saves the same takes as `capture.recorder` to `data/raw/` (`FACE_ID_RAW`) |
| People | `GET/POST/DELETE /v1/users` with `Authorization: Bearer <token>` |
| Device | `GET /v1/device/info` |

`POST /authenticate`, `POST /admin/enroll`, and `DELETE /admin/person/{id}` are the same operations under the names in the PRD. Enrolment stores the id only. Authenticate runs the geometry liveness check and still returns `denied`, because embeddings are not built yet. A flat scene comes back `rejected`. A face that passes the depth shape comes back `live`, with the door still locked.

### Checking distance against a tape measure

1. Point the camera at a flat **black or green** surface so the center box lies
   entirely on it. A white wall clips the IR image and the median stays `n/a`.
2. Measure from the camera's front glass to the surface with a tape.
3. Compare with `center median`. On this camera that matched the tape from
   10 cm to about 100 cm. Past 100 cm is not checked yet.

The colour picture only spans 0.2–1.0 m, so 10 cm and anything past 1 m look
flat in colour. The number under `center median` is the measurement.

The median updates on emitter-on frames only (see below).

## Use it from your own code

```python
from capture.config import load_config
from capture.frame import center_roi_median_m
from capture.realsense_stream import RealSenseStream

cfg = load_config()                      # configs/default.yaml
with RealSenseStream(cfg) as stream:
    for frame in stream:
        frame.ir              # uint8 (H, W), left infrared
        frame.depth           # float32 (H, W), metres, 0 = invalid
        frame.emitter_state   # "on" or "off"
        frame.timestamp       # device time, seconds
        frame.intrinsics      # fx, fy, ppx, ppy, width, height, model, coeffs
        if frame.emitter_state == "on":
            print(center_roi_median_m(frame.depth, 0.1))
```

## Configuration

All settings are in [configs/default.yaml](configs/default.yaml). Capture reads
these sections:

| Section | Keys | Meaning |
| --- | --- | --- |
| `camera` | `width`, `height`, `fps`, `depth_units`, `auto_exposure_limit_us` | Stream size, depth resolution (`0.0001` = 0.1 mm per unit), and the IR exposure cap |
| `filters` | `decimation`, `spatial`, `temporal`, `hole_filling` | Parameters only; the filter order is fixed in code |
| `emitter` | `alternate` | `true` alternates the IR projector on/off every frame |
| `preview` | `center_box_frac` | Center box size as a fraction of the image (0.1 = 10% per side) |

`quality` and `liveness` are read by the geometry check. `match` and `door`
are required to exist and are not used yet.

## Project layout

```
capture/               stream, filters, quality gate
face/                  YuNet detect, 112×112 align
liveness/              depth shape check and the live overlay
api/                   FastAPI console backend
web/                   Astro console
models/                YuNet ONNX (see docs/CITATIONS.md)
configs/default.yaml
tests/                 pytest, no camera required
docs/                  PRD, checklist, journal
```

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `No device connected` / timeout | Camera not on USB 3, or udev rules missing. Run `rs-enumerate-devices`, or check `lsusb` for `8086:0b07`. |
| `depth frame has no frame_laser_power_mode metadata` | Metadata is not available from the kernel/driver. Install the librealsense udev rules and kernel patches. The code refuses to guess the emitter state. |
| Center median shows `n/a` on a white wall | The IR image has clipped to white, so stereo has no texture to match. Use a darker wall. Black and green walls on this camera measured correctly from 10 cm to about 100 cm. |
| fps well below 30 | USB 2 port or cable, or a busy machine. |
