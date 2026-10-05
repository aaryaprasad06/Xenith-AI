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
| 3. Detect + align + quality gate | Not started |
| 4. Baseline identity (embed + match) | Not started |
| 5. Geometry liveness | Not started |
| 6. Door control (API, relay, state machine, log) | Not started |
| 7-10. Learned liveness, tuning, evaluation, hardening | Not started (Pi-specific parts of step 10 deferred) |

The full task-by-task list is in [docs/CHECKLIST.md](docs/CHECKLIST.md).
The decision log is in [docs/journal.md](docs/journal.md).

Only the `capture/` package exists so far. Details: [docs/capture.md](docs/capture.md).

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

`quality`, `liveness`, `match` and `door` are required to exist but are used by
later steps.

## Project layout

```
capture/
  config.py            load and validate configs/default.yaml
  frame.py             Frame, Intrinsics, center_roi_median_m
  filters.py           depth filter chain
  realsense_stream.py  RealSenseStream and the live preview
configs/default.yaml
tests/                 pytest, no camera required
docs/                  PRD and capture notes
```

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `No device connected` / timeout | Camera not on USB 3, or udev rules missing. Run `rs-enumerate-devices`, or check `lsusb` for `8086:0b07`. |
| `depth frame has no frame_laser_power_mode metadata` | Metadata is not available from the kernel/driver. Install the librealsense udev rules and kernel patches. The code refuses to guess the emitter state. |
| Center median shows `n/a` on a white wall | The IR image has clipped to white, so stereo has no texture to match. Use a darker wall. Black and green walls on this camera measured correctly from 10 cm to about 100 cm. |
| fps well below 30 | USB 2 port or cable, or a busy machine. |
