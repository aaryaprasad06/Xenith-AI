# Journal

Running log of decisions, what was built, and what the camera actually did.
The spec is [PRD.md](PRD.md). The task list is [CHECKLIST.md](CHECKLIST.md).
How to run the capture code is in the [README](../README.md) and [capture.md](capture.md).

**Project:** realsense-face-id
**Owner:** Alan Francis Santhosh
**Camera:** Intel RealSense D435, serial `236422073835`, firmware `5.17.3.10`, USB id `8086:0b07`
**Machine:** this x86_64 Linux laptop. Raspberry Pi 4 is the PRD's eventual target and is out of scope for now.

---

## 2026-10-05 — Start

The repo was an empty `uv` app. No git commits. `src/realsense_face_id/__init__.py` only printed hello. Python was already pinned to 3.12. Dependencies already present: `numpy`, `opencv-python`, `pyrealsense2` 2.58.4, `pyyaml`, plus `pytest` and `ruff`.

The spec used was `Downloads/Face-ID PRD (1).md` (PRD v0.3), plus the librealsense repo and the GitHub `d435` topic. The PRD is a 10-step door system: capture, face detect, anti-spoof, identity match, enrolment, unlock, log. Scope chosen for this session was **step 1 only** (the camera stream), not the rest of the pipeline.

## 2026-10-05 — What was built

`pyproject.toml` is a non-package app (`[tool.uv] package = false`) so `uv run python -m capture.realsense_stream` works from the repo root. The hello `src/` package was removed. Python stays 3.12. The PRD's 3.11 note is only for a future Pi image.

| File | Role |
| --- | --- |
| `capture/config.py` | Loads `configs/default.yaml`. Requires `camera`, `quality`, `liveness`, `match`, `door`, `filters`, `emitter`, `preview`. |
| `capture/frame.py` | `Frame`, `Intrinsics`, `center_roi_median_m`. |
| `capture/filters.py` | Depth filter chain. Order is fixed. |
| `capture/realsense_stream.py` | Pipeline, emitter tags, live preview. |
| `configs/default.yaml` | All thresholds. |
| `tests/` | 27 tests. No camera needed. |
| `docs/PRD.md` | Spec, now v0.4. |
| `docs/CHECKLIST.md` | What is done and what is left. |
| `docs/capture.md` | How the capture module behaves. |

Each `Frame` has:

- `ir` — left infrared, `uint8`
- `depth` — metres, `float32`, `0` means invalid
- `intrinsics` — copied off the left-IR profile once at start
- `timestamp` — device time in seconds
- `emitter_state` — `"on"` or `"off"`

Stream settings: depth `z16` and left IR index 1 `y8`, 848×480 at 30 fps. Depth units set to `0.0001` (0.1 mm) and the real scale is read back with `get_depth_scale()`. Depth is aligned to the left IR image. On this D435 that alignment is nearly identity at the same resolution. Decimation magnitude must stay `1` or the depth grid no longer matches the IR image.

Filter order, applied only on emitter-on frames:

```
decimation → depth-to-disparity → spatial → temporal → disparity-to-depth → hole filling
```

The temporal filter must not see a laser-off frame, or it blends a bad depth map into a good one. Emitter-off frames still produce a `Frame`, but their depth is raw. The clean IR image (laser off) is the one later face detection should use. The depth number (laser on) is the one the tape check reads.

The emitter tag comes from `frame_laser_power_mode` metadata (`1` on, `0` off). If that metadata is missing, capture raises. It does not guess from frame number.

A raw depth value of 65535 is "out of range", not a distance. It is stored as `0`. At 0.1 mm per unit, 65535 would otherwise show up as 6.55 m.

Preview: two windows, **IR (left)** and **Depth**. Green center box (`preview.center_box_frac: 0.1`). The Depth window prints `center median` from emitter-on frames only. Colours cover 0.2–1.0 m only. The printed number is the real distance, including outside that colour range. `q` quits.

`uv run pytest`: 27 passed.

Headless check on the connected D435: the emitter tag alternated every frame after the first few, about 24 fps in that run. A later preview showed about 31 fps. The preview windows were not watched in that first run.

## 2026-10-05 — Platform decision

The Pi will not be used for now. All work runs on this machine. Consequences written into PRD v0.4:

- Step 2 (build librealsense on the Pi) is deferred.
- The "< 1.5 s on Pi 4" latency target can only be timed here until a Pi exists.
- The door relay will be a mock that logs lock and unlock. Real GPIO is the same interface, added later.
- `pyrealsense2` has a wheel on this machine, so `uv sync` is enough. The Pi source build stays in the PRD for later.

Installed so far: `numpy`, `opencv-python`, `pyrealsense2`, `pyyaml`, `pytest`, `ruff`. FastAPI, ONNX Runtime, Open3D, cryptography, and PyTorch are not installed yet. They get added when those steps start.

## 2026-10-05 — White wall, then the tape check

First preview at a white wall: IR window almost pure white, Depth window black in the green box, `center median: n/a`. A few coloured pixels sat along the bottom of the depth image. Cause: auto-exposure lengthens the shutter on the dark laser-off frame, then the laser-on frame clips to white. With no dot pattern, stereo returns no depth.

While debugging, setting manual exposure failed with `Permission denied` on a UVC control. Setting `auto_exposure_limit` worked. The camera was hardware-reset once after a bad option change stopped frames. `configs/default.yaml` now has `camera.auto_exposure_limit_us: 3000`, and the preview says `white wall — IR too bright` when the center IR is clipped.

Tape check, done by hand on this camera:

- Black wall and green wall: the center median matched the tape from **10 cm to about 100 cm**.
- Past 100 cm: not checked.
- White wall: `n/a` immediately. Treated as a known limit. A face is not a white wall, so this does not block the door use.

An earlier note in the docs said the D435 cannot measure under about 30 cm. That was wrong for this camera on black and green walls. The 30–60 cm band in the config is the later face-accept range, not the camera's limit.

The colour picture still flattens below 0.2 m and above 1.0 m. Trust `center median`, not the colour, outside that band.

## 2026-10-05 — Where step 1 stands

The capture requirement for now is done. The rest of the face-ID system is not.

Still open inside step 1, none of them blocking the distance check:

- Choose hole-filling mode. Mode `1` (farthest) is in the config. In one early scene it filled holes with the 65535 out-of-range marker, which is why those values are now zeroed. Mode `2` (nearest) filled more, but it also pulled the distance off the raw measurement. Decide from a real face, not from a wall.
- `capture/recorder.py`: save IR and depth together, with labels, under `data/raw/`. Needed before any training data.
- `.gitignore` entry for `data/`, then the first git commit. The repo still has no commits. Face images must not be committed.

Next feature step, when started, is PRD step 3: YuNet face detect, 112×112 align, and the quality gate. Step 2 (Pi) stays deferred.

## How to add an entry

Add a new `## YYYY-MM-DD — title` section at the bottom. Keep the camera serial, the config value, and the pass/fail result in the entry. Update [CHECKLIST.md](CHECKLIST.md) when a box changes.
