# Face-ID Authentication & Anti-Spoofing — Implementation PRD

**Version:** 0.4 | **Owner:** Alan Francis Santhosh | **Progress:** see [CHECKLIST.md](CHECKLIST.md) **Scope:** capture, anti-spoofing, identity matching, enrolment, door control, access log.

---

## 1. What We're Building

A door-side system that decides whether the person at the camera is (a) a real, live human and (b) an enrolled, authorised person. If both are true it unlocks an electric door strike; otherwise it stays locked and logs the reason.

- **Sensor:** Intel RealSense D435 (IR + depth)
- **Edge device:** Raspberry Pi 4 is the eventual target. **For now, development and testing run on the current x86_64 Linux machine; all Pi work is deferred** (see the *Platform scope* note below).
- **Must work:** in low light, with glasses/makeup, against printed-photo, screen-replay and mask attacks
- **Design rule:** liveness runs **before** matching, so a spoof never reaches the gallery

## 2. Success Criteria

| Metric | Target |
| --- | --- |
| Decision latency (capture → relay) | \< 1.5 s on Pi 4 (measure on the dev machine for now; re-measure when the Pi is back in scope) |
| False accept rate (FAR) | \< 0.1% |
| False reject rate (FRR) | \< 5% |
| Print + screen attack rejection | > 95% |
| Low-light operation | Same targets, lights off |
| Soak test | 48 h, no crash or manual restart |
| Privacy | Embeddings only, encrypted, deletable |

**Non-goals:** vault activity monitoring, multi-branch management, mobile app, commercial-grade 3D-mask defence.

## 3. Pipeline

```
D435 -> capture -> filters -> face detect -> quality gate
     -> liveness (geometry + learned IR/depth model) -> fusion
     -> align + embed -> 1:N match -> decision
     -> relay (unlock) + access log
```

The authentication runs as a state machine:

```
IDLE -> FACE_DETECTED -> QUALITY_OK -> LIVENESS_PASS -> MATCHED -> UNLOCKED -> COOLDOWN -> IDLE
                |             |              |              |
                +-------------+--------------+--------------+--> DENIED (log reason) -> IDLE
```

Decide over a short window of frames (e.g. 5–10), not a single frame, and require agreement across them.

## 4. Module Specifications

### 4.1 `capture/`

- Start `rs.pipeline` with depth (z16) and left IR (y8) at 848×480 @ 30 fps.
- Align depth to IR with `rs.align`; set depth units to 0.1 mm for close range.
- Filter chain order: decimation → depth-to-disparity → spatial → temporal → disparity-to-depth → hole-filling.
- Emitter control: alternate `emitter_on_off` so frames are tagged `emitter=on` (use for depth) or `emitter=off` (use for clean IR face image).
- **Output:** `Frame(ir, depth, intrinsics, timestamp, emitter_state)`.
- **Implementation notes (v0.4):**
  - Filters run on emitter-on frames only; emitter-off frames carry raw aligned depth.
  - Decimation magnitude must stay `1` so depth keeps matching the 848×480 IR image.
  - The emitter tag is read from `frame_laser_power_mode` metadata; if unavailable, capture fails rather than guessing.
  - Raw z16 value 65535 is the out-of-range marker and is mapped to `0` (invalid) in `Frame.depth`.
  - Hole-filling mode 1 (farthest) can fill holes with that marker; mode 2 (nearest) may suit faces better. Decide from data.
  - If the camera refuses `emitter_on_off` with auto-exposure on, exposure is frozen at its current value and the set is retried.

### 4.2 `capture/quality_gate.py`

Reject a frame (with a user prompt) if any of these fail:

- no face found, or more than one face
- face distance outside 0.10–0.60 m (median depth in face box)
- face box too small (\< \~80 px) or mostly off-centre
- blur (variance of Laplacian below threshold)
- too many invalid depth pixels (> \~20% holes) in the face region

### 4.3 `face/detect.py` + `align.py`

- YuNet on the IR image → box + 5 landmarks.
- Similarity-transform align to a 112×112 crop; apply the same crop to depth.
- **Output:** `FaceCrop(ir_112, depth_112, landmarks, box)`.

### 4.4 `liveness/`

**Stage A — geometry check (`depth_check.py`, no training needed):**

1. Convert the face-box depth to 3D points with `rs2_deproject_pixel_to_point`.
2. Fit a plane (RANSAC / least squares) to the face region.
3. Compute **relief** = spread of points from the plane (e.g. 5th–95th percentile of signed distance) and the **nose-to-cheek depth gap**.
4. Real face: relief of roughly 1.5–4 cm. Flat photo/screen: under \~0.5 cm. Set thresholds from your own data.
5. Also reject if depth is mostly invalid (screens and glossy prints often give holes).

**Stage B — learned model (`ir_model.py`):** Input: 2-channel tensor (IR crop + normalised depth crop), 112×112. Output: live probability.

```python
import torch, torch.nn as nn

class LivenessNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o): return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(2, 16), block(16, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1))
    def forward(self, x):                 # x: (B, 2, 112, 112)
        return self.head(self.features(x))  # logit; sigmoid -> p(live)
```

Start with this small CNN as the baseline; swap in a small ViT or hybrid later if it helps. Train with BCE, augment brightness/contrast/noise and small rotations, export to ONNX.

**Stage C — fusion (`fusion.py`):** `live = geometry_ok AND (w1*learned + w2*geometry_score > threshold)`. Geometry is a hard gate; the learned score handles what geometry can't (screens with depth glitches, curved masks).

### 4.5 `face/embed.py` + `match.py`

- Baseline: OpenCV SFace on the aligned IR crop → 128-D, L2-normalised.
- Improve by fine-tuning on your IR data (`train_embed.py`) with an ArcFace-style loss, then export to ONNX.
- Gallery: matrix of normalised embeddings + ID list; `scores = gallery @ query`; best score above threshold → match, else `unknown`.
- Enrolment averages several embeddings per person (keep the individual ones too for better recall).

### 4.6 `enroll/`

1. Show consent form; record consent.
2. Capture 10–20 frames passing the quality gate and liveness (so enrolment can't be spoofed either).
3. Embed, store embeddings only, encrypted at rest (e.g. `cryptography` Fernet or SQLCipher).
4. Admin commands: `enroll`, `list`, `update`, `delete` (delete removes all embeddings and metadata).

### 4.7 `control/`

- `POST /authenticate` → runs the state machine, returns `{decision, reason, scores}`.
- `GET /health`, `POST /admin/enroll`, `DELETE /admin/person/{id}` (admin token required).
- `relay.py`: on the dev machine a **mock relay** logs lock/unlock instead of driving hardware; the real GPIO implementation sits behind the same interface and is added with the Pi. Real hardware: GPIO pin drives an opto-isolated relay; strike powered from a separate 12 V supply wired NO/COM for fail-secure; auto re-lock after a configurable delay.
- Lockout: after N failures, refuse attempts for a cooldown.

### 4.8 `logging_/access_log.py`

SQLite table `access_events(id, ts, decision, reason, liveness_geom, liveness_model, match_score, person_id, latency_ms)`. Log every attempt, including rejected spoofs.

## 5. Configuration (`configs/default.yaml`)

```yaml
camera: {width: 848, height: 480, fps: 30, depth_units: 0.0001}
quality: {min_dist_m: 0.10, max_dist_m: 0.60, min_face_px: 80, max_hole_frac: 0.2}
liveness: {relief_min_m: 0.015, flat_max_m: 0.005, model_threshold: 0.5, frames_required: 5}
match: {threshold: 0.45, unknown_label: "unknown"}
door: {gpio_pin: 17, unlock_seconds: 5, max_failures: 5, cooldown_seconds: 60}
```

All thresholds live here and are tuned from your evaluation data, not hard-coded.

## 6. Stack and Licences

| Area | Choice | Licence |
| --- | --- | --- |
| Camera SDK | librealsense2 / `pyrealsense2` | Apache-2.0 |
| Detection + baseline embedding | OpenCV YuNet + SFace | Apache-2.0 |
| Liveness reference | Silent-Face-Anti-Spoofing | Apache-2.0 |
| Geometry | Open3D, NumPy | MIT / BSD |
| Training / inference | PyTorch (train), ONNX Runtime (Pi) | BSD / MIT |
| API | FastAPI | MIT |
| Storage | SQLite | Public domain |
| Packaging | uv | MIT / Apache-2.0 |

Avoid InsightFace pretrained weights if you want a clean licence story (non-commercial only). Record every external model and dataset in `docs/CITATIONS.md`.

## 7. Environment (uv)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # or: yay -S uv
uv init face-id --python 3.12 && cd face-id           # 3.12 is what this repo pins
uv add pyrealsense2 opencv-python numpy open3d fastapi uvicorn \
       onnxruntime pyyaml cryptography
uv add --group train torch torchvision                # laptop/GPU only
uv add --group pi RPi.GPIO                            # Pi only (deferred)
uv add --dev pytest ruff
uv sync
uv run python -m capture.realsense_stream
```

> **Platform scope (v0.4):** the Pi is out of scope for now. Everything runs on this machine (x86_64 Linux, `pyrealsense2` wheel available, so `uv sync` is enough). The `pi` dependency group, the Pi source build and Pi-specific tuning stay in this document for later but are not being worked on.
>
> **Dependencies actually installed so far:** `numpy`, `opencv-python`, `pyrealsense2`, `pyyaml`, plus `pytest` and `ruff` (dev). The rest of the `uv add` line above is added as each step needs it.

- Commit `pyproject.toml` and `uv.lock`.
- **Pi 4 (aarch64) has no `pyrealsense2` wheel:** build librealsense from source with Python bindings, then `uv venv --system-site-packages`.
- Install librealsense udev rules so no root is needed.
- Train on the laptop/GPU, export ONNX, run only ONNX Runtime on the Pi.

## 8. Repository Structure

```
face-id/
├── pyproject.toml / uv.lock
├── configs/          # default.yaml
├── capture/          # realsense_stream.py, filters.py, quality_gate.py, recorder.py
├── face/             # detect.py, align.py, embed.py, match.py, train_embed.py
├── liveness/         # depth_check.py, ir_model.py, fusion.py, train.py
├── enroll/           # enroll.py, store.py, consent_form.md
├── control/          # api.py, relay.py, state_machine.py
├── logging_/         # access_log.py
├── eval/             # spoof_eval.py, far_frr.py, latency.py, soak_test.py
├── data/             # raw/ processed/ (gitignored)
├── docs/             # PRD.md, CITATIONS.md, LICENSES.md
└── tests/
```

## 9. Data Collection Workflow

1. **Recorder tool** (`capture/recorder.py`): saves synchronised IR + depth frames with labels (person ID, lighting, attack type) to `data/raw/`.
2. **Genuine data:** 10–20 volunteers; bright, dim and dark light; with/without glasses; slight pose variation. Written consent, project-limited, withdrawable (DPDP Act 2023).
3. **Attack data (same camera and setup):** matte and glossy prints, phone/tablet photo and video replays, paper masks, plus silicone/3D masks if obtainable.
4. **Splits:** by person and by attack type, never by frame. Hold out one attack type entirely to test generalisation.
5. **Preprocess** (`data/processed/`): detect, align, crop, normalise depth; store as tensors.
6. **Optional:** pretrain liveness on CASIA-SURF (EULA needed; captured on a different RealSense, so expect a domain gap).

## 10. Build Order (by dependency)

Each step is "done" when its check passes, then you move on.

1. **Environment + capture (implemented; tape check passed on black and green, 10 cm to about 100 cm):** `realsense_stream` shows aligned IR/depth live. A white wall reads `n/a`. Not checked past 100 cm.
2. **Pi build (deferred, not in scope for now):** librealsense + Python bindings stream on the Pi. *Check:* 30 fps for 10 minutes with no dropout.
3. **Detect + align + quality gate.** *Check:* correct accept/reject prompts on a test set.
4. **Baseline identity:** embed + match on a small gallery. *Check:* correct match on enrolled people, `unknown` for strangers.
5. **Geometry liveness.** *Check:* rejects printed photos and phone screens in a first attack test.
6. **Door control:** `/authenticate` + relay + state machine + log. *Check:* end-to-end unlock for a live enrolled person; denied for a spoof.
7. **Attack dataset + learned liveness model**, then fusion. *Check:* beats geometry-only on held-out attack types.
8. **Improve identity:** fine-tune the embedding on IR + depth; compare to baseline. *Check:* lower FRR at the same FAR.
9. **Evaluation + threshold tuning** (`eval/`). *Check:* Section 2 targets.
10. **Hardening:** ONNX/INT8 on Pi, latency profiling, soak test, failure handling (camera unplug, USB dropout, power loss).

## 11. Testing

- **Unit:** quality-gate rules, plane-fit relief on synthetic flat/curved point clouds, matching with known embeddings, state-machine transitions.
- **Integration:** recorded frames replayed through the full pipeline (no camera needed) and compared to expected decisions.
- **Evaluation:** ROC and FAR/FRR for identity; APCER, BPCER, ACER per attack type for liveness; ablation of geometry-only vs model-only vs fused; lit vs dark room; glasses/makeup cases.
- **System:** per-stage latency on the Pi, 48-hour soak, USB dropout count, behaviour on power loss (door must stay locked).

## 12. Risks and Fallbacks

| Risk | Fallback |
| --- | --- |
| IR + depth identity less accurate than RGB; few pretrained IR face models | Fine-tune SFace on your IR data; if targets are missed, add the D435's RGB stream as an optional embedding input |
| Learned liveness overfits your attacks | Held-out attack types, keep the geometry gate as a safety net |
| Pi 4 too slow or USB 3 unstable | Lower resolution/fps, powered hub, INT8; fall back to Pi 5 or Jetson Orin Nano |
| Silicone/3D masks defeat the system | State as a known limitation; report per-attack results honestly |
| Scope creep | Must-have path: capture → geometry liveness → match → relay → log. Learned models come after |