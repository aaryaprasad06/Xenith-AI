# Implementation Checklist

Tracks the build against [PRD.md](PRD.md). Tick items as they are done.

**Platform:** current x86_64 Linux machine. Raspberry Pi work is deferred.
**Last updated:** 2026-10-06

Legend: `[x]` done, `[ ]` to do, `[~]` deferred (Pi, out of scope for now).

## Summary

| PRD step | Status |
| --- | --- |
| 1. Environment + capture | Implemented; tape check passed (not on white, not past 100 cm) |
| 2. Pi build | Deferred |
| 3. Detect + align + quality gate | Detect, align, and most quality checks are in. A real face is found, including under 30 cm. Blur is not |
| 4. Baseline identity | To do |
| 5. Geometry liveness | Live check passed on a real face (2026-10-06). Thresholds are still the starting values. A print/screen test is not done |
| 6. Door control | API shell started (no state machine, relay, or log yet) |
| 7. Attack dataset + learned liveness + fusion | To do |
| 8. Improve identity | To do |
| 9. Evaluation + threshold tuning | To do |
| 10. Hardening | To do (Pi parts deferred) |

---

## Step 1: Environment + capture

- [x] uv project, Python 3.12, `pyproject.toml` + `uv.lock`
- [x] Dependencies installed: numpy, opencv-python, pyrealsense2, pyyaml; dev: pytest, ruff
- [x] `configs/default.yaml` with PRD sections plus `filters`, `emitter`, `preview`
- [x] `capture/config.py`: load and validate config
- [x] `capture/frame.py`: `Frame`, `Intrinsics`, `center_roi_median_m`
- [x] `capture/filters.py`: decimation, disparity, spatial, temporal, hole filling
- [x] `capture/realsense_stream.py`: 848×480 @ 30 fps IR + depth, depth units 0.1 mm, align to IR
- [x] Emitter on/off alternation with per-frame tag from metadata
- [x] Live preview with center box, emitter state, fps, median distance
- [x] Unit tests (27, no camera needed)
- [x] Docs: `README.md`, `docs/capture.md`
- [x] Headless smoke test on the real D435 (emitter alternates every frame)
- [x] **Check:** center-box distance matches a tape measure. Checked 2026-10-05: correct from 10 cm to about 100 cm on black and green walls. Not checked past 100 cm. A white wall reads `n/a` (IR image clips, so there is no depth).
- [ ] Decide hole-filling mode (1 vs 2) from real face data
- [ ] `capture/recorder.py`: save synchronised IR + depth with labels to `data/raw/` (also needed for step 7)
- [x] `.gitignore` entry for `data/`; first git commit

## Step 2: Pi build

- [~] Build librealsense + Python bindings on the Pi 4
- [~] 30 fps for 10 minutes with no dropout on the Pi

## Step 3: Detect + align + quality gate

- [x] Download YuNet model; record it in `docs/CITATIONS.md`
- [x] `face/detect.py`: YuNet on IR, box + 5 landmarks
- [x] `face/align.py`: similarity transform to 112×112, same crop on depth
- [x] `capture/quality_gate.py`: no face / multiple faces (applied in `liveness/session.py`)
- [x] Quality gate: distance outside 0.10-0.60 m (median depth in face box)
- [x] Quality gate: face too small (< ~80 px) or off-centre
- [ ] Quality gate: blur (variance of Laplacian)
- [x] Quality gate: too many invalid depth pixels (> ~20%)
- [x] User prompts for each rejection (the auth reason and the `liveness.watch` overlay)
- [x] Unit tests for the gate rules
- [ ] **Check:** correct accept/reject prompts on a test set

## Step 4: Baseline identity

- [ ] Download SFace model; record it in `docs/CITATIONS.md`
- [ ] `face/embed.py`: SFace on aligned IR crop, 128-D, L2-normalised
- [ ] `face/match.py`: gallery matrix, `scores = gallery @ query`, threshold, `unknown`
- [ ] Unit tests with known embeddings
- [ ] **Check:** correct match for enrolled people, `unknown` for strangers

## Step 5: Geometry liveness

- [x] `liveness/depth_check.py`: deproject face-box depth to 3D
- [x] Plane fit (RANSAC / least squares)
- [x] Relief (5th-95th percentile) and nose-to-cheek gap
- [x] Reject when depth is mostly invalid
- [ ] Thresholds from own data (`relief_min_m`, `flat_max_m`). The config still has the starting values: 1.5 cm and 0.5 cm
- [x] Unit tests on synthetic flat and curved point clouds
- [x] **Check:** a real face passes. Confirmed 2026-10-06 on this D435, including closer than 30 cm. The overlay reports the face as live. The door stays locked because identity matching is not built.
- [ ] **Check:** rejects printed photos and phone screens in a first attack test

## Step 6: Door control

- [ ] `control/state_machine.py`: IDLE → FACE_DETECTED → QUALITY_OK → LIVENESS_PASS → MATCHED → UNLOCKED → COOLDOWN, plus DENIED
- [ ] Multi-frame decision (5-10 frames) with agreement
- [ ] `control/relay.py`: relay interface + **mock relay** (logs lock/unlock) for the dev machine
- [ ] Auto re-lock after `unlock_seconds`
- [ ] Lockout after `max_failures`, cooldown `cooldown_seconds`
- [x] FastAPI app in `api/` (RealSenseID-REST shape: `/v1/auth`, `/v1/users`, `/v1/preview`, `/v1/device`), plus PRD aliases `POST /authenticate`, `GET /health`, `POST /admin/enroll`, `DELETE /admin/person/{id}`
- [x] Admin token on enrol and delete (`FACE_ID_ADMIN_TOKEN`)
- [x] Astro console in `web/` (pnpm): preview, people, device
- [ ] `logging_/access_log.py`: SQLite `access_events`, log every attempt including spoofs
- [x] Add fastapi, uvicorn
- [ ] Unit tests: state-machine transitions
- [ ] **Check:** end-to-end unlock for a live enrolled person; denied for a spoof
- [~] Real GPIO relay on the Pi (`RPi.GPIO`, 12 V strike, fail-secure NO/COM)

## Enrolment (needed for steps 4 and 6)

- [ ] `enroll/consent_form.md` and consent recording
- [ ] `enroll/enroll.py`: capture 10-20 frames that pass quality gate and liveness
- [ ] `enroll/store.py`: embeddings only, encrypted at rest (Fernet or SQLCipher)
- [ ] Admin commands: `enroll`, `list`, `update`, `delete` (delete removes embeddings and metadata)
- [ ] Add cryptography

## Step 7: Attack dataset + learned liveness + fusion

- [ ] Genuine data from 10-20 volunteers (bright / dim / dark, glasses, pose), written consent
- [ ] Attack data on the same camera: matte and glossy prints, phone/tablet photo and video, paper masks, silicone/3D masks if available
- [ ] Splits by person and by attack type; one attack type held out entirely
- [ ] Preprocess to `data/processed/` (detect, align, crop, normalise depth)
- [ ] `liveness/ir_model.py`: `LivenessNet` (2-channel 112×112)
- [ ] `liveness/train.py`: BCE, augmentation, ONNX export
- [ ] `liveness/fusion.py`: `geometry_ok AND (w1*learned + w2*geometry > threshold)`
- [ ] Add torch / torchvision (train group), onnxruntime
- [ ] Optional: pretrain on CASIA-SURF (EULA needed)
- [ ] **Check:** beats geometry-only on held-out attack types

## Step 8: Improve identity

- [ ] `face/train_embed.py`: fine-tune SFace on IR (+ depth) with an ArcFace-style loss
- [ ] Export to ONNX
- [ ] **Check:** lower FRR than baseline at the same FAR
- [ ] Fallback if targets are missed: add RGB stream as optional embedding input

## Step 9: Evaluation + threshold tuning

- [ ] `eval/far_frr.py`: ROC, FAR / FRR for identity
- [ ] `eval/spoof_eval.py`: APCER, BPCER, ACER per attack type
- [ ] Ablation: geometry-only vs model-only vs fused
- [ ] Lit vs dark room; glasses and makeup cases
- [ ] `eval/latency.py`: per-stage latency (dev machine for now)
- [ ] Tune thresholds in `configs/default.yaml` from data
- [ ] **Check:** Section 2 targets met (FAR < 0.1%, FRR < 5%, attack rejection > 95%)

## Step 10: Hardening

- [ ] `eval/soak_test.py`: 48 h with no crash or manual restart
- [ ] Failure handling: camera unplug, USB dropout, power loss (door stays locked)
- [ ] USB dropout counter
- [ ] ONNX INT8 quantisation
- [~] Latency profiling and tuning on the Pi (target < 1.5 s)

## Cross-cutting

- [x] `docs/CITATIONS.md`: YuNet is recorded. No dataset is in use yet
- [ ] `docs/LICENSES.md`
- [ ] Integration tests: recorded frames replayed through the full pipeline
- [ ] Privacy: embeddings only, encrypted, deletable (DPDP Act 2023 consent)
- [ ] Make repo layout match PRD section 8 as modules land (`face/`, `liveness/`, `enroll/`, `control/`, `logging_/`, `eval/`, `data/`)
