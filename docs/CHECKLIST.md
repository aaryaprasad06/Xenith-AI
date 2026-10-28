# Implementation Checklist

Tracks the build against [PRD.md](PRD.md). Tick items as they are done.

**Platform:** current x86_64 Linux machine. Raspberry Pi work is deferred.
**Last updated:** 2026-10-28

Legend: `[x]` done, `[ ]` to do, `[~]` deferred (Pi, out of scope for now).

## Summary

| PRD step | Status |
| --- | --- |
| 1. Environment + capture | Implemented; tape check passed (not on white, not past 100 cm) |
| 2. Pi build | Deferred |
| 3. Detect + align + quality gate | Detect, align, and most quality checks are in. A real face is found, including under 30 cm. Blur check is built but off: Laplacian variance did not separate blur on this camera |
| 4. Baseline identity | Embedders and 1:N matcher implemented; live matching and threshold calibration pending |
| 5. Geometry liveness | Live check passed on a real face (2026-10-06). A printed photo and a phone screen were rejected (2026-10-18). Thresholds are still the starting values |
| 6. Door control | API, state machine, mock relay and access log implemented; live end-to-end rehearsal pending |
| 7. Attack dataset + learned liveness + fusion | To do |
| 8. Improve identity | To do |
| 9. Evaluation + threshold tuning | To do |
| 10. Hardening | To do (Pi parts deferred) |
| Phase 2. Silicone masks (thermal) | Later. Not part of the current print/screen target |

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
- [x] `capture/recorder.py`: save synchronised IR + depth (+ RGB) with labels to `data/raw/` (also needed for step 7)
- [x] Console Record page (`/record`): guided head-turn circle, dark and spoof steps (`/v1/recordings/guided`), plus a manual single take (`POST /v1/recordings`), through the API's camera (2026-10-18). First session (p001 s1) recorded on the Linux laptop 2026-10-18. On Ameen's Windows laptop, 2026-10-22: p002 s1 (glasses, print, and screen) and a different person stored as p001 s1, without a print. Those `data/raw` folders are not the Linux laptop's files. Laser-off frames are usable on every genuine take there
- [ ] Tune `guide.reach_min` / `center_max` on the camera
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
- [ ] Quality gate: blur. Laplacian variance was built and measured 2026-10-18 and does not work on this camera, so `blur_min` is unset. Needs a different measure, or may not be needed at a 3 ms exposure
- [x] Quality gate: too many invalid depth pixels (> ~20%)
- [x] User prompts for each rejection (the auth reason and the `liveness.watch` overlay)
- [x] Unit tests for the gate rules
- [ ] **Check:** correct accept/reject prompts on a test set

## Step 4: Baseline identity

- [x] Download SFace model; record it in `docs/CITATIONS.md` (fetched by `python -m face.models`, sha256-pinned, not in git)
- [x] `face/embed.py`: SFace on aligned IR crop, 128-D, L2-normalised. Also ArcFace R50 (buffalo_l), 512-D (2026-10-18), and AdaFace IR-101 (WebFace12M), 512-D, exported to ONNX (2026-10-19)
- [x] AdaFace export rejects non-finite validation output and preserves the existing model on failure; regression tests added (2026-10-28)
- [ ] Pick the model from `eval.far_frr` once 2+ people are recorded
- [x] `face/match.py`: gallery matrix, `scores = gallery @ query`, threshold, `unknown` (2026-10-22). Threshold is still the 0.45 starting value
- [x] Unit tests with known embeddings
- [x] Enroll a saved session from Record into the matching gallery, preserving other identities and updating People (2026-10-28)
- [ ] **Check:** correct match for enrolled people, `unknown` for strangers. Live p001 granted on Linux 2026-10-28 (ArcFace 0.7235, 8/8 agreeing); stranger test still pending

## Step 5: Geometry liveness

- [x] `liveness/depth_check.py`: deproject face-box depth to 3D
- [x] Plane fit (RANSAC / least squares)
- [x] Relief (5th-95th percentile) and nose-to-cheek gap
- [x] Reject when depth is mostly invalid
- [ ] Thresholds from own data (`relief_min_m`, `flat_max_m`). The config still has the starting values: 1.5 cm and 0.5 cm
- [x] Unit tests on synthetic flat and curved point clouds
- [x] **Check:** a real face passes. Confirmed 2026-10-06 on this D435, including closer than 30 cm. The overlay reports the face as live. The door stays locked because identity matching is not built.
- [x] **Check:** rejects printed photos and phone screens in a first attack test. Confirmed 2026-10-18 on this D435. No rejection rate measured yet.

## Step 6: Door control

- [x] `control/state_machine.py`: IDLE → FACE_DETECTED → QUALITY_OK → LIVENESS_PASS → MATCHED → UNLOCKED, plus DENIED; COOLDOWN is the lockout (2026-10-22)
- [x] Multi-frame decision (5-10 frames) with agreement: 5 laser-on frames for liveness, 3+ laser-off faces for identity with 60% agreeing
- [x] `control/relay.py`: relay interface + **mock relay** (logs lock/unlock) for the dev machine
- [x] Auto re-lock after `unlock_seconds`
- [x] Lockout after `max_failures`, cooldown `cooldown_seconds`; disabled for development with `max_failures: 0` (2026-10-28)
- [x] FastAPI app in `api/` (RealSenseID-REST shape: `/v1/auth`, `/v1/users`, `/v1/preview`, `/v1/device`), plus PRD aliases `POST /authenticate`, `GET /health`, `POST /admin/enroll`, `DELETE /admin/person/{id}`
- [x] Admin token on enrol and delete (`FACE_ID_ADMIN_TOKEN`)
- [x] Astro console in `web/` (pnpm): preview, people, device
- [x] Redesign all four console views with responsive forms, accessible navigation, and state-driven guided capture animation with reduced-motion support
- [x] `logging_/access_log.py`: SQLite `access_events`, log every attempt including spoofs; `GET /v1/auth/log`
- [x] Add fastapi, uvicorn
- [x] Unit tests: state-machine transitions
- [ ] **Check:** end-to-end unlock for a live enrolled person; denied for a spoof. Live unlock and subsequent locked state confirmed 2026-10-28 on Linux; live end-to-end spoof rehearsal pending
- [~] Real GPIO relay on the Pi (`RPi.GPIO`, 12 V strike, fail-secure NO/COM)

## Enrolment (needed for steps 4 and 6)

- [x] Stand-in: `enroll/from_takes.py` builds the gallery from recorded takes (unencrypted `data/gallery.npz`) until `enroll.py` exists
- [ ] `enroll/consent_form.md` and consent recording
- [ ] `enroll/enroll.py`: capture 10-20 frames that pass quality gate and liveness
- [ ] `enroll/store.py`: embeddings only, encrypted at rest (Fernet or SQLCipher)
- [ ] Admin commands: `enroll`, `list`, `update`, `delete` (delete removes embeddings and metadata)
- [x] People lists directory IDs and gallery identities with face-enrolment status; Delete revokes gallery access and removes metadata (2026-10-28). Live face enrolment and update still pending
- [ ] Add cryptography

## Step 7: Attack dataset + learned liveness + fusion

- [ ] Genuine data from 10-20 volunteers (bright / dim / dark, glasses, pose), written consent
- [ ] Attack data on the same camera: matte and glossy prints, phone/tablet photo and video, paper masks. Silicone is phase 2 (thermal), not this IR model
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

- [x] `eval/far_frr.py`: ROC, FMR / FNMR, EER, FPIR / FNIR, rule-of-three bound, on recorded takes for any `Embedder` (2026-10-18). First real run 2026-10-22, on Ameen's Windows laptop only: SFace, ArcFace, and AdaFace, `cross-take`, laser-off, that laptop's p001 and p002. ArcFace and AdaFace separate them; SFace overlaps. One person pair cannot resolve a 1% FMR. Not combined with the Linux laptop's 18 October session
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

## Phase 2: Silicone masks (thermal)

Later. Not part of steps 1–10 and not part of the "> 95% rejection" target, which is prints and screens. Research notes: [journal](journal.md), 2026-10-07. The D435 relief check passes a mask worn on a real head. A thermal camera is the add-on that separates a cold silicone mask from skin, and a mask that has already warmed up still gets through a temperature average. The score has to be the heat map (centre against the rim), and the silicone miss rate is reported on its own.

- [ ] FLIR Lepton 3.5 + PureThermal 2 (~$280, 160×120, USB). Not the MLX90640 (32×24 is too coarse) and not the Seek Compact Pro unless Linux support is solved
- [ ] Mount it beside the D435 and store one homography from a warm object
- [ ] Read it as a USB camera. Copy the face box from the IR detector. Score centre versus rim, not a fixed nose temperature
- [ ] Record, on this camera: live faces (glasses, just come in from the cold) and silicone masks cold, worn 30 s, and worn several minutes
- [ ] Split by physical mask, not by frame
- [ ] `eval/spoof_eval.py` reports silicone APCER on its own. Do not fold it into the print/screen number

## Cross-cutting

- [x] `docs/CITATIONS.md`: YuNet is recorded. No dataset is in use yet
- [ ] `docs/LICENSES.md`
- [ ] Integration tests: recorded frames replayed through the full pipeline
- [ ] Privacy: embeddings only, encrypted, deletable (DPDP Act 2023 consent)
- [ ] Make repo layout match PRD section 8 as modules land (`face/`, `liveness/`, `enroll/`, `control/`, `logging_/`, `eval/`, `data/`)
