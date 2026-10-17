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

## 2026-10-06 — API and console, after RealSenseID-REST

Looked at [RealSenseID-REST](https://github.com/realsenseai/RealSenseID-REST). That project is a FastAPI process in front of an F4xx face module: `/v1` routers for auth, users, preview, and device, one device wrapper, and a small console. Their console is NiceGUI, and their utility routes flash F4xx firmware. Neither of those belongs on this D435.

What landed here, in our own code:

- `api/` — FastAPI. `CameraHub` owns the one D435 pipeline. `UserStore` is a SQLite id list in `data/users.db` (gitignored). Routers: `POST /v1/auth`, users enrol/list/delete, `GET /v1/preview/stream` (MJPEG of the left IR image), `GET /v1/device/info`. The PRD names also work: `POST /authenticate`, `GET /health`, `POST /admin/enroll`, `DELETE /admin/person/{id}`. Enrol and delete need `Authorization: Bearer` matching `FACE_ID_ADMIN_TOKEN`.
- `web/` — Astro, package manager pnpm. Pages: Preview, People, Device. Dev server proxies the API so the browser stays on port 4321.
- Auth always denies. The reason is either the camera is down, or liveness and matching are not built. Enrolment does not save a face image and does not compute an embedding.

Still not the door: no state machine, no relay, no access log.

## 2026-10-06 — Geometry liveness

The liveness check is the depth shape of one face, not a neural net. A learned model is not in the decision, and a failed shape check cannot be overridden later.

What landed:

- `models/face_detection_yunet_2023mar.onnx` — YuNet from the OpenCV zoo. Recorded in [CITATIONS.md](CITATIONS.md). `face/detect.py` returns a box and five landmarks on the left IR image. `face/align.py` warps IR and depth to 112×112 for a later embedding step. Alignment is not what the shape check uses.
- `capture/quality_gate.py` — too small, off-centre, more than 20% missing depth, or median distance outside 0.10–0.60 m. The closest allowed distance was lowered from 0.30 m the same day: this camera already measured correctly from 10 cm, and 30 cm was rejecting people who were still in range.
- `liveness/depth_check.py` — deproject the face box, RANSAC plane, relief is the 5th-to-95th percentile of the distance to that plane, plus the nose-to-cheek gap. Cheeks are sampled either side of the nose landmark, not from a cheek landmark (YuNet does not give one).
- `liveness/session.py` — five laser-on frames must all pass, the face must stay roughly still, and the same window must include a laser-off frame. Depth is taken from the laser-on frames. The face is detected on the laser-on IR image, because that is the frame the exposure cap keeps bright.
- `liveness/watch.py` — overlay. `POST /v1/auth` runs the same check. Both still refuse to unlock.

A depth island is not a hard fail. On a white wall the background has no depth, so a real face is an island. That check would lock out a genuine person in the scene we already measured.

Unit tests, no camera: a flat sheet at 0.50 m has relief 0.0 cm and is rejected. A synthetic bulge of 3.5 cm has relief 3.4 cm and a nose-to-cheek gap of 1.4 cm and passes. Thresholds in `configs/default.yaml` are still the starting values (`relief_min_m` 0.015, `flat_max_m` 0.005, `nose_cheek_min_m` 0.008, `relief_max_m` 0.08). They are not tuned on a real face on this D435.

A 25-frame burst on the connected D435 ran the detector and returned `no_face`. The lens was not on a person, so this does not say whether YuNet finds a face in infrared. OpenCV printed a warning that the DNN target is not supported by its new graph engine. Detection still returned a result.

Not done: blur check, a person standing in front of the camera, a printed photo, a phone screen, and any claim about a DepthFake projector. 3D masks remain out of scope.

## 2026-10-06 — Close faces were missed

Standing closer than about 30 cm still failed after the distance floor was lowered to 10 cm. The message was `no face` or `face is too small (32 px, need 120)`. The depth image showed the person. The 32 px box was a background hit, not the face.

YuNet only looks for faces of about 10–300 px. On this camera a face nearer than ~30 cm is larger than 300 px, so the real face was skipped. Detection now also runs on a half-size and a quarter-size copy, drops boxes under 48 px, and if the laser-on image still has no usable face it tries the laser-off image. The smallest accepted face is 80 px, so a face at the far end of 60 cm is not called too small. Restart `liveness.watch` or the API to load this.

## 2026-10-06 — Liveness check passed

Alan confirmed the live liveness check works on this D435 (serial `236422073835`). A real face is detected and the geometry check passes, including closer than 30 cm, which had been failing earlier the same day. The running check is `uv run python -m liveness.watch` (and the same decision on `POST /v1/auth`).

What that pass covers: YuNet finds the face on the IR image, five laser-on frames agree, a laser-off frame is in the window, and the depth shape is accepted under the starting thresholds in `configs/default.yaml` (`relief_min_m` 0.015, `flat_max_m` 0.005, `nose_cheek_min_m` 0.008, closest face 0.10 m, smallest face 80 px).

Still open: blur, tuning those relief numbers from more people, a printed photo, a phone screen, and any DepthFake projector test. A live result still does not unlock the door. Identity matching is not built. 3D masks remain out of scope for this build. The thermal follow-up is [phase 2](CHECKLIST.md).

## 2026-10-07 — Silicone masks, and why thermal waits

Question for this session: can this D435 reject a realistic silicone mask, and would a thermal camera do it. No mask was put in front of the camera. This is from the papers, not a new recording. Decision: silicone stays out of the current build. It is phase 2, and the sensor for that phase is thermal. The print and screen target (above 95%) does not move.

What the D435 already measures will not catch a mask worn on a real head.

- The relief gate (1.5 cm) is for flat things. A mask is a shell a few millimetres thick on a real nose and cheeks. At the door distance, 0.3–0.6 m, stereo depth noise is about that thick, so the shell does not show up as a second surface. Hole filling makes the rim even easier to miss.
- Eyes-open does not help. The wearer's real eyes are in the cutouts.
- The clean IR image is a weak separator for soft silicone. On a RealSense in the same near-infrared band, the ordinary texture stats of a 3D mask overlap a real face (Bhattacharjee and Marcel, BIOSIG 2017). Painted brows disappear. A rigid mask looks smoother. A soft one does not fall out of the live distribution.
- Colour and pulse are the wrong add for this door. A custom silicone mask already fools colour face matchers at more than ten times their normal false-match rate (Bhattacharjee et al., BTAS 2018). Pulse needs visible light and about a second of stillness, and the door has to work with the lights off in under 1.5 s.
- The file `Research_and_Application_of_Face_Anti-Spoofing_Based_on_Depth_Camera_(DMF)_….md` is Aurora Guard (Liu et al.), not a thermal paper. A screen flashes a random colour sequence and a network recovers depth from the reflection. They did not collect 3D masks. On their print and screen set, EER is 1.24% and HTER is 1.91%. Against an SR300 on paper, twisted paper, carved paper, an iPad, and a projector, their false-accept rate is 0.20% and the SR300's is 0.40% (100 tries each). That number is about flat attacks.

Thermal is the cue that actually separates silicone, and only while the mask is cold.

| Study | Setup | Result |
| --- | --- | --- |
| Bhattacharjee et al., BTAS 2018 | Seek Compact Pro. Score = mean temperature in the face box. Custom silicone. | Equal-error rate 7.5%. At 5% of attacks accepted, about 10% of real people are rejected. |
| George et al., TIFS 2019 (WMCA) | SR300 colour, depth, and infrared, plus thermal. Prints, screens, and masks mixed. | Thermal alone 1.85% ACER. All four channels 0.3% ACER. Not a silicone-only number. |
| Bhattacharjee and Marcel, BIOSIG 2017 | Thermal time-lapse of a worn silicone mask, frames 30 s apart. | Forehead, where the mask touches skin, ends near skin temperature. The mask is obviously cold in the first frame. |

A live face is warmed by blood: inner eye corners, forehead, and cheeks warmer, nose tip cooler, nostrils dipping on a breath. Skin and silicone both radiate strongly in long-wave infrared, so the camera reads temperature, not a material colour. After the average matches, that one number no longer separates them. Silicone conducts heat poorly, so the outer surface shows a blurred contact patch, not the blood pattern. A 2020 thermal study that let masks warm on the wearer found detection got worse after that, and full-head flexible masks were the hardest because heat came through more evenly. A mid-wave paper says the plain version: raise the mask temperature and an ordinary thermal image of the mask looks like a face. Polarisation still separated them. A Lepton or a Seek does not measure that.

What can remain after the average has caught up: a sharp cool edge at the neck or hairline if the frame includes it, rings at the eye and nostril holes, and warm blotches that follow contact instead of the warmer-cheek / cooler-nose layout. There is no published rate that says those leftovers stop most warmed, well-fitted masks. A cold mask held up at the door is the case the 7.5% figure describes. A mask put on before walking to the door is the miss.

Hardware, if phase 2 is bought. Prices are street prices on 2026-10-07, not quotes.

| Option | Image | Price | Fit |
| --- | --- | --- | --- |
| MLX90640, 55°, Adafruit | 32×24, I2C | $75 | About 10 pixels across a face at 0.5 m. Whole-face hot or cold only. |
| FLIR Lepton 3.5 + PureThermal 2 | 160×120, USB webcam, 8.7 Hz, sensitivity under 50 mK, absolute error about ±5 °C | $172 + $110 ≈ $280 | About 50 pixels across the face. Shows up as a normal camera, so OpenCV can read it on the Pi. This is the one to buy. |
| Seek Compact Pro | 320×240, export models under 9 Hz | $410–$500 | The camera behind the 7.5% result. Linux support is unofficial. |

Use the shape of the heat map (centre against the rim), not a stored cutoff such as "nose above 32 °C". Absolute error on these modules is several degrees. Sensitivity inside one frame is about 0.05 °C.

Phase 2 work, when it starts: bolt the Lepton next to this D435 (serial `236422073835`), calibrate a homography once, copy the face box from the IR detector, and score centre versus rim. Record cold masks, masks worn 30 s, masks worn several minutes, glasses, and people who just came in from the cold. Report the silicone miss rate on its own. Do not train that score on WMCA weights. Those frames are from an SR300.

## 2026-10-17 — Identity research: what to build, and what accuracy can be claimed

The goal for identity is a 1:N gallery of 100+ people, at accuracy as close as possible to Apple Face ID and the RealSense ID F455. Identity should use IR, depth and RGB, so one modality can cover for another. The model is not chosen yet. Choosing it is a separate research step, measured on our own D435 data. The full plan is phases R0–R5 in `~/.claude/plans/now-lets-brainstorm-the-glimmering-raccoon.md`.

**Decisions taken:**
- A non-commercial model is acceptable (InsightFace buffalo_l / antelopev2, EdgeFace). Models are picked on accuracy, not license.
- About 50 volunteers are available. Adding more is possible but is a large effort, so the plan avoids it. Extra impostor and training data comes from public datasets instead.

**What the reference products claim:**
- **Face ID:** FAR below 1 in 1,000,000 for **one** enrolled face. Higher for twins, look-alike siblings and children under 13. After 5 failed matches the passcode is required. It also checks that the eyes are open and looking at the device, and the capture sequence is randomised. New captures are added to the template only if their quality is good enough, and they expire after a limited number of unlocks. The networks were trained on over a billion images, including IR and depth. ([Face ID security guide](https://www.apple.com/business-docs/FaceID_Security_Guide.pdf), [Apple support](https://support.apple.com/en-us/102381))
- **F455:** FAR 1:1M, TAR 99.76%, spoof accept rate below 0.1%, 500 users, 30–100 cm, 0–50k lux, under 0.8 s. ([datasheet, Dec 2025](https://realsenseai.com/wp-content/uploads/2025/12/RealSense_ID_F45X-Datasheet-Dec-2025.pdf))

**What the F455 actually uses:**
- **Cameras:** two Full-HD colour cameras that also see 850 nm IR. Authentication reads their RAW Bayer pixels. They are not mono IR imagers like the D435's.
- **Illumination:** an 850 nm IR LED flood and an 850 nm IR dot projector.
- **No calibrated depth:** the two cameras are "semi-mechanically aligned" and have no extrinsic calibration, so the device cannot compute calibrated depth. It uses landmark disparity between the two images instead.
- **Pipeline:** pre-processing, anti-spoofing, embedding, matching, all on an on-chip neural accelerator.
- **Anti-spoofing:** three small MobileNetV2-style models (about 1M parameters each): left image, right image and disparity. If any one says spoof, the attempt is rejected.
- **Anti-spoof training data alone:** about 10.7k people and 1.7M live images, plus about 2M spoof images (prints, screens, cut and rolled paper, latex masks), from 20 devices, up to 2 m and ±40° head pose, indoor and outdoor.
- **Unpublished:** the recognition model.
- **Datasheet inconsistency:** it is unclear whether the left or the right camera provides RGB.
- Source for most of the above: the Intel RealSense team's paper, [arXiv 2410.24031](https://arxiv.org/html/2410.24031).

**What can be measured here:**
- With 0 false accepts in n independent impostor trials, the 95% upper bound on FAR is 3/n (the rule of three).
- P people give P(P−1)/2 impostor pairs. **50 volunteers give 1,225 pairs, so the best claim from our own data is FAR ≤ about 2.4e-3.** 100 people would support about 6e-4. Proving 1e-6 would need about 2,450 people.
- In 1:N search, the false-positive identification rate is roughly N × FMR. With 100 users, a 1e-3-per-attempt target needs a 1:1 FMR of about 1e-5.
- So we will report "FAR ≤ 3/n at 95%" and **not** claim parity with 1:1M. The PRD's FAR target (<0.1%) is the realistic one.
- Public datasets used as extra strangers make that bound tighter. They come from other sensors, so they are reported separately.

**Models:**
- Large ArcFace-family backbones (IR50/IR100) trained on RGB transfer to NIR almost unchanged: TAR@FAR=1e-4 of 99.5–99.9% on CASIA NIR-VIS 2.0, Oulu-CASIA and BUAA. Small models degrade (MobileFaceNet 84.6%).
- **Training IR from scratch fails** (12–50%). Naive fine-tuning on NIR hurt (99.89 → 94.3). Only regularised fine-tuning with frozen class centres helped. ([arXiv 2312.00627](https://arxiv.org/pdf/2312.00627))
- **Depth for identity:** Led3D-style networks reach about 99.6% on neutral faces but about 65–78% across sessions, poses and occlusion on Lock3DFace. A depth branch can be trained from scratch (synthetic BFM faces, then Lock3DFace), but only as a supporting score. If it adds nothing, depth stays liveness-only, which is what the F455 appears to do.

**Spoofing (DepthFake, IEEE S&P 2023):**
- **The attack:** estimate the victim's depth from one photo. Then shine a doctored dot pattern from an IR projector onto a printed copy of the photo, which is also adversarially altered for the RGB check.
- **Results:** depth-only liveness was fooled 79.4% of the time, and RGB+depth 59.4%, on an Orbbec Astra Pro (structured light) running Tencent, Baidu and 3DiVi. The attack held for 6+ frames in a row.
- **Once liveness was fooled, matching was fooled 100%,** because matching used RGB only.
- **The D435 is active stereo,** so its depth comes from the real surface and the paper's mechanism does not carry over directly. However, DoubleStar ([USENIX Security 2022](https://www.usenix.org/conference/usenixsecurity22/presentation/zhou-ce)) faked depth on RealSense stereo cameras. Red Bleed ([USENIX Security 2025](https://www.usenix.org/system/files/usenixsecurity25-hu-bowen.pdf)) made an LCD visible in NIR.
- **Lesson:** identity must not rest on RGB alone.

**Decision rule chosen:**
- **Liveness is AND, with no fallback:** depth geometry, the emitter on/off check, and later an IR/depth spoof classifier.
- **Identity is scored per modality and then fused,** with quality weights. IR is mandatory.
- **A modality may drop out only by being absent or low quality.** If it is present, good quality and disagrees, the result is deny. Fewer modalities means a **stricter** threshold, never a looser one.
- **"If one fails, use another" as an OR fallback is ruled out:** it hands the attacker the weakest modality.
- **1:N acceptance also needs** a margin over the second-best person, k-of-n frame agreement, and a lockout after `door.max_failures`.

**Public datasets compared with the F455's own data:**
- **Scale and sensor:** the F455 data is private, large, and captured on the exact device it is deployed on. The public sets are 52–1,000 people captured on other sensors. None used a D435.
- **Depth method:** Kinect v1 and SR300 use structured light, and Kinect v2 uses time-of-flight. The D435 is active stereo, so its depth noise and holes differ.
- **Purpose:** WMCA and CASIA-SURF were built for anti-spoofing, not for identity across sessions.
- **Consequence:** thresholds must come from our own D435 volunteers. Public sets serve only for pretraining (mainly depth) and as extra impostors.

| Dataset | Contents | Use here | Access |
| --- | --- | --- | --- |
| [WMCA](https://www.idiap.ch/en/scientific-research/data/wmca) | 72 people, 1,941 videos, colour/depth/IR/thermal (SR300 among the sensors), prints, replays, masks | Spoof testing, closest sensor | Preprocessed version free on [Zenodo](https://zenodo.org/record/4580313). Full 4.7 TB version for a fee |
| [Lock3DFace](http://irip.buaa.edu.cn/lock3dface/index.html) | About 500 people, Kinect v2 colour and depth, expression/pose/occlusion/time-lapse | Training the depth identity branch | Request |
| [CASIA-SURF](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_A_Dataset_and_Benchmark_for_Large-Scale_Multi-Modal_Face_Anti-Spoofing_CVPR_2019_paper.html) | 1,000 people, 21k videos, RGB/depth/IR from an SR300 | Extra impostors, spoof training | Request. Links not confirmed live ([ChaLearn page](https://chalearnlap.cvc.uab.cat/dataset/33/description/)) |
| [CASIA NIR-VIS 2.0](https://www.idiap.ch/software/bob/docs/bob/bob.db.cbsr_nir_vis_2/stable) | 725 people, NIR and visible, 4 sessions | Extra IR impostors | Release agreement with CASIA |
| [EURECOM KinectFaceDB](https://rgb-d.eurecom.fr/) | 52 people, colour/depth/3D, 2 sessions | Small RGB-D check | Not checked |

These datasets usually require a license agreement signed by a supervisor, not a student. Without them the project still works, with a weaker FAR claim (about 2.4e-3), and depth may stay liveness-only.

**Volunteer protocol (kept lean for 50 people):**
- 2 sessions on different days.
- 5 poses per session (center, left, right, up, down), about 30 frames each, lit.
- One dark take per session, to test IR only.
- A glasses take only for people who wear them.
- About 5 minutes per person per session.
- Spoof takes (print, phone screen) from 10–15 of them.

**Code that landed (R0), not yet run on the camera:**
- **`capture/realsense_stream.py`:** the colour stream is enabled at 848×480 when `camera.rgb: true` (now the default). It is aligned into the IR viewport through depth and exposed as `Frame.rgb` (BGR), with its own `rgb_timestamp`, because the colour sensor is a separate rolling-shutter camera. Pixels with no depth come out black.
- **`capture/recorder.py`:** records labelled takes to `data/raw/<person>/<session>/<pose>_<lighting>[_glasses][_<attack>]/`: a `meta.json` plus one `.npz` per frame. Laser-on and laser-off frames are both kept, in order, so a liveness window can be replayed. It refuses to run without `--consent`. Run it with `uv run python -m capture.recorder --person p001 --session s1 --pose center --lighting lit --consent`. Space records and q quits.
- **Tests:** `tests/test_recorder.py` covers the save/load round trip with and without RGB, overwrite protection, attack takes and label validation. 58 tests pass.
- **To check on the D435:** the colour stream starts at 848×480, and the API and `liveness.watch` still keep frame rate with colour alignment on every frame. If they do not, turn `rgb` off for the API.

Next: R1, the evaluation harness (`eval/far_frr.py`) and one `Embedder` interface (`face/embed.py`), so candidate models can be compared on recorded takes.

## 2026-10-18 — Printed photo and phone screen rejected

Alan tested a printed photo and a phone screen on this D435 (serial `236422073835`) with `uv run python -m liveness.watch`. Both were rejected. A real face still reads `live`.

Config was unchanged: the starting thresholds in `configs/default.yaml` (`relief_min_m` 0.015, `flat_max_m` 0.005, `nose_cheek_min_m` 0.008). A flat target has almost no depth relief, so the shape check rejects it before any matching would run.

This is a first attack test, not a measured rejection rate. The number of tries, distances and relief readings were not recorded. The PRD's above-95% figure needs recorded spoof takes (`capture/recorder.py`) and the spoof evaluation.

Still open: a bent or curved print, a tablet, video replay, glossy against matte prints, and tuning the thresholds from more people.

## 2026-10-18 — Blur check added

`capture/quality_gate.py` now scores sharpness as the variance of the Laplacian of the IR face crop, and rejects the face when it is below `quality.blur_min` in `configs/default.yaml` (placeholder `20.0`). The crop is resized to 112×112 first, so the score does not change with distance, and the area averaging also smooths the projector dots on laser-on frames, which would otherwise read as sharp edges. The check runs only when the IR image is passed in and `blur_min` is set; `check_geometry` and `judge_window` now pass `frame.ir`. The reject reason prints the score.

Not run on the camera yet. Next: read the score on a still face, a moving face and a defocused one, and set `blur_min` between them. 60 tests pass, including a sharp and a blurred synthetic image.

## 2026-10-18 — Blur check measured: Laplacian variance does not work here

Alan ran `uv run python -m liveness.tune_blur` on this D435 (serial `236422073835`), no glasses, holding still, then moving, then leaning in close. Score = variance of the Laplacian of the 112×112 IR face crop.

| Tag | Frames | Laser-on p5 / median / max | Laser-off p5 / median / max |
| --- | --- | --- | --- |
| sharp | 611 | 321.9 / 393.4 / 1406.7 | 2.6 / 4.0 / 37.0 |
| moving | 192 | 371.4 / 499.0 / 738.9 | 2.6 / 5.9 / 29.2 |
| defocused | 95 | 344.2 / 543.1 / 3739.3 | 3.5 / 4.3 / 46.3 |

An earlier run gave the same ordering on laser-on (sharp median 378, moving 452, defocused 1573).

- **Laser-on is backwards.** Moving and close faces score higher than a still one. The score follows the projector dots, which grow brighter and larger as the face comes closer, not the face's focus. Resizing the crop does not remove them.
- **Laser-off has no signal.** All three tags sit at a median of 4–6. Without the projector the image is dark under the 3 ms exposure cap, so there are hardly any edges to measure.
- **Likely little blur to catch.** With `auto_exposure_limit_us: 3000`, a head moving 0.5 m/s travels about 1.5 mm during one exposure, about a pixel at these distances. The IR cameras are fixed-focus.

Decision: `blur_min` is removed from `configs/default.yaml`, so the check is off. The code stays (`sharpness()` in `capture/quality_gate.py`, the gate runs it only when `blur_min` is set) along with the tuning tool. Revisit with a contrast-normalised score on laser-off, or drop the requirement if motion blur never shows up in the recorded takes.

## 2026-10-18 — R1: evaluation harness

Built before any model, so models are chosen by measurement on this D435.

- **`face/embed.py`:** one `Embedder` interface, `embed(ir_crop, depth_crop) -> Embedding(vector, quality)` on the 112×112 crop from `align_crop`. `RawPixelEmbedder` (downscaled pixels) is the floor any real model must beat; it is not a recogniser.
- **`eval/metrics.py`:** FMR / FNMR, threshold at a target FMR, ROC, EER, FPIR / FNIR, rule of three.
- **`eval/far_frr.py`:** `uv run python -m eval.far_frr --embedder raw_pixels`. Each genuine take is one sample (embeddings of its frames averaged), because frames of one take are not independent. Spoof takes are skipped. Protocols: `cross-session` (enrol on each person's first session, probe with the rest; the real test) and `cross-take` (alternate takes; optimistic, for dry runs). Each probe is also searched with its own person removed, which gives the non-mated 1:N searches. The 95% FMR bound counts distinct person pairs, not scores. Targets finer than 1/n are marked unresolvable. Output: `data/eval/<embedder>_<protocol>_<emitter>/` with `report.json`, `roc.csv`, `scores.csv`, `takes.csv`.
- **Frames:** laser-off by default (`--emitter off`), since it has no dot pattern. The blur runs showed laser-off is dark under the 3 ms cap, so check `takes.csv` for how many frames had a face; if few, try `--emitter on`.
- **Tests:** `tests/test_eval.py`, 10 tests. 70 pass.

Not run yet: there are no recordings in `data/raw/`. Next: record 2+ people (2 sessions each for `cross-session`), run the raw-pixel baseline, then add the first real model (R2).

## How to add an entry

Add a new `## YYYY-MM-DD — title` section at the bottom. Keep the camera serial, the config value, and the pass/fail result in the entry. Update [CHECKLIST.md](CHECKLIST.md) when a box changes.
