# Journal

Running log of decisions, what was built, and what the camera actually did.
The spec is [PRD.md](PRD.md). The task list is [CHECKLIST.md](CHECKLIST.md).
How to run the capture code is in the [README](../README.md) and [capture.md](capture.md).

**Project:** realsense-face-id
**Owner:** Alan Francis Santhosh
**Camera:** Intel RealSense D435, serial `236422073835`, firmware `5.17.3.10`, USB id `8086:0b07`
**Machines:** the x86_64 Linux laptop, which is where entries through 2026-10-19 were made. Ameen's Windows laptop (host `ameenas`) is a second machine: the 2026-10-22 recordings and `eval.far_frr` runs were made there, with this same D435 plugged in. `data/raw` and `data/eval` on the two laptops are not the same files. Do not copy one onto the other. `p001/s1` on Ameen's laptop is a different person from Alan's 18 October `p001/s1` on the Linux laptop. Raspberry Pi 4 is the PRD's eventual target and is out of scope for now.

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

## 2026-10-18 — Recording from the console

Takes can now be recorded from the web console, so one person can record alone without the OpenCV window.

- **`POST /v1/recordings`** (admin token) takes the same labels as `capture.recorder` and refuses without `consent: true`. `CameraHub.capture` collects the next N frames from the API's one pipeline, laser-on and laser-off in order, so the recorder and the API never fight over the D435. The files are written by the same `save_take`, so `eval.far_frr` reads them unchanged. The reply counts frames with exactly one face, per emitter state, the same way the harness does, so an unusable take shows up at once.
- **`GET /v1/recordings`** lists takes for a person and session. **`DELETE /v1/recordings/{person}/{session}/{take}`** removes a bad take, and the empty folders above it.
- **Record page (`/record`):** live IR view, labels, a countdown (3 s by default, so you can sit back or hold up a print), the lean protocol as a checklist (five lit poses, one dark, glasses if ticked, print and screen spoofs), and Space to record.
- **Tests:** `tests/test_recordings_api.py`, 10 tests. 80 pass.

Not run on the camera yet. Next: record p001 s1 from the console and check the face counts.

## 2026-10-18 — Guided recording, like phone face enrolment

The Record page now leads the person through the takes instead of asking them to pick labels. A ring of ticks sits round a mirrored live view and fills as the head turns, with prompts such as "Move closer", "Only one face, please" and "Move your head slowly to complete the circle".

- **`face/pose.py`:** a rough head turn from the five YuNet landmarks. The nose offset from the centre of the eyes and mouth, in the eye-line frame (roll does not count) and divided by the eye distance (camera distance does not count). It is not degrees: it is always taken relative to the person's own straight-ahead pose. `left` means the person's own left.
- **`enroll/guide.py`:** the steps. `circle` first holds still for `pairs_per_pose` laser-on/off pairs to learn the straight-ahead pose (moving resets it), then fills `center` plus eight directions (`right`, `up_right` … `down_right`) until each has `pairs_per_pose` pairs past `reach_min`. `dark`, `print` and `screen` hold one pose for 15 pairs. Faces must pass the quality gate, except on spoof steps, where only size is checked because a screen often has no depth. Each direction is saved as an ordinary take, so `eval.far_frr` reads it unchanged.
- **Pairs:** frames are kept as a laser-on frame and the laser-off frame after it (`Pairer`), so a take still has both kinds side by side. The frames of one direction are not one continuous second; they are collected while the head moves.
- **`api/services/guided.py`:** runs the guide in its own thread with its own detector. The camera thread only drops frames into a short queue, so a slow detector skips frames and the preview does not stall. `POST/GET/DELETE /v1/recordings/guided` start, poll and cancel. It refuses to start if any take it would write exists, unless overwrite is set. A run gives up after `guide.timeout_s`.
- **`configs/default.yaml`:** new `guide` section (`center_max` 0.06, `reach_min` 0.15, `pairs_per_pose` 10 (later 6), `ticks` 48, `timeout_s` 180). Starting values, not tuned.
- The single-take form is still there under "Manual take".
- **Tests:** `tests/test_guide.py` (12) and `tests/test_guided_api.py` (6). 99 pass.

Not run on the camera yet. To check: whether `reach_min` is reachable in every direction without losing the face, especially up and the diagonals, and how many pairs per second the worker keeps up with.

## 2026-10-18 — Guided circle did not complete

First try on this D435: the ring looked almost full, but the prompt stayed on "Move your head slowly to complete the circle" and the step never finished.

- **Ticks lied.** A tick lit on the first frame pair seen at that angle, but a direction is only done after `pairs_per_pose` pairs. Several directions were still short while the ring looked full. Now each direction owns the ticks in its 45° sector, and they light in proportion to the pairs collected.
- **Pairs were broken by dropped frames.** Frames went one by one into a 6-slot queue that dropped new frames when full. Once the worker fell behind, the queue held every few frames, so a laser-on frame rarely had its laser-off partner next to it, and lone frames were thrown away. Pairing now happens in the camera callback, and the queue holds whole pairs, dropping the oldest pair.
- **Prompts point the way.** "Good. Hold it there a moment." while a direction is filling, then "Now turn up and to your left." for the nearest direction still short.
- `pairs_per_pose` lowered from 10 to 6 (12 frames per direction take), so each direction needs under a second of holding at 15 pairs/s.
- The page shows the pairs judged per second. About 15 means the worker keeps up with the camera.

## 2026-10-18 — Head pose from depth instead of the 2D landmarks

Second try on this D435: the circle filled faster, but up and down still stuck, and "Move your head slowly" kept showing while the head was already turned.

- **Cause.** The first `face/pose.py` read the turn from the image alone: the nose tip's offset from the middle of the eyes and mouth. Sideways the nose moves a lot, so left/right worked. For a nod it hardly moves in the image, so a plain up or down rarely reached the threshold, while diagonals did because their sideways part added to it.
- **Now.** The eyes and mouth corners are deprojected with the laser-on depth (7×7 median per point), and the face direction is the normal of the plane they span. That gives yaw and pitch in degrees, and pitch is measured as directly as yaw. If depth is missing at any point the pair is skipped with "Keep your whole face in view of the camera."
- **Thresholds in degrees,** as an ellipse: `reach_yaw_deg` 18 and `reach_pitch_deg` 12, because necks nod less than they turn. `center_max_deg` 6 for holding still. Starting values.
- **Prompt.** "A little further." when the turn is at least halfway to the ring but short of it.
- **Pointer.** A dot inside the ring shows where the system thinks the head points; it has to reach the ticks to count. If it moves the wrong way, the sign is wrong.
- Tests build flat tilted depth maps and check the degrees, roll, missing depth, and that a plain nod fills up and down. 103 pass.

## 2026-10-18 — First guided session recorded (p001 s1)

On the Linux laptop, Alan recorded himself with the guided Record page on this D435 (serial `236422073835`). With the depth-based head pose the pointer dot moves the right way and the circle completes, in about 12 s. Takes in `data/raw/p001/s1/` on that laptop: `center_lit` and the eight directions (12 frames each), `center_dark` and `center_lit_screen` (30 frames each). No printed photo was at hand, so `center_lit_print` is missing.

Frames with exactly one face of at least 80 px (YuNet on the stored IR):

| Take | Laser-on | Laser-off |
| --- | --- | --- |
| center_lit | 6/6 | 6/6 |
| up, up_left | 6/6 | 6/6 |
| up_right | 4/6 | 6/6 |
| left / right | 3/6, 2/6 | 6/6 |
| down, down_left, down_right | 0/6 | 6/6 |
| center_dark | 1/15 | 15/15 |
| center_lit_screen | 1/15 | 14/15 |

- **Laser-off is the frame to use for identity.** The face is found in every laser-off frame and often missed on laser-on, where the dot pattern breaks it up, worst when looking down. This answers the open question from R1: `eval.far_frr` should stay on `--emitter off`. The guide still worked because `faces_for_depth_frame` falls back to the laser-off image.
- The dark take works through laser-off IR as well.

Next: session 2 on another day, a printed photo when one is available, and a second person so `eval.far_frr` can run.

## 2026-10-18 — R2 starts: SFace and ArcFace as embedders

Two real face models are now `Embedder`s next to the raw-pixel floor, so `eval.far_frr --embedder sface|arcface_r50` compares them on the same takes.

- **SFace** (OpenCV Zoo, 128-D, Apache-2.0) through `cv2.FaceRecognizerSF`, about 3 ms per face.
- **ArcFace R50** (InsightFace buffalo_l `w600k_r50`, 512-D, non-commercial) through `cv2.dnn`, about 22 ms per face. Chosen because the research found large ArcFace models carry over from colour to near-infrared with little loss, while small models degrade.
- Both run in OpenCV, so no new dependency. The files are too large for git (ArcFace is 175 MB): `face/models.py` pins each by sha256 and `uv run python -m face.models` downloads and checks them. `.gitignore` keeps `models/*.onnx` out except YuNet.

First look on p001 s1 (laser-off frames, each take averaged): cosine similarity of every take to `center_lit`.

| Take | raw_pixels | sface | arcface_r50 |
| --- | --- | --- | --- |
| center_dark | 0.74 | 0.59 | 0.78 |
| left / right | -0.02 / 0.39 | 0.65 / 0.56 | 0.78 / 0.72 |
| up / down | 0.38 / 0.27 | 0.34 / 0.39 | 0.55 / 0.65 |
| diagonals | -0.10 to 0.50 | 0.45 to 0.54 | 0.63 to 0.73 |
| center_lit_screen (spoof) | -0.05 | 0.14 | 0.56 |

- Raw pixels fall apart once the head turns, as expected of a floor. Both models hold the same person across poses and in the dark on IR, and ArcFace holds them highest.
- This is only same-person similarity. Whether the models separate people needs a second person; until then nothing here is a FAR or FRR.
- ArcFace matches the phone-screen photo to Alan (0.56). Identity alone would accept that spoof; the depth liveness check is what stops it, so liveness must stay ahead of matching.

Next: `eval.far_frr` on all three embedders once p002 is recorded.

## 2026-10-19 — AdaFace IR-101 added as a third embedder

AdaFace trains with a margin that adapts to image quality, so dark or turned faces pull the model less. That fits laser-off IR crops. It is now `eval.far_frr --embedder adaface_ir101`.

- **Weights.** The CVLFace IR-101 trained on WebFace12M (512-D). Only PyTorch weights are published, so `face/adaface_export.py` rebuilds the network locally (no `trust_remote_code`), loads the pinned `model.safetensors` strictly, and exports ONNX once. OpenCV matches PyTorch to 8.7e-6.
- **Torch only for the export.** A new `export` dependency group holds torch, safetensors and onnx. The embedder itself runs in `cv2.dnn`, so the normal install is unchanged. Torch is the CUDA 13.2 build (`pytorch-cu132` index; driver 610.57, RTX 4060) so step 7 training can reuse it; the export itself runs on the CPU.
- **Speed.** About 42 ms per face, against 23 ms for ArcFace R50 and 3 ms for SFace.
- **Quality number.** The length of the raw output before normalising is AdaFace's own quality estimate. On p001 s1 it is highest for `center_lit` (24.9) and lowest for the phone screen (22.1). Too few takes to use it for anything yet.

Same check as for ArcFace (p001 s1, laser-off frames, each take averaged, cosine to `center_lit`):

| Take | sface | arcface_r50 | adaface_ir101 |
| --- | --- | --- | --- |
| center_dark | 0.59 | 0.78 | 0.79 |
| left / right | 0.65 / 0.56 | 0.78 / 0.72 | 0.76 / 0.74 |
| up / down | 0.34 / 0.39 | 0.55 / 0.65 | 0.67 / 0.67 |
| diagonals | 0.45 to 0.54 | 0.63 to 0.73 | 0.64 to 0.75 |
| center_lit_screen (spoof) | 0.14 | 0.56 | 0.58 |

- SFace and ArcFace give the same numbers as on 2026-10-18, so the setup is unchanged.
- AdaFace is about level with ArcFace, a little better looking up (0.67 against 0.55). Same-person only; the choice still waits for p002.
- AdaFace also matches the phone screen (0.58). Liveness stays ahead of matching.

## 2026-10-22 — Windows metadata, before any recording

On Ameen's Windows laptop the API came up and `GET /v1/device/info` read this D435 (serial `236422073835`, firmware `5.17.3.10`, USB 3.2). The preview then stopped. `/health` stayed `camera: error`, and `/v1/preview/stream` returned 503. The stream raised `depth frame has no frame_laser_power_mode metadata`. That message names the Linux udev patch. On Windows the same gap is the metadata registry: the depth and RGB nodes were present and `Status OK` on the inbox `usbvideo` driver, and `MetadataBufferSizeInKB*` was missing. Two phantom nodes from an earlier USB port (`6&D043D2C`) were not the camera in use.

The keys are present now, which is what `realsense_metadata_win10.ps1 -op install` writes. Depth interface `6&30854A99`: `MetadataBufferSizeInKB0` through `3`, each 5. RGB interface: `MetadataBufferSizeInKB0` = 5. The recordings below alternate laser-on and laser-off, so emitter tags were available by 05:50 UTC.

## 2026-10-22 — Second person, first session (p002 s1)

Recorded on Ameen's Windows laptop, not the Linux laptop. p002 recorded a full guided session on this D435 (serial `236422073835`), wearing glasses. Takes in `data/raw/p002/s1/` on that laptop only: `center_lit_glasses` and the eight directions (12 frames each), then `center_dark_glasses`, `center_lit_print` and `center_lit_screen` (30 frames each). Every take alternates laser-on then laser-off. The circle finished. Alan's 18 October `p001/s1` on the Linux laptop has no print; this session does. Genuine takes are labelled `glasses: true`. The two spoof takes are not.

Same count as the 18 October check: frames with exactly one face of at least 80 px (YuNet on the stored IR). Misses below are frames with no face. No frame had two faces.

| Take | Laser-on | Laser-off |
| --- | --- | --- |
| center_lit_glasses | 6/6 | 6/6 |
| right, up_right | 5/6, 5/6 | 6/6, 6/6 |
| up | 4/6 | 6/6 |
| up_left | 3/6 | 6/6 |
| left | 2/6 | 6/6 |
| down_left, down_right | 1/6, 1/6 | 6/6, 6/6 |
| down | 0/6 | 6/6 |
| center_dark_glasses | 7/15 | 14/15 |
| center_lit_print | 0/15 | 15/15 |
| center_lit_screen | 0/15 | 15/15 |

- **Laser-off is still the identity frame.** Every genuine laser-off frame has one face, except one dark frame. Looking down still wipes the face out of the laser-on image, the same pattern as Alan's 18 October session on the Linux laptop.
- The print and the screen are found on every laser-off frame and on no laser-on frame. Identity would see both spoofs. Liveness has to reject them.
- The dark take is usable on laser-off IR, and its laser-on count (7/15) is higher than Alan's 18 October `p001` on the Linux laptop (1/15). That comparison is across laptops, not two takes in one folder.
- Embedding weights were not in `models/` on Ameen's laptop yet, so the cosine check from 2026-10-18 was not repeated. They were fetched later the same day, before the `far_frr` runs below. This session is one person with glasses only.

Next: p002 on another day without glasses if they can, and the three embedders on p001 plus p002.

## 2026-10-22 — p001 s1 on Ameen's laptop, no print

A second person was recorded on Ameen's Windows laptop as `p001` `s1` on this D435 (serial `236422073835`), without glasses. This is not Alan's 18 October `p001/s1`. Those files stay on the Linux laptop. The two folders share a name and must not be copied over each other. Takes on Ameen's laptop: `center_lit` and the eight directions, `center_dark`, and `center_lit_screen`. No printed photo, so `center_lit_print` is absent. That does not block `eval.far_frr`: spoof takes are skipped, and p002 on this same laptop already has a print.

Same count: frames with exactly one face of at least 80 px. Misses are frames with no face. No frame had two faces.

| Take | Laser-on | Laser-off |
| --- | --- | --- |
| center_lit | 6/6 | 5/5 |
| right | 6/6 | 6/6 |
| up | 5/6 | 6/6 |
| up_right | 3/6 | 6/6 |
| up_left | 1/6 | 6/6 |
| left | 2/6 | 6/6 |
| down | 3/6 | 6/6 |
| down_left, down_right | 0/6, 0/6 | 6/6, 6/6 |
| center_dark | 4/15 | 12/15 |
| center_lit_screen | 5/15 | 15/15 |

- **Laser-off is usable on every genuine take.** The directions are 6/6. `center_lit` saved 11 frames: the first two are both laser-on, then five on/off pairs, so the laser-off count is 5/5.
- The dark take missed 3 of 15 laser-off frames. Twelve remain, which is enough to embed.
- The screen is found on every laser-off frame. Identity would see it. Liveness has to reject it.
- `p001` and `p002` are now both on disk, so `eval.far_frr --protocol cross-take --emitter off` can run.

## 2026-10-22 — First far_frr on the two sessions

Run on Ameen's Windows laptop, using only the `p001` and `p002` takes recorded there the same day. Alan's 18 October session was not in this run. An earlier run the same morning, with only `p002` on disk, stopped immediately: `need at least 2 enrolled people for impostor scores, have 1`. SFace and ArcFace weights were downloaded on this laptop first (`face.models`). `eval.far_frr --protocol cross-take --emitter off` on `p001` s1 and `p002` s1. Spoofs skipped. `cross-session` was not run: each person has only `s1`. Each person contributes 10 genuine takes; alternate takes enrol and the rest probe, so 10 probes, 10 genuine scores and 10 impostor scores, and only 1 person pair. Reports in `data/eval/sface_cross-take_off` and `data/eval/arcface_r50_cross-take_off`.

| Embedder | Genuine (min–max) | Impostor (min–max) | EER | Rank-1 |
| --- | --- | --- | --- | --- |
| sface | 0.52–0.81 | 0.29–0.65 | 10% at 0.646 | 100% |
| arcface_r50 | 0.69–0.87 | 0.14–0.25 | 0% at 0.691 | 100% |

- ArcFace leaves a gap: the weakest same-person score is 0.69 (`p001` `center_lit`) and the strongest other-person score is 0.25. No probe is closer to the wrong person.
- SFace overlaps. The weakest genuine is 0.52 (`p001` `center_lit`). The strongest impostor is 0.65 (`p002` `down_lit_glasses` against `p001`). Clearing every impostor rejects 3 of 10 genuine probes (FNMR 30% at 0.654). Rank-1 is still 100% because the right person remains the nearer of the two.
- The FMR bound prints `1.00e+00` for both. It is the rule of three on person pairs, `3/1`, capped at 1. One pair cannot show a 1% false-match rate, which is why every target line says "too few impostors to resolve." The 0% EER is "no mistake on these 10 probes," not a door threshold.
- This is the optimistic split: enrol and probe are poses from the same minute. `cross-session` still needs a second day.

## 2026-10-22 — AdaFace on the same two sessions

Run on Ameen's Windows laptop, on the same takes as the SFace and ArcFace runs above. `adaface_ir101` on the same `cross-take`, laser-off split. The ONNX export matches the PyTorch weights to 1.26e-05. Report in `data/eval/adaface_ir101_cross-take_off`.

| Embedder | Genuine (min–max) | Impostor (min–max) | EER | Rank-1 |
| --- | --- | --- | --- | --- |
| sface | 0.52–0.81 | 0.29–0.65 | 10% at 0.646 | 100% |
| arcface_r50 | 0.69–0.87 | 0.14–0.25 | 0% at 0.691 | 100% |
| adaface_ir101 | 0.72–0.87 | 0.11–0.22 | 0% at 0.718 | 100% |

- AdaFace leaves a slightly wider gap than ArcFace. The weakest same-person score is 0.72 (`p001` `up_left_lit`) and the strongest other-person score is 0.22. No probe is closer to the wrong person.
- The FMR bound is again `1.00e+00`, and every target line is unresolved. Same reason: one person pair. The 0% EER is still "no mistake on these 10 probes."
- On this dry run AdaFace and ArcFace both separate the two people, and SFace does not. Choosing between AdaFace and ArcFace waits for a second day and more people.

## 2026-10-24 — Console redesign

- Redesigned Preview, Record, People, and Device around a light workspace, teal actions, Public Sans, and dark camera previews. The supplied face-ring reference informed the capture stage. A saved Obys grid screenshot informed the restrained alignment and information hierarchy.
- The recording ring follows real API ticks and capture states. Running, saving, completion, and failure have distinct treatments. Reduced motion removes animated effects. Setup controls stay locked during active capture, and cancellation and connection recovery give visible feedback.
- Public Sans is bundled locally through Fontsource under the SIL Open Font License. Intel One Mono remains for small sensor readouts.
- The production Astro build passed. Playwright checked all four routes at 1440, 834, 390, and 320 CSS pixels with no horizontal overflow or uncaught page errors in the simulated API states. Visually reviewed desktop, tablet, and phone layouts.
- Browser checks covered required setup, guided start and cancellation, real-tick rendering, saving and completion states, keyboard step selection and retained focus, reduced motion, authentication denial, empty people, enrollment, and device retry. Writes used simulated API responses. No physical camera or door test was performed.
- Trail was unavailable and `.trail/cache/INDEX.md` was absent, so the context graph could not be read or rebuilt.

## 2026-10-24 — Neutral console refinement

- Replaced the teal palette with neutral graphite, off-white, and gray after visual feedback. Removed the avatar illustration, gradients, glow, and decorative preview glyphs. The capture ring now uses short white progress marks around the actual camera viewport and a small camera placeholder when the feed is absent.
- Kept real capture progress, state transitions, keyboard selection, and reduced-motion behavior. The production build passed. Playwright checked all four routes at 1440, 834, 390, and 320 CSS pixels without horizontal overflow. Desktop recording and phone preview were visually reviewed. No physical camera test was performed.

## 2026-10-24 — Typography and workspace composition

- Replaced Public Sans with locally bundled Bricolage Grotesque for headings and Instrument Sans for controls and reading. Both packages include SIL Open Font Licenses. Intel One Mono remains for sensor labels. Removed the unused Public Sans package.
- Kept the navigation and capture ring. Rebuilt Preview as a joined camera and control workspace, with the check action above the instructions and a restrained yellow action color. Directory and hardware views now use indexed sections and a separate dark guidance panel. The device diagram is an illustration, not a live hardware reading.
- The browser confirmed all three font families loaded. Reviewed desktop Preview and Record, phone People, and tablet Device. All four routes fit at 1440, 834, 390, and 320 CSS pixels without horizontal overflow. The main liveness action fits within a 1440 by 700 viewport. Simulated denial, keyboard step selection, and reduced motion still work, with no uncaught page errors during those checks. Physical camera behavior was not retested.

## 2026-10-22 — Matching, door state machine, mock relay and access log

`/authenticate` now goes the whole way: liveness, then 1:N matching, then the door. A live, enrolled face unlocks the mock relay; everything else is denied and logged. This is step 4's matcher and most of step 6, built without the camera so the morning session can test the full path.

**Matching (`face/match.py`).**

- The gallery keeps each person's mean template and every individual enrolment vector, and a person scores their best row. The PRD asks for both. The individual rows help when a pose was only seen once at enrolment.
- One attempt averages the laser-off faces of the window into one query. The per-frame top matches must also agree (`match.agree_frac: 0.6` of at least `match.frames_required: 3` faces). One odd frame cannot open the door on its own.
- Laser-off frames are used because the face is found in every one of them, and the dot pattern breaks it up on laser-on (2026-10-17).
- A laser-off face only counts if it sits where the live face was (within `MAX_MOTION_FRAC` of the image width). The matcher cannot be shown a different face than the one whose depth was checked.
- Liveness still runs first. A spoof is never embedded, and a test checks that.
- The threshold is still the PRD starting value, 0.45. Same-person ArcFace scores were 0.55 to 0.79, and the phone screen scored 0.56 (2026-10-18), so 0.45 is low. It waits for p002 and `eval.far_frr`.

**Door (`control/`).**

- `DoorController` walks the PRD states and records the path for the log. It takes a clock as an argument, so the tests drive time and do not sleep.
- COOLDOWN is used for the lockout, not as a step after every unlock. The config already pairs `cooldown_seconds` with `max_failures`. After an unlock the door goes back to IDLE when `unlock_seconds` runs out.
- Only failures that say something about the person count toward the lockout: a measured wrong face shape, an unknown face, or frames that disagree. A face too far away, two faces, motion, or a missing gallery do not count. Otherwise a real user standing badly would lock themselves out. These attempts never reach the matcher, so not counting them gives an attacker nothing to probe. `shape_failed` on the liveness result tells the two kinds apart.
- A lockout also closes a door that is still open. A grant while the door is open keeps it open for longer. Stopping the API locks the door.
- The mock relay starts locked and only logs `UNLOCK` / `LOCK`. A thread in `DoorService` calls `poll` every 0.2 s so the re-lock happens even if no request comes in.

**Access log (`logging_/access_log.py`).** These are the PRD columns. `liveness_geom` is the relief in metres and `liveness_model` stays NULL until step 7. Three columns were added: `status`, `state` and `path`, so a spoof can be told apart from a badly framed face. Every call is logged, including no-face and camera-down calls. Admins read it with `GET /v1/auth/log`.

**API answer.** `decision` is now `granted` or `denied`, and `status` says how far the attempt got. `user_id` is only filled in on a grant, so a denied caller learns nothing about who is enrolled.

**Gallery for now (`enroll/from_takes.py`).** Real enrolment (`enroll.py`, the encrypted store) is not built. Until then this builds `data/gallery.npz` from recorded takes, each person from their first session. The file is not encrypted, and deleting a user in the API does not remove them from it. Both are for `enroll/store.py`. The API reloads the file when it changes. The embedder is whichever one built the gallery, so the config cannot name a different one.

**Replay on p001 s1** (ArcFace gallery from s1, each take's first 16 frames through `evaluate_window`, the same code as the API):

| Takes | Result |
| --- | --- |
| center_lit, center_dark, right, up, up_left, up_right | matched p001, 230-380 ms per attempt |
| center_lit_screen | rejected ("move to the centre of the camera") |
| down, down_left, down_right, left | rejected by liveness: nose-to-cheek gap 0.3-0.7 cm (need 0.8). down_left read -76 cm, a bad depth sample |

- The 1.00 scores only mean each take was in the gallery. This checks the wiring, not accuracy.
- **Watch:** looking down fails the nose-to-cheek check, and that counts toward the lockout. If it happens live tomorrow, either `nose_cheek_min_m` is too strict for head-down poses, or the nose sample needs to be more robust (the -76 cm value).

Tests: 146 pass, all without the camera.

Next (morning, on the camera): rebuild the gallery, then try `/authenticate` live: enrolled face, a stranger if one is around, the phone screen, and the print. Note how often a real face gets a counted failure.

## 2026-10-28 — Review AdaFace export validation

Reviewed commit `511c163` separately from the existing uncommitted door changes.
Its 112 tests pass with the local embedding models present. The AdaFace weights
match their pinned SHA256, and a fresh ONNX export agrees with PyTorch to
8.66e-6, within the export's 1e-3 tolerance.

The validation could accept NaN output because `max(0.0, nan)` returns 0.0.
It now rejects non-finite outputs from either backend and removes the temporary
file even if export raises, preserving any existing final model on failure.
Eleven regression cases cover these paths; eight fail on the original code.
The working tree passes 157 tests. The API tests need to run outside the agent
sandbox here because TestClient startup hangs inside it. Lint passes on the
changed Python files; repository lint still reports two pre-existing TRY004
findings in `capture/config.py`. No camera or new identity recordings were used.

## 2026-10-28 — Demo preparation

No RealSense camera is connected. Built `data/gallery.npz` from the existing
consented p001 s1 genuine takes using ArcFace R50 (10 take vectors). The local
API configuration points to this gallery. `pnpm --dir web build` passes.

Replayed each take's first 16 frames through `DoorService`, using a temporary
access log and advancing the controller clock between takes. Centre lit/dark,
right and upward poses grant access. Left/downward poses fail the nose-to-cheek
gate; down-left still has the -76 cm outlier. The screen take is denied for
framing, so this replay does not prove geometry rejection of that attack.
Gallery and replay use the same session; these are wiring checks, not accuracy.

The live rehearsal is still pending. `docs/demo.md` gives startup steps,
rehearsal checks and an explicit camera-free fallback.

## 2026-10-28 — Integrate the updated console on Linux

Fast-forwarded to `acd07ee`, including the Windows evaluation notes and the
console redesign. Preserved the local matching and door changes. Resolved the
checklist and journal conflicts by retaining both sets of work. Updated Preview,
People and README wording to describe gallery matching and mock relay control;
the People directory still does not add or revoke gallery embeddings.

An initial failed guided-status request enabled capture despite unknown service
state. Saving a valid admin token refreshed options but never retried guided
status, leaving an existing capture without its cancellation control. Capture
now remains disabled until status is known, and saving the token retries status.
Playwright reproduced the failure and verified recovery and cancellation after
the fix, using simulated API responses and no identity writes.

All 157 backend tests pass. Browser checks pass authentication grant/denial,
directory add/delete, device retry, and all four routes at widths 1440, 834, 390
and 320 pixels without horizontal overflow or uncaught page errors. No physical
camera was used. The two laptops' recordings were not copied or combined.

## 2026-10-28 — Live authentication and visible mock door state on Linux

`POST /authenticate` and `POST /v1/auth` already ran liveness, matching, the
controller and access logging. Added `GET /v1/auth/state` and a polling indicator
on Preview so the demo shows the actual lock, unlock and cooldown state. The
controller snapshot reads the timer-adjusted state, relay and failures together.

On the Linux laptop, D435 `236422073835`, firmware `5.17.3.10`, streamed live.
Early attempts found two or three faces and were denied. After a USB disconnect
and application restart, a live `/authenticate` call granted p001: ArcFace score
0.7235 against threshold 0.45, 8/8 laser-off frames agreed, relief 0.0469 m and
nose-to-cheek gap 0.0233 m. The subsequent state read reported locked and idle.
This is one live grant, not a measured success rate or a live spoof rehearsal.

The disconnect exposed a cleanup bug: `pipeline.stop()` raised after the device
had already stopped and hid the useful disconnection message. Cleanup now keeps
the original capture exception. Preview polls camera health and restores its
stream when the restarted service becomes available. Replugging still requires
restarting the API; automatic camera reconnection was not added.

All 164 tests pass, including route grants for both aliases, timed re-lock,
logging, spoof denial, lockout and disconnect cleanup. A separate HTTP check
using the local recorded centre take and actual models granted p001 and observed
the real five-second mock re-lock. This recorded check used temporary logs and
did not claim live input. Browser checks pass the visible door state transition,
existing console flows and responsive routes. The frontend build passes.

## 2026-10-28 — People page reflects the actual face gallery

On the Linux laptop, People said nobody was enrolled while p001 could
authenticate. `/v1/users` only read the metadata database, but the matcher read
the separate gallery. Listing now combines both sources, deduplicates IDs and
returns `face_enrolled`. Gallery-only identities have no invented creation date.
The page labels each row Face enrolled or ID only and calls the metadata form
Add ID. Live face capture/enrolment is still a separate unfinished workflow.

Delete, including the PRD alias, now removes gallery embeddings and metadata.
Gallery updates are serialized with authentication, written through a temporary
file and atomically replaced, and invalidate the cached matcher. Removing the
last face removes the gallery file. Failed writes preserve the old gallery and
metadata. Saved takes remain; an explicit gallery rebuild can re-enroll them.

All 174 tests pass, including ten directory/gallery cases: gallery-only listing,
mixed IDs, duplicate IDs, access revocation with a cached matcher, last-person
deletion, and corrupt-gallery/write-failure handling. The frontend build passes.
The live API lists the existing p001 as face enrolled and rejects anonymous
listing. No actual enrolled identity or recorded sample was deleted during
verification.

## 2026-10-28 — Disable failed-attempt lockout during development

On the Linux development machine, `door.max_failures: 0` now disables the
failed-attempt cooldown so authentication can be retried immediately. A positive
value restores the existing lockout. Authentication denials, access logging and
automatic re-locking remain in place. Verification uses fake frames and a mock
relay; no camera measurement was made.
All 179 camera-free tests pass, including repeated spoof denials followed by a
successful authentication without a cooldown. Ruff checks pass for the changed
Python files.

## 2026-10-28 — Finish enrollment from a saved recording session

The Record page saved p002's takes but never enrolled their embeddings, so
People still listed only p001. Added an explicit Finish enrollment action and
admin-only saved-session endpoint. It uses genuine laser-off samples with the
gallery's existing embedder (ArcFace R50 for a new gallery), replaces only the
selected person's rows, writes atomically, and invalidates the cached matcher.
Consent must be confirmed. Capture completion now points to enrollment instead
of implying that saved recordings already provide face access.

Recovered p002 on this Linux machine from all nine usable s1 takes with glasses.
The ArcFace gallery now lists p001 and p002; p001's vectors are unchanged.
No new camera capture or physical authentication test was made. All 185 tests
pass, including session enrollment, repeated enrollment, matcher refresh,
preserving other identities, consent, invalid/missing sessions, attack-only
sessions and failed writes. The console build and changed-file Ruff checks pass.

## How to add an entry

Add a new `## YYYY-MM-DD — title` section at the bottom. Keep the camera serial, the laptop, the config value, and the pass/fail result in the entry. Name the laptop when it is not the Linux one. Update [CHECKLIST.md](CHECKLIST.md) when a box changes.
