# Capture module

`capture/` turns the D435 into a stream of `Frame` objects: a left-IR image and
a depth map on the same pixel grid, each tagged with whether the IR projector
was on or off for that frame.

## Pipeline

```
D435 (z16 depth + left IR y8, 848x480 @ 30 fps)
  -> read laser metadata from the raw depth frame
  -> rs.align(infrared)
  -> emitter on:  filter chain on depth
     emitter off: raw aligned depth
  -> Frame
```

## Frame

| Field | Type | Notes |
| --- | --- | --- |
| `ir` | `uint8` (H, W) | Left infrared. Emitter-on frames show the projector dot pattern; emitter-off frames are clean for face images. |
| `depth` | `float32` (H, W), metres | `0` means invalid. |
| `intrinsics` | `Intrinsics` | Left-IR camera. Plain data copied once at start. |
| `timestamp` | `float` | Device time in seconds. |
| `emitter_state` | `"on"` or `"off"` | From the `frame_laser_power_mode` frame metadata. |

Arrays are copies, so they stay valid after the next frame arrives.

`center_roi_median_m(depth, frac)` returns the median of the non-zero depth in a
centered box (`frac` of each side), or `NaN` if the box has no valid depth.

## Emitter on / off

With `emitter.alternate: true` the projector toggles every frame.

- **Emitter on**: the dot pattern gives good depth. Use these frames for depth.
- **Emitter off**: no dots on the face, so the IR image is clean. Use these frames
  for the face image. Their depth is raw (unfiltered) and noisier.

The state comes from per-frame metadata, never from frame parity. If the driver
does not provide it, the stream raises instead of guessing.

If the camera refuses `emitter_on_off` while auto-exposure is on, the stream
turns auto-exposure off, keeps the current exposure, and retries. Fixed
exposure is worth remembering when you test in very different lighting.

## Depth filter chain

Applied to emitter-on frames only, so the temporal filter never mixes laser-on
and laser-off frames. Order is fixed in `capture/filters.py`:

1. decimation (magnitude must be `1`; anything else is refused so depth keeps
   matching the 848x480 IR image)
2. depth to disparity
3. spatial
4. temporal (`persistency` is written to the temporal filter's `holes_fill`
   option, which is how pyrealsense2 2.58 exposes the persistency index)
5. disparity to depth
6. hole filling (`mode`: 0 fill from left, 1 farthest from around, 2 nearest from around)

### Out-of-range depth

A raw z16 value of 65535 (6.5535 m at 0.1 mm units) is the out-of-range marker,
not a measurement. It is set to `0` in `Frame.depth`.

Hole-filling mode 1 (farthest from around) can fill holes with that marker, so in
a scene with many missing pixels it adds little. For faces near a background,
mode 2 may behave better; try it in `configs/default.yaml` and check against
the tape measure.

## Depth units

`camera.depth_units` is set on the sensor before streaming and the real scale is
read back with `get_depth_scale()`. At `0.0001` the range tops out near 6.55 m,
which is fine for the 0.3-0.6 m door distance.

Tape check on this D435 (2026-10-05): the center median matched a tape from
10 cm to about 100 cm on black and green walls. Past 100 cm was not checked.
A white wall reads `n/a` because the IR image clips and the stereo match fails.
The preview colours cover 0.2–1.0 m only; the printed median is the real value.

## Tests

```bash
uv run pytest
```

| File | Covers |
| --- | --- |
| `tests/test_config.py` | default config loads; missing sections / keys fail |
| `tests/test_roi.py` | center median, invalid-pixel handling, NaN case |
| `tests/test_filters.py` | stage order and options read back from the filter objects |
| `tests/test_emitter_metadata.py` | metadata `1`/`0` to `on`/`off`, other values rejected |

No camera is needed for any of them.
