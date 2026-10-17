from pathlib import Path

import numpy as np
import pytest

from capture.frame import Frame, Intrinsics
from capture.recorder import TakeLabels, iter_takes, load_take, save_take

INTR = Intrinsics(fx=420.0, fy=420.0, ppx=424.0, ppy=240.0, width=8, height=6,
                  model="distortion.brown_conrady", coeffs=(0.0, 0.0, 0.0, 0.0, 0.0))


def _frame(i: int, *, rgb: bool) -> Frame:
    rng = np.random.default_rng(i)
    return Frame(
        ir=rng.integers(0, 255, (6, 8), dtype=np.uint8),
        depth=rng.random((6, 8), dtype=np.float32),
        intrinsics=INTR,
        timestamp=100.0 + i / 30,
        emitter_state="on" if i % 2 == 0 else "off",
        rgb=rng.integers(0, 255, (6, 8, 3), dtype=np.uint8) if rgb else None,
        rgb_timestamp=100.001 + i / 30 if rgb else None,
    )


def _labels(**kw) -> TakeLabels:
    return TakeLabels(**{"person": "p001", "session": "s1", "pose": "center", "lighting": "lit", **kw})


@pytest.mark.parametrize("rgb", [True, False])
def test_take_round_trip(tmp_path: Path, rgb: bool):
    frames = [_frame(i, rgb=rgb) for i in range(4)]
    out = save_take(tmp_path, _labels(glasses=True), frames)
    assert out == tmp_path / "p001" / "s1" / "center_lit_glasses"

    take = load_take(out)
    assert take.labels == _labels(glasses=True)
    assert len(take.frames) == 4
    for saved, loaded in zip(frames, take.frames, strict=True):
        np.testing.assert_array_equal(saved.ir, loaded.ir)
        np.testing.assert_array_equal(saved.depth, loaded.depth)
        assert loaded.emitter_state == saved.emitter_state
        assert loaded.timestamp == saved.timestamp
        assert loaded.intrinsics == INTR
        if rgb:
            np.testing.assert_array_equal(saved.rgb, loaded.rgb)
            assert loaded.rgb_timestamp == saved.rgb_timestamp
        else:
            assert loaded.rgb is None


def test_existing_take_is_not_overwritten(tmp_path: Path):
    save_take(tmp_path, _labels(), [_frame(0, rgb=False)])
    with pytest.raises(FileExistsError):
        save_take(tmp_path, _labels(), [_frame(1, rgb=False)])
    save_take(tmp_path, _labels(), [_frame(1, rgb=False)], overwrite=True)


def test_attack_take_has_its_own_directory(tmp_path: Path):
    real = save_take(tmp_path, _labels(), [_frame(0, rgb=False)])
    spoof = save_take(tmp_path, _labels(attack="print"), [_frame(0, rgb=False)])
    assert spoof.name == "center_lit_print"
    assert iter_takes(tmp_path) == sorted([real, spoof])


@pytest.mark.parametrize(
    "bad",
    [{"person": "../x"}, {"session": ""}, {"pose": "sideways"}, {"lighting": "sunny"}, {"attack": "deepfake"}],
)
def test_bad_labels_rejected(bad: dict):
    with pytest.raises(ValueError):
        _labels(**bad)


def test_empty_take_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        save_take(tmp_path, _labels(), [])
