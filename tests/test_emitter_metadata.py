import pytest

from capture.realsense_stream import emitter_state_from_metadata


def test_laser_on_off_mapping():
    assert emitter_state_from_metadata(1) == "on"
    assert emitter_state_from_metadata(0) == "off"


@pytest.mark.parametrize("value", [-1, 2, 3, 255])
def test_unknown_value_rejected(value: int):
    with pytest.raises(ValueError, match="frame_laser_power_mode"):
        emitter_state_from_metadata(value)
