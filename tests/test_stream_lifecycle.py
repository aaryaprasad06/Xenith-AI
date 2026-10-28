"""Disconnect cleanup must preserve the useful capture error; no camera used."""

from types import SimpleNamespace

import pytest

from capture.realsense_stream import RealSenseStream


class OpenStream(RealSenseStream):
    def __enter__(self):
        return self


def stream_with_stop(stop):
    stream = object.__new__(OpenStream)
    stream._pipeline = SimpleNamespace(stop=stop)
    return stream


def failed_stop():
    raise RuntimeError("stop() cannot be called before start()")


def test_disconnect_is_not_hidden_by_cleanup_failure():
    stream = stream_with_stop(failed_stop)
    with pytest.raises(RuntimeError, match="Device disconnected"), stream:
        raise RuntimeError("Device disconnected")
    assert stream._pipeline is None


def test_stop_failure_is_reported_when_no_capture_error_exists():
    stream = stream_with_stop(failed_stop)
    with pytest.raises(RuntimeError, match="stop"), stream:
        pass
    assert stream._pipeline is None


def test_cleanup_stops_once_and_can_be_repeated():
    stopped = []
    stream = stream_with_stop(lambda: stopped.append(True))
    with stream:
        pass
    stream.__exit__(None, None, None)
    assert stopped == [True]
