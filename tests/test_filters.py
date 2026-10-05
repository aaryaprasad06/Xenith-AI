import copy

import pyrealsense2 as rs
import pytest

from capture.config import load_config
from capture.filters import STAGE_ORDER, DepthFilterChain


def _stage(chain: DepthFilterChain, name: str):
    return dict(chain.stages)[name]


def test_stage_order_matches_prd():
    chain = DepthFilterChain(load_config()["filters"])
    assert chain.stage_names == STAGE_ORDER
    assert STAGE_ORDER == (
        "decimation",
        "depth_to_disparity",
        "spatial",
        "temporal",
        "disparity_to_depth",
        "hole_filling",
    )


def test_options_come_from_config():
    cfg = load_config()["filters"]
    chain = DepthFilterChain(cfg)

    spatial = _stage(chain, "spatial")
    assert spatial.get_option(rs.option.filter_magnitude) == cfg["spatial"]["magnitude"]
    assert spatial.get_option(rs.option.filter_smooth_alpha) == pytest.approx(
        cfg["spatial"]["smooth_alpha"]
    )
    assert spatial.get_option(rs.option.filter_smooth_delta) == cfg["spatial"]["smooth_delta"]
    assert spatial.get_option(rs.option.holes_fill) == cfg["spatial"]["holes_fill"]

    temporal = _stage(chain, "temporal")
    assert temporal.get_option(rs.option.filter_smooth_alpha) == pytest.approx(
        cfg["temporal"]["smooth_alpha"]
    )
    assert temporal.get_option(rs.option.filter_smooth_delta) == cfg["temporal"]["smooth_delta"]
    assert temporal.get_option(rs.option.holes_fill) == cfg["temporal"]["persistency"]

    assert _stage(chain, "hole_filling").get_option(rs.option.holes_fill) == cfg["hole_filling"]["mode"]
    assert _stage(chain, "decimation").get_option(rs.option.filter_magnitude) == 1


def test_changed_config_is_applied():
    cfg = copy.deepcopy(load_config()["filters"])
    cfg["spatial"]["magnitude"] = 4
    cfg["temporal"]["persistency"] = 5
    chain = DepthFilterChain(cfg)
    assert _stage(chain, "spatial").get_option(rs.option.filter_magnitude) == 4
    assert _stage(chain, "temporal").get_option(rs.option.holes_fill) == 5


def test_decimation_other_than_one_is_refused():
    cfg = copy.deepcopy(load_config()["filters"])
    cfg["decimation"]["magnitude"] = 2
    with pytest.raises(ValueError, match="magnitude"):
        DepthFilterChain(cfg)
