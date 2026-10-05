"""Depth post-processing chain.

Order is fixed (Intel's recommended scheme for D4xx stereo cameras):

    decimation -> depth-to-disparity -> spatial -> temporal
               -> disparity-to-depth -> hole filling

Only parameters come from the config (`filters:` section).
"""

from __future__ import annotations

from typing import Any

import pyrealsense2 as rs

STAGE_ORDER = (
    "decimation",
    "depth_to_disparity",
    "spatial",
    "temporal",
    "disparity_to_depth",
    "hole_filling",
)


class DepthFilterChain:
    def __init__(self, filters_cfg: dict[str, Any]) -> None:
        dec_cfg = filters_cfg.get("decimation", {})
        magnitude = int(dec_cfg.get("magnitude", 1))
        if magnitude != 1:
            # Depth is aligned to the 848x480 left-IR image; a coarser depth
            # grid would no longer match it.
            raise ValueError(
                f"filters.decimation.magnitude must be 1, got {magnitude}"
            )

        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, magnitude)

        spatial_cfg = filters_cfg.get("spatial", {})
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, spatial_cfg["magnitude"])
        spatial.set_option(rs.option.filter_smooth_alpha, spatial_cfg["smooth_alpha"])
        spatial.set_option(rs.option.filter_smooth_delta, spatial_cfg["smooth_delta"])
        spatial.set_option(rs.option.holes_fill, spatial_cfg["holes_fill"])

        temporal_cfg = filters_cfg.get("temporal", {})
        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, temporal_cfg["smooth_alpha"])
        temporal.set_option(rs.option.filter_smooth_delta, temporal_cfg["smooth_delta"])
        # In pyrealsense2 the persistency index is the temporal filter's holes_fill.
        temporal.set_option(rs.option.holes_fill, temporal_cfg["persistency"])

        hole_cfg = filters_cfg.get("hole_filling", {})
        hole_filling = rs.hole_filling_filter()
        hole_filling.set_option(rs.option.holes_fill, hole_cfg["mode"])

        self.stages: list[tuple[str, Any]] = [
            ("decimation", decimation),
            ("depth_to_disparity", rs.disparity_transform(True)),
            ("spatial", spatial),
            ("temporal", temporal),
            ("disparity_to_depth", rs.disparity_transform(False)),
            ("hole_filling", hole_filling),
        ]

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.stages)

    def process(self, depth_frame: Any) -> Any:
        for _, stage in self.stages:
            depth_frame = stage.process(depth_frame)
        return depth_frame
