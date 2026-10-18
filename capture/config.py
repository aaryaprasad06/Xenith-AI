from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"

REQUIRED_SECTIONS = (
    "camera",
    "quality",
    "liveness",
    "match",
    "door",
    "filters",
    "emitter",
    "preview",
    "guide",
)
REQUIRED_CAMERA_KEYS = ("width", "height", "fps", "depth_units")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"{cfg_path}: top level must be a mapping")
    for section in REQUIRED_SECTIONS:
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"{cfg_path}: missing section '{section}'")
    missing = [k for k in REQUIRED_CAMERA_KEYS if k not in cfg["camera"]]
    if missing:
        raise ValueError(f"{cfg_path}: camera section missing {missing}")
    return cfg
