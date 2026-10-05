from pathlib import Path

import pytest
import yaml

from capture.config import DEFAULT_CONFIG_PATH, REQUIRED_SECTIONS, load_config


def test_default_config_loads():
    cfg = load_config()
    for section in REQUIRED_SECTIONS:
        assert isinstance(cfg[section], dict)
    assert cfg["camera"]["width"] == 848
    assert cfg["camera"]["depth_units"] == 0.0001


@pytest.mark.parametrize("section", REQUIRED_SECTIONS)
def test_missing_section_fails(tmp_path: Path, section: str):
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    del cfg[section]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match=section):
        load_config(path)


def test_missing_camera_key_fails(tmp_path: Path):
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    del cfg["camera"]["fps"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="fps"):
        load_config(path)
