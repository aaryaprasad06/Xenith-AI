"""Process settings. Environment variables override defaults. A local `.env` fills in anything unset."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or Path(".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass(frozen=True)
class Settings:
    api_v1_prefix: str = "/v1"
    admin_token: str = ""
    camera_enabled: bool = True
    cors_origins: tuple[str, ...] = (
        "http://127.0.0.1:4321",
        "http://localhost:4321",
    )
    db_path: Path = Path("data/users.db")
    config_path: Path | None = None


def load_settings() -> Settings:
    load_dotenv()
    origins = os.environ.get("FACE_ID_CORS_ORIGINS")
    return Settings(
        admin_token=os.environ.get("FACE_ID_ADMIN_TOKEN", ""),
        camera_enabled=os.environ.get("FACE_ID_CAMERA", "1") != "0",
        cors_origins=tuple(o.strip() for o in origins.split(",")) if origins else Settings.cors_origins,
        db_path=Path(os.environ.get("FACE_ID_DB", "data/users.db")),
        config_path=Path(os.environ["FACE_ID_CONFIG"]) if os.environ.get("FACE_ID_CONFIG") else None,
    )
