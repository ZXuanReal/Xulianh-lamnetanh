"""
Application configuration and persistent settings management.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys
from typing import Any, Dict


def get_base_dir() -> Path:
    """Return base directory supporting both source execution and PyInstaller frozen builds."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


BASE_DIR: Path = get_base_dir()
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
LOGS_DIR: Path = BASE_DIR / "logs"
MODELS_DIR: Path = BASE_DIR / "models"
ASSETS_DIR: Path = BASE_DIR / "assets"
ICONS_DIR: Path = ASSETS_DIR / "icons"
CONFIG_FILE: Path = BASE_DIR / "config" / "user_settings.json"
HISTORY_DB_FILE: Path = BASE_DIR / "outputs" / "history.sqlite3"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_PIXELS = 64_000_000  # ~64 MP safety limit (e.g. 8000x8000)
WARN_IMAGE_PIXELS = 16_000_000  # ~16 MP warning threshold


@dataclass
class AppSettings:
    """User-configurable persistent application settings."""

    default_output_dir: str = field(default_factory=lambda: str(OUTPUTS_DIR))
    default_mode: str = "auto"  # auto, sharpen, denoise, restore_ai, product, document
    compute_device: str = "auto"  # auto, cpu, cuda
    default_export_format: str = "PNG"  # PNG, JPEG, WEBP, ORIGINAL
    jpeg_quality: int = 92
    webp_quality: int = 92
    preserve_metadata: bool = True
    enable_notifications: bool = True
    language: str = "vi"
    tile_size: int = 256
    tile_pad: int = 16
    max_batch_workers: int = 1
    models_dir: str = field(default_factory=lambda: str(MODELS_DIR))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppSettings":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


class SettingsManager:
    """Loads, validates, saves, and resets application settings."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = Path(config_path) if config_path else CONFIG_FILE
        self.ensure_directories()
        self.settings: AppSettings = self.load()

    def ensure_directories(self) -> None:
        """Create required project directories if they do not exist."""
        for directory in (OUTPUTS_DIR, LOGS_DIR, MODELS_DIR, ASSETS_DIR, ICONS_DIR, self.config_path.parent):
            directory.mkdir(parents=True, exist_ok=True)

    def load(self) -> AppSettings:
        """Load settings from JSON or return defaults on missing/corrupt file."""
        if not self.config_path.exists():
            default_settings = AppSettings()
            self.save(default_settings)
            return default_settings
        try:
            raw = self.config_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("Settings root must be a JSON object.")
            settings = AppSettings.from_dict(data)
            return settings
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Không thể đọc file cấu hình %s (%s). Khôi phục cấu hình mặc định.",
                self.config_path,
                exc,
            )
            default_settings = AppSettings()
            self.save(default_settings)
            return default_settings

    def save(self, settings: AppSettings | None = None) -> None:
        """Persist settings to disk with UTF-8 encoding."""
        if settings is not None:
            self.settings = settings
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        Path(self.settings.default_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.settings.models_dir).mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps(self.settings.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def reset_to_defaults(self) -> AppSettings:
        """Reset all settings to factory defaults and persist."""
        self.settings = AppSettings()
        self.save(self.settings)
        return self.settings
