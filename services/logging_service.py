"""
Centralized logging service with rotating file handler and in-memory log buffer for UI display.
"""
from __future__ import annotations

from collections import deque
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Deque, List

from config.settings import LOGS_DIR


class UIMemoryLogHandler(logging.Handler):
    """Custom logging handler that stores recent log entries and notifies UI listeners."""

    def __init__(self, max_records: int = 500) -> None:
        super().__init__()
        self.records: Deque[str] = deque(maxlen=max_records)
        self._callbacks: List[Callable[[str], None]] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.records.append(msg)
            for callback in list(self._callbacks):
                try:
                    callback(msg)
                except Exception:
                    pass
        except Exception:
            self.handleError(record)

    def add_callback(self, callback: Callable[[str], None]) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[str], None]) -> None:
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_recent_logs(self) -> List[str]:
        return list(self.records)


_ui_log_handler: UIMemoryLogHandler | None = None
_initialized: bool = False


def setup_logging(log_dir: Path | None = None, level: int = logging.INFO) -> UIMemoryLogHandler:
    """Configure application-wide logging to file, console, and UI buffer."""
    global _ui_log_handler, _initialized
    target_dir = Path(log_dir) if log_dir else LOGS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    log_file = target_dir / "image_enhancement_studio.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not _initialized:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        _ui_log_handler = UIMemoryLogHandler(max_records=500)
        _ui_log_handler.setFormatter(formatter)
        root_logger.addHandler(_ui_log_handler)
        _initialized = True

    assert _ui_log_handler is not None
    return _ui_log_handler


def get_ui_log_handler() -> UIMemoryLogHandler:
    """Return the active UI memory log handler, initializing logging if needed."""
    global _ui_log_handler
    if _ui_log_handler is None:
        return setup_logging()
    return _ui_log_handler
