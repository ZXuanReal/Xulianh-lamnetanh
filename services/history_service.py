"""
Local SQLite history service for tracking image enhancement operations.
Stores processing records strictly locally with no network access.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List

from config.settings import HISTORY_DB_FILE

logger = logging.getLogger(__name__)


@dataclass
class HistoryRecord:
    """Represents a single image enhancement processing record."""

    id: int | None
    timestamp: str
    source_filename: str
    source_path: str
    method: str
    parameters: Dict[str, Any]
    output_path: str
    status: str  # "SUCCESS" or "FAILED"
    duration_sec: float = 0.0
    error_message: str = ""
    input_resolution: str = ""
    output_resolution: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HistoryService:
    """Manages local processing history using SQLite."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else HISTORY_DB_FILE
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        try:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS processing_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        source_filename TEXT NOT NULL,
                        source_path TEXT NOT NULL,
                        method TEXT NOT NULL,
                        parameters TEXT NOT NULL,
                        output_path TEXT NOT NULL,
                        status TEXT NOT NULL,
                        duration_sec REAL DEFAULT 0.0,
                        error_message TEXT DEFAULT '',
                        input_resolution TEXT DEFAULT '',
                        output_resolution TEXT DEFAULT ''
                    )
                    """
                )
                conn.commit()
        except Exception as exc:
            logger.error("Không thể khởi tạo cơ sở dữ liệu lịch sử %s: %s", self.db_path, exc)

    def add_record(
        self,
        source_path: str | Path,
        method: str,
        parameters: Dict[str, Any],
        output_path: str | Path = "",
        status: str = "SUCCESS",
        duration_sec: float = 0.0,
        error_message: str = "",
        input_resolution: str = "",
        output_resolution: str = "",
    ) -> int:
        """Insert a new record into local history and return its row ID."""
        src_path_obj = Path(source_path) if source_path else Path("unknown")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        params_json = json.dumps(parameters, ensure_ascii=False)
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO processing_history (
                        timestamp, source_filename, source_path, method,
                        parameters, output_path, status, duration_sec,
                        error_message, input_resolution, output_resolution
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        src_path_obj.name,
                        str(source_path),
                        method,
                        params_json,
                        str(output_path) if output_path else "",
                        status,
                        round(float(duration_sec), 3),
                        error_message,
                        input_resolution,
                        output_resolution,
                    ),
                )
                conn.commit()
                return int(cursor.lastrowid or 0)
        except Exception as exc:
            logger.error("Lỗi khi lưu lịch sử xử lý: %s", exc)
            return -1

    def list_records(self, limit: int = 200, status_filter: str | None = None) -> List[HistoryRecord]:
        """Fetch recent history records ordered from newest to oldest."""
        records: List[HistoryRecord] = []
        try:
            with self._get_conn() as conn:
                if status_filter:
                    cursor = conn.execute(
                        "SELECT * FROM processing_history WHERE status = ? ORDER BY id DESC LIMIT ?",
                        (status_filter, limit),
                    )
                else:
                    cursor = conn.execute(
                        "SELECT * FROM processing_history ORDER BY id DESC LIMIT ?",
                        (limit,),
                    )
                for row in cursor.fetchall():
                    try:
                        params = json.loads(row["parameters"]) if row["parameters"] else {}
                    except Exception:
                        params = {}
                    records.append(
                        HistoryRecord(
                            id=row["id"],
                            timestamp=row["timestamp"],
                            source_filename=row["source_filename"],
                            source_path=row["source_path"],
                            method=row["method"],
                            parameters=params,
                            output_path=row["output_path"],
                            status=row["status"],
                            duration_sec=float(row["duration_sec"] or 0.0),
                            error_message=row["error_message"] or "",
                            input_resolution=row["input_resolution"] or "",
                            output_resolution=row["output_resolution"] or "",
                        )
                    )
        except Exception as exc:
            logger.error("Lỗi khi đọc lịch sử xử lý: %s", exc)
        return records

    def delete_record(self, record_id: int) -> bool:
        """Delete a single history record by ID."""
        try:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM processing_history WHERE id = ?", (record_id,))
                conn.commit()
            return True
        except Exception as exc:
            logger.error("Lỗi khi xóa bản ghi lịch sử #%s: %s", record_id, exc)
            return False

    def clear_history(self) -> bool:
        """Clear all history records."""
        try:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM processing_history")
                conn.commit()
            logger.info("Đã xóa toàn bộ lịch sử xử lý ảnh.")
            return True
        except Exception as exc:
            logger.error("Lỗi khi xóa toàn bộ lịch sử: %s", exc)
            return False
