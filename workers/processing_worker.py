"""
Qt QThread background workers for Single Image Enhancement, Batch Processing,
and User-Initiated AI Model Downloading.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List

from PySide6.QtCore import QObject, QThread, Signal

from ai.model_manager import ModelLoadError, ModelManager
from ai.super_resolution import TaskCancelledError
from core.batch_processor import BatchItem, BatchProcessor, BatchSummary
from core.enhancement_pipeline import EnhancementParams, EnhancementPipeline, PipelineResult
from core.image_io import ImageContainer

logger = logging.getLogger(__name__)


def build_remediation_hint(error_text: str) -> str:
    """Return an actionable Vietnamese troubleshooting tip for common errors."""
    lower = error_text.lower()
    if "trọng số" in lower or ".pth" in lower or "model" in lower:
        return (
            "Cách khắc phục: Chuyển sang thẻ 'Phục hồi AI' hoặc 'Cài đặt' -> nhấn 'Tải mô hình' "
            "để tải trọng số chính thức, hoặc tắt tùy chọn 'Sử dụng AI Super-Resolution'."
        )
    if "bộ nhớ" in lower or "memory" in lower:
        return (
            "Cách khắc phục: Giảm kích thước Tile (ví dụ 192px) trong phần Cài đặt, đóng bớt ứng dụng nặng, "
            "hoặc chọn tỷ lệ phóng to 2x thay vì 4x."
        )
    if "ghi đè" in lower or "overwrite" in lower:
        return "Cách khắc phục: Hãy chọn tên file hoặc thư mục đầu ra khác với file ảnh gốc."
    if "hỏng" in lower or "corrupt" in lower or "giải mã" in lower:
        return "Cách khắc phục: Kiểm tra lại file ảnh nguồn bằng trình xem ảnh của Windows; file có thể bị lỗi định dạng."
    return "Cách khắc phục: Kiểm tra nhật ký chi tiết tại bảng trạng thái hoặc thử lại với chế độ 'Làm nét tiêu chuẩn'."


class SingleProcessingWorker(QThread):
    """Background worker for processing a single image without freezing the UI."""

    progress_changed = Signal(int, str)
    finished_result = Signal(object)  # PipelineResult
    cancelled = Signal(str)
    failed_error = Signal(str, str)  # (error_message, remediation_hint)

    def __init__(
        self,
        pipeline: EnhancementPipeline,
        source_container: ImageContainer,
        params: EnhancementParams,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.pipeline = pipeline
        self.source_container = source_container
        self.params = params
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Request safe cancellation of the active processing task."""
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def run(self) -> None:
        try:
            result: PipelineResult = self.pipeline.process(
                source=self.source_container,
                params=self.params,
                progress_callback=lambda pct, msg: self.progress_changed.emit(int(pct), str(msg)),
                cancel_check=self.is_cancelled,
            )
            if self.is_cancelled():
                self.cancelled.emit("Đã hủy tác vụ xử lý ảnh.")
                return
            self.finished_result.emit(result)
        except TaskCancelledError as exc:
            logger.info("Tác vụ đơn lẻ đã bị hủy: %s", exc)
            self.cancelled.emit(str(exc))
        except ModelLoadError as exc:
            err_msg = str(exc)
            logger.warning("Thiếu mô hình AI: %s", err_msg)
            self.failed_error.emit(err_msg, build_remediation_hint(err_msg))
        except Exception as exc:
            err_msg = str(exc)
            logger.exception("Lỗi khi xử lý ảnh: %s", err_msg)
            self.failed_error.emit(err_msg, build_remediation_hint(err_msg))
        finally:
            self.pipeline.device_manager.release_memory()


class BatchProcessingWorker(QThread):
    """Background worker for batch image processing."""

    item_updated = Signal(int, object)  # (index, BatchItem)
    batch_progress = Signal(int, int, str)  # (completed, total, status_text)
    batch_finished = Signal(object)  # BatchSummary
    batch_failed = Signal(str, str)

    def __init__(
        self,
        batch_processor: BatchProcessor,
        items: List[BatchItem],
        params: EnhancementParams,
        output_dir: str | Path,
        target_format: str = "PNG",
        jpeg_quality: int = 92,
        webp_quality: int = 92,
        preserve_metadata: bool = True,
        only_failed: bool = False,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.batch_processor = batch_processor
        self.items = items
        self.params = params
        self.output_dir = Path(output_dir)
        self.target_format = target_format
        self.jpeg_quality = jpeg_quality
        self.webp_quality = webp_quality
        self.preserve_metadata = preserve_metadata
        self.only_failed = only_failed
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def run(self) -> None:
        try:
            summary: BatchSummary = self.batch_processor.run_batch(
                items=self.items,
                params=self.params,
                output_dir=self.output_dir,
                target_format=self.target_format,
                jpeg_quality=self.jpeg_quality,
                webp_quality=self.webp_quality,
                preserve_metadata=self.preserve_metadata,
                only_failed=self.only_failed,
                on_item_updated=lambda idx, item: self.item_updated.emit(int(idx), item),
                on_batch_progress=lambda cur, tot, msg: self.batch_progress.emit(int(cur), int(tot), str(msg)),
                cancel_check=self.is_cancelled,
            )
            self.batch_finished.emit(summary)
        except Exception as exc:
            err_msg = str(exc)
            logger.exception("Lỗi nghiêm trọng trong BatchProcessingWorker: %s", err_msg)
            self.batch_failed.emit(err_msg, build_remediation_hint(err_msg))


class ModelDownloadWorker(QThread):
    """Background worker for downloading AI model weights upon user request."""

    download_progress = Signal(int, int, str)  # (downloaded_bytes, total_bytes, model_name)
    download_finished = Signal(str, str)  # (model_name, saved_path)
    download_failed = Signal(str, str)  # (model_name, error_message)

    def __init__(
        self,
        model_manager: ModelManager,
        model_name: str,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.model_manager = model_manager
        self.model_name = model_name

    def run(self) -> None:
        try:
            saved_path = self.model_manager.download_model(
                model_name=self.model_name,
                progress_callback=lambda dl, tot: self.download_progress.emit(int(dl), int(tot), self.model_name),
            )
            self.download_finished.emit(self.model_name, str(saved_path))
        except Exception as exc:
            self.download_failed.emit(self.model_name, str(exc))
