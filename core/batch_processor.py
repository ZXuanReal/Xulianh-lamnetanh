"""
Batch image processing engine with per-file error isolation, safe output naming,
history recording, resource release, and cancellation support.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Callable, List

from config.settings import SUPPORTED_EXTENSIONS
from core.enhancement_pipeline import EnhancementParams, EnhancementPipeline
from core.image_io import format_file_size, load_image
from services.export_service import ExportService
from services.history_service import HistoryService

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """Represents a single image file in the batch queue."""

    source_path: Path
    file_size_str: str = ""
    status: str = "PENDING"  # PENDING, PROCESSING, SUCCESS, FAILED, SKIPPED, CANCELLED
    output_path: Path | None = None
    error_message: str = ""
    duration_sec: float = 0.0
    input_resolution: str = ""
    output_resolution: str = ""

    def __post_init__(self) -> None:
        self.source_path = Path(self.source_path)
        if not self.file_size_str and self.source_path.exists() and self.source_path.is_file():
            try:
                self.file_size_str = format_file_size(self.source_path.stat().st_size)
            except Exception:
                self.file_size_str = "N/A"


@dataclass
class BatchSummary:
    """Summary report of a batch run."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    cancelled: int = 0
    total_duration_sec: float = 0.0
    items: List[BatchItem] = field(default_factory=list)


def collect_supported_images(paths: List[str | Path], recursive: bool = False) -> List[Path]:
    """Scan a list of file or directory paths and return unique supported image files."""
    collected: List[Path] = []
    seen: set[Path] = set()

    for raw in paths:
        p = Path(raw)
        if not p.exists():
            continue
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                collected.append(resolved)
        elif p.is_dir():
            iterator = p.rglob("*") if recursive else p.iterdir()
            for child in sorted(iterator):
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                    resolved = child.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        collected.append(resolved)
    return collected


class BatchProcessor:
    """
    Executes batch enhancement sequentially (or bounded) to prevent CPU/GPU overload,
    isolating per-file errors and freeing memory after each image.
    """

    def __init__(
        self,
        pipeline: EnhancementPipeline,
        export_service: ExportService,
        history_service: HistoryService | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.export_service = export_service
        self.history_service = history_service

    def run_batch(
        self,
        items: List[BatchItem],
        params: EnhancementParams,
        output_dir: str | Path,
        target_format: str = "PNG",
        jpeg_quality: int = 92,
        webp_quality: int = 92,
        preserve_metadata: bool = True,
        only_failed: bool = False,
        on_item_updated: Callable[[int, BatchItem], None] | None = None,
        on_batch_progress: Callable[[int, int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> BatchSummary:
        """
        Process a list of BatchItems and export results to `output_dir`.
        If `only_failed=True`, only items with status == 'FAILED' are re-processed.
        """
        batch_start = time.perf_counter()
        out_folder = Path(output_dir).resolve()
        out_folder.mkdir(parents=True, exist_ok=True)

        summary = BatchSummary(total=len(items), items=items)

        target_indices = [
            idx
            for idx, item in enumerate(items)
            if (item.status == "FAILED" if only_failed else item.status in ("PENDING", "FAILED", "CANCELLED"))
        ]
        total_to_run = len(target_indices)

        for step_idx, item_idx in enumerate(target_indices, start=1):
            item = items[item_idx]
            if cancel_check and cancel_check():
                item.status = "CANCELLED"
                item.error_message = "Đã dừng tác vụ hàng loạt theo yêu cầu người dùng."
                if on_item_updated:
                    on_item_updated(item_idx, item)
                continue

            item.status = "PROCESSING"
            item.error_message = ""
            if on_item_updated:
                on_item_updated(item_idx, item)

            if on_batch_progress:
                on_batch_progress(
                    step_idx - 1,
                    max(1, total_to_run),
                    f"Đang xử lý ({step_idx}/{total_to_run}): {item.source_path.name}",
                )

            item_start = time.perf_counter()
            try:
                src_container = load_image(item.source_path)
                item.input_resolution = src_container.resolution_str

                res = self.pipeline.process(
                    source=src_container,
                    params=params,
                    progress_callback=None,
                    cancel_check=cancel_check,
                )
                item.output_resolution = res.container.resolution_str

                saved_path, _ = self.export_service.export_auto_named(
                    container=res.container,
                    output_dir=out_folder,
                    target_format=target_format,
                    suffix="_enhanced",
                    jpeg_quality=jpeg_quality,
                    webp_quality=webp_quality,
                    preserve_metadata=preserve_metadata,
                )

                item.output_path = saved_path
                item.duration_sec = round(time.perf_counter() - item_start, 3)
                item.status = "SUCCESS"

                if self.history_service is not None:
                    self.history_service.add_record(
                        source_path=item.source_path,
                        method=f"[Batch] {res.method_summary}",
                        parameters=res.applied_params,
                        output_path=saved_path,
                        status="SUCCESS",
                        duration_sec=item.duration_sec,
                        input_resolution=item.input_resolution,
                        output_resolution=item.output_resolution,
                    )
            except Exception as exc:
                item.duration_sec = round(time.perf_counter() - item_start, 3)
                if "hủy" in str(exc).lower() or (cancel_check and cancel_check()):
                    item.status = "CANCELLED"
                    item.error_message = str(exc)
                else:
                    item.status = "FAILED"
                    item.error_message = str(exc)
                    logger.error("Lỗi khi xử lý file hàng loạt '%s': %s", item.source_path, exc)
                    if self.history_service is not None:
                        self.history_service.add_record(
                            source_path=item.source_path,
                            method=f"[Batch] {params.preset}",
                            parameters=params.to_dict(),
                            output_path="",
                            status="FAILED",
                            duration_sec=item.duration_sec,
                            error_message=str(exc),
                            input_resolution=item.input_resolution,
                        )
            finally:
                if on_item_updated:
                    on_item_updated(item_idx, item)
                self.pipeline.device_manager.release_memory()

        for item in items:
            if item.status == "SUCCESS":
                summary.succeeded += 1
            elif item.status == "FAILED":
                summary.failed += 1
            elif item.status == "SKIPPED":
                summary.skipped += 1
            elif item.status == "CANCELLED":
                summary.cancelled += 1

        summary.total_duration_sec = round(time.perf_counter() - batch_start, 3)
        if on_batch_progress:
            on_batch_progress(
                total_to_run,
                max(1, total_to_run),
                f"Hoàn tất lô xử lý: {summary.succeeded} thành công, {summary.failed} lỗi ({summary.total_duration_sec:.2f}s).",
            )
        return summary
