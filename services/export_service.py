"""
Export service for managing image output formatting, safe naming, and metadata handling.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

from core.image_io import ImageContainer, resolve_safe_output_path, save_image

logger = logging.getLogger(__name__)


class ExportService:
    """Service responsible for exporting processed ImageContainer objects safely."""

    def __init__(
        self,
        default_output_dir: str | Path,
        default_format: str = "PNG",
        jpeg_quality: int = 92,
        webp_quality: int = 92,
        preserve_metadata: bool = True,
    ) -> None:
        self.default_output_dir = Path(default_output_dir)
        self.default_format = default_format
        self.jpeg_quality = jpeg_quality
        self.webp_quality = webp_quality
        self.preserve_metadata = preserve_metadata

    def export_auto_named(
        self,
        container: ImageContainer,
        output_dir: str | Path | None = None,
        target_format: str | None = None,
        suffix: str = "_enhanced",
        jpeg_quality: int | None = None,
        webp_quality: int | None = None,
        preserve_metadata: bool | None = None,
    ) -> Tuple[Path, Dict[str, Any]]:
        """Export an image to output_dir using an automatically deduplicated safe filename."""
        out_dir = Path(output_dir) if output_dir else self.default_output_dir
        fmt = target_format or self.default_format
        safe_path = resolve_safe_output_path(
            output_dir=out_dir,
            source_path=container.source_path,
            target_format=fmt,
            suffix=suffix,
            overwrite=False,
        )
        return self.export_to_path(
            container=container,
            output_path=safe_path,
            target_format=fmt,
            jpeg_quality=jpeg_quality,
            webp_quality=webp_quality,
            preserve_metadata=preserve_metadata,
        )

    def export_to_path(
        self,
        container: ImageContainer,
        output_path: str | Path,
        target_format: str | None = None,
        jpeg_quality: int | None = None,
        webp_quality: int | None = None,
        preserve_metadata: bool | None = None,
    ) -> Tuple[Path, Dict[str, Any]]:
        """Export an image to an explicit file path while guarding against source overwrite."""
        fmt = target_format or self.default_format
        jq = self.jpeg_quality if jpeg_quality is None else jpeg_quality
        wq = self.webp_quality if webp_quality is None else webp_quality
        pm = self.preserve_metadata if preserve_metadata is None else preserve_metadata

        saved_path, report = save_image(
            container=container,
            output_path=output_path,
            target_format=fmt,
            jpeg_quality=jq,
            webp_quality=wq,
            preserve_metadata=pm,
            source_path_guard=container.source_path,
        )
        logger.info(
            "Đã xuất ảnh thành công: %s (%s, %s bytes)",
            saved_path,
            report.get("resolution"),
            report.get("file_size_bytes"),
        )
        return saved_path, report
