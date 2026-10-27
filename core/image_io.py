"""
Robust Image I/O module with full Unicode/Vietnamese path support,
alpha channel preservation, metadata retention, and memory safety checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from config.settings import MAX_IMAGE_PIXELS, SUPPORTED_EXTENSIONS, WARN_IMAGE_PIXELS

logger = logging.getLogger(__name__)

# Raise Pillow's decompression bomb limit slightly above our controlled MAX_IMAGE_PIXELS
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS * 2


class ImageIOError(Exception):
    """Raised when reading or writing an image fails."""


@dataclass
class ImageContainer:
    """
    Holds decoded image arrays in BGR (uint8) format alongside optional Alpha channel
    and source metadata (EXIF, ICC profile, DPI, original path, format).
    """

    bgr: np.ndarray  # H x W x 3, uint8, BGR order for OpenCV compatibility
    alpha: np.ndarray | None = None  # H x W, uint8 if source had transparency
    source_path: Path | None = None
    original_format: str = "PNG"
    color_mode: str = "RGB"  # "RGB", "RGBA", "L", etc.
    file_size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def height(self) -> int:
        return int(self.bgr.shape[0])

    @property
    def width(self) -> int:
        return int(self.bgr.shape[1])

    @property
    def has_alpha(self) -> bool:
        return self.alpha is not None

    @property
    def megapixels(self) -> float:
        return round((self.width * self.height) / 1_000_000.0, 2)

    @property
    def resolution_str(self) -> str:
        return f"{self.width} x {self.height}"

    def clone(self) -> "ImageContainer":
        """Return a deep copy of the image container so original data is never mutated."""
        return ImageContainer(
            bgr=self.bgr.copy(),
            alpha=self.alpha.copy() if self.alpha is not None else None,
            source_path=self.source_path,
            original_format=self.original_format,
            color_mode=self.color_mode,
            file_size_bytes=self.file_size_bytes,
            metadata=dict(self.metadata),
            warnings=list(self.warnings),
        )

    def to_rgb(self) -> np.ndarray:
        """Return H x W x 3 uint8 RGB array."""
        return cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB)

    def to_rgba(self) -> np.ndarray:
        """Return H x W x 4 uint8 RGBA array."""
        rgb = self.to_rgb()
        if self.alpha is not None:
            alpha_ch = self.alpha
            if alpha_ch.shape[:2] != rgb.shape[:2]:
                alpha_ch = cv2.resize(alpha_ch, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            return np.dstack([rgb, alpha_ch])
        full_alpha = np.full((rgb.shape[0], rgb.shape[1]), 255, dtype=np.uint8)
        return np.dstack([rgb, full_alpha])


def format_file_size(num_bytes: int) -> str:
    """Format byte size into human-readable string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024.0:.1f} KB"
    return f"{num_bytes / (1024.0 * 1024.0):.2f} MB"


def load_image(path: str | Path, max_pixels: int = MAX_IMAGE_PIXELS) -> ImageContainer:
    """
    Load an image safely from disk supporting Vietnamese/Unicode paths,
    EXIF orientation transposition, alpha channel separation, and memory guards.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise ImageIOError(f"File không tồn tại: {file_path}")
    if not file_path.is_file():
        raise ImageIOError(f"Đường dẫn không phải là file hợp lệ: {file_path}")

    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ImageIOError(
            f"Định dạng '{ext}' không được hỗ trợ. Chỉ hỗ trợ: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    file_size = file_path.stat().st_size
    if file_size == 0:
        raise ImageIOError(f"File ảnh rỗng (0 bytes): {file_path.name}")

    warnings: list[str] = []
    try:
        with Image.open(file_path) as pil_img:
            pil_img.verify()

        with Image.open(file_path) as pil_img:
            orig_format = (pil_img.format or ext.lstrip(".").upper()).upper()
            if orig_format == "JPG":
                orig_format = "JPEG"

            width, height = pil_img.size
            total_pixels = width * height
            if total_pixels <= 0:
                raise ImageIOError(f"Kích thước ảnh không hợp lệ ({width}x{height}).")
            if total_pixels > max_pixels:
                raise ImageIOError(
                    f"Ảnh quá lớn ({width}x{height} = {total_pixels / 1e6:.1f} MP), "
                    f"vượt giới hạn an toàn bộ nhớ ({max_pixels / 1e6:.1f} MP)."
                )
            if total_pixels > WARN_IMAGE_PIXELS:
                warnings.append(
                    f"Ảnh có độ phân giải lớn ({width}x{height}). Quá trình xử lý có thể tốn nhiều RAM/thời gian."
                )

            # Capture metadata before orientation transpose
            metadata: Dict[str, Any] = {}
            if "exif" in pil_img.info:
                metadata["exif"] = pil_img.info["exif"]
            if "icc_profile" in pil_img.info:
                metadata["icc_profile"] = pil_img.info["icc_profile"]
            if "dpi" in pil_img.info:
                metadata["dpi"] = pil_img.info["dpi"]

            # Apply EXIF orientation safely if present
            try:
                pil_img = ImageOps.exif_transpose(pil_img)
            except Exception:
                pass

            orig_mode = pil_img.mode
            alpha: np.ndarray | None = None

            if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
                rgba_img = pil_img.convert("RGBA")
                rgba_arr = np.array(rgba_img, dtype=np.uint8)
                rgb_arr = rgba_arr[:, :, :3]
                alpha = rgba_arr[:, :, 3].copy()
                bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
                color_mode = "RGBA"
            else:
                rgb_img = pil_img.convert("RGB")
                rgb_arr = np.array(rgb_img, dtype=np.uint8)
                bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
                color_mode = "RGB" if orig_mode != "L" else "L"

            return ImageContainer(
                bgr=bgr_arr,
                alpha=alpha,
                source_path=file_path.resolve(),
                original_format=orig_format,
                color_mode=color_mode,
                file_size_bytes=file_size,
                metadata=metadata,
                warnings=warnings,
            )
    except ImageIOError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageIOError(f"File ảnh bị hỏng hoặc không thể giải mã ({file_path.name}): {exc}") from exc
    except MemoryError as exc:
        raise ImageIOError(f"Không đủ bộ nhớ RAM để mở ảnh {file_path.name}.") from exc


def resolve_safe_output_path(
    output_dir: str | Path,
    source_path: str | Path | None,
    target_format: str = "PNG",
    suffix: str = "_enhanced",
    overwrite: bool = False,
) -> Path:
    """
    Generate a safe output file path that NEVER overwrites the source file
    and avoids accidental collisions with existing files in output_dir.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt_upper = target_format.upper()
    if fmt_upper == "ORIGINAL" and source_path is not None:
        ext = Path(source_path).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            ext = ".png"
    elif fmt_upper in ("JPG", "JPEG"):
        ext = ".jpg"
    elif fmt_upper == "WEBP":
        ext = ".webp"
    else:
        ext = ".png"

    stem = Path(source_path).stem if source_path else "output_image"
    candidate = out_dir / f"{stem}{suffix}{ext}"

    src_resolved = Path(source_path).resolve() if source_path and Path(source_path).exists() else None

    # Never allow overwriting the original source file even if overwrite=True
    if src_resolved is not None and candidate.resolve() == src_resolved:
        candidate = out_dir / f"{stem}{suffix}_copy{ext}"

    if not overwrite:
        counter = 1
        while candidate.exists() or (src_resolved is not None and candidate.resolve() == src_resolved):
            candidate = out_dir / f"{stem}{suffix}_{counter}{ext}"
            counter += 1

    return candidate


def save_image(
    container: ImageContainer,
    output_path: str | Path,
    target_format: str | None = None,
    jpeg_quality: int = 92,
    webp_quality: int = 92,
    preserve_metadata: bool = True,
    source_path_guard: str | Path | None = None,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Save an ImageContainer to disk with full Unicode path support,
    alpha channel preservation (for PNG/WEBP), and optional metadata retention.
    Returns (saved_path, save_report).
    """
    dest = Path(output_path).resolve()
    guard_path = Path(source_path_guard).resolve() if source_path_guard else container.source_path
    if guard_path is not None and dest == guard_path.resolve():
        raise ImageIOError(
            f"Từ chối ghi đè trực tiếp lên file ảnh gốc ({guard_path.name}). Vui lòng chọn tên hoặc thư mục khác."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)

    if target_format is None or target_format.upper() == "ORIGINAL":
        ext = dest.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            fmt = "JPEG"
        elif ext == ".webp":
            fmt = "WEBP"
        else:
            fmt = "PNG"
    else:
        fmt = target_format.upper()
        if fmt == "JPG":
            fmt = "JPEG"

    report: Dict[str, Any] = {
        "format": fmt,
        "alpha_preserved": False,
        "metadata_preserved": False,
        "warnings": [],
    }

    try:
        if container.has_alpha and fmt in ("PNG", "WEBP"):
            rgba = container.to_rgba()
            pil_out = Image.fromarray(rgba, mode="RGBA")
            report["alpha_preserved"] = True
        else:
            if container.has_alpha and fmt == "JPEG":
                # Composite RGBA over white background for JPEG
                rgba = container.to_rgba().astype(np.float32)
                alpha_norm = rgba[:, :, 3:4] / 255.0
                white_bg = np.full_like(rgba[:, :, :3], 255.0)
                composited = (rgba[:, :, :3] * alpha_norm + white_bg * (1.0 - alpha_norm)).clip(0, 255).astype(np.uint8)
                pil_out = Image.fromarray(composited, mode="RGB")
                report["warnings"].append("Định dạng JPEG không hỗ trợ kênh trong suốt (Alpha); đã gộp trên nền trắng.")
            else:
                rgb = container.to_rgb()
                pil_out = Image.fromarray(rgb, mode="RGB")

        save_kwargs: Dict[str, Any] = {}
        if fmt == "JPEG":
            save_kwargs["quality"] = int(np.clip(jpeg_quality, 10, 100))
            save_kwargs["subsampling"] = 0 if jpeg_quality >= 90 else 2
            save_kwargs["optimize"] = True
        elif fmt == "WEBP":
            save_kwargs["quality"] = int(np.clip(webp_quality, 10, 100))
            save_kwargs["method"] = 4
        elif fmt == "PNG":
            save_kwargs["compress_level"] = 6

        if preserve_metadata and container.metadata:
            if "icc_profile" in container.metadata:
                save_kwargs["icc_profile"] = container.metadata["icc_profile"]
                report["metadata_preserved"] = True
            if "dpi" in container.metadata and fmt in ("JPEG", "PNG"):
                save_kwargs["dpi"] = container.metadata["dpi"]
                report["metadata_preserved"] = True
            if "exif" in container.metadata and fmt in ("JPEG", "WEBP", "PNG"):
                save_kwargs["exif"] = container.metadata["exif"]
                report["metadata_preserved"] = True

        try:
            pil_out.save(dest, format=fmt, **save_kwargs)
        except Exception as metadata_exc:
            # If saving with EXIF fails due to modified dimensions/corrupt tags, retry without EXIF
            if "exif" in save_kwargs:
                save_kwargs.pop("exif", None)
                report["metadata_preserved"] = False
                report["warnings"].append(
                    f"Không thể giữ nguyên EXIF gốc sau khi thay đổi cấu trúc ảnh ({metadata_exc}); đã lưu ảnh không kèm EXIF."
                )
                pil_out.save(dest, format=fmt, **save_kwargs)
            else:
                raise

        report["file_size_bytes"] = dest.stat().st_size
        report["resolution"] = f"{pil_out.width} x {pil_out.height}"
        return dest, report
    except ImageIOError:
        raise
    except OSError as exc:
        raise ImageIOError(f"Không thể ghi file đầu ra '{dest}': {exc}") from exc
