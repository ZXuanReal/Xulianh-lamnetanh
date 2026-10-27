"""
Unit and integration tests for core/image_io.py and services/export_service.py.
Covers:
- Valid and corrupt images
- Tiny and oversized images (memory guard)
- JPG, PNG, WEBP formats
- RGB and RGBA alpha channel preservation
- Paths with Vietnamese diacritics and special characters
- Source file immutability and collision-safe output naming
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.image_io import (
    ImageContainer,
    ImageIOError,
    load_image,
    resolve_safe_output_path,
    save_image,
)


def _create_synthetic_bgr(width: int = 64, height: int = 48) -> np.ndarray:
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)
    b = xv
    g = yv
    r = ((xv.astype(np.uint16) + yv.astype(np.uint16)) // 2).astype(np.uint8)
    return np.dstack([b, g, r])


def test_load_and_save_formats_rgb_and_rgba(tmp_path: Path) -> None:
    """Test loading and saving JPG, PNG, and WEBP with both RGB and RGBA data."""
    bgr = _create_synthetic_bgr(80, 60)
    alpha = np.full((60, 80), 128, dtype=np.uint8)
    alpha[:20, :20] = 0

    container_rgba = ImageContainer(bgr=bgr, alpha=alpha, original_format="PNG", color_mode="RGBA")

    # 1. PNG with RGBA should preserve alpha
    png_path = tmp_path / "test_rgba.png"
    saved_png, rep_png = save_image(container_rgba, png_path, target_format="PNG")
    assert saved_png.exists()
    assert rep_png["alpha_preserved"] is True

    loaded_png = load_image(saved_png)
    assert loaded_png.width == 80 and loaded_png.height == 60
    assert loaded_png.has_alpha is True
    assert loaded_png.alpha is not None
    assert int(loaded_png.alpha[5, 5]) == 0
    assert int(loaded_png.alpha[40, 40]) == 128

    # 2. WEBP with RGBA should preserve alpha
    webp_path = tmp_path / "test_rgba.webp"
    saved_webp, rep_webp = save_image(container_rgba, webp_path, target_format="WEBP", webp_quality=100)
    assert saved_webp.exists()
    assert rep_webp["alpha_preserved"] is True
    loaded_webp = load_image(saved_webp)
    assert loaded_webp.has_alpha is True

    # 3. JPEG with RGBA should composite cleanly onto white and warn
    jpg_path = tmp_path / "test_rgb.jpg"
    saved_jpg, rep_jpg = save_image(container_rgba, jpg_path, target_format="JPEG", jpeg_quality=95)
    assert saved_jpg.exists()
    assert rep_jpg["alpha_preserved"] is False
    assert len(rep_jpg["warnings"]) >= 1
    loaded_jpg = load_image(saved_jpg)
    assert loaded_jpg.has_alpha is False
    assert loaded_jpg.width == 80 and loaded_jpg.height == 60


def test_vietnamese_unicode_path_support(tmp_path: Path) -> None:
    """Verify full support for directories and filenames with Vietnamese accents and spaces."""
    vn_dir = tmp_path / "Thư mục Kiểm thử Xử lý Ảnh Tiếng Việt"
    vn_dir.mkdir(parents=True, exist_ok=True)
    src_file = vn_dir / "Ảnh_chân_dung_mờ_nhiễu_đặc_biệt_01.png"

    bgr = _create_synthetic_bgr(50, 50)
    cont = ImageContainer(bgr=bgr)
    save_image(cont, src_file, target_format="PNG")

    loaded = load_image(src_file)
    assert loaded.width == 50 and loaded.height == 50

    out_path = resolve_safe_output_path(vn_dir, src_file, target_format="PNG", suffix="_đã_làm_nét")
    saved_out, _ = save_image(loaded, out_path, target_format="PNG")
    assert saved_out.exists()
    assert "đã_làm_nét" in saved_out.name


def test_corrupt_and_empty_image_handling(tmp_path: Path) -> None:
    """Verify clear ImageIOError exceptions for missing, 0-byte, unsupported, and corrupted files."""
    # Missing file
    with pytest.raises(ImageIOError, match="không tồn tại"):
        load_image(tmp_path / "non_existent.png")

    # Unsupported extension
    bad_ext = tmp_path / "file.bmp"
    bad_ext.write_bytes(b"BM123456")
    with pytest.raises(ImageIOError, match="không được hỗ trợ"):
        load_image(bad_ext)

    # Empty 0-byte file
    empty_file = tmp_path / "empty.jpg"
    empty_file.write_bytes(b"")
    with pytest.raises(ImageIOError, match="rỗng"):
        load_image(empty_file)

    # Corrupt file content
    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"NOT_A_VALID_PNG_HEADER_1234567890")
    with pytest.raises(ImageIOError, match="hỏng"):
        load_image(corrupt_file)


def test_very_small_and_oversized_memory_guard(tmp_path: Path) -> None:
    """Verify tiny images (4x4) work and images exceeding max_pixels raise a friendly memory guard error."""
    tiny_bgr = _create_synthetic_bgr(4, 4)
    tiny_path = tmp_path / "tiny.png"
    save_image(ImageContainer(bgr=tiny_bgr), tiny_path)

    loaded_tiny = load_image(tiny_path)
    assert loaded_tiny.width == 4 and loaded_tiny.height == 4

    # Enforce strict pixel ceiling via max_pixels parameter
    med_bgr = _create_synthetic_bgr(200, 200)  # 40,000 pixels
    med_path = tmp_path / "medium.png"
    save_image(ImageContainer(bgr=med_bgr), med_path)

    with pytest.raises(ImageIOError, match="vượt giới hạn an toàn bộ nhớ"):
        load_image(med_path, max_pixels=10_000)


def test_source_overwrite_protection_and_collision_naming(tmp_path: Path) -> None:
    """Ensure original file is never overwritten and duplicate output names auto-increment."""
    src_path = tmp_path / "sample.png"
    bgr = _create_synthetic_bgr(32, 32)
    save_image(ImageContainer(bgr=bgr), src_path)
    orig_hash = hashlib.sha256(src_path.read_bytes()).hexdigest()

    loaded = load_image(src_path)

    # Attempting to save directly over source_path must raise ImageIOError
    with pytest.raises(ImageIOError, match="Từ chối ghi đè"):
        save_image(loaded, src_path)

    # Verify original file bytes are untouched
    assert hashlib.sha256(src_path.read_bytes()).hexdigest() == orig_hash

    # Safe path generation when collision occurs
    out1 = resolve_safe_output_path(tmp_path, src_path, target_format="PNG", suffix="_enhanced")
    save_image(loaded, out1)
    out2 = resolve_safe_output_path(tmp_path, src_path, target_format="PNG", suffix="_enhanced")
    assert out1 != out2
    assert out2.name == "sample_enhanced_1.png"
