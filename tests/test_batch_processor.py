"""
Comprehensive integration tests for:
- core/batch_processor.py (multi-format batching, error isolation, retry failed, cancellation, source immutability)
- ai/model_manager.py, ai/device_manager.py, and ai/super_resolution.py (missing weights error, CPU fallback, tiled AI SR inference, model caching)
- services/history_service.py and config/settings.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from ai.device_manager import DeviceManager
from ai.model_manager import MODEL_CATALOG, ModelLoadError, ModelManager
from ai.super_resolution import SRVGGNetCompact, SuperResolutionEngine, TaskCancelledError
from config.settings import AppSettings, SettingsManager
from core.batch_processor import BatchItem, BatchProcessor, collect_supported_images
from core.enhancement_pipeline import EnhancementParams, EnhancementPipeline
from core.image_io import ImageContainer, save_image
from services.export_service import ExportService
from services.history_service import HistoryService


def _make_sample_image(path: Path, fmt: str = "PNG", width: int = 80, height: int = 60) -> str:
    img = np.full((height, width, 3), 200, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (width - 10, height - 10), (40, 90, 210), 2)
    save_image(ImageContainer(bgr=img), path, target_format=fmt)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batch_processing_error_isolation_and_source_immutability(tmp_path: Path) -> None:
    """
    Verify batch processing handles valid JPG/PNG/WEBP files alongside a corrupt file,
    isolates the failure without crashing, preserves original files untouched, and logs to SQLite history.
    """
    in_dir = tmp_path / "inputs_tiếng_Việt"
    out_dir = tmp_path / "outputs_kết_quả"
    in_dir.mkdir()
    out_dir.mkdir()

    p_png = in_dir / "img1.png"
    p_jpg = in_dir / "img2.jpg"
    p_webp = in_dir / "img3.webp"
    p_corrupt = in_dir / "corrupt.png"

    h_png = _make_sample_image(p_png, "PNG")
    h_jpg = _make_sample_image(p_jpg, "JPEG")
    h_webp = _make_sample_image(p_webp, "WEBP")
    p_corrupt.write_bytes(b"CORRUPTED_BINARY_CONTENT")

    found_files = collect_supported_images([in_dir])
    assert len(found_files) == 4

    items = [BatchItem(source_path=p) for p in found_files]

    db_path = tmp_path / "test_history.sqlite3"
    hist_svc = HistoryService(db_path=db_path)
    exp_svc = ExportService(default_output_dir=out_dir)
    pipeline = EnhancementPipeline(
        model_manager=ModelManager(models_dir=tmp_path / "models"),
        device_manager=DeviceManager("cpu"),
    )
    processor = BatchProcessor(pipeline=pipeline, export_service=exp_svc, history_service=hist_svc)

    params = EnhancementParams(
        mode="manual",
        preset="standard_sharpen",
        sharpen_method="unsharp",
        sharpen_strength=1.2,
        denoise_strength=2.0,
    )

    summary = processor.run_batch(
        items=items,
        params=params,
        output_dir=out_dir,
        target_format="PNG",
    )

    assert summary.total == 4
    assert summary.succeeded == 3
    assert summary.failed == 1

    # Verify source files were never modified
    assert hashlib.sha256(p_png.read_bytes()).hexdigest() == h_png
    assert hashlib.sha256(p_jpg.read_bytes()).hexdigest() == h_jpg
    assert hashlib.sha256(p_webp.read_bytes()).hexdigest() == h_webp

    # Verify history records were written
    records = hist_svc.list_records()
    assert len(records) == 4
    succ_records = hist_svc.list_records(status_filter="SUCCESS")
    fail_records = hist_svc.list_records(status_filter="FAILED")
    assert len(succ_records) == 3
    assert len(fail_records) == 1

    # Fix the corrupt file and test retry with only_failed=True
    _make_sample_image(p_corrupt, "PNG")
    retry_summary = processor.run_batch(
        items=items,
        params=params,
        output_dir=out_dir,
        target_format="PNG",
        only_failed=True,
    )
    assert retry_summary.succeeded == 4
    assert retry_summary.failed == 0


def test_batch_cancellation(tmp_path: Path) -> None:
    """Verify batch processing stops cleanly when cancel_check returns True."""
    in_dir = tmp_path / "cancel_in"
    out_dir = tmp_path / "cancel_out"
    in_dir.mkdir()
    for i in range(4):
        _make_sample_image(in_dir / f"f_{i}.png", "PNG")

    items = [BatchItem(source_path=p) for p in collect_supported_images([in_dir])]
    pipeline = EnhancementPipeline(
        model_manager=ModelManager(models_dir=tmp_path / "m"),
        device_manager=DeviceManager("cpu"),
    )
    processor = BatchProcessor(pipeline=pipeline, export_service=ExportService(out_dir))

    processed_count = 0

    def cancel_after_first() -> bool:
        return processed_count >= 1

    def on_updated(_idx: int, item: BatchItem) -> None:
        nonlocal processed_count
        if item.status == "SUCCESS":
            processed_count += 1

    summary = processor.run_batch(
        items=items,
        params=EnhancementParams(mode="manual", sharpen_strength=0.8),
        output_dir=out_dir,
        on_item_updated=on_updated,
        cancel_check=cancel_after_first,
    )
    assert summary.succeeded == 1
    assert summary.cancelled == 3


def test_ai_missing_model_error_and_device_fallback(tmp_path: Path) -> None:
    """Verify clear error when AI weights are missing and automatic CPU fallback when CUDA is unavailable."""
    dev_mgr = DeviceManager(preferred="cuda")
    if not torch.cuda.is_available():
        assert dev_mgr.status.selected_device == "cpu"
        assert "CPU" in dev_mgr.status.fallback_warning

    empty_models_dir = tmp_path / "empty_models"
    model_mgr = ModelManager(models_dir=empty_models_dir)
    assert model_mgr.any_sr_model_installed() is False

    sr_engine = SuperResolutionEngine(model_mgr, dev_mgr)
    dummy_bgr = np.zeros((32, 32, 3), dtype=np.uint8)

    with pytest.raises(ModelLoadError, match="Chưa tìm thấy trọng số AI"):
        sr_engine.upscale(dummy_bgr, outscale=2, model_name="realesr-general-x4v3")


def test_ai_super_resolution_tiled_inference_and_caching(tmp_path: Path) -> None:
    """
    Create a valid SRVGGNetCompact checkpoint on disk, verify 2x and 4x tiled AI inference,
    verify in-memory model caching reuse, and verify mid-tile cancellation.
    """
    models_dir = tmp_path / "models_valid"
    models_dir.mkdir(parents=True, exist_ok=True)
    spec = MODEL_CATALOG["realesr-general-x4v3"]

    # Initialize deterministic identity-like weights so output is valid and clean
    net = SRVGGNetCompact(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=spec.num_feat,
        num_conv=spec.num_conv,
        upscale=spec.scale,
    )
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.zeros_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    weight_path = models_dir / spec.filename
    torch.save({"params": net.state_dict()}, str(weight_path))
    # Pad file size > 100KB so ModelManager.is_weight_installed recognizes it as a complete checkpoint
    assert weight_path.stat().st_size > 100_000

    model_mgr = ModelManager(models_dir=models_dir)
    dev_mgr = DeviceManager("cpu")
    assert model_mgr.is_weight_installed("realesr-general-x4v3") is True

    pipeline = EnhancementPipeline(model_manager=model_mgr, device_manager=dev_mgr)
    src_bgr = np.full((48, 64, 3), 128, dtype=np.uint8)
    src_alpha = np.full((48, 64), 200, dtype=np.uint8)
    container = ImageContainer(bgr=src_bgr, alpha=src_alpha)

    # 1. Test 2x AI Super-Resolution with tiled inference (tile_size=32 -> multiple tiles)
    params_2x = EnhancementParams(
        mode="restore_ai",
        preset="ai_restoration",
        upscale_factor=2,
        use_ai_sr=True,
        ai_model_name="realesr-general-x4v3",
        tile_size=32,
        tile_pad=8,
        sharpen_strength=0.5,
    )
    res_2x = pipeline.process(container, params_2x)
    assert res_2x.container.width == 128
    assert res_2x.container.height == 96
    assert res_2x.container.alpha is not None
    assert res_2x.container.alpha.shape == (96, 128)

    # Verify model was cached in memory
    status = model_mgr.get_status_summary()
    assert len(status["cached_in_memory"]) == 1

    # 2. Test 4x AI Super-Resolution reusing cached model
    params_4x = EnhancementParams(
        mode="restore_ai",
        preset="ai_restoration",
        upscale_factor=4,
        use_ai_sr=True,
        ai_model_name="realesr-general-x4v3",
        tile_size=32,
        tile_pad=8,
        sharpen_strength=0.0,
    )
    res_4x = pipeline.process(container, params_4x)
    assert res_4x.container.width == 256
    assert res_4x.container.height == 192

    # 3. Test mid-tile cancellation
    with pytest.raises(TaskCancelledError):
        pipeline.sr_engine.upscale(
            src_bgr,
            outscale=4,
            model_name="realesr-general-x4v3",
            tile_size=16,
            cancel_check=lambda: True,
        )


def test_settings_persistence_and_reset(tmp_path: Path) -> None:
    """Verify SettingsManager saves, reloads, and resets configuration properly."""
    cfg_file = tmp_path / "settings.json"
    mgr = SettingsManager(config_path=cfg_file)
    custom = AppSettings(default_mode="strong_sharpen", jpeg_quality=85, tile_size=512)
    mgr.save(custom)

    mgr2 = SettingsManager(config_path=cfg_file)
    assert mgr2.settings.default_mode == "strong_sharpen"
    assert mgr2.settings.jpeg_quality == 85
    assert mgr2.settings.tile_size == 512

    reset_cfg = mgr2.reset_to_defaults()
    assert reset_cfg.default_mode == "auto"
    assert reset_cfg.jpeg_quality == 92


def test_rrdbnet_2x_4x_and_face_restoration() -> None:
    """Verify RRDBNet 2x (pixel_unshuffle) and 4x architectures and FaceRestorer execution."""
    from ai.face_restoration import FaceRestorer
    from ai.super_resolution import RRDBNet

    x = torch.zeros((1, 3, 16, 16), dtype=torch.float32)
    net_2x = RRDBNet(num_in_ch=3, num_out_ch=3, scale=2, num_feat=16, num_block=2, num_grow_ch=8)
    net_2x.eval()
    with torch.no_grad():
        out_2x = net_2x(x)
    assert out_2x.shape == (1, 3, 32, 32)

    net_4x = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=16, num_block=2, num_grow_ch=8)
    net_4x.eval()
    with torch.no_grad():
        out_4x = net_4x(x)
    assert out_4x.shape == (1, 3, 64, 64)

    restorer = FaceRestorer()
    portrait = np.full((120, 120, 3), 180, dtype=np.uint8)
    res = restorer.restore_faces(portrait, fidelity=0.65)
    assert res.bgr.shape == portrait.shape
    assert isinstance(res.note, str)

