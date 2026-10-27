"""
AI Model Registry, weight file verification, in-memory model caching,
and user-initiated downloading from trusted official open-source repositories.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List
import urllib.request

import torch

from config.settings import MODELS_DIR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Metadata and trusted specification for a supported AI model."""

    name: str
    display_name: str
    filename: str
    arch: str  # "srvgg" or "rrdb" or "gfpgan"
    scale: int
    url: str
    license_name: str
    source_repo: str
    approx_size_mb: float
    description: str
    num_feat: int = 64
    num_block: int = 23
    num_grow_ch: int = 32
    num_conv: int = 32


MODEL_CATALOG: Dict[str, ModelSpec] = {
    "realesr-general-x4v3": ModelSpec(
        name="realesr-general-x4v3",
        display_name="Real-ESRGAN Compact v3 (4x - Nhanh, Nhẹ ~4.7MB)",
        filename="realesr-general-x4v3.pth",
        arch="srvgg",
        scale=4,
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        license_name="BSD-3-Clause",
        source_repo="https://github.com/xinntao/Real-ESRGAN",
        approx_size_mb=4.7,
        description="Mô hình SRVGGNetCompact gọn nhẹ, tối ưu cho cả CPU và GPU, phục hồi chi tiết tổng quát.",
        num_feat=64,
        num_conv=32,
    ),
    "RealESRGAN_x2plus": ModelSpec(
        name="RealESRGAN_x2plus",
        display_name="Real-ESRGAN x2plus (2x - Chi tiết cao ~64MB)",
        filename="RealESRGAN_x2plus.pth",
        arch="rrdb",
        scale=2,
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        license_name="BSD-3-Clause",
        source_repo="https://github.com/xinntao/Real-ESRGAN",
        approx_size_mb=64.0,
        description="Mô hình RRDBNet 23 khối tăng độ phân giải 2x chất lượng cao.",
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
    ),
    "RealESRGAN_x4plus": ModelSpec(
        name="RealESRGAN_x4plus",
        display_name="Real-ESRGAN x4plus (4x - Chuẩn Studio ~64MB)",
        filename="RealESRGAN_x4plus.pth",
        arch="rrdb",
        scale=4,
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        license_name="BSD-3-Clause",
        source_repo="https://github.com/xinntao/Real-ESRGAN",
        approx_size_mb=64.0,
        description="Mô hình RRDBNet 23 khối tăng độ phân giải 4x dành cho ảnh chụp thực tế.",
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
    ),
    "RealESRGAN_x4plus_anime_6B": ModelSpec(
        name="RealESRGAN_x4plus_anime_6B",
        display_name="Real-ESRGAN Anime/Illustration 6B (4x - Minh họa/Logo ~17MB)",
        filename="RealESRGAN_x4plus_anime_6B.pth",
        arch="rrdb",
        scale=4,
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        license_name="BSD-3-Clause",
        source_repo="https://github.com/xinntao/Real-ESRGAN",
        approx_size_mb=17.1,
        description="Mô hình RRDBNet 6 khối tối ưu cho poster, logo, hình minh họa, thiết kế đồ họa.",
        num_feat=64,
        num_block=6,
        num_grow_ch=32,
    ),
}


class ModelLoadError(Exception):
    """Raised when an AI model weight file is missing, incompatible, or corrupted."""


class ModelManager:
    """
    Manages AI model weights on disk and caches instantiated PyTorch models in memory
    so weights are loaded once and reused across tasks.
    """

    def __init__(self, models_dir: str | Path | None = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Any] = {}

    def get_model_path(self, model_name: str) -> Path:
        spec = MODEL_CATALOG.get(model_name)
        if spec is None:
            raise ModelLoadError(f"Mô hình '{model_name}' không nằm trong danh mục hỗ trợ.")
        return self.models_dir / spec.filename

    def is_weight_installed(self, model_name: str) -> bool:
        """Return True if the model weight file exists and has a valid non-trivial size."""
        if model_name not in MODEL_CATALOG:
            return False
        path = self.get_model_path(model_name)
        return path.exists() and path.is_file() and path.stat().st_size > 100_000

    def get_installed_sr_models(self) -> List[str]:
        """Return list of installed Super-Resolution model names."""
        return [name for name in MODEL_CATALOG if self.is_weight_installed(name)]

    def any_sr_model_installed(self) -> bool:
        return len(self.get_installed_sr_models()) > 0

    def get_status_summary(self) -> Dict[str, Any]:
        """Return comprehensive status dictionary of all registered models."""
        installed = self.get_installed_sr_models()
        return {
            "models_dir": str(self.models_dir),
            "installed_models": installed,
            "any_installed": len(installed) > 0,
            "cached_in_memory": list(self._cache.keys()),
            "catalog": {
                name: {
                    "display_name": spec.display_name,
                    "filename": spec.filename,
                    "installed": self.is_weight_installed(name),
                    "path": str(self.get_model_path(name)),
                    "scale": spec.scale,
                    "approx_size_mb": spec.approx_size_mb,
                    "license": spec.license_name,
                    "source_repo": spec.source_repo,
                    "url": spec.url,
                }
                for name, spec in MODEL_CATALOG.items()
            },
        }

    def load_state_dict(self, model_name: str, device: torch.device) -> Dict[str, Any]:
        """Load and validate a PyTorch state dict from disk with safe weights handling."""
        path = self.get_model_path(model_name)
        if not path.exists() or path.stat().st_size < 100_000:
            spec = MODEL_CATALOG[model_name]
            raise ModelLoadError(
                f"Chưa tìm thấy trọng số AI '{spec.filename}' tại thư mục '{self.models_dir}'. "
                f"Vui lòng mở mục Phục hồi AI / Cài đặt để tải mô hình từ nguồn chính thức ({spec.source_repo}) "
                f"hoặc đặt file '{spec.filename}' vào thư mục '{self.models_dir}'."
            )
        try:
            try:
                checkpoint = torch.load(str(path), map_location=device, weights_only=True)
            except TypeError:
                checkpoint = torch.load(str(path), map_location=device)
            if not isinstance(checkpoint, dict):
                raise ModelLoadError(f"Cấu trúc trọng số không hợp lệ trong file {path.name}.")
            if "params_ema" in checkpoint and isinstance(checkpoint["params_ema"], dict):
                return checkpoint["params_ema"]
            if "params" in checkpoint and isinstance(checkpoint["params"], dict):
                return checkpoint["params"]
            return checkpoint
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(
                f"Không thể nạp trọng số mô hình '{path.name}' (file có thể bị hỏng hoặc không tương thích): {exc}"
            ) from exc

    def get_cached_model(self, cache_key: str) -> Any | None:
        return self._cache.get(cache_key)

    def set_cached_model(self, cache_key: str, model: Any) -> None:
        self._cache[cache_key] = model

    def clear_cache(self) -> None:
        """Unload all cached models from RAM/VRAM."""
        self._cache.clear()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        logger.info("Đã giải phóng bộ nhớ đệm các mô hình AI.")

    def download_model(
        self,
        model_name: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """
        Download a model weight file from its official release URL ONLY when explicitly triggered by the user.
        Downloads to a temporary '.part' file first and atomically renames on completion.
        """
        if model_name not in MODEL_CATALOG:
            raise ModelLoadError(f"Mô hình không hợp lệ: {model_name}")

        spec = MODEL_CATALOG[model_name]
        target_path = self.get_model_path(model_name)
        temp_path = target_path.with_suffix(".pth.part")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Bắt đầu tải mô hình %s từ %s ...", spec.name, spec.url)
        try:
            req = urllib.request.Request(
                spec.url,
                headers={"User-Agent": "ImageEnhancementStudio/1.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                total_size = int(response.info().get("Content-Length", 0) or 0)
                downloaded = 0
                chunk_size = 64 * 1024
                with open(temp_path, "wb") as out_file:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(downloaded, total_size)

            if not temp_path.exists() or temp_path.stat().st_size < 100_000:
                raise ModelLoadError("File tải về không đầy đủ hoặc bị lỗi.")

            if target_path.exists():
                target_path.unlink()
            temp_path.rename(target_path)
            logger.info("Đã tải xong mô hình %s tại %s", spec.name, target_path)
            return target_path
        except Exception as exc:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            raise ModelLoadError(f"Không thể tải mô hình '{spec.display_name}': {exc}") from exc
