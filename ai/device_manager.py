"""
Hardware device detection, CPU/GPU selection, and RAM/VRAM resource management.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import logging
from typing import Tuple

import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass
class DeviceStatus:
    """Detailed hardware and compute device status."""

    selected_device: str  # "cpu" or "cuda"
    device_name: str
    cuda_available: bool
    torch_version: str
    total_ram_gb: float
    available_ram_gb: float
    total_vram_gb: float = 0.0
    free_vram_gb: float = 0.0
    fallback_warning: str = ""

    @property
    def badge_text(self) -> str:
        if self.selected_device == "cuda":
            return f"GPU: {self.device_name} ({self.free_vram_gb:.1f}/{self.total_vram_gb:.1f} GB VRAM)"
        return f"CPU ({self.available_ram_gb:.1f}/{self.total_ram_gb:.1f} GB RAM)"


class DeviceManager:
    """Manages PyTorch execution device (CPU/CUDA) and memory cleanup."""

    def __init__(self, preferred: str = "auto") -> None:
        self._preferred = preferred.lower().strip()
        self._status = self.detect_device(self._preferred)

    @property
    def status(self) -> DeviceStatus:
        return self._status

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self._status.selected_device)

    def set_preferred_device(self, preferred: str) -> DeviceStatus:
        """Switch compute device preference ('auto', 'cpu', 'cuda') and re-evaluate status."""
        self._preferred = preferred.lower().strip()
        self._status = self.detect_device(self._preferred)
        return self._status

    def detect_device(self, preferred: str = "auto") -> DeviceStatus:
        """Inspect system RAM, CUDA availability, and resolve active compute device."""
        vm = psutil.virtual_memory()
        total_ram = round(vm.total / (1024 ** 3), 2)
        avail_ram = round(vm.available / (1024 ** 3), 2)
        cuda_ok = bool(torch.cuda.is_available())
        torch_ver = str(torch.__version__)

        pref = preferred.lower().strip()
        warning = ""

        if pref == "cuda" and not cuda_ok:
            warning = (
                "GPU CUDA không khả dụng trên hệ thống này. Đã tự động chuyển sang CPU "
                "(thời gian xử lý AI trên CPU sẽ lâu hơn)."
            )
            logger.warning(warning)
            selected = "cpu"
        elif pref in ("auto", "cuda") and cuda_ok:
            selected = "cuda"
        else:
            selected = "cpu"
            if pref == "auto" and not cuda_ok:
                warning = "Đang chạy trên CPU (không phát hiện GPU NVIDIA CUDA). Tác vụ AI có thể mất nhiều thời gian hơn."

        if selected == "cuda":
            try:
                dev_name = torch.cuda.get_device_name(0)
                free_b, total_b = torch.cuda.mem_get_info(0)
                total_vram = round(total_b / (1024 ** 3), 2)
                free_vram = round(free_b / (1024 ** 3), 2)
            except Exception:
                dev_name = "CUDA GPU"
                total_vram = 0.0
                free_vram = 0.0
        else:
            dev_name = "CPU Đa luồng (PyTorch)"
            total_vram = 0.0
            free_vram = 0.0

        return DeviceStatus(
            selected_device=selected,
            device_name=dev_name,
            cuda_available=cuda_ok,
            torch_version=torch_ver,
            total_ram_gb=total_ram,
            available_ram_gb=avail_ram,
            total_vram_gb=total_vram,
            free_vram_gb=free_vram,
            fallback_warning=warning,
        )

    def refresh_memory_stats(self) -> DeviceStatus:
        """Update dynamic RAM/VRAM metrics."""
        self._status = self.detect_device(self._preferred)
        return self._status

    @staticmethod
    def release_memory() -> None:
        """Force garbage collection and empty PyTorch CUDA cache if applicable."""
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def recommend_tile_size(self) -> Tuple[int, int]:
        """Recommend safe (tile_size, tile_pad) based on available VRAM/RAM."""
        self.refresh_memory_stats()
        if self._status.selected_device == "cuda":
            if self._status.free_vram_gb >= 6.0:
                return 512, 16
            if self._status.free_vram_gb >= 3.0:
                return 256, 16
            return 192, 16
        # On CPU, 256x256 tiles keep peak RAM low and provide responsive progress updates
        if self._status.available_ram_gb < 4.0:
            return 192, 16
        return 256, 16
