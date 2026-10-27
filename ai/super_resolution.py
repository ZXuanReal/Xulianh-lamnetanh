"""
Super-Resolution & AI Image Restoration Engine.
Implements native PyTorch architectures (RRDBNet and SRVGGNetCompact) compatible with
official Real-ESRGAN weights (eliminating legacy basicsr incompatibilities on Python 3.12+ / Torch 2.x),
complete with memory-safe Tiled Inference, progress reporting, and cancellation support.
"""
from __future__ import annotations

import logging
import math
from typing import Callable, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ai.device_manager import DeviceManager
from ai.model_manager import MODEL_CATALOG, ModelLoadError, ModelManager

logger = logging.getLogger(__name__)


class TaskCancelledError(Exception):
    """Raised when a user cancels an active processing task."""


# ---------------------------------------------------------------------------
# Native PyTorch Real-ESRGAN Architectures (RRDBNet & SRVGGNetCompact)
# ---------------------------------------------------------------------------


def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Downsample spatial dimensions and fold into channels (for scale=2 or scale=1 RRDBNet)."""
    b, c, hh, hw = x.size()
    out_channel = c * (scale ** 2)
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class ResidualDenseBlock(nn.Module):
    """5-convolution Residual Dense Block used inside RRDB."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """
    Official Real-ESRGAN RRDBNet generator architecture compatible with:
    - RealESRGAN_x4plus.pth (scale=4, num_block=23)
    - RealESRGAN_x2plus.pth (scale=2, num_block=23)
    - RealESRGAN_x4plus_anime_6B.pth (scale=4, num_block=6)
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.ModuleList([RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = feat
        for block in self.body:
            body_feat = block(body_feat)
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat

        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class SRVGGNetCompact(nn.Module):
    """
    Compact VGG-style Super-Resolution Network compatible with official
    realesr-general-x4v3.pth weights.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 32,
        upscale: int = 4,
    ) -> None:
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale

        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))

        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))

        self.body.append(nn.Conv2d(num_feat, num_out_ch * (upscale ** 2), 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


# ---------------------------------------------------------------------------
# High-Quality Edge-Directed Traditional Upscaler (when AI is disabled)
# ---------------------------------------------------------------------------


def edge_directed_upscale(bgr: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    High-precision non-AI super-resolution using Lanczos-4 interpolation followed by
    guided edge back-projection in LAB luminance space to reduce interpolation blur.
    """
    if scale <= 1:
        return bgr.copy()

    h, w = bgr.shape[:2]
    target_w, target_h = w * scale, h * scale

    upscaled = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
    l_up, a_up, b_up = cv2.split(lab)

    # Iterative back-projection step on luminance to recover edge acutance
    orig_lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_orig = orig_lab[:, :, 0].astype(np.float32)

    l_curr = l_up.astype(np.float32)
    downprojected = cv2.resize(l_curr, (w, h), interpolation=cv2.INTER_AREA)
    residual_lr = l_orig - downprojected
    residual_hr = cv2.resize(residual_lr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    l_refined = np.clip(l_curr + 0.65 * residual_hr, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l_refined, a_up, b_up]), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Super-Resolution Engine with Tiled Inference and Model Caching
# ---------------------------------------------------------------------------


class SuperResolutionEngine:
    """
    Manages loading, caching, and tiled inference of Super-Resolution models on CPU or GPU.
    """

    def __init__(self, model_manager: ModelManager, device_manager: DeviceManager) -> None:
        self.model_manager = model_manager
        self.device_manager = device_manager

    def build_model(self, model_name: str, device: torch.device) -> Tuple[nn.Module, int]:
        """Instantiate and load weights for the specified model, returning (model, native_scale)."""
        cache_key = f"{model_name}:{device.type}"
        cached = self.model_manager.get_cached_model(cache_key)
        if cached is not None:
            spec = MODEL_CATALOG[model_name]
            return cached, spec.scale

        if model_name not in MODEL_CATALOG:
            raise ModelLoadError(f"Mô hình '{model_name}' không được hỗ trợ.")

        spec = MODEL_CATALOG[model_name]
        state_dict = self.model_manager.load_state_dict(model_name, device=device)

        if spec.arch == "srvgg":
            net: nn.Module = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=spec.num_feat,
                num_conv=spec.num_conv,
                upscale=spec.scale,
            )
        elif spec.arch == "rrdb":
            net = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                scale=spec.scale,
                num_feat=spec.num_feat,
                num_block=spec.num_block,
                num_grow_ch=spec.num_grow_ch,
            )
        else:
            raise ModelLoadError(f"Kiến trúc mô hình '{spec.arch}' chưa được hỗ trợ.")

        try:
            net.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            raise ModelLoadError(f"Trọng số không khớp với kiến trúc {model_name}: {exc}") from exc

        net.eval()
        net.to(device)
        self.model_manager.set_cached_model(cache_key, net)
        logger.info("Đã nạp mô hình AI '%s' lên thiết bị %s (đã lưu cache).", model_name, device)
        return net, spec.scale

    def upscale(
        self,
        bgr: np.ndarray,
        outscale: int = 2,
        model_name: str = "realesr-general-x4v3",
        tile_size: int = 256,
        tile_pad: int = 16,
        progress_callback: Callable[[int, int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> np.ndarray:
        """
        Perform tiled AI Super-Resolution on a BGR uint8 image.

        Args:
            bgr: Input HxWx3 uint8 BGR image.
            outscale: Desired output scale factor (1, 2, or 4).
            model_name: Registered model name in MODEL_CATALOG.
            tile_size: Spatial tile dimension (e.g., 256) to prevent RAM/VRAM exhaustion.
            tile_pad: Overlap padding around each tile to eliminate seam artifacts.
            progress_callback: Optional callback receiving (current_step, total_steps, message).
            cancel_check: Optional callable returning True if user requested cancellation.
        """
        if cancel_check and cancel_check():
            raise TaskCancelledError("Tác vụ đã bị hủy bởi người dùng.")

        device = self.device_manager.torch_device
        model, net_scale = self.build_model(model_name, device=device)

        h_in, w_in = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)

        # Ensure input dimensions are divisible by mod_scale (for pixel_unshuffle when net_scale=2)
        mod_scale = 2 if net_scale == 2 else (4 if net_scale == 1 else 1)
        pad_h = (mod_scale - (h_in % mod_scale)) % mod_scale
        pad_w = (mod_scale - (w_in % mod_scale)) % mod_scale
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")

        try:
            output_tensor = self._tiled_forward(
                img_tensor=img_tensor,
                model=model,
                device=device,
                net_scale=net_scale,
                tile_size=max(64, int(tile_size)),
                tile_pad=max(8, int(tile_pad)),
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                self.device_manager.release_memory()
                raise RuntimeError(
                    "Thiếu bộ nhớ RAM/VRAM khi chạy mô hình AI. Hãy giảm kích thước Tile trong Cài đặt hoặc chuyển sang ảnh nhỏ hơn."
                ) from exc
            raise

        # Remove padding in scaled coordinates
        valid_h = h_in * net_scale
        valid_w = w_in * net_scale
        output_tensor = output_tensor[:, :, :valid_h, :valid_w]

        out_rgb = output_tensor.squeeze(0).clamp_(0.0, 1.0).cpu().numpy()
        out_rgb = (np.transpose(out_rgb, (1, 2, 0)) * 255.0).round().astype(np.uint8)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        # Rescale if user requested outscale != net_scale (e.g. 2x output from a 4x model)
        if outscale != net_scale and outscale > 0:
            target_w = int(round(w_in * outscale))
            target_h = int(round(h_in * outscale))
            interp = cv2.INTER_AREA if outscale < net_scale else cv2.INTER_LANCZOS4
            out_bgr = cv2.resize(out_bgr, (target_w, target_h), interpolation=interp)

        return out_bgr

    def _tiled_forward(
        self,
        img_tensor: torch.Tensor,
        model: nn.Module,
        device: torch.device,
        net_scale: int,
        tile_size: int,
        tile_pad: int,
        progress_callback: Callable[[int, int, str], None] | None,
        cancel_check: Callable[[], bool] | None,
    ) -> torch.Tensor:
        """Process image tensor in overlapping tiles to bound memory usage."""
        batch, channel, height, width = img_tensor.shape
        output_height = height * net_scale
        output_width = width * net_scale
        output = torch.zeros((batch, channel, output_height, output_width), dtype=torch.float32)

        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)
        total_tiles = tiles_x * tiles_y
        current_tile = 0

        with torch.no_grad():
            for y in range(tiles_y):
                for x in range(tiles_x):
                    if cancel_check and cancel_check():
                        raise TaskCancelledError("Đã hủy tác vụ AI theo yêu cầu người dùng.")

                    current_tile += 1
                    if progress_callback:
                        progress_callback(
                            current_tile,
                            total_tiles,
                            f"Đang xử lý AI vùng {current_tile}/{total_tiles} trên {device.type.upper()}...",
                        )

                    ofs_x = x * tile_size
                    ofs_y = y * tile_size
                    input_start_x = ofs_x
                    input_end_x = min(ofs_x + tile_size, width)
                    input_start_y = ofs_y
                    input_end_y = min(ofs_y + tile_size, height)

                    input_start_x_pad = max(input_start_x - tile_pad, 0)
                    input_end_x_pad = min(input_end_x + tile_pad, width)
                    input_start_y_pad = max(input_start_y - tile_pad, 0)
                    input_end_y_pad = min(input_end_y + tile_pad, height)

                    # Ensure padded tile dimensions are multiples of 2 for scale=2 RRDBNet
                    tile_h = input_end_y_pad - input_start_y_pad
                    tile_w = input_end_x_pad - input_start_x_pad
                    extra_h = tile_h % 2
                    extra_w = tile_w % 2

                    input_tile = img_tensor[
                        :, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad
                    ].to(device)
                    if extra_h > 0 or extra_w > 0:
                        input_tile = F.pad(input_tile, (0, extra_w, 0, extra_h), mode="reflect")

                    output_tile = model(input_tile).cpu()
                    if extra_h > 0 or extra_w > 0:
                        output_tile = output_tile[:, :, : tile_h * net_scale, : tile_w * net_scale]

                    output_start_x = input_start_x * net_scale
                    output_end_x = input_end_x * net_scale
                    output_start_y = input_start_y * net_scale
                    output_end_y = input_end_y * net_scale

                    output_start_x_tile = (input_start_x - input_start_x_pad) * net_scale
                    output_end_x_tile = output_start_x_tile + (input_end_x - input_start_x) * net_scale
                    output_start_y_tile = (input_start_y - input_start_y_pad) * net_scale
                    output_end_y_tile = output_start_y_tile + (input_end_y - input_start_y) * net_scale

                    output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
                        :, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile
                    ]

        return output
