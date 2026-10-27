"""
Image denoising and contrast/exposure enhancement algorithms using OpenCV.
Supports Non-Local Means, Bilateral Filtering, Chrominance Denoising, CLAHE, and Gamma/Brightness.
"""
from __future__ import annotations

import cv2
import numpy as np


def denoise_image(
    bgr: np.ndarray,
    strength: float = 0.0,
    color_strength: float | None = None,
    method: str = "nlm",
) -> np.ndarray:
    """
    Reduce image noise while preserving edges and fine structures.

    Args:
        bgr: Input HxWx3 uint8 BGR image.
        strength: Denoising intensity (0.0 = disabled, 1.0..25.0 range; normalized 0..100 UI slider mapped to 0..20).
        color_strength: Chrominance denoising strength. Defaults to strength * 0.85.
        method: 'nlm' (Non-Local Means), 'bilateral' (Edge-preserving bilateral), or 'median'.
    """
    if strength <= 0.05:
        return bgr.copy()

    h_luma = float(np.clip(strength, 0.1, 30.0))
    h_color = float(np.clip(color_strength if color_strength is not None else h_luma * 0.85, 0.1, 30.0))
    total_pixels = bgr.shape[0] * bgr.shape[1]

    method_key = method.lower().strip()

    if method_key == "median":
        ksize = 3 if h_luma < 10.0 else 5
        median_out = cv2.medianBlur(bgr, ksize)
        alpha = float(np.clip(h_luma / 15.0, 0.15, 0.85))
        return cv2.addWeighted(bgr, 1.0 - alpha, median_out, alpha, 0)

    if method_key == "bilateral" or total_pixels > 6_000_000:
        # Bilateral filter in LAB space avoids color bleeding and runs fast on large images
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        d = 5 if h_luma <= 8.0 else 7
        sigma_color = h_luma * 3.2
        sigma_space = 5.0 + h_luma * 0.5
        l_den = cv2.bilateralFilter(l_chan, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        a_den = cv2.GaussianBlur(a_chan, (3, 3), sigmaX=max(0.5, h_color * 0.15))
        b_den = cv2.GaussianBlur(b_chan, (3, 3), sigmaX=max(0.5, h_color * 0.15))
        return cv2.cvtColor(cv2.merge([l_den, a_den, b_den]), cv2.COLOR_LAB2BGR)

    # Default: Non-Local Means Colored Denoising with tuned search/template windows
    template_win = 7
    search_win = 15 if total_pixels > 2_000_000 else 21
    return cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        h=h_luma,
        hColor=h_color,
        templateWindowSize=template_win,
        searchWindowSize=search_win,
    )


def enhance_contrast_clahe(
    bgr: np.ndarray,
    clip_limit: float = 1.5,
    tile_grid_size: int = 8,
    blend: float = 1.0,
) -> np.ndarray:
    """
    Enhance local contrast using CLAHE on the CIE LAB L-channel to preserve original hue and saturation.

    Args:
        bgr: Input HxWx3 uint8 BGR image.
        clip_limit: CLAHE clip limit (0.5..4.0). Values <= 0.05 disable CLAHE.
        tile_grid_size: Grid size (e.g., 8x8).
        blend: Blend factor (0.0..1.0) between original L and CLAHE L to avoid harsh over-enhancement.
    """
    if clip_limit <= 0.05 or blend <= 0.01:
        return bgr.copy()

    eff_clip = float(np.clip(clip_limit, 0.5, 5.0))
    grid = max(2, int(tile_grid_size))
    blend_ratio = float(np.clip(blend, 0.0, 1.0))

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=eff_clip, tileGridSize=(grid, grid))
    l_clahe = clahe.apply(l_chan)

    if blend_ratio < 0.999:
        l_out = cv2.addWeighted(l_chan, 1.0 - blend_ratio, l_clahe, blend_ratio, 0)
    else:
        l_out = l_clahe

    return cv2.cvtColor(cv2.merge([l_out, a_chan, b_chan]), cv2.COLOR_LAB2BGR)


def adjust_brightness_gamma(
    bgr: np.ndarray,
    brightness: float = 0.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Apply brightness offset (-100..+100) and gamma correction (0.2..3.0) via fast 256-entry LUT.
    """
    if abs(brightness) < 0.5 and abs(gamma - 1.0) < 0.01:
        return bgr.copy()

    eff_gamma = float(np.clip(gamma, 0.2, 3.0))
    inv_gamma = 1.0 / eff_gamma
    offset = float(np.clip(brightness, -100.0, 100.0))

    indices = np.arange(256, dtype=np.float32)
    lut = ((indices / 255.0) ** inv_gamma) * 255.0 + offset
    lut_u8 = np.clip(lut, 0.0, 255.0).astype(np.uint8)

    return cv2.LUT(bgr, lut_u8)
