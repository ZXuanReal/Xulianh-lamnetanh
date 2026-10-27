"""
Traditional image sharpening algorithms using OpenCV.
Operates primarily on the CIE LAB luminance channel (L) to preserve color accuracy
and incorporates soft-knee halo suppression and flat-area noise thresholding.
"""
from __future__ import annotations

import cv2
import numpy as np


def unsharp_mask(
    bgr: np.ndarray,
    strength: float = 1.0,
    radius: float = 1.5,
    threshold: int = 4,
    halo_limit: float = 28.0,
    luminance_only: bool = True,
) -> np.ndarray:
    """
    Apply Unsharp Masking with halo suppression and noise threshold protection.

    Args:
        bgr: Input HxWx3 uint8 BGR image.
        strength: Sharpening amount (0.0 = no change, 0.3..3.0 typical).
        radius: Gaussian blur sigma in pixels (0.5..5.0).
        threshold: Minimum luminance difference (0..50) required to sharpen a pixel,
                   preventing noise amplification in flat regions.
        halo_limit: Maximum overshoot delta in luminance units to suppress bright/dark halos.
        luminance_only: If True, sharpens only the L channel in LAB color space.
    """
    if strength <= 1e-4:
        return bgr.copy()

    sigma = max(0.3, float(radius))
    amount = float(np.clip(strength, 0.0, 5.0))

    if luminance_only and bgr.ndim == 3 and bgr.shape[2] == 3:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_float = l_chan.astype(np.float32)

        blurred = cv2.GaussianBlur(l_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
        detail = l_float - blurred

        # Suppress noise amplification in flat areas using soft thresholding
        if threshold > 0:
            abs_detail = np.abs(detail)
            mask = np.clip((abs_detail - float(threshold)) / max(1.0, float(threshold)), 0.0, 1.0)
            detail = detail * mask

        delta = detail * amount

        # Soft-knee halo clamping using hyperbolic tangent compression
        if halo_limit > 0:
            limit = float(halo_limit)
            delta = limit * np.tanh(delta / limit)

        sharpened_l = np.clip(l_float + delta, 0.0, 255.0).astype(np.uint8)
        merged_lab = cv2.merge([sharpened_l, a_chan, b_chan])
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    # Full-channel fallback
    img_float = bgr.astype(np.float32)
    blurred = cv2.GaussianBlur(img_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
    detail = img_float - blurred
    if threshold > 0:
        mask = np.clip((np.abs(detail) - float(threshold)) / max(1.0, float(threshold)), 0.0, 1.0)
        detail = detail * mask
    delta = detail * amount
    if halo_limit > 0:
        delta = float(halo_limit) * np.tanh(delta / float(halo_limit))
    return np.clip(img_float + delta, 0.0, 255.0).astype(np.uint8)


def kernel_sharpen(
    bgr: np.ndarray,
    strength: float = 1.0,
    luminance_only: bool = True,
) -> np.ndarray:
    """
    Apply 3x3 isotropic Laplacian convolution kernel sharpening with adjustable blending
    and halo clamping.

    Args:
        bgr: Input HxWx3 uint8 BGR image.
        strength: Blend factor (0.0 = original, 1.0 = standard kernel, up to 2.5).
        luminance_only: Operate on LAB L-channel to prevent color fringing.
    """
    if strength <= 1e-4:
        return bgr.copy()

    alpha = float(np.clip(strength, 0.0, 3.0))
    # Isotropic 3x3 sharpening kernel (center=5, neighbors=-1) scaled by alpha
    base_kernel = np.array(
        [
            [-0.15, -0.35, -0.15],
            [-0.35, 3.00, -0.35],
            [-0.15, -0.35, -0.15],
        ],
        dtype=np.float32,
    )
    identity = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    eff_kernel = identity + alpha * (base_kernel - identity)

    if luminance_only and bgr.ndim == 3 and bgr.shape[2] == 3:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_float = l_chan.astype(np.float32)
        filtered = cv2.filter2D(l_float, cv2.CV_32F, eff_kernel, borderType=cv2.BORDER_REFLECT_101)
        # Clamp per-pixel change to prevent harsh ringing around text/edges
        delta = np.clip(filtered - l_float, -32.0, 32.0)
        l_out = np.clip(l_float + delta, 0.0, 255.0).astype(np.uint8)
        return cv2.cvtColor(cv2.merge([l_out, a_chan, b_chan]), cv2.COLOR_LAB2BGR)

    img_float = bgr.astype(np.float32)
    filtered = cv2.filter2D(img_float, cv2.CV_32F, eff_kernel, borderType=cv2.BORDER_REFLECT_101)
    delta = np.clip(filtered - img_float, -32.0, 32.0)
    return np.clip(img_float + delta, 0.0, 255.0).astype(np.uint8)


def adaptive_document_sharpen(bgr: np.ndarray, strength: float = 1.2) -> np.ndarray:
    """
    Specialized sharpening for document/text photos: combines background illumination
    normalization with edge-preserving unsharp masking to make small text crisp.
    """
    if strength <= 1e-4:
        return bgr.copy()

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_float = l_chan.astype(np.float32)

    # Estimate slow illumination background via large morphological/Gaussian smoothing
    bg = cv2.GaussianBlur(l_float, (0, 0), sigmaX=25.0, sigmaY=25.0)
    normalized = np.clip((l_float / np.maximum(bg, 15.0)) * 235.0, 0.0, 255.0)

    # Blend normalized luminance with original according to strength
    blend_ratio = float(np.clip(strength * 0.35, 0.0, 0.65))
    l_blended = ((1.0 - blend_ratio) * l_float + blend_ratio * normalized).astype(np.uint8)

    merged = cv2.cvtColor(cv2.merge([l_blended, a_chan, b_chan]), cv2.COLOR_LAB2BGR)
    return unsharp_mask(merged, strength=strength, radius=1.1, threshold=3, halo_limit=35.0)
