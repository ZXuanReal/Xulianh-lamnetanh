"""
Tests for core/sharpen.py using synthetic reference images, controlled blur degradation,
and quantitative evaluation via PSNR, SSIM, and edge sharpness metrics.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.image_analyzer import compute_edge_sharpness, compute_psnr, compute_ssim
from core.sharpen import adaptive_document_sharpen, kernel_sharpen, unsharp_mask


def _build_reference_pattern(width: int = 240, height: int = 180) -> np.ndarray:
    """Create a synthetic reference image containing edges, geometric shapes, and text."""
    img = np.full((height, width, 3), 220, dtype=np.uint8)
    cv2.rectangle(img, (25, 25), (110, 95), (45, 70, 180), -1)
    cv2.circle(img, (175, 65), 35, (30, 140, 60), -1)
    cv2.putText(img, "SHARP 123", (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 25), 2, cv2.LINE_AA)
    return img


def test_unsharp_mask_improves_blurred_reference_metrics() -> None:
    """Verify Unsharp Mask increases edge sharpness and improves PSNR/SSIM toward reference."""
    ref = _build_reference_pattern()
    blurred = cv2.GaussianBlur(ref, (5, 5), 1.3)
    blurred_copy = blurred.copy()

    restored = unsharp_mask(blurred, strength=1.35, radius=1.3, threshold=2, halo_limit=30.0)

    # Ensure input array was never mutated
    assert np.array_equal(blurred, blurred_copy)
    assert restored.shape == ref.shape
    assert restored.dtype == np.uint8

    sharp_blur = compute_edge_sharpness(blurred)
    sharp_restored = compute_edge_sharpness(restored)
    assert sharp_restored > sharp_blur

    psnr_blur = compute_psnr(ref, blurred)
    psnr_restored = compute_psnr(ref, restored)
    ssim_blur = compute_ssim(ref, blurred)
    ssim_restored = compute_ssim(ref, restored)

    assert psnr_restored > psnr_blur
    assert ssim_restored >= ssim_blur


def test_kernel_sharpen_and_zero_strength_identity() -> None:
    """Verify convolution kernel sharpening and zero-strength identity behavior."""
    ref = _build_reference_pattern()
    blurred = cv2.GaussianBlur(ref, (5, 5), 1.2)

    identity_out = kernel_sharpen(blurred, strength=0.0)
    assert np.array_equal(identity_out, blurred)

    sharpened = kernel_sharpen(blurred, strength=1.0)
    assert compute_edge_sharpness(sharpened) > compute_edge_sharpness(blurred)
    assert compute_psnr(ref, sharpened) > compute_psnr(ref, blurred)


def test_halo_limit_suppression() -> None:
    """Verify soft-knee halo clamping prevents extreme overshoot on high-contrast edges."""
    step_img = np.full((100, 100, 3), 100, dtype=np.uint8)
    step_img[:, 50:] = 160
    blurred_step = cv2.GaussianBlur(step_img, (5, 5), 1.5)

    tight_halo = unsharp_mask(blurred_step, strength=2.5, radius=1.5, threshold=0, halo_limit=10.0)
    loose_halo = unsharp_mask(blurred_step, strength=2.5, radius=1.5, threshold=0, halo_limit=80.0)

    assert int(tight_halo.max()) <= int(loose_halo.max())


def test_adaptive_document_sharpen() -> None:
    """Verify document sharpening enhances uneven illumination text images."""
    doc = np.full((120, 200, 3), 200, dtype=np.uint8)
    cv2.putText(doc, "Document Text", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    blurred_doc = cv2.GaussianBlur(doc, (3, 3), 1.0)

    out_doc = adaptive_document_sharpen(blurred_doc, strength=1.2)
    assert out_doc.shape == doc.shape
    assert compute_edge_sharpness(out_doc) > compute_edge_sharpness(blurred_doc)
