"""
Tests for core/denoise.py and core/image_analyzer.py using synthetic noisy images
and PSNR / SSIM / noise sigma verification.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.denoise import adjust_brightness_gamma, denoise_image, enhance_contrast_clahe
from core.image_analyzer import (
    analyze_image,
    compute_psnr,
    compute_ssim,
    estimate_noise_sigma,
    recommend_auto_enhance,
)


def _create_clean_test_image(width: int = 200, height: int = 160) -> np.ndarray:
    img = np.full((height, width, 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (95, 130), (60, 110, 200), -1)
    cv2.circle(img, (145, 80), 40, (80, 180, 90), -1)
    return img


def test_nlm_and_bilateral_denoising_improve_psnr_and_ssim() -> None:
    """Verify NLM and Bilateral denoising reduce estimated noise sigma and increase PSNR/SSIM."""
    clean = _create_clean_test_image()
    rng = np.random.default_rng(12345)
    noise = rng.normal(0.0, 14.0, size=clean.shape).astype(np.float32)
    noisy = np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    noisy_sigma = estimate_noise_sigma(cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY))
    psnr_noisy = compute_psnr(clean, noisy)
    ssim_noisy = compute_ssim(clean, noisy)

    # 1. Non-Local Means
    denoised_nlm = denoise_image(noisy, strength=11.0, method="nlm")
    nlm_sigma = estimate_noise_sigma(cv2.cvtColor(denoised_nlm, cv2.COLOR_BGR2GRAY))
    assert nlm_sigma < noisy_sigma
    assert compute_psnr(clean, denoised_nlm) > psnr_noisy
    assert compute_ssim(clean, denoised_nlm) > ssim_noisy

    # 2. Bilateral
    denoised_bil = denoise_image(noisy, strength=10.0, method="bilateral")
    bil_sigma = estimate_noise_sigma(cv2.cvtColor(denoised_bil, cv2.COLOR_BGR2GRAY))
    assert bil_sigma < noisy_sigma
    assert compute_psnr(clean, denoised_bil) > psnr_noisy

    # 3. Median
    denoised_med = denoise_image(noisy, strength=9.0, method="median")
    assert compute_psnr(clean, denoised_med) > psnr_noisy


def test_clahe_contrast_and_brightness_gamma() -> None:
    """Verify CLAHE increases RMS contrast on low-contrast images and brightness/gamma adjust luminance."""
    low_contrast = np.full((100, 120, 3), 120, dtype=np.uint8)
    low_contrast[20:80, 30:90] = 145

    enhanced = enhance_contrast_clahe(low_contrast, clip_limit=2.2, blend=1.0)
    diag_before = analyze_image(low_contrast)
    diag_after = analyze_image(enhanced)
    assert diag_after.rms_contrast > diag_before.rms_contrast

    brightened = adjust_brightness_gamma(low_contrast, brightness=25.0, gamma=1.2)
    assert float(brightened.mean()) > float(low_contrast.mean())


def test_auto_enhance_recommendation_rules() -> None:
    """Verify Auto Enhance selects explainable, appropriate methods for low-res, noisy, and blurry inputs."""
    # 1. Low-res image (without AI weights and with AI weights)
    low_res = _create_clean_test_image(320, 240)
    rec_no_ai = recommend_auto_enhance(low_res, ai_available=False)
    assert rec_no_ai.recommended_params["upscale_factor"] == 2
    assert rec_no_ai.recommended_params["use_ai_sr"] is False

    rec_with_ai = recommend_auto_enhance(low_res, ai_available=True)
    assert rec_with_ai.recommended_params["upscale_factor"] == 2
    assert rec_with_ai.recommended_params["use_ai_sr"] is True

    # 2. High-noise standard-resolution image
    clean_hd = _create_clean_test_image(800, 640)
    rng = np.random.default_rng(99)
    noisy_hd = np.clip(clean_hd.astype(np.float32) + rng.normal(0.0, 18.0, clean_hd.shape), 0, 255).astype(np.uint8)
    rec_noisy = recommend_auto_enhance(noisy_hd, ai_available=False)
    assert rec_noisy.preset_key == "denoise_focus"
    assert rec_noisy.recommended_params["denoise_strength"] >= 5.0
