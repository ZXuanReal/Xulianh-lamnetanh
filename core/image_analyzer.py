"""
Image quality analyzer, reference metrics (PSNR, SSIM), and rule-based Auto Enhance advisor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

import cv2
import numpy as np

from core.image_io import ImageContainer


@dataclass
class ImageDiagnostics:
    """Quantitative characteristics estimated from an input image."""

    width: int
    height: int
    megapixels: float
    laplacian_var: float
    tenengrad_score: float
    noise_sigma: float
    rms_contrast: float
    dynamic_range: float
    mean_brightness: float
    is_low_res: bool
    is_blurry: bool
    is_very_blurry: bool
    is_noisy: bool
    is_low_contrast: bool
    is_dark: bool
    is_bright: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AutoRecommendation:
    """Explainable recommendation produced by the Auto Enhance engine."""

    preset_key: str
    method_title: str
    explanation: str
    confidence_note: str
    recommended_params: Dict[str, Any] = field(default_factory=dict)
    diagnostics: ImageDiagnostics | None = None


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """
    Estimate Gaussian noise standard deviation using Immerkaer's fast 3x3 Laplacian operator
    and Median Absolute Deviation (MAD), robust against natural image edges.
    """
    gray_f = gray.astype(np.float32)
    kernel = np.array(
        [
            [1.0, -2.0, 1.0],
            [-2.0, 4.0, -2.0],
            [1.0, -2.0, 1.0],
        ],
        dtype=np.float32,
    )
    conv = cv2.filter2D(gray_f, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT_101)
    # Normalization factor for 3x3 Laplacian operator: sqrt(pi/2) / 6 = 0.208885689
    mad = float(np.median(np.abs(conv)))
    sigma = mad * 0.208885689 * 1.4826
    return round(float(sigma), 2)


def compute_edge_sharpness(bgr: np.ndarray) -> float:
    """Compute Tenengrad (Sobel gradient magnitude mean) sharpness index."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    return round(float(np.mean(mag)), 2)


def compute_psnr(ref_bgr: np.ndarray, target_bgr: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (dB) between reference and target BGR images.
    If dimensions differ, target is resized to reference dimensions for comparison.
    """
    if ref_bgr.shape[:2] != target_bgr.shape[:2]:
        target_bgr = cv2.resize(
            target_bgr, (ref_bgr.shape[1], ref_bgr.shape[0]), interpolation=cv2.INTER_CUBIC
        )
    mse = float(np.mean((ref_bgr.astype(np.float64) - target_bgr.astype(np.float64)) ** 2))
    if mse <= 1e-10:
        return 100.0
    psnr = 10.0 * np.log10((255.0 ** 2) / mse)
    return round(float(psnr), 2)


def compute_ssim(ref_bgr: np.ndarray, target_bgr: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM) on luminance using an 11x11 Gaussian window.
    Returns a float in [-1.0, 1.0], where 1.0 indicates identical structure.
    """
    if ref_bgr.shape[:2] != target_bgr.shape[:2]:
        target_bgr = cv2.resize(
            target_bgr, (ref_bgr.shape[1], ref_bgr.shape[0]), interpolation=cv2.INTER_CUBIC
        )

    x = (cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY) if ref_bgr.ndim == 3 else ref_bgr).astype(np.float64)
    y = (cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY) if target_bgr.ndim == 3 else target_bgr).astype(np.float64)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu1 = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(y, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu1_mu2

    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / np.maximum(denominator, 1e-12)
    return round(float(np.mean(ssim_map)), 4)


def analyze_image(container: ImageContainer | np.ndarray) -> ImageDiagnostics:
    """
    Analyze an image container or BGR array to extract blur, noise, contrast, and resolution indicators.
    """
    bgr = container.bgr if isinstance(container, ImageContainer) else container
    h, w = bgr.shape[:2]
    mp = round((w * h) / 1_000_000.0, 3)

    # Downsample very large images for fast diagnostic estimation
    max_dim = max(h, w)
    if max_dim > 1600:
        scale = 1600.0 / max_dim
        sample_bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        sample_bgr = bgr

    gray = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2GRAY)

    # Smooth slightly with 3x3 Gaussian before Laplacian so fine grain noise isn't mistaken for sharpness
    gray_smooth = cv2.GaussianBlur(gray, (3, 3), 0.8)
    lap = cv2.Laplacian(gray_smooth, cv2.CV_64F)
    lap_var = round(float(lap.var()), 2)

    tenengrad = compute_edge_sharpness(sample_bgr)
    noise_sigma = estimate_noise_sigma(gray)

    rms_contrast = round(float(np.std(gray.astype(np.float32))), 2)
    p2, p98 = np.percentile(gray, [2, 98])
    dyn_range = round(float(p98 - p2), 2)
    mean_luma = round(float(np.mean(gray)), 2)

    is_low_res = (w < 640 or h < 640) or (w * h < 350_000)
    is_very_blurry = lap_var < 55.0
    is_blurry = lap_var < 140.0
    is_noisy = noise_sigma > 4.2
    is_low_contrast = rms_contrast < 36.0 or dyn_range < 130.0
    is_dark = mean_luma < 75.0
    is_bright = mean_luma > 205.0

    return ImageDiagnostics(
        width=w,
        height=h,
        megapixels=mp,
        laplacian_var=lap_var,
        tenengrad_score=tenengrad,
        noise_sigma=noise_sigma,
        rms_contrast=rms_contrast,
        dynamic_range=dyn_range,
        mean_brightness=mean_luma,
        is_low_res=is_low_res,
        is_blurry=is_blurry,
        is_very_blurry=is_very_blurry,
        is_noisy=is_noisy,
        is_low_contrast=is_low_contrast,
        is_dark=is_dark,
        is_bright=is_bright,
    )


def recommend_auto_enhance(
    container: ImageContainer | np.ndarray,
    ai_available: bool = False,
) -> AutoRecommendation:
    """
    Inspect image diagnostics and return a balanced, non-destructive enhancement configuration.
    Follows strict guardrails: avoids stacking heavy sharpening on noisy images, and only activates
    AI Super-Resolution when input resolution is low and AI models are ready.
    """
    diag = analyze_image(container)

    # Start with neutral/safe parameters
    params: Dict[str, Any] = {
        "sharpen_method": "unsharp",
        "sharpen_strength": 0.8,
        "sharpen_radius": 1.3,
        "sharpen_threshold": 4,
        "denoise_method": "nlm",
        "denoise_strength": 0.0,
        "contrast_clahe": False,
        "clahe_clip_limit": 1.4,
        "brightness": 0.0,
        "gamma": 1.0,
        "upscale_factor": 1,
        "use_ai_sr": False,
        "use_face_restore": False,
    }

    reasons: list[str] = []

    # Case 1: Low-resolution image
    if diag.is_low_res:
        if ai_available:
            params["upscale_factor"] = 2
            params["use_ai_sr"] = True
            params["sharpen_strength"] = 0.5
            preset_key = "ai_restoration"
            method_title = "Phục hồi & Phóng to AI 2x (Auto)"
            reasons.append(
                f"Ảnh có độ phân giải thấp ({diag.width}x{diag.height}); kích hoạt AI Super-Resolution 2x kết hợp làm nét nhẹ."
            )
        else:
            params["upscale_factor"] = 2
            params["use_ai_sr"] = False
            params["sharpen_strength"] = 1.05
            params["sharpen_radius"] = 1.2
            preset_key = "standard_sharpen"
            method_title = "Phóng to 2x & Làm nét cân bằng (Auto)"
            reasons.append(
                f"Ảnh có độ phân giải thấp ({diag.width}x{diag.height}); chưa có trọng số AI nên sử dụng Lanczos4 + Edge Sharpening an toàn."
            )
    # Case 2: High noise detected -> prioritize denoising and moderate thresholded sharpening
    elif diag.is_noisy:
        denoise_val = min(14.0, max(5.0, diag.noise_sigma * 1.4))
        params["denoise_strength"] = round(denoise_val, 1)
        params["sharpen_strength"] = 0.75 if diag.is_blurry else 0.45
        params["sharpen_threshold"] = 8  # higher threshold so residual noise is not sharpened
        preset_key = "denoise_focus"
        method_title = "Khử nhiễu ưu tiên & Giữ cạnh (Auto)"
        reasons.append(
            f"Phát hiện mức nhiễu cao (σ ≈ {diag.noise_sigma}); ưu tiên khử nhiễu NLM ({params['denoise_strength']}) và tăng ngưỡng bảo vệ vùng phẳng."
        )
    # Case 3: Very blurry image with clean signal
    elif diag.is_very_blurry:
        params["sharpen_strength"] = 1.55
        params["sharpen_radius"] = 1.6
        params["sharpen_threshold"] = 3
        preset_key = "strong_sharpen"
        method_title = "Làm nét chuyên sâu chống Halo (Auto)"
        reasons.append(
            f"Chỉ số sắc nét thấp (Laplacian Var = {diag.laplacian_var}); áp dụng Unsharp Mask cường độ cao kèm kìm hãm viền sáng."
        )
    # Case 4: Moderately blurry image
    elif diag.is_blurry:
        params["sharpen_strength"] = 1.1
        params["sharpen_radius"] = 1.4
        params["sharpen_threshold"] = 4
        preset_key = "standard_sharpen"
        method_title = "Làm nét tiêu chuẩn (Auto)"
        reasons.append(
            f"Ảnh hơi mềm nét (Laplacian Var = {diag.laplacian_var}); áp dụng làm nét tiêu chuẩn trên kênh độ sáng LAB."
        )
    # Case 5: Already sharp image -> light micro-contrast & gentle polish
    else:
        params["sharpen_strength"] = 0.55
        params["sharpen_radius"] = 1.0
        params["sharpen_threshold"] = 5
        preset_key = "light_sharpen"
        method_title = "Làm nét nhẹ bảo toàn chi tiết (Auto)"
        reasons.append(
            f"Ảnh đã có độ nét khá tốt (Laplacian Var = {diag.laplacian_var}); chỉ tăng cường chi tiết vi mô nhẹ để tránh hiện tượng ảnh quá gắt."
        )

    # Adjust contrast gently if flat
    if diag.is_low_contrast:
        params["contrast_clahe"] = True
        params["clahe_clip_limit"] = 1.5
        reasons.append(f"Độ tương phản thấp (RMS = {diag.rms_contrast}); bật cân bằng tương phản cục bộ CLAHE nhẹ.")

    # Gentle exposure correction if noticeably dark
    if diag.is_dark:
        params["gamma"] = 1.15
        reasons.append(f"Ảnh thiếu sáng nhẹ (Luminance TB = {diag.mean_brightness}); nâng Gamma = 1.15.")

    explanation = " ".join(reasons)
    confidence_note = (
        "Lưu ý: Đây là cấu hình ước lượng tự động từ thông số thống kê của ảnh. "
        "Bạn có thể tinh chỉnh thêm trên bảng điều khiển để phù hợp nhất với mục đích sử dụng."
    )

    return AutoRecommendation(
        preset_key=preset_key,
        method_title=method_title,
        explanation=explanation,
        confidence_note=confidence_note,
        recommended_params=params,
        diagnostics=diag,
    )
