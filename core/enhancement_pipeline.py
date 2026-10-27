"""
Unified Image Enhancement Pipeline combining traditional OpenCV signal processing
and AI Super-Resolution / Face Restoration in optimal order.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
import time
from typing import Any, Callable, Dict, List

import cv2

from ai.device_manager import DeviceManager
from ai.face_restoration import FaceRestorer
from ai.model_manager import ModelLoadError, ModelManager
from ai.super_resolution import SuperResolutionEngine, TaskCancelledError, edge_directed_upscale
from core.denoise import adjust_brightness_gamma, denoise_image, enhance_contrast_clahe
from core.image_analyzer import compute_edge_sharpness, recommend_auto_enhance
from core.image_io import ImageContainer
from core.sharpen import adaptive_document_sharpen, kernel_sharpen, unsharp_mask

logger = logging.getLogger(__name__)


@dataclass
class EnhancementParams:
    """Complete set of parameters controlling a single or batch enhancement run."""

    mode: str = "auto"  # auto, manual, sharpen, restore_ai
    preset: str = "auto"
    sharpen_method: str = "unsharp"  # unsharp, kernel, document, none
    sharpen_strength: float = 1.0  # 0.0 .. 3.0
    sharpen_radius: float = 1.4  # 0.5 .. 5.0
    sharpen_threshold: int = 4  # 0 .. 30
    denoise_method: str = "nlm"  # nlm, bilateral, median, none
    denoise_strength: float = 0.0  # 0.0 .. 25.0
    contrast_clahe: bool = False
    clahe_clip_limit: float = 1.5  # 0.5 .. 4.0
    brightness: float = 0.0  # -100 .. +100
    gamma: float = 1.0  # 0.2 .. 3.0
    upscale_factor: int = 1  # 1, 2, 4
    use_ai_sr: bool = False
    ai_model_name: str = "realesr-general-x4v3"
    tile_size: int = 256
    tile_pad: int = 16
    use_face_restore: bool = False
    face_restore_fidelity: float = 0.65

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnhancementParams":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass(frozen=True)
class PresetDefinition:
    """Starter preset definition with user-friendly title, description, and default parameters."""

    key: str
    title: str
    description: str
    params: Dict[str, Any]


PRESETS: Dict[str, PresetDefinition] = {
    "auto": PresetDefinition(
        key="auto",
        title="Tự động phân tích (Auto Enhance)",
        description="Phân tích độ mờ, nhiễu, tương phản và kích thước ảnh để tự động chọn thông số an toàn.",
        params={"mode": "auto", "preset": "auto"},
    ),
    "light_sharpen": PresetDefinition(
        key="light_sharpen",
        title="Làm nét nhẹ",
        description="Tăng nhẹ độ sắc nét cạnh, giữ nguyên kết cấu tự nhiên cho ảnh chân dung và sinh hoạt.",
        params={
            "mode": "manual",
            "preset": "light_sharpen",
            "sharpen_method": "unsharp",
            "sharpen_strength": 0.65,
            "sharpen_radius": 1.1,
            "sharpen_threshold": 5,
            "denoise_strength": 0.0,
            "contrast_clahe": False,
            "upscale_factor": 1,
            "use_ai_sr": False,
        },
    ),
    "standard_sharpen": PresetDefinition(
        key="standard_sharpen",
        title="Làm nét tiêu chuẩn",
        description="Cấu hình cân bằng giữa độ nét và kiểm soát viền sáng halo cho đa số ảnh hơi mờ.",
        params={
            "mode": "manual",
            "preset": "standard_sharpen",
            "sharpen_method": "unsharp",
            "sharpen_strength": 1.15,
            "sharpen_radius": 1.4,
            "sharpen_threshold": 4,
            "denoise_strength": 2.0,
            "contrast_clahe": False,
            "upscale_factor": 1,
            "use_ai_sr": False,
        },
    ),
    "strong_sharpen": PresetDefinition(
        key="strong_sharpen",
        title="Làm nét mạnh",
        description="Tăng cường chi tiết biên mạnh mẽ kèm bộ lọc chống nhiễu nhẹ và nén viền sáng.",
        params={
            "mode": "manual",
            "preset": "strong_sharpen",
            "sharpen_method": "unsharp",
            "sharpen_strength": 1.85,
            "sharpen_radius": 1.7,
            "sharpen_threshold": 4,
            "denoise_strength": 3.5,
            "contrast_clahe": True,
            "clahe_clip_limit": 1.3,
            "upscale_factor": 1,
            "use_ai_sr": False,
        },
    ),
    "denoise_focus": PresetDefinition(
        key="denoise_focus",
        title="Giảm nhiễu",
        description="Làm sạch nhiễu hạt ISO cao hoặc ảnh chụp thiếu sáng trong khi vẫn giữ biên sắc nét.",
        params={
            "mode": "manual",
            "preset": "denoise_focus",
            "sharpen_method": "unsharp",
            "sharpen_strength": 0.6,
            "sharpen_radius": 1.2,
            "sharpen_threshold": 8,
            "denoise_method": "nlm",
            "denoise_strength": 9.5,
            "contrast_clahe": False,
            "upscale_factor": 1,
            "use_ai_sr": False,
        },
    ),
    "product_clarity": PresetDefinition(
        key="product_clarity",
        title="Làm rõ ảnh sản phẩm",
        description="Tăng độ trong trẻo, tương phản vi mô (CLAHE) và độ nét bề mặt cho ảnh thương mại điện tử.",
        params={
            "mode": "manual",
            "preset": "product_clarity",
            "sharpen_method": "unsharp",
            "sharpen_strength": 1.3,
            "sharpen_radius": 1.3,
            "sharpen_threshold": 3,
            "denoise_method": "bilateral",
            "denoise_strength": 3.0,
            "contrast_clahe": True,
            "clahe_clip_limit": 1.6,
            "gamma": 1.04,
            "upscale_factor": 1,
            "use_ai_sr": False,
        },
    ),
    "document_enhance": PresetDefinition(
        key="document_enhance",
        title="Cải thiện ảnh tài liệu",
        description="Làm phẳng nền sáng, tăng độ tương phản chữ nhỏ và làm sắc nét nét chữ chụp từ tài liệu.",
        params={
            "mode": "manual",
            "preset": "document_enhance",
            "sharpen_method": "document",
            "sharpen_strength": 1.45,
            "sharpen_radius": 1.1,
            "sharpen_threshold": 3,
            "denoise_method": "median",
            "denoise_strength": 3.0,
            "contrast_clahe": True,
            "clahe_clip_limit": 2.0,
            "upscale_factor": 1,
            "use_ai_sr": False,
        },
    ),
    "ai_restoration": PresetDefinition(
        key="ai_restoration",
        title="Phục hồi & Tăng phân giải AI (2x)",
        description="Sử dụng mạng nơ-ron sâu Real-ESRGAN để phóng to 2x và tái tạo chi tiết cho ảnh phân giải thấp.",
        params={
            "mode": "restore_ai",
            "preset": "ai_restoration",
            "sharpen_method": "unsharp",
            "sharpen_strength": 0.45,
            "sharpen_radius": 1.1,
            "sharpen_threshold": 5,
            "denoise_strength": 0.0,
            "contrast_clahe": False,
            "upscale_factor": 2,
            "use_ai_sr": True,
            "ai_model_name": "realesr-general-x4v3",
        },
    ),
}


@dataclass
class PipelineResult:
    """Output of a complete enhancement pipeline execution."""

    container: ImageContainer
    method_summary: str
    explanation: str
    applied_params: Dict[str, Any]
    steps_executed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_sec: float = 0.0
    sharpness_before: float = 0.0
    sharpness_after: float = 0.0


class EnhancementPipeline:
    """
    Executes the multi-stage enhancement workflow without mutating the source image.
    Order of operations:
    1. Auto-analysis (if mode == 'auto')
    2. Noise reduction (before contrast/sharpening so noise is not amplified)
    3. Brightness & Gamma adjustment
    4. Local contrast enhancement (CLAHE on LAB L-channel)
    5. Super-Resolution / Upscaling (AI Real-ESRGAN or Edge-Directed Lanczos)
    6. Optional Face Restoration (when explicitly enabled by user)
    7. Luminance Sharpening (Unsharp Mask / Kernel / Adaptive Document)
    8. Alpha channel synchronization
    """

    def __init__(
        self,
        model_manager: ModelManager | None = None,
        device_manager: DeviceManager | None = None,
    ) -> None:
        self.model_manager = model_manager or ModelManager()
        self.device_manager = device_manager or DeviceManager()
        self.sr_engine = SuperResolutionEngine(self.model_manager, self.device_manager)
        self.face_restorer = FaceRestorer()

    def process(
        self,
        source: ImageContainer,
        params: EnhancementParams,
        progress_callback: Callable[[int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> PipelineResult:
        """Run the enhancement pipeline on a deep clone of `source`."""
        start_t = time.perf_counter()

        def report(pct: int, msg: str) -> None:
            if cancel_check and cancel_check():
                raise TaskCancelledError("Tác vụ xử lý ảnh đã bị hủy bởi người dùng.")
            if progress_callback:
                progress_callback(pct, msg)

        report(5, "Đang khởi tạo bản sao dữ liệu ảnh an toàn...")
        working = source.clone()
        bgr = working.bgr
        sharpness_before = compute_edge_sharpness(bgr)

        eff_params = EnhancementParams.from_dict(params.to_dict())
        steps: List[str] = []
        warnings: List[str] = list(working.warnings)
        explanation = ""
        method_summary = PRESETS.get(eff_params.preset, PRESETS["standard_sharpen"]).title

        # Step 1: Auto Enhance resolution
        if eff_params.mode == "auto" or eff_params.preset == "auto":
            report(10, "Đang phân tích độ mờ, nhiễu và tương phản (Auto Enhance)...")
            ai_ready = self.model_manager.is_weight_installed(eff_params.ai_model_name)
            rec = recommend_auto_enhance(working, ai_available=ai_ready)
            for k, v in rec.recommended_params.items():
                if hasattr(eff_params, k):
                    setattr(eff_params, k, v)
            method_summary = rec.method_title
            explanation = f"{rec.explanation} {rec.confidence_note}"
            steps.append(f"Auto Enhance: {rec.method_title}")
        else:
            preset_obj = PRESETS.get(eff_params.preset)
            if preset_obj:
                method_summary = preset_obj.title
                explanation = preset_obj.description
            else:
                method_summary = "Tùy chỉnh thủ công"
                explanation = "Áp dụng bộ thông số do người dùng cấu hình."

        # Step 2: Denoising
        if eff_params.denoise_strength > 0.05 and eff_params.denoise_method != "none":
            report(22, f"Đang khử nhiễu ({eff_params.denoise_method.upper()}, cường độ {eff_params.denoise_strength:.1f})...")
            bgr = denoise_image(
                bgr,
                strength=eff_params.denoise_strength,
                method=eff_params.denoise_method,
            )
            steps.append(f"Giảm nhiễu ({eff_params.denoise_method}, mức {eff_params.denoise_strength:.1f})")

        # Step 3: Brightness & Gamma
        if abs(eff_params.brightness) >= 0.5 or abs(eff_params.gamma - 1.0) >= 0.02:
            report(35, "Đang cân chỉnh độ sáng và Gamma...")
            bgr = adjust_brightness_gamma(bgr, brightness=eff_params.brightness, gamma=eff_params.gamma)
            steps.append(f"Độ sáng ({eff_params.brightness:+.0f}), Gamma ({eff_params.gamma:.2f})")

        # Step 4: Contrast Enhancement (CLAHE)
        if eff_params.contrast_clahe and eff_params.clahe_clip_limit > 0.05:
            report(45, f"Đang cải thiện tương phản cục bộ CLAHE (Clip={eff_params.clahe_clip_limit:.1f})...")
            bgr = enhance_contrast_clahe(bgr, clip_limit=eff_params.clahe_clip_limit, blend=0.85)
            steps.append(f"Tương phản CLAHE (Clip={eff_params.clahe_clip_limit:.1f})")

        # Step 5: Super-Resolution / Upscaling
        scale = int(eff_params.upscale_factor)
        if eff_params.use_ai_sr or scale > 1:
            eff_scale = max(2, scale) if eff_params.use_ai_sr and scale <= 1 else scale
            if eff_params.use_ai_sr:
                dev_status = self.device_manager.status
                if dev_status.fallback_warning:
                    warnings.append(dev_status.fallback_warning)
                warnings.append(
                    "Lưu ý AI: Mô hình AI tái tạo chi tiết dựa trên học sâu và có thể tạo kết cấu hơi khác ảnh gốc. "
                    "Hãy kiểm tra bằng thanh trượt Before/After."
                )

                def sr_progress(curr_t: int, total_t: int, msg: str) -> None:
                    pct = 50 + int((curr_t / max(1, total_t)) * 30)
                    report(pct, msg)

                report(50, f"Đang chạy AI Super-Resolution ({eff_params.ai_model_name}, {eff_scale}x)...")
                bgr = self.sr_engine.upscale(
                    bgr=bgr,
                    outscale=eff_scale,
                    model_name=eff_params.ai_model_name,
                    tile_size=eff_params.tile_size,
                    tile_pad=eff_params.tile_pad,
                    progress_callback=sr_progress,
                    cancel_check=cancel_check,
                )
                steps.append(
                    f"AI Super-Resolution {eff_scale}x ({eff_params.ai_model_name} trên {dev_status.selected_device.upper()})"
                )
            elif eff_scale > 1:
                report(65, f"Đang phóng to ảnh {eff_scale}x (Edge-Directed Lanczos4)...")
                bgr = edge_directed_upscale(bgr, scale=eff_scale)
                steps.append(f"Phóng to truyền thống {eff_scale}x (Edge-Directed Lanczos4)")

        # Step 6: Optional Face Restoration
        if eff_params.use_face_restore:
            report(82, "Đang dò tìm và phục hồi khuôn mặt...")
            face_res = self.face_restorer.restore_faces(bgr, fidelity=eff_params.face_restore_fidelity)
            bgr = face_res.bgr
            steps.append(f"Phục hồi khuôn mặt ({face_res.faces_detected} khuôn mặt)")
            warnings.append(face_res.note)

        # Step 7: Sharpening
        if eff_params.sharpen_strength > 0.01 and eff_params.sharpen_method != "none":
            report(90, f"Đang làm nét ảnh ({eff_params.sharpen_method}, mức {eff_params.sharpen_strength:.2f})...")
            if eff_params.sharpen_method == "kernel":
                bgr = kernel_sharpen(bgr, strength=eff_params.sharpen_strength)
                steps.append(f"Làm nét Kernel tích chập (mức {eff_params.sharpen_strength:.2f})")
            elif eff_params.sharpen_method == "document":
                bgr = adaptive_document_sharpen(bgr, strength=eff_params.sharpen_strength)
                steps.append(f"Làm rõ tài liệu & văn bản (mức {eff_params.sharpen_strength:.2f})")
            else:
                bgr = unsharp_mask(
                    bgr,
                    strength=eff_params.sharpen_strength,
                    radius=eff_params.sharpen_radius,
                    threshold=eff_params.sharpen_threshold,
                )
                steps.append(
                    f"Unsharp Mask (Strength={eff_params.sharpen_strength:.2f}, Radius={eff_params.sharpen_radius:.1f}, Threshold={eff_params.sharpen_threshold})"
                )

        # Step 8: Synchronize Alpha channel if resolution changed
        out_alpha = working.alpha
        if out_alpha is not None and (out_alpha.shape[0] != bgr.shape[0] or out_alpha.shape[1] != bgr.shape[1]):
            out_alpha = cv2.resize(
                out_alpha,
                (bgr.shape[1], bgr.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )

        working.bgr = bgr
        working.alpha = out_alpha
        working.warnings = warnings

        sharpness_after = compute_edge_sharpness(bgr)
        elapsed = round(time.perf_counter() - start_t, 3)

        if not steps:
            steps.append("Giữ nguyên ảnh gốc (không có bộ lọc nào được bật)")

        report(100, f"Hoàn tất xử lý trong {elapsed:.2f}s.")
        return PipelineResult(
            container=working,
            method_summary=method_summary,
            explanation=explanation,
            applied_params=eff_params.to_dict(),
            steps_executed=steps,
            warnings=warnings,
            duration_sec=elapsed,
            sharpness_before=sharpness_before,
            sharpness_after=sharpness_after,
        )
