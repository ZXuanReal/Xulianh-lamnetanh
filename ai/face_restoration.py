"""
Optional Face Restoration module.
Activates strictly when explicitly enabled by the user. Detects faces and applies
region-aware skin-preserving frequency separation + facial feature enhancement
with soft elliptical feather blending and configurable fidelity.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from core.sharpen import unsharp_mask

logger = logging.getLogger(__name__)


@dataclass
class FaceRestorationResult:
    """Result of the optional face restoration pass."""

    bgr: np.ndarray
    faces_detected: int
    face_boxes: List[Tuple[int, int, int, int]]
    note: str


class FaceRestorer:
    """
    Detects human faces in portraits/group photos and applies targeted restoration
    (skin noise suppression + high-frequency feature refinement around eyes/mouth)
    blended smoothly via a cosine-feathered elliptical mask.
    """

    def __init__(self) -> None:
        self._cascade: cv2.CascadeClassifier | None = None
        self._init_detector()

    def _init_detector(self) -> None:
        try:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if cascade_path.exists():
                self._cascade = cv2.CascadeClassifier(str(cascade_path))
        except Exception as exc:
            logger.warning("Không thể khởi tạo bộ nhận diện khuôn mặt Haar Cascade: %s", exc)
            self._cascade = None

    def detect_faces(self, bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect frontal faces and return list of (x, y, w, h) bounding boxes."""
        if self._cascade is None or self._cascade.empty():
            return []

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        min_dim = max(32, int(min(h, w) * 0.06))

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_dim, min_dim),
        )
        if faces is None or len(faces) == 0:
            return []
        return [(int(x), int(y), int(fw), int(fh)) for (x, y, fw, fh) in faces]

    def restore_faces(
        self,
        bgr: np.ndarray,
        fidelity: float = 0.65,
    ) -> FaceRestorationResult:
        """
        Restore detected faces in the image.

        Args:
            bgr: Input HxWx3 uint8 BGR image.
            fidelity: Balance between original appearance (1.0) and strong restoration (0.0).
                      Default 0.65 preserves identity while cleaning skin & sharpening eyes/lips.
        """
        boxes = self.detect_faces(bgr)
        if not boxes:
            return FaceRestorationResult(
                bgr=bgr.copy(),
                faces_detected=0,
                face_boxes=[],
                note="Không phát hiện khuôn mặt rõ ràng trong ảnh; giữ nguyên vùng ảnh gốc.",
            )

        out = bgr.copy()
        h_img, w_img = out.shape[:2]
        strength = float(np.clip(1.15 - 0.6 * float(np.clip(fidelity, 0.0, 1.0)), 0.35, 1.15))

        for (x, y, fw, fh) in boxes:
            # Expand bounding box slightly to include forehead and chin
            pad_x = int(fw * 0.15)
            pad_y = int(fh * 0.18)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_img, x + fw + pad_x)
            y2 = min(h_img, y + fh + pad_y)

            roi = out[y1:y2, x1:x2]
            rh, rw = roi.shape[:2]
            if rh < 16 or rw < 16:
                continue

            # 1. Frequency separation on LAB luminance: smooth skin blotches while retaining structure
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)

            smooth_l = cv2.bilateralFilter(l_ch, d=7, sigmaColor=24.0 * strength, sigmaSpace=9.0)
            # Detect high-gradient features (eyes, eyebrows, lips, nostrils) via Sobel magnitude
            gx = cv2.Sobel(l_ch, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(l_ch, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.hypot(gx, gy)
            feature_mask = np.clip((grad_mag - 18.0) / 45.0, 0.0, 1.0)

            # Enhance local contrast & sharpness on facial features while keeping smooth skin on flat areas
            sharpened_roi = unsharp_mask(
                roi,
                strength=1.25 * strength,
                radius=1.2,
                threshold=3,
                halo_limit=22.0,
            )
            smooth_bgr = cv2.cvtColor(cv2.merge([smooth_l, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

            feat_3c = feature_mask[:, :, np.newaxis]
            restored_roi = (
                feat_3c * sharpened_roi.astype(np.float32) + (1.0 - feat_3c) * smooth_bgr.astype(np.float32)
            )

            # Blend with original ROI according to user fidelity weight
            fid_weight = float(np.clip(fidelity, 0.15, 0.9))
            blended_roi = (1.0 - fid_weight * 0.45) * restored_roi + (fid_weight * 0.45) * roi.astype(np.float32)

            # Create soft elliptical mask so face boundary transitions invisibly
            yy, xx = np.ogrid[:rh, :rw]
            cy, cx = rh / 2.0, rw / 2.0
            ry, rx = max(1.0, rh * 0.46), max(1.0, rw * 0.46)
            dist = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
            ellipse_mask = np.clip((1.05 - dist) / 0.35, 0.0, 1.0).astype(np.float32)[:, :, np.newaxis]

            final_roi = ellipse_mask * blended_roi + (1.0 - ellipse_mask) * roi.astype(np.float32)
            out[y1:y2, x1:x2] = np.clip(final_roi, 0.0, 255.0).astype(np.uint8)

        note = (
            f"Đã phục hồi {len(boxes)} khuôn mặt (Fidelity={fidelity:.2f}). "
            "Lưu ý: Phục hồi khuôn mặt có thể làm thay đổi nhẹ kết cấu da; hãy dùng thanh trượt Before/After để kiểm tra."
        )
        return FaceRestorationResult(
            bgr=out,
            faces_detected=len(boxes),
            face_boxes=boxes,
            note=note,
        )
