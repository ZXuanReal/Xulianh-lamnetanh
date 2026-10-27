"""
Right-hand enhancement parameter control panel and application settings page widget.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ai.model_manager import MODEL_CATALOG, ModelManager
from config.settings import AppSettings
from core.enhancement_pipeline import PRESETS, EnhancementParams


class EnhancementControlPanel(QFrame):
    """
    Right-hand studio panel allowing users to select presets, tweak sharpening,
    denoising, contrast, exposure, AI super-resolution, face restoration, and export options.
    """

    process_requested = Signal(object)  # EnhancementParams
    auto_enhance_requested = Signal()
    cancel_requested = Signal()
    export_requested = Signal(str, int)  # (format, quality)
    download_model_requested = Signal(str)  # model_name

    def __init__(self, model_manager: ModelManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ControlPanel")
        self.setFixedWidth(355)
        self.model_manager = model_manager
        self._updating_ui = False

        self._build_ui()
        self.refresh_model_status()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("BẢNG ĐIỀU KHIỂN XỬ LÝ")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        # 1. Preset & Auto Mode Group
        grp_preset = QGroupBox("1. Chế độ & Preset khởi đầu")
        vp = QVBoxLayout(grp_preset)

        self.cmb_preset = QComboBox()
        for key, preset_def in PRESETS.items():
            self.cmb_preset.addItem(preset_def.title, userData=key)
        self.cmb_preset.currentIndexChanged.connect(self._on_preset_changed)
        vp.addWidget(self.cmb_preset)

        self.lbl_preset_desc = QLabel(PRESETS["auto"].description)
        self.lbl_preset_desc.setObjectName("SubTitle")
        self.lbl_preset_desc.setWordWrap(True)
        vp.addWidget(self.lbl_preset_desc)

        self.btn_auto = QPushButton("⚡ Phân tích & Đề xuất Tự động (Auto)")
        self.btn_auto.clicked.connect(self.auto_enhance_requested.emit)
        vp.addWidget(self.btn_auto)

        layout.addWidget(grp_preset)

        # 2. Sharpening Group
        grp_sharp = QGroupBox("2. Làm nét ảnh (Sharpening)")
        vs = QVBoxLayout(grp_sharp)

        row_m = QHBoxLayout()
        row_m.addWidget(QLabel("Thuật toán:"))
        self.cmb_sharpen_method = QComboBox()
        self.cmb_sharpen_method.addItem("Unsharp Mask (Chống Halo)", "unsharp")
        self.cmb_sharpen_method.addItem("Kernel Tích chập 3x3", "kernel")
        self.cmb_sharpen_method.addItem("Tối ưu Tài liệu & Chữ nhỏ", "document")
        self.cmb_sharpen_method.addItem("Tắt làm nét", "none")
        self.cmb_sharpen_method.currentIndexChanged.connect(self._mark_manual_if_edited)
        row_m.addWidget(self.cmb_sharpen_method, 1)
        vs.addLayout(row_m)

        # Sharpen Strength (0.00 .. 3.00 -> slider 0..300)
        self.lbl_sharp_str = QLabel("Mức độ làm nét: 1.00")
        self.sld_sharp_str = QSlider(Qt.Orientation.Horizontal)
        self.sld_sharp_str.setRange(0, 300)
        self.sld_sharp_str.setValue(100)
        self.sld_sharp_str.valueChanged.connect(self._on_sharp_str_changed)
        vs.addWidget(self.lbl_sharp_str)
        vs.addWidget(self.sld_sharp_str)

        # Sharpen Radius (0.5 .. 4.0 -> slider 5..40)
        self.lbl_sharp_rad = QLabel("Bán kính (Radius): 1.4 px")
        self.sld_sharp_rad = QSlider(Qt.Orientation.Horizontal)
        self.sld_sharp_rad.setRange(5, 40)
        self.sld_sharp_rad.setValue(14)
        self.sld_sharp_rad.valueChanged.connect(self._on_sharp_rad_changed)
        vs.addWidget(self.lbl_sharp_rad)
        vs.addWidget(self.sld_sharp_rad)

        # Flat-region Threshold (0 .. 25)
        self.lbl_sharp_thr = QLabel("Ngưỡng bảo vệ vùng phẳng: 4")
        self.sld_sharp_thr = QSlider(Qt.Orientation.Horizontal)
        self.sld_sharp_thr.setRange(0, 25)
        self.sld_sharp_thr.setValue(4)
        self.sld_sharp_thr.valueChanged.connect(self._on_sharp_thr_changed)
        vs.addWidget(self.lbl_sharp_thr)
        vs.addWidget(self.sld_sharp_thr)

        layout.addWidget(grp_sharp)

        # 3. Denoising & Contrast Group
        grp_denoise = QGroupBox("3. Giảm nhiễu & Tương phản")
        vd = QVBoxLayout(grp_denoise)

        row_dm = QHBoxLayout()
        row_dm.addWidget(QLabel("Giảm nhiễu:"))
        self.cmb_denoise_method = QComboBox()
        self.cmb_denoise_method.addItem("Non-Local Means (NLM)", "nlm")
        self.cmb_denoise_method.addItem("Bilateral (Giữ biên nhanh)", "bilateral")
        self.cmb_denoise_method.addItem("Median (Khử nhiễu muối tiêu)", "median")
        self.cmb_denoise_method.currentIndexChanged.connect(self._mark_manual_if_edited)
        row_dm.addWidget(self.cmb_denoise_method, 1)
        vd.addLayout(row_dm)

        self.lbl_denoise_str = QLabel("Cường độ giảm nhiễu: 0.0")
        self.sld_denoise_str = QSlider(Qt.Orientation.Horizontal)
        self.sld_denoise_str.setRange(0, 200)
        self.sld_denoise_str.setValue(0)
        self.sld_denoise_str.valueChanged.connect(self._on_denoise_changed)
        vd.addWidget(self.lbl_denoise_str)
        vd.addWidget(self.sld_denoise_str)

        self.chk_clahe = QCheckBox("Cải thiện tương phản cục bộ (CLAHE)")
        self.chk_clahe.toggled.connect(self._mark_manual_if_edited)
        vd.addWidget(self.chk_clahe)

        self.lbl_clahe_clip = QLabel("Mức độ tương phản CLAHE: 1.5")
        self.sld_clahe_clip = QSlider(Qt.Orientation.Horizontal)
        self.sld_clahe_clip.setRange(5, 40)
        self.sld_clahe_clip.setValue(15)
        self.sld_clahe_clip.valueChanged.connect(self._on_clahe_changed)
        vd.addWidget(self.lbl_clahe_clip)
        vd.addWidget(self.sld_clahe_clip)

        self.lbl_brightness = QLabel("Độ sáng (Brightness): 0")
        self.sld_brightness = QSlider(Qt.Orientation.Horizontal)
        self.sld_brightness.setRange(-50, 50)
        self.sld_brightness.setValue(0)
        self.sld_brightness.valueChanged.connect(self._on_brightness_changed)
        vd.addWidget(self.lbl_brightness)
        vd.addWidget(self.sld_brightness)

        self.lbl_gamma = QLabel("Gamma: 1.00")
        self.sld_gamma = QSlider(Qt.Orientation.Horizontal)
        self.sld_gamma.setRange(50, 200)
        self.sld_gamma.setValue(100)
        self.sld_gamma.valueChanged.connect(self._on_gamma_changed)
        vd.addWidget(self.lbl_gamma)
        vd.addWidget(self.sld_gamma)

        layout.addWidget(grp_denoise)

        # 4. Super-Resolution & AI Restoration Group
        grp_ai = QGroupBox("4. Tăng phân giải & Phục hồi AI")
        vai = QVBoxLayout(grp_ai)

        row_scale = QHBoxLayout()
        row_scale.addWidget(QLabel("Tỷ lệ phóng to:"))
        self.cmb_scale = QComboBox()
        self.cmb_scale.addItem("1x (Giữ nguyên kích thước)", 1)
        self.cmb_scale.addItem("2x (Tăng gấp đôi chiều rộng/cao)", 2)
        self.cmb_scale.addItem("4x (Siêu phân giải 4x)", 4)
        self.cmb_scale.currentIndexChanged.connect(self._mark_manual_if_edited)
        row_scale.addWidget(self.cmb_scale, 1)
        vai.addLayout(row_scale)

        self.chk_use_ai = QCheckBox("Sử dụng mô hình AI Super-Resolution")
        self.chk_use_ai.toggled.connect(self._on_ai_toggled)
        vai.addWidget(self.chk_use_ai)

        self.cmb_ai_model = QComboBox()
        for name, spec in MODEL_CATALOG.items():
            self.cmb_ai_model.addItem(spec.display_name, userData=name)
        self.cmb_ai_model.currentIndexChanged.connect(self.refresh_model_status)
        vai.addWidget(self.cmb_ai_model)

        self.lbl_ai_status = QLabel("")
        self.lbl_ai_status.setWordWrap(True)
        self.lbl_ai_status.setObjectName("SubTitle")
        vai.addWidget(self.lbl_ai_status)

        self.btn_download_model = QPushButton("⬇ Tải trọng số AI đã chọn từ GitHub chính thức")
        self.btn_download_model.clicked.connect(self._on_download_clicked)
        vai.addWidget(self.btn_download_model)

        self.chk_face_restore = QCheckBox("Bật phục hồi khuôn mặt (Face Restoration)")
        self.chk_face_restore.toggled.connect(self._mark_manual_if_edited)
        vai.addWidget(self.chk_face_restore)

        self.lbl_face_fid = QLabel("Độ trung thực khuôn mặt (Fidelity): 0.65")
        self.sld_face_fid = QSlider(Qt.Orientation.Horizontal)
        self.sld_face_fid.setRange(20, 90)
        self.sld_face_fid.setValue(65)
        self.sld_face_fid.valueChanged.connect(self._on_face_fid_changed)
        vai.addWidget(self.lbl_face_fid)
        vai.addWidget(self.sld_face_fid)

        warn_lbl = QLabel(
            "⚠ Lưu ý: AI có thể tái tạo chi tiết không hoàn toàn giống ảnh gốc. "
            "Hãy so sánh kết quả bằng thanh trượt Before/After."
        )
        warn_lbl.setWordWrap(True)
        warn_lbl.setStyleSheet("color: #FBBF24; font-size: 11px;")
        vai.addWidget(warn_lbl)

        layout.addWidget(grp_ai)

        # 5. Export Format & Quality Group
        grp_exp = QGroupBox("5. Định dạng & Chất lượng xuất file")
        vexp = QVBoxLayout(grp_exp)

        row_fmt = QHBoxLayout()
        row_fmt.addWidget(QLabel("Định dạng:"))
        self.cmb_export_fmt = QComboBox()
        self.cmb_export_fmt.addItem("PNG (Không nén mất dữ liệu, giữ Alpha)", "PNG")
        self.cmb_export_fmt.addItem("JPEG (Dung lượng nhẹ, phổ biến)", "JPEG")
        self.cmb_export_fmt.addItem("WEBP (Hiện đại, hỗ trợ Alpha)", "WEBP")
        self.cmb_export_fmt.addItem("Giữ định dạng gốc (ORIGINAL)", "ORIGINAL")
        row_fmt.addWidget(self.cmb_export_fmt, 1)
        vexp.addLayout(row_fmt)

        self.lbl_export_qual = QLabel("Chất lượng nén (JPEG/WEBP): 92%")
        self.sld_export_qual = QSlider(Qt.Orientation.Horizontal)
        self.sld_export_qual.setRange(30, 100)
        self.sld_export_qual.setValue(92)
        self.sld_export_qual.valueChanged.connect(
            lambda v: self.lbl_export_qual.setText(f"Chất lượng nén (JPEG/WEBP): {v}%")
        )
        vexp.addWidget(self.lbl_export_qual)
        vexp.addWidget(self.sld_export_qual)

        layout.addWidget(grp_exp)
        layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        # Sticky Bottom Action Bar
        action_bar = QFrame()
        action_bar.setObjectName("TopBar")
        act_layout = QVBoxLayout(action_bar)
        act_layout.setContentsMargins(12, 10, 12, 10)
        act_layout.setSpacing(6)

        row_proc = QHBoxLayout()
        self.btn_process = QPushButton("✨ Xử lý ảnh ngay")
        self.btn_process.setObjectName("PrimaryButton")
        self.btn_process.clicked.connect(lambda: self.process_requested.emit(self.get_params()))

        self.btn_cancel = QPushButton("🛑 Hủy")
        self.btn_cancel.setObjectName("DangerButton")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_requested.emit)

        row_proc.addWidget(self.btn_process, 3)
        row_proc.addWidget(self.btn_cancel, 1)
        act_layout.addLayout(row_proc)

        self.btn_export = QPushButton("💾 Lưu ảnh kết quả...")
        self.btn_export.setObjectName("SuccessButton")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(
            lambda: self.export_requested.emit(
                str(self.cmb_export_fmt.currentData()),
                int(self.sld_export_qual.value()),
            )
        )
        act_layout.addWidget(self.btn_export)

        outer.addWidget(action_bar)

    def _on_sharp_str_changed(self, val: int) -> None:
        self.lbl_sharp_str.setText(f"Mức độ làm nét: {val / 100.0:.2f}")
        self._mark_manual_if_edited()

    def _on_sharp_rad_changed(self, val: int) -> None:
        self.lbl_sharp_rad.setText(f"Bán kính (Radius): {val / 10.0:.1f} px")
        self._mark_manual_if_edited()

    def _on_sharp_thr_changed(self, val: int) -> None:
        self.lbl_sharp_thr.setText(f"Ngưỡng bảo vệ vùng phẳng: {val}")
        self._mark_manual_if_edited()

    def _on_denoise_changed(self, val: int) -> None:
        self.lbl_denoise_str.setText(f"Cường độ giảm nhiễu: {val / 10.0:.1f}")
        self._mark_manual_if_edited()

    def _on_clahe_changed(self, val: int) -> None:
        self.lbl_clahe_clip.setText(f"Mức độ tương phản CLAHE: {val / 10.0:.1f}")
        self._mark_manual_if_edited()

    def _on_brightness_changed(self, val: int) -> None:
        self.lbl_brightness.setText(f"Độ sáng (Brightness): {val:+d}")
        self._mark_manual_if_edited()

    def _on_gamma_changed(self, val: int) -> None:
        self.lbl_gamma.setText(f"Gamma: {val / 100.0:.2f}")
        self._mark_manual_if_edited()

    def _on_face_fid_changed(self, val: int) -> None:
        self.lbl_face_fid.setText(f"Độ trung thực khuôn mặt (Fidelity): {val / 100.0:.2f}")
        self._mark_manual_if_edited()

    def _on_ai_toggled(self, checked: bool) -> None:
        if checked and int(self.cmb_scale.currentData()) == 1:
            self.cmb_scale.setCurrentIndex(1)  # default to 2x when AI SR is toggled on
        self.refresh_model_status()
        self._mark_manual_if_edited()

    def _on_download_clicked(self) -> None:
        model_name = str(self.cmb_ai_model.currentData())
        self.download_model_requested.emit(model_name)

    def refresh_model_status(self) -> None:
        model_name = str(self.cmb_ai_model.currentData())
        spec = MODEL_CATALOG.get(model_name)
        if spec is None:
            return
        installed = self.model_manager.is_weight_installed(model_name)
        if installed:
            self.lbl_ai_status.setText(
                f"✅ Đã cài đặt: {spec.filename} (Giấy phép: {spec.license_name})"
            )
            self.lbl_ai_status.setStyleSheet("color: #34D399; font-size: 12px;")
            self.btn_download_model.setText("✔ Trọng số AI đã sẵn sàng (Nhấn để tải lại)")
        else:
            self.lbl_ai_status.setText(
                f"⚠ Chưa có file '{spec.filename}' (~{spec.approx_size_mb} MB). "
                f"Nhấn nút bên dưới để tải từ GitHub chính thức ({spec.license_name})."
            )
            self.lbl_ai_status.setStyleSheet("color: #FBBF24; font-size: 12px;")
            self.btn_download_model.setText(f"⬇ Tải mô hình {spec.name} (~{spec.approx_size_mb} MB)")

    def _on_preset_changed(self, _index: int) -> None:
        if self._updating_ui:
            return
        key = str(self.cmb_preset.currentData())
        preset_def = PRESETS.get(key)
        if not preset_def:
            return
        self.lbl_preset_desc.setText(preset_def.description)
        if key != "auto":
            self.apply_params_dict(preset_def.params, keep_preset_selection=True)

    def _mark_manual_if_edited(self) -> None:
        if self._updating_ui:
            return
        # If preset was "auto", switching a manual slider switches mode to manual adjustment
        if str(self.cmb_preset.currentData()) == "auto":
            self.lbl_preset_desc.setText("Đang tinh chỉnh thông số thủ công theo ý muốn người dùng.")

    def apply_params_dict(self, data: Dict[str, Any], keep_preset_selection: bool = False) -> None:
        """Populate UI sliders and dropdowns from a parameter dictionary."""
        self._updating_ui = True
        try:
            if not keep_preset_selection and "preset" in data:
                idx = self.cmb_preset.findData(data["preset"])
                if idx >= 0:
                    self.cmb_preset.setCurrentIndex(idx)

            if "sharpen_method" in data:
                idx = self.cmb_sharpen_method.findData(data["sharpen_method"])
                if idx >= 0:
                    self.cmb_sharpen_method.setCurrentIndex(idx)

            if "sharpen_strength" in data:
                v = int(round(float(data["sharpen_strength"]) * 100))
                self.sld_sharp_str.setValue(v)
                self.lbl_sharp_str.setText(f"Mức độ làm nét: {v / 100.0:.2f}")

            if "sharpen_radius" in data:
                v = int(round(float(data["sharpen_radius"]) * 10))
                self.sld_sharp_rad.setValue(v)
                self.lbl_sharp_rad.setText(f"Bán kính (Radius): {v / 10.0:.1f} px")

            if "sharpen_threshold" in data:
                v = int(data["sharpen_threshold"])
                self.sld_sharp_thr.setValue(v)
                self.lbl_sharp_thr.setText(f"Ngưỡng bảo vệ vùng phẳng: {v}")

            if "denoise_method" in data:
                idx = self.cmb_denoise_method.findData(data["denoise_method"])
                if idx >= 0:
                    self.cmb_denoise_method.setCurrentIndex(idx)

            if "denoise_strength" in data:
                v = int(round(float(data["denoise_strength"]) * 10))
                self.sld_denoise_str.setValue(v)
                self.lbl_denoise_str.setText(f"Cường độ giảm nhiễu: {v / 10.0:.1f}")

            if "contrast_clahe" in data:
                self.chk_clahe.setChecked(bool(data["contrast_clahe"]))

            if "clahe_clip_limit" in data:
                v = int(round(float(data["clahe_clip_limit"]) * 10))
                self.sld_clahe_clip.setValue(v)
                self.lbl_clahe_clip.setText(f"Mức độ tương phản CLAHE: {v / 10.0:.1f}")

            if "brightness" in data:
                v = int(round(float(data["brightness"])))
                self.sld_brightness.setValue(v)
                self.lbl_brightness.setText(f"Độ sáng (Brightness): {v:+d}")

            if "gamma" in data:
                v = int(round(float(data["gamma"]) * 100))
                self.sld_gamma.setValue(v)
                self.lbl_gamma.setText(f"Gamma: {v / 100.0:.2f}")

            if "upscale_factor" in data:
                idx = self.cmb_scale.findData(int(data["upscale_factor"]))
                if idx >= 0:
                    self.cmb_scale.setCurrentIndex(idx)

            if "use_ai_sr" in data:
                self.chk_use_ai.setChecked(bool(data["use_ai_sr"]))

            if "ai_model_name" in data:
                idx = self.cmb_ai_model.findData(data["ai_model_name"])
                if idx >= 0:
                    self.cmb_ai_model.setCurrentIndex(idx)

            if "use_face_restore" in data:
                self.chk_face_restore.setChecked(bool(data["use_face_restore"]))

            if "face_restore_fidelity" in data:
                v = int(round(float(data["face_restore_fidelity"]) * 100))
                self.sld_face_fid.setValue(v)
                self.lbl_face_fid.setText(f"Độ trung thực khuôn mặt (Fidelity): {v / 100.0:.2f}")

            self.refresh_model_status()
        finally:
            self._updating_ui = False

    def get_params(self, tile_size: int = 256, tile_pad: int = 16) -> EnhancementParams:
        """Read current UI control state and return an EnhancementParams instance."""
        preset_key = str(self.cmb_preset.currentData())
        mode = "auto" if preset_key == "auto" else "manual"
        return EnhancementParams(
            mode=mode,
            preset=preset_key,
            sharpen_method=str(self.cmb_sharpen_method.currentData()),
            sharpen_strength=round(self.sld_sharp_str.value() / 100.0, 2),
            sharpen_radius=round(self.sld_sharp_rad.value() / 10.0, 2),
            sharpen_threshold=int(self.sld_sharp_thr.value()),
            denoise_method=str(self.cmb_denoise_method.currentData()),
            denoise_strength=round(self.sld_denoise_str.value() / 10.0, 2),
            contrast_clahe=self.chk_clahe.isChecked(),
            clahe_clip_limit=round(self.sld_clahe_clip.value() / 10.0, 2),
            brightness=float(self.sld_brightness.value()),
            gamma=round(self.sld_gamma.value() / 100.0, 2),
            upscale_factor=int(self.cmb_scale.currentData()),
            use_ai_sr=self.chk_use_ai.isChecked(),
            ai_model_name=str(self.cmb_ai_model.currentData()),
            tile_size=tile_size,
            tile_pad=tile_pad,
            use_face_restore=self.chk_face_restore.isChecked(),
            face_restore_fidelity=round(self.sld_face_fid.value() / 100.0, 2),
        )

    def set_busy_state(self, busy: bool) -> None:
        self.btn_process.setEnabled(not busy)
        self.btn_auto.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)


class AppSettingsPage(QWidget):
    """
    Full settings page allowing the user to configure default folders, compute device,
    tile size, default export format/quality, metadata retention, and reset to defaults.
    """

    settings_saved = Signal(object)  # AppSettings
    clear_model_cache_requested = Signal()

    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._build_ui()
        self.load_from_settings(settings)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        lbl_head = QLabel("CÀI ĐẶT HỆ THỐNG & CẤU HÌNH MẶC ĐỊNH")
        lbl_head.setObjectName("SectionTitle")
        root.addWidget(lbl_head)

        card = QFrame()
        card.setObjectName("CardPanel")
        form = QFormLayout(card)
        form.setContentsMargins(20, 20, 20, 20)
        form.setSpacing(12)

        # Default Output Dir
        out_row = QHBoxLayout()
        self.edt_output_dir = QLineEdit()
        btn_browse_out = QPushButton("📁 Chọn thư mục...")
        btn_browse_out.clicked.connect(self._browse_output_dir)
        out_row.addWidget(self.edt_output_dir, 1)
        out_row.addWidget(btn_browse_out)
        form.addRow("Thư mục đầu ra mặc định:", out_row)

        # Models Dir
        mod_row = QHBoxLayout()
        self.edt_models_dir = QLineEdit()
        btn_browse_mod = QPushButton("📁 Chọn thư mục...")
        btn_browse_mod.clicked.connect(self._browse_models_dir)
        mod_row.addWidget(self.edt_models_dir, 1)
        mod_row.addWidget(btn_browse_mod)
        form.addRow("Thư mục chứa trọng số AI:", mod_row)

        # Default Mode
        self.cmb_default_mode = QComboBox()
        for key, preset_def in PRESETS.items():
            self.cmb_default_mode.addItem(preset_def.title, userData=key)
        form.addRow("Chế độ xử lý mặc định:", self.cmb_default_mode)

        # Compute Device
        self.cmb_device = QComboBox()
        self.cmb_device.addItem("Tự động ưu tiên GPU CUDA (Auto)", "auto")
        self.cmb_device.addItem("Chỉ sử dụng CPU (Ổn định mọi máy)", "cpu")
        self.cmb_device.addItem("Bắt buộc thử GPU CUDA (Chuyển về CPU nếu thiếu)", "cuda")
        form.addRow("Thiết bị tính toán (CPU/GPU):", self.cmb_device)

        # Tile Size for Tiled Inference
        self.spn_tile_size = QSpinBox()
        self.spn_tile_size.setRange(128, 1024)
        self.spn_tile_size.setSingleStep(64)
        self.spn_tile_size.setSuffix(" px")
        form.addRow("Kích thước Tile xử lý AI (Tiled Inference):", self.spn_tile_size)

        # Default Export Format
        self.cmb_default_fmt = QComboBox()
        self.cmb_default_fmt.addItem("PNG", "PNG")
        self.cmb_default_fmt.addItem("JPEG", "JPEG")
        self.cmb_default_fmt.addItem("WEBP", "WEBP")
        self.cmb_default_fmt.addItem("ORIGINAL", "ORIGINAL")
        form.addRow("Định dạng xuất ảnh mặc định:", self.cmb_default_fmt)

        # Default JPEG & WEBP Quality
        self.spn_jpeg_q = QSpinBox()
        self.spn_jpeg_q.setRange(30, 100)
        self.spn_jpeg_q.setSuffix(" %")
        form.addRow("Chất lượng JPEG mặc định:", self.spn_jpeg_q)

        self.spn_webp_q = QSpinBox()
        self.spn_webp_q.setRange(30, 100)
        self.spn_webp_q.setSuffix(" %")
        form.addRow("Chất lượng WEBP mặc định:", self.spn_webp_q)

        # Metadata & Notifications
        self.chk_preserve_meta = QCheckBox("Bảo toàn thông tin EXIF / ICC Profile khi định dạng hỗ trợ")
        form.addRow("Metadata ảnh:", self.chk_preserve_meta)

        self.chk_notifications = QCheckBox("Hiển thị hộp thoại thông báo khi hoàn tất hoặc có cảnh báo")
        form.addRow("Thông báo giao diện:", self.chk_notifications)

        root.addWidget(card)

        # Privacy & Security Disclosure Card
        priv_card = QFrame()
        priv_card.setObjectName("BannerInfo")
        priv_layout = QVBoxLayout(priv_card)
        lbl_priv = QLabel(
            "🔒 Cam kết An toàn & Bảo mật Dữ liệu Cục bộ:\n"
            "• Mọi thao tác làm nét, khử nhiễu, phân tích và phục hồi AI đều chạy 100% cục bộ trên máy tính của bạn.\n"
            "• Ứng dụng KHÔNG tải ảnh, đường dẫn cá nhân hay metadata lên bất kỳ máy chủ bên ngoài nào.\n"
            "• Kết nối mạng chỉ được sử dụng khi bạn chủ động nhấn nút 'Tải mô hình AI' từ GitHub Releases chính thức."
        )
        lbl_priv.setWordWrap(True)
        priv_layout.addWidget(lbl_priv)
        root.addWidget(priv_card)

        # Action buttons
        btn_row = QHBoxLayout()
        self.btn_save = QPushButton("💾 Lưu cài đặt")
        self.btn_save.setObjectName("PrimaryButton")
        self.btn_save.clicked.connect(self._save_clicked)

        self.btn_free_cache = QPushButton("🧹 Giải phóng bộ nhớ đệm AI (RAM/VRAM)")
        self.btn_free_cache.clicked.connect(self.clear_model_cache_requested.emit)

        self.btn_reset = QPushButton("🔄 Khôi phục cấu hình mặc định")
        self.btn_reset.setObjectName("DangerButton")
        self.btn_reset.clicked.connect(self._reset_clicked)

        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_free_cache)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_reset)
        root.addLayout(btn_row)
        root.addStretch(1)

    def _browse_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục đầu ra mặc định", self.edt_output_dir.text())
        if folder:
            self.edt_output_dir.setText(folder)

    def _browse_models_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục chứa mô hình AI", self.edt_models_dir.text())
        if folder:
            self.edt_models_dir.setText(folder)

    def load_from_settings(self, s: AppSettings) -> None:
        self._settings = s
        self.edt_output_dir.setText(s.default_output_dir)
        self.edt_models_dir.setText(s.models_dir)
        idx_m = self.cmb_default_mode.findData(s.default_mode)
        if idx_m >= 0:
            self.cmb_default_mode.setCurrentIndex(idx_m)
        idx_d = self.cmb_device.findData(s.compute_device)
        if idx_d >= 0:
            self.cmb_device.setCurrentIndex(idx_d)
        self.spn_tile_size.setValue(int(s.tile_size))
        idx_f = self.cmb_default_fmt.findData(s.default_export_format)
        if idx_f >= 0:
            self.cmb_default_fmt.setCurrentIndex(idx_f)
        self.spn_jpeg_q.setValue(int(s.jpeg_quality))
        self.spn_webp_q.setValue(int(s.webp_quality))
        self.chk_preserve_meta.setChecked(bool(s.preserve_metadata))
        self.chk_notifications.setChecked(bool(s.enable_notifications))

    def _save_clicked(self) -> None:
        updated = AppSettings(
            default_output_dir=self.edt_output_dir.text().strip() or str(Path.cwd() / "outputs"),
            default_mode=str(self.cmb_default_mode.currentData()),
            compute_device=str(self.cmb_device.currentData()),
            default_export_format=str(self.cmb_default_fmt.currentData()),
            jpeg_quality=int(self.spn_jpeg_q.value()),
            webp_quality=int(self.spn_webp_q.value()),
            preserve_metadata=self.chk_preserve_meta.isChecked(),
            enable_notifications=self.chk_notifications.isChecked(),
            tile_size=int(self.spn_tile_size.value()),
            models_dir=self.edt_models_dir.text().strip() or str(Path.cwd() / "models"),
        )
        self.settings_saved.emit(updated)

    def _reset_clicked(self) -> None:
        defaults = AppSettings()
        self.load_from_settings(defaults)
        self.settings_saved.emit(defaults)
