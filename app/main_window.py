"""
Main Application Window for Image Enhancement Studio.
Integrates Sidebar navigation, Top Status Bar, Before/After Image Studio Workspace,
Batch Processing View, Processing History View, AI Model Hub, and Settings Page.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ai.device_manager import DeviceManager
from ai.model_manager import MODEL_CATALOG, ModelManager
from app.styles import STUDIO_STYLESHEET
from app.widgets.before_after_view import BeforeAfterWorkspace
from app.widgets.progress_panel import StatusAndProgressPanel
from app.widgets.settings_panel import AppSettingsPage, EnhancementControlPanel
from config.settings import SettingsManager
from core.batch_processor import BatchItem, BatchProcessor, BatchSummary, collect_supported_images
from core.enhancement_pipeline import PRESETS, EnhancementParams, EnhancementPipeline, PipelineResult
from core.image_analyzer import analyze_image, recommend_auto_enhance
from core.image_io import ImageContainer, ImageIOError, format_file_size, load_image, resolve_safe_output_path
from services.export_service import ExportService
from services.history_service import HistoryRecord, HistoryService
from services.logging_service import get_ui_log_handler
from workers.processing_worker import (
    BatchProcessingWorker,
    ModelDownloadWorker,
    SingleProcessingWorker,
)

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main desktop window for Image Enhancement Studio."""

    def __init__(self, settings_manager: SettingsManager | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Image Enhancement Studio — Làm Nét & Phục Hồi Ảnh Chuyên Nghiệp")
        self.resize(1420, 880)
        self.setMinimumSize(1120, 720)
        self.setStyleSheet(STUDIO_STYLESHEET)

        # Core managers and services
        self.settings_mgr = settings_manager or SettingsManager()
        self.settings = self.settings_mgr.settings

        self.device_mgr = DeviceManager(preferred=self.settings.compute_device)
        self.model_mgr = ModelManager(models_dir=self.settings.models_dir)
        self.pipeline = EnhancementPipeline(model_manager=self.model_mgr, device_manager=self.device_mgr)
        self.history_service = HistoryService()
        self.export_service = ExportService(
            default_output_dir=self.settings.default_output_dir,
            default_format=self.settings.default_export_format,
            jpeg_quality=self.settings.jpeg_quality,
            webp_quality=self.settings.webp_quality,
            preserve_metadata=self.settings.preserve_metadata,
        )
        self.batch_processor = BatchProcessor(
            pipeline=self.pipeline,
            export_service=self.export_service,
            history_service=self.history_service,
        )

        # Runtime state
        self._current_source: ImageContainer | None = None
        self._current_result: PipelineResult | None = None
        self._single_worker: SingleProcessingWorker | None = None
        self._batch_worker: BatchProcessingWorker | None = None
        self._download_worker: ModelDownloadWorker | None = None
        self._batch_items: List[BatchItem] = []

        self._build_ui()
        self._connect_logging()
        self._refresh_top_bar_badges()
        self._refresh_history_table()

        if self.device_mgr.status.fallback_warning:
            self.progress_panel.show_warning(self.device_mgr.status.fallback_warning)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Top Header Bar
        top_bar = self._create_top_bar()
        root_layout.addWidget(top_bar)

        # 2. Body Area (Left Sidebar + Center Stacked Pages)
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self.sidebar = self._create_sidebar()
        body_layout.addWidget(self.sidebar)

        self.main_stack = QStackedWidget()
        # Page 0: Studio Workspace (shared by Tổng quan, Làm nét ảnh, Phục hồi AI)
        self.studio_page = self._create_studio_page()
        self.main_stack.addWidget(self.studio_page)

        # Page 1: Batch Processing Workspace
        self.batch_page = self._create_batch_page()
        self.main_stack.addWidget(self.batch_page)

        # Page 2: Processing History Page
        self.history_page = self._create_history_page()
        self.main_stack.addWidget(self.history_page)

        # Page 3: Settings Page
        self.settings_page = AppSettingsPage(self.settings)
        self.settings_page.settings_saved.connect(self._on_settings_saved)
        self.settings_page.clear_model_cache_requested.connect(self._on_clear_model_cache)
        self.main_stack.addWidget(self.settings_page)

        body_layout.addWidget(self.main_stack, 1)
        root_layout.addWidget(body, 1)

        # 3. Bottom Status & Progress Panel
        self.progress_panel = StatusAndProgressPanel()
        root_layout.addWidget(self.progress_panel)

    def _create_top_bar(self) -> QFrame:
        top = QFrame()
        top.setObjectName("TopBar")
        top.setFixedHeight(56)
        layout = QHBoxLayout(top)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)

        lbl_title = QLabel("✨ IMAGE ENHANCEMENT STUDIO")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: 800; color: #FFFFFF; letter-spacing: 0.5px;")
        layout.addWidget(lbl_title)

        self.lbl_top_notice = QLabel("Chế độ: Xử lý cục bộ an toàn (Offline-First)")
        self.lbl_top_notice.setObjectName("SubTitle")
        layout.addWidget(self.lbl_top_notice)

        layout.addStretch(1)

        self.btn_demo_img = QPushButton("🧪 Tạo ảnh mẫu thử nghiệm")
        self.btn_demo_img.setToolTip("Tạo một ảnh mẫu có chi tiết văn bản, chân dung hình học và nhiễu để thử nghiệm ngay")
        self.btn_demo_img.clicked.connect(self._load_synthetic_demo_image)
        layout.addWidget(self.btn_demo_img)

        self.lbl_model_badge = QLabel("AI: Đang kiểm tra...")
        self.lbl_model_badge.setObjectName("BadgeModelMissing")
        layout.addWidget(self.lbl_model_badge)

        self.lbl_device_badge = QLabel("Thiết bị: CPU")
        self.lbl_device_badge.setObjectName("BadgeDevice")
        layout.addWidget(self.lbl_device_badge)

        return top

    def _create_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(210)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 14, 10, 14)
        layout.setSpacing(6)

        lbl_nav = QLabel("DANH MỤC CHỨC NĂNG")
        lbl_nav.setStyleSheet("color: #6B7280; font-size: 11px; font-weight: 700; padding-left: 8px;")
        layout.addWidget(lbl_nav)

        self.nav_buttons: List[QPushButton] = []
        nav_items = [
            ("📊  Tổng quan", 0, "overview"),
            ("🔪  Làm nét ảnh", 1, "sharpen"),
            ("🤖  Phục hồi AI", 2, "ai"),
            ("📦  Xử lý hàng loạt", 3, "batch"),
            ("🕒  Lịch sử xử lý", 4, "history"),
            ("⚙  Cài đặt", 5, "settings"),
        ]

        for label, idx, mode_tag in nav_items:
            btn = QPushButton(label)
            btn.setObjectName("NavButton")
            btn.setProperty("active", "true" if idx == 0 else "false")
            btn.clicked.connect(lambda _checked=False, i=idx, tag=mode_tag: self._switch_navigation(i, tag))
            layout.addWidget(btn)
            self.nav_buttons.append(btn)

        layout.addStretch(1)

        # Quick diagnostic summary card inside sidebar bottom
        diag_frame = QFrame()
        diag_frame.setObjectName("CardPanel")
        df_layout = QVBoxLayout(diag_frame)
        df_layout.setContentsMargins(10, 10, 10, 10)
        df_layout.setSpacing(4)

        lbl_diag_head = QLabel("CHẨN ĐOÁN ẢNH HIỆN TẠI")
        lbl_diag_head.setStyleSheet("color: #A5B4FC; font-size: 11px; font-weight: 700;")
        df_layout.addWidget(lbl_diag_head)

        self.lbl_diag_summary = QLabel("Chưa nạp ảnh.\nHãy chọn ảnh để xem độ nét, độ nhiễu và đề xuất tự động.")
        self.lbl_diag_summary.setObjectName("SubTitle")
        self.lbl_diag_summary.setWordWrap(True)
        df_layout.addWidget(self.lbl_diag_summary)

        layout.addWidget(diag_frame)
        return sidebar

    def _create_studio_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Center area: Contextual banner + BeforeAfterWorkspace
        center_col = QWidget()
        center_layout = QVBoxLayout(center_col)
        center_layout.setContentsMargins(6, 6, 6, 0)
        center_layout.setSpacing(4)

        self.studio_banner = QFrame()
        self.studio_banner.setObjectName("BannerInfo")
        sb_layout = QHBoxLayout(self.studio_banner)
        sb_layout.setContentsMargins(12, 6, 12, 6)
        self.lbl_studio_banner = QLabel(
            "💡 Tổng quan Studio: Kéo thả ảnh vào khung bên dưới hoặc nhấn 'Chọn ảnh'. "
            "Sử dụng 'Phân tích & Đề xuất Tự động (Auto)' để nhận gợi ý thông số tối ưu."
        )
        self.lbl_studio_banner.setWordWrap(True)
        sb_layout.addWidget(self.lbl_studio_banner, 1)
        center_layout.addWidget(self.studio_banner)

        self.workspace_view = BeforeAfterWorkspace()
        self.workspace_view.file_selected.connect(self.load_source_image_from_path)
        self.workspace_view.open_file_dialog_requested.connect(self._open_file_dialog)
        center_layout.addWidget(self.workspace_view, 1)

        layout.addWidget(center_col, 1)

        # Right-hand Control Panel
        self.control_panel = EnhancementControlPanel(model_manager=self.model_mgr)
        self.control_panel.process_requested.connect(self._start_single_processing)
        self.control_panel.auto_enhance_requested.connect(self._run_auto_enhance_analysis)
        self.control_panel.cancel_requested.connect(self._cancel_single_processing)
        self.control_panel.export_requested.connect(self._export_current_result)
        self.control_panel.download_model_requested.connect(self._start_model_download)
        layout.addWidget(self.control_panel)

        return page

    def _create_batch_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        lbl_head = QLabel("XỬ LÝ HÀNG LOẠT (BATCH PROCESSING)")
        lbl_head.setObjectName("SectionTitle")
        layout.addWidget(lbl_head)

        # Top Action Bar for Batch Queue
        top_row = QHBoxLayout()
        btn_add_files = QPushButton("➕ Thêm các file ảnh...")
        btn_add_files.setObjectName("PrimaryButton")
        btn_add_files.clicked.connect(self._batch_add_files)

        btn_add_folder = QPushButton("📁 Thêm cả thư mục...")
        btn_add_folder.clicked.connect(self._batch_add_folder)

        btn_remove_sel = QPushButton("🗑 Bỏ file đã chọn")
        btn_remove_sel.clicked.connect(self._batch_remove_selected)

        btn_clear_all = QPushButton("🧹 Xóa danh sách")
        btn_clear_all.clicked.connect(self._batch_clear_all)

        for b in (btn_add_files, btn_add_folder, btn_remove_sel, btn_clear_all):
            top_row.addWidget(b)
        top_row.addStretch(1)

        self.lbl_batch_summary = QLabel("Tổng số: 0 ảnh")
        self.lbl_batch_summary.setObjectName("SubTitle")
        top_row.addWidget(self.lbl_batch_summary)
        layout.addLayout(top_row)

        # Batch Configuration Card
        cfg_card = QFrame()
        cfg_card.setObjectName("CardPanel")
        cfg_layout = QHBoxLayout(cfg_card)
        cfg_layout.setContentsMargins(12, 10, 12, 10)
        cfg_layout.setSpacing(12)

        cfg_layout.addWidget(QLabel("Phương pháp xử lý chung:"))
        self.cmb_batch_preset = QComboBox()
        for key, pdef in PRESETS.items():
            self.cmb_batch_preset.addItem(pdef.title, userData=key)
        self.cmb_batch_preset.setCurrentIndex(2)  # standard_sharpen default
        cfg_layout.addWidget(self.cmb_batch_preset)

        cfg_layout.addWidget(QLabel("Định dạng xuất:"))
        self.cmb_batch_fmt = QComboBox()
        self.cmb_batch_fmt.addItem("PNG", "PNG")
        self.cmb_batch_fmt.addItem("JPEG", "JPEG")
        self.cmb_batch_fmt.addItem("WEBP", "WEBP")
        self.cmb_batch_fmt.addItem("ORIGINAL", "ORIGINAL")
        cfg_layout.addWidget(self.cmb_batch_fmt)

        cfg_layout.addWidget(QLabel("Chất lượng:"))
        self.spn_batch_qual = QSpinBox()
        self.spn_batch_qual.setRange(30, 100)
        self.spn_batch_qual.setValue(int(self.settings.jpeg_quality))
        self.spn_batch_qual.setSuffix("%")
        cfg_layout.addWidget(self.spn_batch_qual)

        cfg_layout.addWidget(QLabel("Thư mục lưu:"))
        self.edt_batch_out = QLineEdit(self.settings.default_output_dir)
        cfg_layout.addWidget(self.edt_batch_out, 1)

        btn_batch_dir = QPushButton("Chọn...")
        btn_batch_dir.clicked.connect(self._batch_choose_output_dir)
        cfg_layout.addWidget(btn_batch_dir)

        layout.addWidget(cfg_card)

        # Batch Items Table
        self.tbl_batch = QTableWidget(0, 6)
        self.tbl_batch.setHorizontalHeaderLabels(
            ["Tên file", "Dung lượng", "Kích thước (Trước → Sau)", "Trạng thái", "Thời gian", "Đường dẫn kết quả / Nguyên nhân lỗi"]
        )
        self.tbl_batch.setAlternatingRowColors(True)
        self.tbl_batch.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_batch.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.tbl_batch.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.tbl_batch, 1)

        # Bottom Batch Execution Controls
        bot_row = QHBoxLayout()
        self.btn_batch_start = QPushButton("▶ Bắt đầu xử lý hàng loạt")
        self.btn_batch_start.setObjectName("PrimaryButton")
        self.btn_batch_start.clicked.connect(lambda: self._start_batch_processing(only_failed=False))

        self.btn_batch_retry = QPushButton("🔄 Thử lại các file lỗi")
        self.btn_batch_retry.clicked.connect(lambda: self._start_batch_processing(only_failed=True))

        self.btn_batch_stop = QPushButton("🛑 Dừng tác vụ lô")
        self.btn_batch_stop.setObjectName("DangerButton")
        self.btn_batch_stop.setEnabled(False)
        self.btn_batch_stop.clicked.connect(self._stop_batch_processing)

        btn_open_out = QPushButton("📂 Mở thư mục kết quả")
        btn_open_out.clicked.connect(self._open_batch_output_folder)

        bot_row.addWidget(self.btn_batch_start)
        bot_row.addWidget(self.btn_batch_retry)
        bot_row.addWidget(self.btn_batch_stop)
        bot_row.addStretch(1)
        bot_row.addWidget(btn_open_out)
        layout.addLayout(bot_row)

        return page

    def _create_history_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        top_row = QHBoxLayout()
        lbl_head = QLabel("LỊCH SỬ XỬ LÝ ẢNH CỤC BỘ")
        lbl_head.setObjectName("SectionTitle")
        top_row.addWidget(lbl_head)
        top_row.addStretch(1)

        btn_refresh = QPushButton("🔄 Làm mới")
        btn_refresh.clicked.connect(self._refresh_history_table)

        btn_open_sel = QPushButton("📂 Mở file kết quả đã chọn")
        btn_open_sel.clicked.connect(self._open_selected_history_output)

        btn_del_sel = QPushButton("🗑 Xóa dòng đã chọn")
        btn_del_sel.clicked.connect(self._delete_selected_history_record)

        btn_clear_hist = QPushButton("🧹 Xóa toàn bộ lịch sử")
        btn_clear_hist.setObjectName("DangerButton")
        btn_clear_hist.clicked.connect(self._clear_all_history)

        for b in (btn_refresh, btn_open_sel, btn_del_sel, btn_clear_hist):
            top_row.addWidget(b)
        layout.addLayout(top_row)

        self.tbl_history = QTableWidget(0, 7)
        self.tbl_history.setHorizontalHeaderLabels(
            ["ID", "Thời gian", "File nguồn", "Phương pháp", "Kích thước", "Trạng thái", "Đường dẫn kết quả / Ghi chú"]
        )
        self.tbl_history.setAlternatingRowColors(True)
        self.tbl_history.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_history.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        h_header = self.tbl_history.horizontalHeader()
        h_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.tbl_history, 1)

        return page

    # ------------------------------------------------------------------
    # Navigation & Status Helpers
    # ------------------------------------------------------------------

    def _switch_navigation(self, nav_index: int, mode_tag: str) -> None:
        for idx, btn in enumerate(self.nav_buttons):
            btn.setProperty("active", "true" if idx == nav_index else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        if mode_tag in ("overview", "sharpen", "ai"):
            self.main_stack.setCurrentIndex(0)
            if mode_tag == "overview":
                self.lbl_studio_banner.setText(
                    "📊 Tổng quan Studio: Kéo thả ảnh để xem chẩn đoán tự động (Độ nét, Độ nhiễu, Tương phản) "
                    "và so sánh trực quan Trước / Sau."
                )
            elif mode_tag == "sharpen":
                self.lbl_studio_banner.setText(
                    "🔪 Chế độ Làm Nét & Giảm Nhiễu (OpenCV): Điều chỉnh Unsharp Mask chống Halo, Kernel tích chập, "
                    "khử nhiễu NLM/Bilateral và tương phản cục bộ CLAHE."
                )
                if str(self.control_panel.cmb_preset.currentData()) == "ai_restoration":
                    idx = self.control_panel.cmb_preset.findData("standard_sharpen")
                    if idx >= 0:
                        self.control_panel.cmb_preset.setCurrentIndex(idx)
            elif mode_tag == "ai":
                self.lbl_studio_banner.setText(
                    "🤖 Chế độ Phục hồi AI (Real-ESRGAN & Face Restoration): Tăng độ phân giải 2x/4x bằng mạng nơ-ron sâu "
                    "với cơ chế Tiled Inference tiết kiệm RAM/VRAM. Lưu ý: AI có thể tạo chi tiết tổng hợp."
                )
                idx = self.control_panel.cmb_preset.findData("ai_restoration")
                if idx >= 0:
                    self.control_panel.cmb_preset.setCurrentIndex(idx)
        elif mode_tag == "batch":
            self.main_stack.setCurrentIndex(1)
        elif mode_tag == "history":
            self._refresh_history_table()
            self.main_stack.setCurrentIndex(2)
        elif mode_tag == "settings":
            self.settings_page.load_from_settings(self.settings)
            self.main_stack.setCurrentIndex(3)

    def _connect_logging(self) -> None:
        handler = get_ui_log_handler()
        for line in handler.get_recent_logs():
            self.progress_panel.append_log_line(line)
        handler.add_callback(self.progress_panel.append_log_line)

    def _refresh_top_bar_badges(self) -> None:
        dev_stat = self.device_mgr.refresh_memory_stats()
        self.lbl_device_badge.setText(f"🖥 {dev_stat.badge_text}")

        installed = self.model_mgr.get_installed_sr_models()
        if installed:
            self.lbl_model_badge.setText(f"🤖 AI: Sẵn sàng ({len(installed)} mô hình)")
            self.lbl_model_badge.setObjectName("BadgeModelReady")
        else:
            self.lbl_model_badge.setText("🤖 AI: Chưa có trọng số (Có thể tải 1 chạm)")
            self.lbl_model_badge.setObjectName("BadgeModelMissing")
        self.lbl_model_badge.style().unpolish(self.lbl_model_badge)
        self.lbl_model_badge.style().polish(self.lbl_model_badge)

        self.control_panel.refresh_model_status()

    # ------------------------------------------------------------------
    # Single Image Loading, Diagnostics & Processing
    # ------------------------------------------------------------------

    def _open_file_dialog(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh cần làm nét hoặc phục hồi",
            "",
            "Image Files (*.jpg *.jpeg *.png *.webp);;All Files (*)",
        )
        if file_path:
            self.load_source_image_from_path(file_path)

    def load_source_image_from_path(self, file_path: str | Path) -> None:
        try:
            container = load_image(file_path)
            self._set_active_source_container(container)
        except ImageIOError as exc:
            self.progress_panel.fail_task(str(exc), "Hãy chọn file ảnh hợp lệ (JPG, PNG, WEBP).")
            QMessageBox.warning(self, "Không thể mở ảnh", str(exc))

    def _load_synthetic_demo_image(self) -> None:
        """Create a realistic multi-region test image on disk and load it for instant demonstration."""
        demo_dir = Path(self.settings.default_output_dir) / "samples"
        demo_dir.mkdir(parents=True, exist_ok=True)
        demo_path = demo_dir / "anh_mau_thu_nghiem.png"

        canvas = np.full((540, 760, 3), 238, dtype=np.uint8)
        # Gradient background sky/studio card
        for y in range(260):
            canvas[y, :, 0] = int(60 + y * 0.45)
            canvas[y, :, 1] = int(45 + y * 0.35)
            canvas[y, :, 2] = int(35 + y * 0.25)

        # Crisp geometric product/poster shapes
        cv2.rectangle(canvas, (40, 40), (340, 230), (245, 245, 250), -1)
        cv2.rectangle(canvas, (40, 40), (340, 230), (80, 90, 220), 3)
        cv2.putText(canvas, "STUDIO TEST CARD", (62, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (25, 30, 45), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Fine Text 12pt - Kiem tra do sac net", (62, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 60), 1, cv2.LINE_AA)
        cv2.putText(canvas, "0123456789 ABCDEFGHIJKLMN", (62, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 70, 90), 1, cv2.LINE_AA)

        # Concentric resolution rings
        for r in range(15, 85, 12):
            cv2.circle(canvas, (540, 140), r, (240, 210, 120), 2, cv2.LINE_AA)

        # Document / receipt section at bottom
        cv2.rectangle(canvas, (40, 285), (720, 500), (250, 250, 248), -1)
        cv2.putText(canvas, "PHUC HOI ANH & LAM NET CHUYEN NGHIEP", (65, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, "- Khu nhieu hat (Denoising) & Tang tuong phan CLAHE", (65, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (45, 45, 45), 1, cv2.LINE_AA)
        cv2.putText(canvas, "- Unsharp Mask chong vien sang Halo & Kernel 3x3", (65, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (45, 45, 45), 1, cv2.LINE_AA)
        cv2.putText(canvas, "- AI Super-Resolution (Real-ESRGAN 2x / 4x)", (65, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (45, 45, 45), 1, cv2.LINE_AA)

        # Simulate realistic defocus blur + mild sensor noise
        blurred = cv2.GaussianBlur(canvas, (5, 5), 1.35)
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, 4.5, size=blurred.shape).astype(np.float32)
        degraded = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        from PIL import Image
        Image.fromarray(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)).save(demo_path, format="PNG")
        self.load_source_image_from_path(demo_path)

    def _set_active_source_container(self, container: ImageContainer) -> None:
        self._current_source = container
        self._current_result = None
        self.workspace_view.set_source_image(container)
        self.control_panel.btn_export.setEnabled(False)

        diag = analyze_image(container)
        ai_ok = self.model_mgr.any_sr_model_installed()
        rec = recommend_auto_enhance(container, ai_available=ai_ok)

        self.lbl_diag_summary.setText(
            f"• Độ phân giải: {diag.width}x{diag.height}\n"
            f"• Độ nét (Laplacian): {diag.laplacian_var}\n"
            f"• Độ sắc cạnh (Sobel): {diag.tenengrad_score}\n"
            f"• Độ nhiễu ước lượng (σ): {diag.noise_sigma}\n"
            f"• Tương phản (RMS): {diag.rms_contrast}\n"
            f"• Đề xuất: {rec.method_title}"
        )

        if container.warnings:
            self.progress_panel.show_warning(" | ".join(container.warnings))
        else:
            self.progress_panel.finish_task(
                f"Đã tải ảnh '{container.source_path.name if container.source_path else 'Ảnh'}' ({container.resolution_str}).",
                duration_sec=0.0,
            )

    def _run_auto_enhance_analysis(self) -> None:
        if self._current_source is None:
            QMessageBox.information(self, "Chưa có ảnh", "Vui lòng chọn hoặc kéo thả một bức ảnh trước khi phân tích tự động.")
            return

        model_name = str(self.control_panel.cmb_ai_model.currentData())
        ai_ok = self.model_mgr.is_weight_installed(model_name)
        rec = recommend_auto_enhance(self._current_source, ai_available=ai_ok)

        self.control_panel.apply_params_dict(rec.recommended_params, keep_preset_selection=False)
        self.lbl_studio_banner.setText(f"⚡ Auto Enhance đã chọn '{rec.method_title}': {rec.explanation}")

        # Immediately trigger processing with the recommended parameters
        params = self.control_panel.get_params(
            tile_size=self.settings.tile_size,
            tile_pad=self.settings.tile_pad,
        )
        params.mode = "manual"
        self._start_single_processing(params)

    def _start_single_processing(self, params: EnhancementParams) -> None:
        if self._current_source is None:
            QMessageBox.information(self, "Chưa chọn ảnh", "Hãy chọn một bức ảnh hoặc nhấn 'Tạo ảnh mẫu thử nghiệm' để bắt đầu.")
            return

        if self._single_worker is not None and self._single_worker.isRunning():
            return

        if params.use_ai_sr and not self.model_mgr.is_weight_installed(params.ai_model_name):
            spec = MODEL_CATALOG.get(params.ai_model_name)
            size_txt = f"~{spec.approx_size_mb} MB" if spec else ""
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Chưa có trọng số mô hình AI")
            msg_box.setIcon(QMessageBox.Icon.Question)
            msg_box.setText(
                f"Bạn đang bật chế độ AI Super-Resolution nhưng mô hình '{params.ai_model_name}' ({size_txt}) "
                f"chưa được tải về thư mục '{self.model_mgr.models_dir}'.\n\n"
                "Bạn muốn thực hiện theo phương án nào?"
            )
            btn_dl = msg_box.addButton(f"⬇ Tải mô hình AI ({size_txt})", QMessageBox.ButtonRole.AcceptRole)
            btn_fallback = msg_box.addButton(
                "⚡ Phóng to & Làm nét ngay (Không cần tải AI)", QMessageBox.ButtonRole.ActionRole
            )
            msg_box.addButton("Hủy", QMessageBox.ButtonRole.RejectRole)
            msg_box.exec()

            clicked = msg_box.clickedButton()
            if clicked == btn_dl:
                self._start_model_download(params.ai_model_name)
                return
            elif clicked == btn_fallback:
                params.use_ai_sr = False
                self.control_panel.chk_use_ai.setChecked(False)
            else:
                return

        params.tile_size = int(self.settings.tile_size)
        params.tile_pad = int(self.settings.tile_pad)

        self.control_panel.set_busy_state(True)
        self.progress_panel.start_task("Đang khởi chạy chuỗi xử lý ảnh...")

        self._single_worker = SingleProcessingWorker(
            pipeline=self.pipeline,
            source_container=self._current_source,
            params=params,
            parent=self,
        )
        self._single_worker.progress_changed.connect(self.progress_panel.update_progress)
        self._single_worker.finished_result.connect(self._on_single_processing_finished)
        self._single_worker.cancelled.connect(self._on_single_processing_cancelled)
        self._single_worker.failed_error.connect(self._on_single_processing_failed)
        self._single_worker.start()

    def _cancel_single_processing(self) -> None:
        if self._single_worker is not None and self._single_worker.isRunning():
            self._single_worker.cancel()

    def _on_single_processing_finished(self, result: PipelineResult) -> None:
        self.control_panel.set_busy_state(False)
        self._current_result = result
        self.control_panel.btn_export.setEnabled(True)

        self.workspace_view.set_result_image(
            result_container=result.container,
            method_summary=result.method_summary,
            sharpness_before=result.sharpness_before,
            sharpness_after=result.sharpness_after,
        )

        self.lbl_studio_banner.setText(
            f"✅ Đã áp dụng: {result.method_summary} | Các bước: {' → '.join(result.steps_executed)}"
        )
        self.progress_panel.finish_task(
            f"Hoàn thành ({result.method_summary}) — Kích thước: {result.container.resolution_str}",
            duration_sec=result.duration_sec,
        )

        if result.warnings:
            self.progress_panel.show_warning(" ".join(result.warnings))

        # Record in local history
        if self._current_source is not None:
            self.history_service.add_record(
                source_path=self._current_source.source_path or "memory_image",
                method=result.method_summary,
                parameters=result.applied_params,
                output_path="(Đã xem trước - Chưa lưu file)",
                status="SUCCESS",
                duration_sec=result.duration_sec,
                input_resolution=self._current_source.resolution_str,
                output_resolution=result.container.resolution_str,
            )

        self._refresh_top_bar_badges()

    def _on_single_processing_cancelled(self, message: str) -> None:
        self.control_panel.set_busy_state(False)
        self.progress_panel.cancel_task(message)

    def _on_single_processing_failed(self, error_msg: str, hint: str) -> None:
        self.control_panel.set_busy_state(False)
        self.progress_panel.fail_task(error_msg, hint)
        if self._current_source is not None:
            self.history_service.add_record(
                source_path=self._current_source.source_path or "memory_image",
                method="Lỗi xử lý",
                parameters=self.control_panel.get_params().to_dict(),
                output_path="",
                status="FAILED",
                error_message=error_msg,
                input_resolution=self._current_source.resolution_str,
            )

    def _export_current_result(self, fmt: str, quality: int) -> None:
        if self._current_result is None:
            QMessageBox.information(self, "Chưa có kết quả", "Hãy nhấn 'Xử lý ảnh ngay' trước khi lưu kết quả.")
            return

        default_candidate = resolve_safe_output_path(
            output_dir=self.settings.default_output_dir,
            source_path=self._current_result.container.source_path,
            target_format=fmt,
            suffix="_enhanced",
            overwrite=False,
        )

        save_path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu ảnh kết quả",
            str(default_candidate),
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;WEBP Image (*.webp)",
        )
        if not save_path_str:
            return

        try:
            saved_path, report = self.export_service.export_to_path(
                container=self._current_result.container,
                output_path=save_path_str,
                target_format=fmt,
                jpeg_quality=quality,
                webp_quality=quality,
                preserve_metadata=self.settings.preserve_metadata,
            )
            size_str = format_file_size(int(report.get("file_size_bytes", 0)))
            self.progress_panel.finish_task(
                f"Đã lưu kết quả tại: {saved_path} ({report.get('resolution')}, {size_str})"
            )
            if report.get("warnings"):
                self.progress_panel.show_warning(" ".join(report["warnings"]))

            self.history_service.add_record(
                source_path=self._current_result.container.source_path or "memory_image",
                method=f"[Xuất file] {self._current_result.method_summary}",
                parameters=self._current_result.applied_params,
                output_path=saved_path,
                status="SUCCESS",
                duration_sec=self._current_result.duration_sec,
                input_resolution=self._current_source.resolution_str if self._current_source else "",
                output_resolution=self._current_result.container.resolution_str,
            )
            if self.settings.enable_notifications:
                QMessageBox.information(
                    self,
                    "Lưu ảnh thành công",
                    f"Đã lưu ảnh kết quả thành công!\n\n"
                    f"• Đường dẫn: {saved_path}\n"
                    f"• Độ phân giải: {report.get('resolution')}\n"
                    f"• Dung lượng: {size_str}",
                )
        except Exception as exc:
            self.progress_panel.fail_task(str(exc), "Hãy chọn tên file khác với ảnh gốc và đảm bảo thư mục có quyền ghi.")
            QMessageBox.critical(self, "Lỗi khi lưu ảnh", str(exc))

    # ------------------------------------------------------------------
    # AI Model Download Handler
    # ------------------------------------------------------------------

    def _start_model_download(self, model_name: str) -> None:
        if self._download_worker is not None and self._download_worker.isRunning():
            QMessageBox.information(self, "Đang tải", "Một tiến trình tải mô hình AI đang chạy. Vui lòng đợi hoàn tất.")
            return

        spec = MODEL_CATALOG.get(model_name)
        if spec is None:
            return

        reply = QMessageBox.question(
            self,
            "Xác nhận tải mô hình AI từ nguồn chính thức",
            f"Bạn có muốn tải trọng số mô hình AI sau về máy tính không?\n\n"
            f"• Tên mô hình: {spec.display_name}\n"
            f"• Nguồn chính thức: {spec.source_repo}\n"
            f"• URL: {spec.url}\n"
            f"• Dung lượng ước tính: ~{spec.approx_size_mb} MB\n"
            f"• Giấy phép mã nguồn mở: {spec.license_name}\n"
            f"• Thư mục lưu: {self.model_mgr.models_dir}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.control_panel.btn_download_model.setEnabled(False)
        self.progress_panel.start_task(f"Đang tải mô hình {spec.name} từ GitHub Releases...")

        self._download_worker = ModelDownloadWorker(self.model_mgr, model_name, parent=self)
        self._download_worker.download_progress.connect(self._on_model_download_progress)
        self._download_worker.download_finished.connect(self._on_model_download_finished)
        self._download_worker.download_failed.connect(self._on_model_download_failed)
        self._download_worker.start()

    def _on_model_download_progress(self, downloaded: int, total: int, model_name: str) -> None:
        if total > 0:
            pct = int((downloaded / total) * 100)
            dl_mb = downloaded / (1024 * 1024)
            tot_mb = total / (1024 * 1024)
            self.progress_panel.update_progress(
                pct,
                f"Đang tải {model_name}: {dl_mb:.1f} / {tot_mb:.1f} MB ({pct}%)",
            )
        else:
            dl_mb = downloaded / (1024 * 1024)
            self.progress_panel.update_progress(50, f"Đang tải {model_name}: {dl_mb:.1f} MB...")

    def _on_model_download_finished(self, model_name: str, saved_path: str) -> None:
        self.control_panel.btn_download_model.setEnabled(True)
        self._refresh_top_bar_badges()
        self.progress_panel.finish_task(f"Đã tải xong mô hình AI '{model_name}' tại {saved_path}")
        if self.settings.enable_notifications:
            QMessageBox.information(
                self,
                "Tải mô hình thành công",
                f"Trọng số AI '{model_name}' đã sẵn sàng tại:\n{saved_path}",
            )

    def _on_model_download_failed(self, model_name: str, error_msg: str) -> None:
        self.control_panel.btn_download_model.setEnabled(True)
        self.progress_panel.fail_task(
            f"Không thể tải mô hình {model_name}: {error_msg}",
            "Kiểm tra kết nối Internet hoặc tải file .pth thủ công theo hướng dẫn trong README.md.",
        )

    # ------------------------------------------------------------------
    # Batch Processing Handlers
    # ------------------------------------------------------------------

    def _batch_add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Chọn các file ảnh để xử lý hàng loạt",
            "",
            "Supported Images (*.jpg *.jpeg *.png *.webp)",
        )
        if files:
            self._append_batch_paths(files)

    def _batch_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục chứa ảnh")
        if folder:
            found = collect_supported_images([folder], recursive=False)
            if not found:
                QMessageBox.information(self, "Không tìm thấy ảnh", "Thư mục đã chọn không chứa file JPG, JPEG, PNG hoặc WEBP.")
                return
            self._append_batch_paths(found)

    def _append_batch_paths(self, paths: List[str | Path]) -> None:
        existing = {item.source_path.resolve() for item in self._batch_items}
        for p in collect_supported_images(paths):
            if p not in existing:
                existing.add(p)
                self._batch_items.append(BatchItem(source_path=p))
        self._refresh_batch_table()

    def _batch_remove_selected(self) -> None:
        rows = sorted({idx.row() for idx in self.tbl_batch.selectedIndexes()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self._batch_items):
                self._batch_items.pop(r)
        self._refresh_batch_table()

    def _batch_clear_all(self) -> None:
        if self._batch_worker and self._batch_worker.isRunning():
            return
        self._batch_items.clear()
        self._refresh_batch_table()

    def _batch_choose_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu kết quả hàng loạt", self.edt_batch_out.text())
        if folder:
            self.edt_batch_out.setText(folder)

    def _refresh_batch_table(self) -> None:
        self.tbl_batch.setRowCount(len(self._batch_items))
        for idx, item in enumerate(self._batch_items):
            self._populate_batch_row(idx, item)
        self._update_batch_summary_label()

    def _populate_batch_row(self, row: int, item: BatchItem) -> None:
        res_text = item.input_resolution or "-"
        if item.output_resolution:
            res_text = f"{item.input_resolution} → {item.output_resolution}"

        status_map = {
            "PENDING": "⏳ Chờ xử lý",
            "PROCESSING": "🔄 Đang xử lý...",
            "SUCCESS": "✅ Thành công",
            "FAILED": "❌ Lỗi",
            "SKIPPED": "⏭ Bỏ qua",
            "CANCELLED": "🛑 Đã dừng",
        }
        status_display = status_map.get(item.status, item.status)
        detail = str(item.output_path) if item.status == "SUCCESS" and item.output_path else item.error_message

        self.tbl_batch.setItem(row, 0, QTableWidgetItem(item.source_path.name))
        self.tbl_batch.setItem(row, 1, QTableWidgetItem(item.file_size_str))
        self.tbl_batch.setItem(row, 2, QTableWidgetItem(res_text))
        self.tbl_batch.setItem(row, 3, QTableWidgetItem(status_display))
        self.tbl_batch.setItem(row, 4, QTableWidgetItem(f"{item.duration_sec:.2f}s" if item.duration_sec else "-"))
        self.tbl_batch.setItem(row, 5, QTableWidgetItem(detail))

    def _update_batch_summary_label(self) -> None:
        tot = len(self._batch_items)
        succ = sum(1 for x in self._batch_items if x.status == "SUCCESS")
        fail = sum(1 for x in self._batch_items if x.status == "FAILED")
        self.lbl_batch_summary.setText(f"Tổng số: {tot} ảnh | Thành công: {succ} | Lỗi: {fail}")

    def _start_batch_processing(self, only_failed: bool = False) -> None:
        if not self._batch_items:
            QMessageBox.information(self, "Danh sách trống", "Vui lòng thêm ảnh hoặc thư mục vào danh sách xử lý hàng loạt.")
            return
        if self._batch_worker is not None and self._batch_worker.isRunning():
            return

        preset_key = str(self.cmb_batch_preset.currentData())
        preset_def = PRESETS.get(preset_key, PRESETS["standard_sharpen"])
        params = EnhancementParams.from_dict(preset_def.params)
        params.tile_size = int(self.settings.tile_size)
        params.tile_pad = int(self.settings.tile_pad)

        out_dir = self.edt_batch_out.text().strip() or self.settings.default_output_dir
        fmt = str(self.cmb_batch_fmt.currentData())
        qual = int(self.spn_batch_qual.value())

        self.btn_batch_start.setEnabled(False)
        self.btn_batch_retry.setEnabled(False)
        self.btn_batch_stop.setEnabled(True)
        self.progress_panel.start_task("Đang bắt đầu xử lý lô ảnh...")

        self._batch_worker = BatchProcessingWorker(
            batch_processor=self.batch_processor,
            items=self._batch_items,
            params=params,
            output_dir=out_dir,
            target_format=fmt,
            jpeg_quality=qual,
            webp_quality=qual,
            preserve_metadata=self.settings.preserve_metadata,
            only_failed=only_failed,
            parent=self,
        )
        self._batch_worker.item_updated.connect(self._on_batch_item_updated)
        self._batch_worker.batch_progress.connect(self._on_batch_progress)
        self._batch_worker.batch_finished.connect(self._on_batch_finished)
        self._batch_worker.batch_failed.connect(self._on_batch_failed)
        self._batch_worker.start()

    def _stop_batch_processing(self) -> None:
        if self._batch_worker is not None and self._batch_worker.isRunning():
            self._batch_worker.cancel()

    def _on_batch_item_updated(self, index: int, item: BatchItem) -> None:
        if 0 <= index < self.tbl_batch.rowCount():
            self._populate_batch_row(index, item)
        self._update_batch_summary_label()

    def _on_batch_progress(self, completed: int, total: int, message: str) -> None:
        pct = int((completed / max(1, total)) * 100)
        self.progress_panel.update_progress(pct, message)

    def _on_batch_finished(self, summary: BatchSummary) -> None:
        self.btn_batch_start.setEnabled(True)
        self.btn_batch_retry.setEnabled(True)
        self.btn_batch_stop.setEnabled(False)
        self._update_batch_summary_label()
        self.progress_panel.finish_task(
            f"Hoàn tất xử lý hàng loạt: {summary.succeeded}/{summary.total} thành công, {summary.failed} lỗi.",
            duration_sec=summary.total_duration_sec,
        )
        if self.settings.enable_notifications:
            QMessageBox.information(
                self,
                "Kết quả xử lý hàng loạt",
                f"Đã hoàn thành lô xử lý ảnh!\n\n"
                f"• Tổng số file: {summary.total}\n"
                f"• Thành công: {summary.succeeded}\n"
                f"• Thất bại: {summary.failed}\n"
                f"• Đã dừng/hủy: {summary.cancelled}\n"
                f"• Tổng thời gian: {summary.total_duration_sec:.2f}s",
            )

    def _on_batch_failed(self, err_msg: str, hint: str) -> None:
        self.btn_batch_start.setEnabled(True)
        self.btn_batch_retry.setEnabled(True)
        self.btn_batch_stop.setEnabled(False)
        self.progress_panel.fail_task(err_msg, hint)

    def _open_batch_output_folder(self) -> None:
        out_dir = Path(self.edt_batch_out.text().strip() or self.settings.default_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(out_dir))  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("Không thể mở thư mục %s: %s", out_dir, exc)

    # ------------------------------------------------------------------
    # History & Settings Handlers
    # ------------------------------------------------------------------

    def _refresh_history_table(self) -> None:
        records: List[HistoryRecord] = self.history_service.list_records(limit=250)
        self.tbl_history.setRowCount(len(records))
        for row, rec in enumerate(records):
            res_str = rec.input_resolution
            if rec.output_resolution:
                res_str = f"{rec.input_resolution} → {rec.output_resolution}"
            status_str = "✅ Thành công" if rec.status == "SUCCESS" else "❌ Thất bại"
            detail = rec.output_path if rec.status == "SUCCESS" else rec.error_message

            self.tbl_history.setItem(row, 0, QTableWidgetItem(str(rec.id or "")))
            self.tbl_history.setItem(row, 1, QTableWidgetItem(rec.timestamp))
            self.tbl_history.setItem(row, 2, QTableWidgetItem(rec.source_filename))
            self.tbl_history.setItem(row, 3, QTableWidgetItem(rec.method))
            self.tbl_history.setItem(row, 4, QTableWidgetItem(res_str))
            self.tbl_history.setItem(row, 5, QTableWidgetItem(status_str))
            self.tbl_history.setItem(row, 6, QTableWidgetItem(detail))

    def _open_selected_history_output(self) -> None:
        row = self.tbl_history.currentRow()
        if row < 0:
            return
        item = self.tbl_history.item(row, 6)
        if not item:
            return
        path_text = item.text().strip()
        p = Path(path_text)
        if p.exists():
            try:
                os.startfile(str(p))  # type: ignore[attr-defined]
            except Exception:
                pass
        else:
            QMessageBox.information(self, "Thông báo", f"Đường dẫn '{path_text}' không phải là file đã lưu trên đĩa.")

    def _delete_selected_history_record(self) -> None:
        row = self.tbl_history.currentRow()
        if row < 0:
            return
        id_item = self.tbl_history.item(row, 0)
        if id_item and id_item.text().isdigit():
            self.history_service.delete_record(int(id_item.text()))
            self._refresh_history_table()

    def _clear_all_history(self) -> None:
        reply = QMessageBox.question(
            self,
            "Xóa toàn bộ lịch sử",
            "Bạn có chắc chắn muốn xóa toàn bộ lịch sử xử lý ảnh cục bộ không?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.history_service.clear_history()
            self._refresh_history_table()

    def _on_settings_saved(self, new_settings) -> None:
        self.settings_mgr.save(new_settings)
        self.settings = self.settings_mgr.settings

        self.device_mgr.set_preferred_device(self.settings.compute_device)
        self.model_mgr.models_dir = Path(self.settings.models_dir)
        self.export_service.default_output_dir = Path(self.settings.default_output_dir)
        self.export_service.default_format = self.settings.default_export_format
        self.export_service.jpeg_quality = self.settings.jpeg_quality
        self.export_service.webp_quality = self.settings.webp_quality
        self.export_service.preserve_metadata = self.settings.preserve_metadata

        self.edt_batch_out.setText(self.settings.default_output_dir)
        self._refresh_top_bar_badges()
        self.progress_panel.finish_task("Đã lưu và áp dụng cấu hình hệ thống mới.")

    def _on_clear_model_cache(self) -> None:
        self.model_mgr.clear_cache()
        self.device_mgr.release_memory()
        self._refresh_top_bar_badges()
        self.progress_panel.finish_task("Đã giải phóng bộ nhớ đệm mô hình AI (RAM/VRAM).")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._single_worker and self._single_worker.isRunning():
            self._single_worker.cancel()
            self._single_worker.wait(1500)
        if self._batch_worker and self._batch_worker.isRunning():
            self._batch_worker.cancel()
            self._batch_worker.wait(1500)
        self.device_mgr.release_memory()
        super().closeEvent(event)
