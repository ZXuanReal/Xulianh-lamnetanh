"""
Before / After comparison workspace widget supporting:
1. Interactive draggable curtain slider comparison
2. Synchronized side-by-side split view
3. Full-frame Before / After view modes
4. Zoom, pan, drag-and-drop file loading, and live image metadata inspection.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.widgets.image_viewer import InteractiveImageCanvas, container_to_qpixmap
from config.settings import SUPPORTED_EXTENSIONS
from core.image_io import ImageContainer, format_file_size


class BeforeAfterCurtainCanvas(QWidget):
    """
    Custom QWidget that renders Before and After QPixmaps aligned to the same spatial
    bounding box with a draggable vertical split curtain (0.0 .. 1.0), zoom, and pan.
    """

    file_dropped = Signal(str)
    open_requested = Signal()
    split_changed = Signal(float)
    zoom_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._before_pix = QPixmap()
        self._after_pix = QPixmap()
        self._split_ratio = 0.5  # 0.0 .. 1.0
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._dragging_slider = False
        self._dragging_pan = False
        self._last_pos = QPointF(0.0, 0.0)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setMinimumSize(400, 320)

    def set_images(self, before_pix: QPixmap, after_pix: QPixmap | None = None, fit: bool = True) -> None:
        self._before_pix = before_pix
        self._after_pix = after_pix if (after_pix is not None and not after_pix.isNull()) else QPixmap()
        if fit and not self._before_pix.isNull():
            self.fit_to_window()
        else:
            self.update()

    def set_split_ratio(self, ratio: float) -> None:
        self._split_ratio = float(np.clip(ratio, 0.0, 1.0))
        self.update()

    def fit_to_window(self) -> None:
        ref = self._after_pix if not self._after_pix.isNull() else self._before_pix
        if ref.isNull() or self.width() <= 20 or self.height() <= 20:
            self._zoom = 1.0
            self._pan = QPointF(0.0, 0.0)
            self.update()
            return
        sx = (self.width() - 32) / max(1, ref.width())
        sy = (self.height() - 32) / max(1, ref.height())
        self._zoom = float(np.clip(min(sx, sy), 0.05, 4.0))
        self._pan = QPointF(0.0, 0.0)
        self.zoom_changed.emit(self._zoom)
        self.update()

    def reset_zoom_100(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.zoom_changed.emit(self._zoom)
        self.update()

    def zoom_by(self, factor: float) -> None:
        self._zoom = float(np.clip(self._zoom * factor, 0.05, 16.0))
        self.zoom_changed.emit(self._zoom)
        self.update()

    @property
    def zoom_factor(self) -> float:
        return self._zoom

    def _split_x_coord(self) -> float:
        return float(self.width() * self._split_ratio)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                local = url.toLocalFile()
                if local and Path(local).suffix.lower() in SUPPORTED_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                local = url.toLocalFile()
                if local and Path(local).suffix.lower() in SUPPORTED_EXTENSIONS:
                    self.file_dropped.emit(local)
                    event.acceptProposedAction()
                    return

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._before_pix.isNull():
            return
        delta = event.angleDelta().y()
        self.zoom_by(1.15 if delta > 0 else (1.0 / 1.15))
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._before_pix.isNull():
            if event.button() == Qt.MouseButton.LeftButton:
                self.open_requested.emit()
            return

        pos = event.position()
        if event.button() == Qt.MouseButton.LeftButton:
            if not self._after_pix.isNull() and abs(pos.x() - self._split_x_coord()) <= 18.0:
                self._dragging_slider = True
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self._dragging_pan = True
                self._last_pos = pos
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._dragging_pan = True
            self._last_pos = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if self._dragging_slider:
            ratio = float(np.clip(pos.x() / max(1.0, float(self.width())), 0.02, 0.98))
            self._split_ratio = ratio
            self.split_changed.emit(ratio)
            self.update()
            return

        if self._dragging_pan:
            delta = pos - self._last_pos
            self._last_pos = pos
            self._pan += delta
            self.update()
            return

        if not self._before_pix.isNull() and not self._after_pix.isNull():
            if abs(pos.x() - self._split_x_coord()) <= 18.0:
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:
        self._dragging_slider = False
        self._dragging_pan = False
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._zoom < 2.5)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect()
        painter.fillRect(rect, QColor("#10131A"))

        if self._before_pix.isNull():
            pen = QPen(QColor("#363D52"), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            drop_rect = rect.adjusted(32, 32, -32, -32)
            painter.drawRoundedRect(drop_rect, 14, 14)

            painter.setPen(QColor("#E5E7EB"))
            painter.drawText(
                drop_rect,
                Qt.AlignmentFlag.AlignCenter,
                "KÉO & THẢ ẢNH VÀO KHUNG NÀY (JPG, JPEG, PNG, WEBP)\n"
                "hoặc Nhấn chuột để mở ảnh từ máy tính",
            )
            return

        # Use the reference dimensions of After (if available) so super-resolved 2x/4x images align spatially
        ref_pix = self._after_pix if not self._after_pix.isNull() else self._before_pix
        draw_w = ref_pix.width() * self._zoom
        draw_h = ref_pix.height() * self._zoom
        cx = rect.width() / 2.0 + self._pan.x()
        cy = rect.height() / 2.0 + self._pan.y()
        target_rect = QRectF(cx - draw_w / 2.0, cy - draw_h / 2.0, draw_w, draw_h)

        if self._after_pix.isNull():
            painter.drawPixmap(target_rect, self._before_pix, QRectF(self._before_pix.rect()))
            self._draw_badge(painter, QRectF(14, 14, 130, 28), "ẢNH GỐC (Chưa xử lý)")
            return

        split_x = self._split_x_coord()

        # Draw Before on the left of split_x
        painter.save()
        painter.setClipRect(QRectF(0, 0, split_x, rect.height()))
        painter.drawPixmap(target_rect, self._before_pix, QRectF(self._before_pix.rect()))
        painter.restore()

        # Draw After on the right of split_x
        painter.save()
        painter.setClipRect(QRectF(split_x, 0, rect.width() - split_x, rect.height()))
        painter.drawPixmap(target_rect, self._after_pix, QRectF(self._after_pix.rect()))
        painter.restore()

        # Draw dividing vertical line and circular drag handle
        line_pen = QPen(QColor("#6366F1"), 2)
        painter.setPen(line_pen)
        painter.drawLine(QPointF(split_x, 0), QPointF(split_x, rect.height()))

        handle_y = rect.height() / 2.0
        handle_rect = QRectF(split_x - 18, handle_y - 18, 36, 36)
        painter.setBrush(QColor("#6366F1"))
        painter.setPen(QPen(QColor("#FFFFFF"), 2))
        painter.drawEllipse(handle_rect)
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(handle_rect, Qt.AlignmentFlag.AlignCenter, "◄►")

        self._draw_badge(painter, QRectF(14, 14, 135, 28), "TRƯỚC (Ảnh gốc)")
        self._draw_badge(painter, QRectF(rect.width() - 155, 14, 141, 28), "SAU (Kết quả)")

    @staticmethod
    def _draw_badge(painter: QPainter, badge_rect: QRectF, text: str) -> None:
        painter.save()
        painter.setPen(QPen(QColor("#363D52"), 1))
        painter.setBrush(QColor(15, 17, 23, 215))
        painter.drawRoundedRect(badge_rect, 6, 6)
        painter.setPen(QColor("#F3F4F6"))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()


class BeforeAfterWorkspace(QWidget):
    """
    Complete center viewport combining toolbar, Curtain/Side-by-Side/Single modes,
    and rich image metadata inspection bar.
    """

    file_selected = Signal(str)
    open_file_dialog_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source_container: ImageContainer | None = None
        self._result_container: ImageContainer | None = None
        self._before_pix = QPixmap()
        self._after_pix = QPixmap()

        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # 1. Viewport Top Toolbar
        toolbar = QFrame()
        toolbar.setObjectName("CardPanel")
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(10, 6, 10, 6)
        tb_layout.setSpacing(6)

        self.btn_open = QPushButton("📂 Chọn ảnh")
        self.btn_open.setObjectName("PrimaryButton")
        self.btn_open.clicked.connect(self.open_file_dialog_requested.emit)
        tb_layout.addWidget(self.btn_open)

        self.btn_mode_curtain = QPushButton("🔀 Thanh trượt Before/After")
        self.btn_mode_curtain.setCheckable(True)
        self.btn_mode_curtain.setChecked(True)
        self.btn_mode_curtain.clicked.connect(lambda: self.set_view_mode(0))

        self.btn_mode_side = QPushButton("◫ Chia đôi song song")
        self.btn_mode_side.setCheckable(True)
        self.btn_mode_side.clicked.connect(lambda: self.set_view_mode(1))

        self.btn_mode_orig = QPushButton("🖼 Ảnh gốc")
        self.btn_mode_orig.setCheckable(True)
        self.btn_mode_orig.clicked.connect(lambda: self.set_view_mode(2))

        self.btn_mode_res = QPushButton("✨ Ảnh kết quả")
        self.btn_mode_res.setCheckable(True)
        self.btn_mode_res.clicked.connect(lambda: self.set_view_mode(3))

        for btn in (self.btn_mode_curtain, self.btn_mode_side, self.btn_mode_orig, self.btn_mode_res):
            tb_layout.addWidget(btn)

        # Slider control for curtain position
        self.lbl_slider = QLabel("Vị trí trượt:")
        self.lbl_slider.setObjectName("SubTitle")
        self.curtain_slider = QSlider(Qt.Orientation.Horizontal)
        self.curtain_slider.setRange(2, 98)
        self.curtain_slider.setValue(50)
        self.curtain_slider.setFixedWidth(100)
        self.curtain_slider.valueChanged.connect(self._on_slider_bar_moved)
        tb_layout.addWidget(self.lbl_slider)
        tb_layout.addWidget(self.curtain_slider)

        tb_layout.addStretch(1)

        # Zoom controls
        self.btn_zoom_out = QPushButton("➖")
        self.btn_zoom_out.setFixedWidth(34)
        self.btn_zoom_out.setToolTip("Thu nhỏ")
        self.btn_zoom_out.clicked.connect(lambda: self.zoom_by(1.0 / 1.2))

        self.lbl_zoom = QLabel("100%")
        self.lbl_zoom.setFixedWidth(50)
        self.lbl_zoom.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_zoom_in = QPushButton("➕")
        self.btn_zoom_in.setFixedWidth(34)
        self.btn_zoom_in.setToolTip("Phóng to")
        self.btn_zoom_in.clicked.connect(lambda: self.zoom_by(1.2))

        self.btn_zoom_100 = QPushButton("1:1")
        self.btn_zoom_100.setFixedWidth(42)
        self.btn_zoom_100.setToolTip("Kích thước thật 100%")
        self.btn_zoom_100.clicked.connect(self.reset_zoom_100)

        self.btn_zoom_fit = QPushButton("⛶ Vừa khung")
        self.btn_zoom_fit.clicked.connect(self.fit_to_window)

        for w in (self.btn_zoom_out, self.lbl_zoom, self.btn_zoom_in, self.btn_zoom_100, self.btn_zoom_fit):
            tb_layout.addWidget(w)

        root.addWidget(toolbar)

        # 2. Stacked Viewport Area
        self.stack = QStackedWidget()

        # Page 0: Curtain Slider Canvas
        self.curtain_canvas = BeforeAfterCurtainCanvas()
        self.curtain_canvas.file_dropped.connect(self.file_selected.emit)
        self.curtain_canvas.open_requested.connect(self.open_file_dialog_requested.emit)
        self.curtain_canvas.split_changed.connect(self._on_canvas_split_dragged)
        self.curtain_canvas.zoom_changed.connect(self._update_zoom_label)
        self.stack.addWidget(self.curtain_canvas)

        # Page 1: Side-by-Side Dual Canvas
        side_widget = QWidget()
        side_layout = QHBoxLayout(side_widget)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(6)
        self.side_left = InteractiveImageCanvas("TRƯỚC (Ảnh gốc)")
        self.side_right = InteractiveImageCanvas("SAU (Kết quả)")
        self.side_left.file_dropped.connect(self.file_selected.emit)
        self.side_right.file_dropped.connect(self.file_selected.emit)
        self.side_left.open_requested.connect(self.open_file_dialog_requested.emit)
        self.side_right.open_requested.connect(self.open_file_dialog_requested.emit)
        self.side_left.view_changed.connect(self._sync_from_left)
        self.side_right.view_changed.connect(self._sync_from_right)
        side_layout.addWidget(self.side_left, 1)
        side_layout.addWidget(self.side_right, 1)
        self.stack.addWidget(side_widget)

        # Page 2: Single Original Canvas
        self.single_orig = InteractiveImageCanvas("ẢNH GỐC")
        self.single_orig.file_dropped.connect(self.file_selected.emit)
        self.single_orig.open_requested.connect(self.open_file_dialog_requested.emit)
        self.single_orig.view_changed.connect(lambda z, _p: self._update_zoom_label(z))
        self.stack.addWidget(self.single_orig)

        # Page 3: Single Result Canvas
        self.single_res = InteractiveImageCanvas("ẢNH KẾT QUẢ")
        self.single_res.file_dropped.connect(self.file_selected.emit)
        self.single_res.open_requested.connect(self.open_file_dialog_requested.emit)
        self.single_res.view_changed.connect(lambda z, _p: self._update_zoom_label(z))
        self.stack.addWidget(self.single_res)

        root.addWidget(self.stack, 1)

        # 3. Bottom Metadata & Inspection Bar
        info_bar = QFrame()
        info_bar.setObjectName("CardPanel")
        info_layout = QHBoxLayout(info_bar)
        info_layout.setContentsMargins(12, 6, 12, 6)

        self.lbl_source_meta = QLabel("Ảnh gốc: Chưa chọn ảnh")
        self.lbl_source_meta.setObjectName("SubTitle")
        self.lbl_result_meta = QLabel("Kết quả: Chưa xử lý")
        self.lbl_result_meta.setObjectName("SubTitle")
        self.lbl_result_meta.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        info_layout.addWidget(self.lbl_source_meta, 1)
        info_layout.addWidget(self.lbl_result_meta, 1)
        root.addWidget(info_bar)

    def _sync_from_left(self, zoom: float, pan: QPointF) -> None:
        self.side_right.set_view_state(zoom, pan)
        self._update_zoom_label(zoom)

    def _sync_from_right(self, zoom: float, pan: QPointF) -> None:
        self.side_left.set_view_state(zoom, pan)
        self._update_zoom_label(zoom)

    def _on_slider_bar_moved(self, value: int) -> None:
        self.curtain_canvas.set_split_ratio(value / 100.0)

    def _on_canvas_split_dragged(self, ratio: float) -> None:
        self.curtain_slider.blockSignals(True)
        self.curtain_slider.setValue(int(round(ratio * 100)))
        self.curtain_slider.blockSignals(False)

    def _update_zoom_label(self, zoom: float) -> None:
        self.lbl_zoom.setText(f"{int(round(zoom * 100))}%")

    def set_view_mode(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        modes = [self.btn_mode_curtain, self.btn_mode_side, self.btn_mode_orig, self.btn_mode_res]
        for idx, btn in enumerate(modes):
            btn.setChecked(idx == index)
        self.curtain_slider.setEnabled(index == 0)

    def set_source_image(self, container: ImageContainer) -> None:
        """Load a new source image into the workspace and clear previous result."""
        self._source_container = container
        self._result_container = None
        self._before_pix = container_to_qpixmap(container)
        self._after_pix = QPixmap()

        self.curtain_canvas.set_images(self._before_pix, None, fit=True)
        self.side_left.set_pixmap(self._before_pix, fit=True)
        self.side_right.set_pixmap(QPixmap(), fit=True)
        self.single_orig.set_pixmap(self._before_pix, fit=True)
        self.single_res.set_pixmap(QPixmap(), fit=True)

        name = container.source_path.name if container.source_path else "Ảnh bộ nhớ"
        size_str = format_file_size(container.file_size_bytes)
        alpha_tag = " (Có kênh Alpha)" if container.has_alpha else ""
        self.lbl_source_meta.setText(
            f"📌 Ảnh gốc: {name} | Kích thước: {container.width}x{container.height} ({container.megapixels} MP) "
            f"| Định dạng: {container.original_format} {container.color_mode}{alpha_tag} | Dung lượng: {size_str}"
        )
        self.lbl_result_meta.setText("⏳ Kết quả: Chưa xử lý (Nhấn 'Xử lý ảnh' hoặc 'Auto Enhance')")

    def set_result_image(
        self,
        result_container: ImageContainer,
        method_summary: str = "",
        sharpness_before: float = 0.0,
        sharpness_after: float = 0.0,
    ) -> None:
        """Update the workspace with the newly enhanced result image."""
        self._result_container = result_container
        self._after_pix = container_to_qpixmap(result_container)

        self.curtain_canvas.set_images(self._before_pix, self._after_pix, fit=False)
        self.side_right.set_pixmap(self._after_pix, fit=True)
        self.single_res.set_pixmap(self._after_pix, fit=True)

        sharp_info = ""
        if sharpness_before > 0 or sharpness_after > 0:
            sharp_info = f" | Độ sắc nét biên: {sharpness_before:.1f} → {sharpness_after:.1f}"

        self.lbl_result_meta.setText(
            f"✅ Kết quả ({method_summary}): {result_container.width}x{result_container.height} "
            f"({result_container.megapixels} MP){sharp_info}"
        )

    def zoom_by(self, factor: float) -> None:
        idx = self.stack.currentIndex()
        if idx == 0:
            self.curtain_canvas.zoom_by(factor)
        elif idx == 1:
            self.side_left.zoom_by(factor)
        elif idx == 2:
            self.single_orig.zoom_by(factor)
        elif idx == 3:
            self.single_res.zoom_by(factor)

    def fit_to_window(self) -> None:
        self.curtain_canvas.fit_to_window()
        self.side_left.fit_to_window()
        self.side_right.fit_to_window()
        self.single_orig.fit_to_window()
        self.single_res.fit_to_window()

    def reset_zoom_100(self) -> None:
        self.curtain_canvas.reset_zoom_100()
        self.side_left.reset_zoom_100()
        self.side_right.reset_zoom_100()
        self.single_orig.reset_zoom_100()
        self.single_res.reset_zoom_100()
