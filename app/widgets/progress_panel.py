"""
Bottom status bar, progress tracker, elapsed execution timer, error troubleshooting banner,
and live log console drawer.
"""
from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class StatusAndProgressPanel(QFrame):
    """
    Bottom status panel displaying task progress, elapsed time, completion status,
    actionable error guidance, and an expandable log console.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("TopBar")
        self._start_time: float | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._update_elapsed_timer)

        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 6, 12, 6)
        root.setSpacing(6)

        # Error / Remediation Banner (hidden unless an error or warning occurs)
        self.error_banner = QFrame()
        self.error_banner.setObjectName("BannerError")
        err_layout = QHBoxLayout(self.error_banner)
        err_layout.setContentsMargins(10, 6, 10, 6)

        self.lbl_error_text = QLabel("")
        self.lbl_error_text.setWordWrap(True)
        self.lbl_error_text.setStyleSheet("color: #FCA5A5; font-size: 12px;")

        btn_close_err = QPushButton("✖")
        btn_close_err.setFixedWidth(28)
        btn_close_err.clicked.connect(lambda: self.error_banner.setVisible(False))

        err_layout.addWidget(self.lbl_error_text, 1)
        err_layout.addWidget(btn_close_err)
        self.error_banner.setVisible(False)
        root.addWidget(self.error_banner)

        # Collapsible Log Console
        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumBlockCount(400)
        self.log_console.setFixedHeight(130)
        self.log_console.setStyleSheet(
            "background-color: #0B0D13; color: #9CA3AF; font-family: Consolas, monospace; font-size: 11px; border: 1px solid #2E3446;"
        )
        self.log_console.setVisible(False)
        root.addWidget(self.log_console)

        # Main Status Row
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)

        self.lbl_status = QLabel("🟢 Sẵn sàng. Hãy chọn hoặc kéo thả ảnh vào khung làm việc.")
        self.lbl_status.setMinimumWidth(280)
        row.addWidget(self.lbl_status, 2)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(240)
        row.addWidget(self.progress_bar)

        self.lbl_elapsed = QLabel("⏱ Thời gian: 0.00s")
        self.lbl_elapsed.setFixedWidth(130)
        self.lbl_elapsed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self.lbl_elapsed)

        self.btn_toggle_logs = QPushButton("📜 Nhật ký (Logs)")
        self.btn_toggle_logs.setCheckable(True)
        self.btn_toggle_logs.toggled.connect(self.log_console.setVisible)
        row.addWidget(self.btn_toggle_logs)

        root.addLayout(row)

    def start_task(self, message: str = "Đang xử lý...") -> None:
        self.error_banner.setVisible(False)
        self._start_time = time.perf_counter()
        self._timer.start()
        self.progress_bar.setValue(5)
        self.lbl_status.setText(f"⏳ {message}")
        self.lbl_elapsed.setText("⏱ Thời gian: 0.00s")

    def update_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(int(max(0, min(100, percent))))
        self.lbl_status.setText(f"⏳ {message}")
        self._update_elapsed_timer()

    def finish_task(self, message: str, duration_sec: float | None = None) -> None:
        self._timer.stop()
        elapsed = duration_sec if duration_sec is not None else (
            time.perf_counter() - self._start_time if self._start_time else 0.0
        )
        self.progress_bar.setValue(100)
        self.lbl_status.setText(f"✅ {message}")
        self.lbl_elapsed.setText(f"⏱ Thời gian: {elapsed:.2f}s")

    def show_warning(self, warning_text: str) -> None:
        self.error_banner.setObjectName("BannerWarn")
        self.error_banner.style().unpolish(self.error_banner)
        self.error_banner.style().polish(self.error_banner)
        self.lbl_error_text.setStyleSheet("color: #FDE68A; font-size: 12px;")
        self.lbl_error_text.setText(f"⚠ Lưu ý: {warning_text}")
        self.error_banner.setVisible(True)

    def fail_task(self, error_message: str, remediation_hint: str = "") -> None:
        self._timer.stop()
        self.progress_bar.setValue(0)
        self.lbl_status.setText(f"❌ Lỗi: {error_message}")

        self.error_banner.setObjectName("BannerError")
        self.error_banner.style().unpolish(self.error_banner)
        self.error_banner.style().polish(self.error_banner)
        self.lbl_error_text.setStyleSheet("color: #FCA5A5; font-size: 12px;")
        combined = f"❌ Lỗi: {error_message}"
        if remediation_hint:
            combined += f"\n💡 {remediation_hint}"
        self.lbl_error_text.setText(combined)
        self.error_banner.setVisible(True)

    def cancel_task(self, message: str = "Đã hủy tác vụ.") -> None:
        self._timer.stop()
        self.progress_bar.setValue(0)
        self.lbl_status.setText(f"🛑 {message}")

    def append_log_line(self, line: str) -> None:
        self.log_console.appendPlainText(line)

    def _update_elapsed_timer(self) -> None:
        if self._start_time is not None:
            elapsed = time.perf_counter() - self._start_time
            self.lbl_elapsed.setText(f"⏱ Thời gian: {elapsed:.2f}s")
