"""
Entry point for Image Enhancement Studio desktop application.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from config.settings import SettingsManager
from services.logging_service import setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image Enhancement Studio — Làm nét & Phục hồi ảnh bằng Python")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Khởi tạo giao diện và tạo ảnh mẫu thử nghiệm rồi tự động đóng (phục vụ kiểm thử tự động).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Đường dẫn ảnh cần mở ngay khi khởi động ứng dụng.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging()
    logger = logging.getLogger("main")
    logger.info("Khởi động Image Enhancement Studio...")

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Image Enhancement Studio")
    app.setOrganizationName("ImageEnhancementStudio")

    settings_mgr = SettingsManager()
    window = MainWindow(settings_manager=settings_mgr)
    window.show()

    if args.image and Path(args.image).exists():
        window.load_source_image_from_path(args.image)

    if args.smoke_test:
        logger.info("Chạy kiểm thử giao diện (--smoke-test)...")
        window._load_synthetic_demo_image()
        app.processEvents()
        window.close()
        logger.info("Hoàn tất --smoke-test thành công.")
        return 0

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
