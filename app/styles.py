"""
Modern Creative Studio QSS stylesheet and color palette definitions.
"""
from __future__ import annotations

PALETTE = {
    "bg_darkest": "#0F1117",
    "bg_main": "#151821",
    "bg_panel": "#1C202B",
    "bg_card": "#232836",
    "bg_input": "#13161F",
    "border": "#2E3446",
    "border_focus": "#6366F1",
    "accent": "#6366F1",
    "accent_hover": "#818CF8",
    "accent_pressed": "#4F46E5",
    "text_primary": "#F3F4F6",
    "text_secondary": "#9CA3AF",
    "text_muted": "#6B7280",
    "success": "#10B981",
    "warning": "#F59E0B",
    "error": "#EF4444",
}

STUDIO_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #151821;
    color: #F3F4F6;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}

QFrame#TopBar {
    background-color: #0F1117;
    border-bottom: 1px solid #282E3E;
}

QFrame#Sidebar {
    background-color: #0F1117;
    border-right: 1px solid #282E3E;
}

QFrame#ControlPanel {
    background-color: #1C202B;
    border-left: 1px solid #282E3E;
}

QFrame#CardPanel {
    background-color: #1C202B;
    border: 1px solid #2E3446;
    border-radius: 8px;
}

QFrame#BannerInfo {
    background-color: #1E243B;
    border: 1px solid #4338CA;
    border-radius: 6px;
    padding: 6px;
}

QFrame#BannerWarn {
    background-color: #2D2318;
    border: 1px solid #B45309;
    border-radius: 6px;
    padding: 6px;
}

QFrame#BannerError {
    background-color: #2F191E;
    border: 1px solid #B91C1C;
    border-radius: 6px;
    padding: 6px;
}

QPushButton {
    background-color: #262B3A;
    color: #F3F4F6;
    border: 1px solid #363D52;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #31384B;
    border-color: #4B5563;
}

QPushButton:pressed {
    background-color: #1E222E;
}

QPushButton:disabled {
    background-color: #191C24;
    color: #555D70;
    border-color: #242834;
}

QPushButton#PrimaryButton {
    background-color: #6366F1;
    color: #FFFFFF;
    border: 1px solid #4F46E5;
    font-weight: 600;
}

QPushButton#PrimaryButton:hover {
    background-color: #818CF8;
}

QPushButton#PrimaryButton:pressed {
    background-color: #4F46E5;
}

QPushButton#SuccessButton {
    background-color: #059669;
    color: #FFFFFF;
    border: 1px solid #047857;
    font-weight: 600;
}

QPushButton#SuccessButton:hover {
    background-color: #10B981;
}

QPushButton#DangerButton {
    background-color: #DC2626;
    color: #FFFFFF;
    border: 1px solid #B91C1C;
    font-weight: 600;
}

QPushButton#DangerButton:hover {
    background-color: #EF4444;
}

QPushButton#NavButton {
    background-color: transparent;
    color: #9CA3AF;
    border: none;
    border-radius: 6px;
    text-align: left;
    padding: 10px 14px;
    font-size: 13px;
    font-weight: 500;
}

QPushButton#NavButton:hover {
    background-color: #1C202B;
    color: #F3F4F6;
}

QPushButton#NavButton[active="true"] {
    background-color: #252945;
    color: #FFFFFF;
    border-left: 3px solid #6366F1;
    font-weight: 600;
}

QLabel#SectionTitle {
    color: #F3F4F6;
    font-size: 15px;
    font-weight: 700;
}

QLabel#SubTitle {
    color: #9CA3AF;
    font-size: 12px;
}

QLabel#BadgeDevice {
    background-color: #1F2937;
    color: #60A5FA;
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 600;
}

QLabel#BadgeModelReady {
    background-color: #064E3B;
    color: #34D399;
    border: 1px solid #047857;
    border-radius: 12px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 600;
}

QLabel#BadgeModelMissing {
    background-color: #451A03;
    color: #FBBF24;
    border: 1px solid #B45309;
    border-radius: 12px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 600;
}

QGroupBox {
    background-color: #1C202B;
    border: 1px solid #2E3446;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 12px;
    font-weight: 600;
    color: #E5E7EB;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #A5B4FC;
}

QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #13161F;
    color: #F3F4F6;
    border: 1px solid #32394E;
    border-radius: 5px;
    padding: 5px 8px;
}

QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #6366F1;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #13161F;
    border: 1px solid #2E3446;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #6366F1;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #F3F4F6;
    border: 2px solid #6366F1;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #FFFFFF;
    border-color: #818CF8;
}

QProgressBar {
    background-color: #13161F;
    border: 1px solid #2E3446;
    border-radius: 5px;
    text-align: center;
    color: #F3F4F6;
    font-weight: 600;
    height: 18px;
}

QProgressBar::chunk {
    background-color: #6366F1;
    border-radius: 4px;
}

QTableWidget {
    background-color: #13161F;
    alternate-background-color: #191D28;
    color: #F3F4F6;
    border: 1px solid #2E3446;
    border-radius: 6px;
    gridline-color: #262B3A;
}

QHeaderView::section {
    background-color: #1C202B;
    color: #9CA3AF;
    padding: 6px 8px;
    border: none;
    border-bottom: 1px solid #2E3446;
    font-weight: 600;
}

QScrollBar:vertical {
    background: #151821;
    width: 10px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #32394E;
    min-height: 24px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #4B5563;
}

QCheckBox {
    spacing: 8px;
    color: #E5E7EB;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #4B5563;
    background-color: #13161F;
}

QCheckBox::indicator:checked {
    background-color: #6366F1;
    border-color: #6366F1;
}
"""
