@echo off
chcp 65001 >nul
title Dong goi Image Enhancement Studio (PyInstaller)
cd /d "%~dp0"

echo [1/2] Kiem tra PyInstaller...
python -m PyInstaller --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [WARN] Chua cai dat PyInstaller. Chay: pip install pyinstaller
    exit /b 1
)

echo [2/2] Dang dong goi ung dung Windows (che do --onedir, khong nhung file .pth dung luong lon)...
python -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --windowed ^
    --onedir ^
    --name "ImageEnhancementStudio" ^
    --add-data "assets;assets" ^
    --exclude-module "PySide6.QtWebEngineCore" ^
    --exclude-module "PySide6.QtWebEngineWidgets" ^
    --exclude-module "PySide6.Qt3DCore" ^
    main.py

if exist "dist\ImageEnhancementStudio" (
    if not exist "dist\ImageEnhancementStudio\models" mkdir "dist\ImageEnhancementStudio\models"
    if not exist "dist\ImageEnhancementStudio\outputs" mkdir "dist\ImageEnhancementStudio\outputs"
    if not exist "dist\ImageEnhancementStudio\logs" mkdir "dist\ImageEnhancementStudio\logs"
    echo.
    echo [SUCCESS] Dong goi hoan tat tai: dist\ImageEnhancementStudio\ImageEnhancementStudio.exe
    echo Hay dat cac file trong so .pth vao thu muc dist\ImageEnhancementStudio\models\
)
