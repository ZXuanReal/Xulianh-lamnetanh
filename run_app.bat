@echo off
chcp 65001 >nul
title Image Enhancement Studio
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    echo [INFO] Dang khoi chay bang moi truong ao .venv...
    ".venv\Scripts\python.exe" main.py %*
) else (
    echo [INFO] Dang khoi chay bang Python he thong...
    python main.py %*
)
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Ung dung ket thuc voi ma loi %ERRORLEVEL%. Kiem tra logs\image_enhancement_studio.log.
    pause
)
