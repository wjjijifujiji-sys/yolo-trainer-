@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo   YOLO Trainer
echo ========================================
echo.

py main.py

if errorlevel 1 (
    echo.
    echo Launch failed!
    echo Please run "Install Dependencies" first.
    echo.
    pause
)
