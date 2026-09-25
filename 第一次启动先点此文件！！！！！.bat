@echo off
cd /d "%~dp0"
chcp 65001 >nul

echo ========================================
echo   YOLO Trainer - First Time Setup
echo ========================================
echo.

echo Checking Python...
py --version
if errorlevel 1 (
    echo Python not found!
    echo Please install Python 3.10+ from:
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
echo.

echo Checking dependencies...
py -c "import PyQt6; import cv2; import ultralytics; import numpy; import PIL; import tqdm; print('All OK')"
if errorlevel 1 (
    echo.
    echo Missing dependencies detected.
    echo.
    echo Select download source:
    echo   1. China mirror
    echo   2. Official PyPI
    echo.
    set /p "MIRROR=Enter 1 or 2: "
    
    if "%MIRROR%"=="1" (
        py -m pip install PyQt6 opencv-python ultralytics numpy Pillow tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    ) else (
        py -m pip install PyQt6 opencv-python ultralytics numpy Pillow tqdm
    )
    
    if errorlevel 1 (
        echo.
        echo Install failed!
        pause
        exit /b 1
    )
    echo.
    echo Dependencies installed!
)

echo.
echo Checking GPU...
py -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>nul
if errorlevel 1 (
    echo No GPU detected or PyTorch not installed
)

echo.
echo ========================================
echo   Setup Done! Start YOLO-Trainer.exe---jijifujiji_sys
echo ========================================
echo.
pause
