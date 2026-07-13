@echo off
chcp 65001 >nul

echo ========================================
echo   YOLO Trainer - Install Dependencies
echo ========================================
echo.

set "MISSING="

py -c "import PyQt6" >nul 2>&1
if errorlevel 1 set "MISSING=%MISSING% PyQt6"

py -c "import cv2" >nul 2>&1
if errorlevel 1 set "MISSING=%MISSING% opencv-python"

py -c "import ultralytics" >nul 2>&1
if errorlevel 1 set "MISSING=%MISSING% ultralytics"

py -c "import numpy" >nul 2>&1
if errorlevel 1 set "MISSING=%MISSING% numpy"

py -c "import PIL" >nul 2>&1
if errorlevel 1 set "MISSING=%MISSING% Pillow"

py -c "import tqdm" >nul 2>&1
if errorlevel 1 set "MISSING=%MISSING% tqdm"

if "%MISSING%"=="" (
    echo [OK] All basic dependencies installed!
    echo.
    goto :check_cuda
)

echo [MISS] %MISSING%
echo.
echo Select download source:
echo   1. China mirror (fast)
echo   2. Official PyPI (abroad)
echo.
set /p "MIRROR=  Enter 1 or 2: "

if "%MIRROR%"=="1" (
    set "MIRROR_URL=https://pypi.tuna.tsinghua.edu.cn/simple"
    set "MIRROR_HOST=pypi.tuna.tsinghua.edu.cn"
    goto :install
)
if "%MIRROR%"=="2" (
    set "MIRROR_URL=https://pypi.org/simple"
    set "MIRROR_HOST=pypi.org"
    goto :install
)
echo Invalid input!
pause
exit /b 1

:install
echo.
echo Installing: %MISSING%
echo.
py -m pip install %MISSING% -i %MIRROR_URL% --trusted-host %MIRROR_HOST%

if errorlevel 1 (
    echo.
    echo Install failed!
    pause
    exit /b 1
)

echo.
echo [OK] Dependencies installed!
echo.

:check_cuda
echo ========================================
echo   GPU / CUDA Check
echo ========================================
echo.

py -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>nul
if errorlevel 1 (
    echo [INFO] PyTorch not installed or no GPU detected
    echo.
    goto :cuda_guide
)

py -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
if errorlevel 1 (
    echo.
    echo [WARNING] NVIDIA GPU detected but CUDA not available!
    echo.
    echo Your GPU needs CUDA-accelerated PyTorch.
    echo Please ask your AI assistant:
    echo.
    echo   "How to install PyTorch with CUDA for my GPU?"
    echo.
    echo Or visit: https://pytorch.org/get-started/locally/
    echo.
) else (
    echo.
    echo [OK] CUDA is ready! GPU training enabled.
    echo.
)

:cuda_guide
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo   - Double click YOLO-Trainer.exe to start
echo   - First time training may download model weights
echo.
pause
