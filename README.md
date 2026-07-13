# YOLO Trainer

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

> 一站式 YOLO 目标检测工具：视频抽帧 → 图片标注 → 模型训练 → 目标检测

## 功能

### 视频抽帧
- 支持 MP4/AVI/MOV/MKV 等格式
- 4种抽帧模式：按帧率、按间隔、场景变化检测、运动检测
- 实时预览，拖拽支持

### 图片标注
- 鼠标拖拽绘制矩形标注框
- 多类别管理
- 一键导出 YOLO 格式数据集（自动划分 train/val）

### 模型训练
- 支持 YOLOv8 / YOLO11 全系列（n/s/m/l/x）
- GPU/CPU 自动检测，支持 CUDA 加速
- 可调参数：Epochs、Batch Size、Image Size、Device

### 目标检测
- 支持图片、视频、摄像头实时检测
- 可调置信度和 IoU 阈值
- 导出 CSV 检测结果
- 保存标注后的图片/视频

### 其他
- 中英文双语界面
- 深色主题，现代 UI 设计
- 开箱即用，无需配置环境

## 截图

![主界面](screenshot.png)

## 快速开始

### 方式一：直接使用（推荐）

1. 下载 [Releases](https://github.com/your-username/yolo-trainer/releases) 中的 `YOLO-Trainer.zip`
2. 解压后双击 `第一次启动先点此文件！！！！！.bat` 安装依赖
3. 双击 `YOLO-Trainer.exe` 启动

### 方式二：源码运行

```bash
# 克隆仓库
git clone https://github.com/your-username/yolo-trainer.git
cd yolo-trainer

# 安装依赖
pip install -r requirements.txt

# 运行
python main.py
```

## 使用流程

```
视频抽帧 → 图片标注 → 导出数据集 → 模型训练 → 目标检测
```

1. **视频抽帧**：导入视频，选择抽帧模式，导出图片
2. **图片标注**：导入图片文件夹，添加类别，绘制标注框
3. **导出数据集**：一键导出 YOLO 格式，自动划分训练集/验证集
4. **模型训练**：选择数据集和模型，开始训练
5. **目标检测**：加载训练好的模型，检测图片/视频/摄像头

## 支持的模型

| 模型 | 参数量 | 推荐场景 |
|------|--------|----------|
| yolov8n / yolo11n | ~3M | 快速训练、边缘部署 |
| yolov8s / yolo11s | ~11M | 平衡速度与精度 |
| yolov8m / yolo11m | ~25M | 中等规模数据集 |
| yolov8l / yolo11l | ~43M | 高精度需求 |
| yolov8x / yolo11x | ~68M | 最高精度 |

## GPU 支持

### NVIDIA GPU (CUDA)

1. 安装 [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. 安装 [cuDNN](https://developer.nvidia.com/cudnn)
3. 安装 PyTorch with CUDA：
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

### AMD GPU (ROCm)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
```

### CPU Only

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 项目结构

```
yolo-trainer/
├── main.py              # 程序入口
├── requirements.txt     # Python 依赖
├── 启动.bat             # Windows 启动脚本
├── ui/                  # 界面模块
│   ├── main_window.py   # 主窗口
│   ├── extract_page.py  # 视频抽帧页面
│   ├── annotate_page.py # 图片标注页面
│   ├── detect_page.py   # 目标检测页面
│   ├── train_page.py    # 模型训练页面
│   └── components.py    # 通用组件
├── utils/               # 工具模块
│   ├── annotations.py   # 标注管理
│   ├── dataset_parser.py # 数据集解析
│   └── i18n.py          # 国际化
└── tools/               # 模型权重（需自行下载）
```

## 依赖

- PyQt6 >= 6.6.0
- ultralytics >= 8.0.0
- Pillow >= 10.0.0
- tqdm >= 4.65.0
- opencv-python >= 4.8.0

## 许可证

[MIT License](LICENSE)

## 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO 模型实现
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - GUI 框架
