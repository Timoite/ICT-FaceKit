# M1 Mac 环境设置指南

## ✅ 安装完成

你的环境已经成功配置！所有依赖已安装到系统 Python 环境中。

## 🔥 GPU 加速状态

- **PyTorch 版本**: 2.10.0
- **MPS (Apple Silicon GPU) 可用**: ✅ 是
- **MPS 构建**: ✅ 是

这意味着 PyTorch 会自动使用你的 M1 GPU 加速！

## 📦 已安装的关键依赖

### 核心依赖
- numpy, scipy, trimesh, matplotlib
- imageio, imageio-ffmpeg, tqdm

### GPU 加速
- torch 2.10.0 (带 MPS 支持)
- torchaudio 2.10.0

### 渲染
- pyrender (CPU-based OpenGL 渲染)
- PyOpenGL

### 舌头动画 (WavLM)
- transformers
- loralib

### 数据集工具
- huggingface_hub

### VSR (Visual Speech Recognition)
- hydra-core
- opencv-python
- scikit-image
- av
- espnet
- espnet-model-zoo

## 🚀 如何运行代码

### 方法 1: 使用系统 Python (推荐)

由于依赖已安装到系统 Python (`/Users/iitealpha/anaconda3/bin/python3`)，你可以直接：

```bash
# 舌头动画渲染
cd tongue_scripts
python3 test_tongue_grid_search_25fps.py

# VSR 评估
cd ADFA_EVALUATION
python3 Visual_Speech_Recognition_for_Multiple_Languages/infer.py \
    config_filename=configs/LRS3_V_WER19.1.ini \
    data_filename=tongue_scripts/batch_videos/video.mp4 \
    detector=mediapipe
```

### 方法 2: 使用 uv (但需跳过项目安装)

```bash
# 舌头动画渲染
cd tongue_scripts
uv run --no-project python3 test_tongue_grid_search_25fps.py

# VSR 评估
cd ADFA_EVALUATION
uv run --no-project python3 Visual_Speech_Recognition_for_Multiple_Languages/infer.py \
    config_filename=configs/LRS3_V_WER19.1.ini \
    data_filename=tongue_scripts/batch_videos/video.mp4 \
    detector=mediapipe
```

## 💡 在代码中启用 MPS 加速

PyTorch 会自动使用 MPS。如果需要手动指定设备：

```python
import torch

# 检查 MPS 是否可用
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"✅ 使用 MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ 使用 CUDA (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    print(f"⚠️ 使用 CPU")

# 将模型移到 GPU
model.to(device)

# 将数据移到 GPU
data = data.to(device)
```

## 📝 项目说明

此项目不是一个标准的 Python 包，而是一个工具包集合。因此：
- 无需 `pip install -e .`
- 直接运行脚本即可
- 所有依赖已全局安装

## ⚠️ 已知限制

1. **librosa 未安装**: 由于 numba 在 Python 3.11+ 的兼容性问题，跳过了 librosa。这主要影响音频分析功能，对核心功能（舌头动画和 VSR）无影响。

2. **pytorch3d 未安装**: PyTorch3D 对 Apple Silicon 支持有限。如果需要，可以尝试：
   ```bash
   pip install pytorch3d --no-deps
   ```
   但可能会有兼容性问题。

## 🔄 重新安装

如需重新安装环境：

```bash
# 运行设置脚本
./setup_m1.sh

# 或者手动安装
pip install numpy scipy trimesh matplotlib imageio imageio-ffmpeg tqdm
pip install "torch>=2.0" "torchaudio>=2.0"
pip install "pyrender>=0.1.45" "PyOpenGL>=3.1"
pip install "transformers>=4.30" "loralib>=0.1"
pip install "huggingface_hub>=0.20"
pip install hydra-core opencv-python scikit-image av six pillow
pip install "espnet>=0.10.0" "espnet-model-zoo>=0.1.0"
```

## 📚 更多信息

- 详见 `CLAUDE.md` 了解项目结构和使用方法
- 详见各个子目录中的 README 和脚本注释
