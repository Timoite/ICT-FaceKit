#!/bin/bash
# ICT-FaceKit M1 Mac 环境设置脚本

set -e

echo "🍎 为 M1 Mac 设置 ICT-FaceKit 环境..."
echo ""

# 检查 uv 是否安装
if ! command -v uv &> /dev/null; then
    echo "❌ uv 未安装。请先安装 uv:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ uv 已安装"
echo ""

# 创建虚拟环境（如果不存在）
if [ ! -d ".venv" ]; then
    echo "📦 创建虚拟环境 (Python 3.10)..."
    uv venv --python 3.10
else
    echo "✅ 虚拟环境已存在"
fi

echo ""
echo "📥 安装基础依赖..."
uv pip install numpy scipy trimesh matplotlib imageio imageio-ffmpeg tqdm

echo ""
echo "🔥 安装 PyTorch with MPS 支持 (Apple Silicon GPU)..."
uv pip install "torch>=2.0" "torchaudio>=2.0"

echo ""
echo "🎨 安装 Pyrender (CPU 渲染)..."
uv pip install "pyrender>=0.1.45" "PyOpenGL>=3.1"

echo ""
echo "🗣️ 安装 WavLM 依赖..."
uv pip install "transformers>=4.30" "loralib>=0.1"

echo ""
echo "📚 安装 BEAT 数据集工具..."
uv pip install "huggingface_hub>=0.20"

echo ""
echo "👁️ 安装 VSR (Visual Speech Recognition) 依赖..."
uv pip install \
    "hydra-core>=1.3.2" \
    "opencv-python>=4.5.5.62" \
    "scikit-image>=0.13.0" \
    "av>=10.0.0" \
    "six>=1.16.0" \
    "pillow>=9.0"

# 跳过 librosa (numba 在 Python 3.11+ 有兼容性问题)
echo ""
echo "⚠️  跳过 librosa (可选，用于音频分析)"

echo ""
echo "📦 安装 ESPnet (用于 VSR 模型)..."
uv pip install "espnet>=0.10.0" "espnet-model-zoo>=0.1.0"

echo ""
echo "✅ 环境设置完成！"
echo ""
echo "📝 重要提示:"
echo "   1. PyTorch 将自动使用 MPS (Apple Silicon GPU) 加速"
echo "   2. 对于舌头动画和 VSR 推理，GPU 加速将自动启用"
echo "   3. 使用 'uv run python' 来运行脚本"
echo ""
echo "🎯 快速开始:"
echo ""
echo "   # 舌头动画渲染"
echo "   cd tongue_scripts"
echo "   uv run python test_tongue_grid_search_25fps.py"
echo ""
echo "   # VSR 评估"
echo "   cd ADFA_EVALUATION"
echo "   uv run python Visual_Speech_Recognition_for_Multiple_Languages/infer.py \\"
echo "       config_filename=configs/LRS3_V_WER19.1.ini \\"
echo "       data_filename=tongue_scripts/batch_videos/video.mp4 \\"
echo "       detector=mediapipe"
echo ""
