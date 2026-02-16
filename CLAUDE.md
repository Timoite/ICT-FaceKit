# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

ICT-FaceKit 是 ICT Vision and Graphics Lab 的可变形面部模型和工具包,包含:
- **ICT Face Model Light**: 基础面部拓扑 + 100 个 PCA 身份模式 + 53 个表情混合形状
- **Tongue Animation Pipeline**: 舌头动画系统,使用 WavLM 从音频生成舌头运动
- **Visual Speech Recognition (VSR)**: 基于 AutoAVSR 的多语言视觉语音识别评估系统

## 安装依赖

```bash
# 基础依赖
pip install numpy scipy trimesh matplotlib imageio imageio-ffmpeg tqdm pyrender

# 可选依赖组
pip install "ict-facekit[pyrender]"      # Pyrender 渲染
pip install "ict-facekit[pytorch3d]"     # PyTorch3D 渲染
pip install "ict-facekit[wavlm]"         # WavLM 语音反演
pip install "ict-facekit[beat]"          # BEAT 数据集下载
pip install "ict-facekit[all]"           # 所有功能

# VSR 系统额外依赖
cd ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages
pip install -r requirements.txt
```

## 核心架构

### 面部模型结构 (FaceXModel/)

```
FaceXModel/
├── generic_neutral_mesh.obj    # 基础中性网格 (26719 顶点)
├── identity000-099.obj         # 100 个 PCA 身份模式
├── *_L.obj, *_R.obj            # 53 个表情混合形状 (左右分离)
└── vertex_indices.json         # 顶点索引映射
```

**几何分区:**
| 区域 | 顶点索引 | 说明 |
|------|---------|------|
| Face | [0:9408] | 面部主体 |
| Head and Neck | [9409:11247] | 头颈部 |
| Mouth socket | [11248:13293] | 口腔 |
| Eye sockets | [13294:14061] | 眼眶 (左右) |
| **Gums and tongue** | [14062:17038] | 牙龈和舌头 |
| Teeth | [17039:21450] | 牙齿 (32 颗) |
| Eyeballs | [21451:24590] | 眼球 (左右) |
| Eyelashes | [25351:26718] | 睫毛 (左右) |

**舌头控制点:**
- `T4` (16661): 后部 - 最靠后
- `T3` (16696): 背部 - 中后部
- `T2` (16755): 叶部 - 中前部
- `T1` (16758): 尖部 - 最靠前

### 数据流

```
音频 (.wav) ──► WavLM LoRA ──► EMA .npy (舌头运动)
                              │
                              ▼
BEAT JSON ◄── TextGrid ◄── Ground Truth
     │                         │
     ▼                         ▼
process_beat_data()    load_ema_motion()
     │                         │
     └──────────┬──────────────┘
                ▼
         FaceKitTongueRig.deform()
                │
                ▼
          PyrenderRenderer ──► MP4 (25 fps)
                │
                ▼
         AutoAVSR (VSR) ──► 转录文本
                │
                ▼
            WER 分数
```

## 常用命令

### 面部模型基础操作

```bash
# 加载和查看模型
cd toolkit_scripts
python extract_ground_truth.py

# 读取身份模式
cd Scripts
python read_identity.py

# 生成随机样本
python sample_random.py
```

### 舌头动画生成

```bash
cd tongue_scripts

# 单个动画渲染 (25 fps 用于 VSR)
uv run python test_tongue_grid_search_25fps.py

# 批量渲染
uv run python batch_render_corrected.py

# 参数网格搜索 (27 种配置)
uv run python test_tongue_grid_search_25fps.py

# 分析下颌-舌头时序
uv run python jaw_tongue_sync_analysis.py --dataset-id 1_wayne_0_75_75

# 音素级延迟探测
uv run python phoneme_lag_probe.py --dataset-id 1_wayne_0_75_75 --clip-idx 63

# 交互式真实值编辑器
uv run python ground_truth_tools/tongue_gt_editor.py --dataset-id 1_wayne_0_75_75
```

### VSR 评估

```bash
cd ADFA_EVALUATION

# 运行 VSR 推理
uv run python Visual_Speech_Recognition_for_Multiple_Languages/infer.py \
    config_filename=configs/LRS3_V_WER19.1.ini \
    data_filename=tongue_scripts/batch_videos/video.mp4 \
    detector=mediapipe

# 计算 WER
uv run python compute_wer.py \
    --predicted-root transcripts/ \
    --textgrid-root Visual_Speech_Recognition_for_Multiple_Languages/data/beat_textgrids/

# 批量 WER 评估
uv run python jiwer_directory_wer.py
```

### BEAT 数据集

```bash
cd ADFA_EVALUATION

# 下载 BEAT 数据
uv run python download_beat_data.py

# 规范化转录文本
uv run python normalize_transcripts.py

# 可视化 WER 结果
uv run python visualize_wer.py
```

## 关键配置

### 舌头形变参数 (`TONGUE_CONFIG`)

| 参数 | 默认值 | 范围 | 效果 |
|------|--------|------|------|
| `rotation_deg` | 5 | 0-20 | 舌头俯仰角 |
| `thickness` | 1.2 | 1.0-4.0 | 舌头垂直厚度 |
| `std_scalar` | 0.20 | 0.1-0.4 | 运动幅度缩放 |
| `shift_y` | 0 | -5 到 +5 | 垂直偏移 |
| `shift_z` | 0 | -5 到 +5 | 前后偏移 |

### 帧率设置 (CRITICAL)

**VSR 模型期望 25fps。** 使用 50fps + `speed_rate=2.0` 会导致输出乱码。

| FPS | speed_rate | VSR 结果 |
|-----|------------|----------|
| 25 | 1.0 | ✅ 正确 |
| 50 | 2.0 | ❌ 乱码 (时间混叠) |

始终使用 `batch_render_corrected.py` (25fps) 或 `test_tongue_grid_search_25fps.py` 进行 VSR 评估。

## 文件依赖

### 输入文件 (每个片段)

| 文件 | 路径 | 来源 |
|------|------|------|
| EMA 运动 | `outputs/{clip_id}.npy` | WavLM 推理 |
| BEAT 混合形状 | `inputs/{clip_id}.json` | BEAT 数据集 |
| 音频 | `inputs/{clip_id}.wav` | BEAT 数据集 |
| 归一化 | `normalising_vectors/JW13_4points_std.npy` | 训练数据 |

### 模型文件

| 文件 | 路径 | 大小 |
|------|------|------|
| 面部模型 | `FaceXModel/` | ~100MB |
| VSR 模型 | `ADFA_EVALUATION/.../LRS3_V_WER19.1/` | ~891MB |
| 语言模型 | `ADFA_EVALUATION/.../lm_en_subword/` | ~191MB |

## 延迟优化工作流

### 1. 分析当前时序

```bash
uv run python jaw_tongue_sync_analysis.py --dataset-id 1_wayne_0_75_75
# 输出: jaw_tongue_sync/1_wayne_0_75_75_jaw_tongue_sync.json
# 关键字段: corr_at_zero, best_lag_s, corr_at_best
```

### 2. 定义真实值 (可选)

```bash
uv run python ground_truth_tools/tongue_gt_editor.py --dataset-id 1_wayne_0_75_75
# 输出: *_tongue_gt.json (每个音素的锚点位置)
```

### 3. 查找最优全局偏移

```bash
uv run python ground_truth_tools/tongue_gt_compare.py \
    --gt-json jaw_tongue_sync/clip_tongue_gt.json \
    --max-shift-s 0.5
# 输出: best_global_shift_s (例如 +0.05s = 舌头延迟 50ms)
```

### 4. 渲染偏移动画

```bash
uv run python jaw_tongue_sync_render_shift.py \
    --dataset-id 1_wayne_0_75_75 \
    --shift-seconds 0.05
```

### 5. 评估 WER

```bash
uv run python ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py \
    config_filename=configs/LRS3_V_WER19.1.ini \
    data_filename=shifted.mp4 detector=mediapipe

uv run python ADFA_EVALUATION/compute_wer.py \
    --predicted-root transcripts/ \
    --textgrid-root ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/data/beat_textgrids/
```

## 已知问题和修复

### 问题: VSR 输出重复乱码
**原因:** 50fps 渲染 + `speed_rate=2.0`
**修复:** 使用 25fps 脚本 (`batch_render_corrected.py`, `test_tongue_grid_search_25fps.py`)

### 问题: 嘴唇不能完全闭合
**原因:** `jawOpen` 混合形状有非零最小值
**修复:** 在 `batch_render_corrected.py` 中应用最小值校正:
```python
face_seq[:, jaw_idx] = np.maximum(0, face_seq[:, jaw_idx] - np.min(face_seq[:, jaw_idx]))
```

### 问题: 舌头运动太小/太大
**原因:** `std_scalar` 未校准
**修复:** 运行网格搜索或手动调整 (典型范围 0.15-0.25)

### 问题: 全局相关性接近零
**原因:** 下颌和舌头有不同的发音功能
**修复:** 使用音素级分析 (`phoneme_lag_probe.py`) 而非全局相关性

## 成功指标

| WER | 质量 |
|-----|------|
| < 20% | 优秀 (接近人类唇读) |
| 20-40% | 良好 (可理解) |
| 40-60% | 中等 (可见问题) |
| > 60% | 差 (主要缺陷) |

**当前基线:** ~100% (乱码) 使用损坏的 50fps 管道
**目标:** <50% 使用校正的 25fps + 优化时序

## 目录结构

```
ICT-FaceKit/
├── FaceXModel/                          # 面部模型 OBJ 文件
├── tongue_scripts/                      # 舌头动画系统
│   ├── tongue_animation/                # 核心动画模块
│   │   ├── face_model_io_trimesh.py    # 模型 I/O
│   │   ├── generate_tongue_animation.py # 生成动画
│   │   └── render_face_animation_trimesh.py # 渲染
│   ├── ground_truth_tools/              # 真实值工具
│   ├── jaw_tongue_sync_script/          # 时序分析
│   └── batch_videos/                    # 批量输出
├── toolkit_scripts/                     # 基础工具
├── ADFA_EVALUATION/                     # VSR 评估系统
│   └── Visual_Speech_Recognition_for_Multiple_Languages/
│       ├── configs/                     # VSR 模型配置
│       ├── pipelines/                   # 推理管道
│       └── benchmarks/                  # 评估数据集
└── evaluation_script/                   # 评估脚本
```

## 表情混合形状命名

表情混合形状使用 Apple ARKit 命名约定,但使用 `_L` 和 `_R` 指定左右。

| FACS 单位 | 对应形状 |
|-----------|----------|
| AU1 内眉上抬 | browInnerUp_L + browInnerUp_R |
| AU4 眉下压 | browDown_L + browDown_R |
| AU12 嘴角拉动 | mouthSmile_L + mouthSmile_R |
| AU27 嘴伸展 | jawOpen |
| AU45 眨眼 | eyeBlink_L + eyeBlink_R |

完整映射见 README.md。

## 引用

如果在研究中使用 ICT-FaceKit,请引用:

```bibtex
@misc{li2020learning,
title={Learning Formation of Physically-Based Face Attributes},
author={Ruilong Li and Karl Bladin and Yajie Zhao and Chinmay Chinara and Owen Ingraham and Pengda Xiang and Xinglei Ren and Pratusha Prasad and Bipin Kishore and Jun Xing and Hao Li},
year={2020},
eprint={2004.03458},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
```
