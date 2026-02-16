# CLAUDE.md - tongue_scripts

This file provides guidance to Claude Code (claude.ai/code) when working with the tongue animation pipeline.

## 目标

**优化和偏移舌头动画延迟以提高 VSR/WER 评估分数。**

渲染的面部动画应该足够清晰(WER 越低 = 唇/舌同步越好)。

## 快速参考

### 按任务分类的关键脚本

| 任务 | 脚本 | 描述 |
|------|------|------|
| **生成 EMA** | `invert.py` | 从单个音频文件生成 EMA (.npy) - 简单脚本 |
| **批量生成 EMA** | `wavlm_lora.py` | WavLM 模型包装器,支持批量处理 |
| **渲染动画** | `generate_tongue_animation.py` | 主舌头绑定 + 面部动画渲染器 |
| **网格搜索** | `test_tongue_grid_search_25fps.py` | 最佳 WER 参数扫描 |
| **分析延迟** | `jaw_tongue_sync_analysis.py` | 下颌和舌头之间的相关性分析 |
| **音素探测** | `phoneme_lag_probe.py` | 每音素延迟估计 |
| **GT 编辑器** | `ground_truth_tools/tongue_gt_editor.py` | 交互式真实值舌头位置 |
| **GT 比较** | `ground_truth_tools/tongue_gt_compare.py` | 比较 EMA 与真实值,查找最优偏移 |
| **渲染偏移** | `jaw_tongue_sync_render_shift.py` | 使用时间偏移的舌头渲染 |

### 常用命令

```bash
# 从单个音频文件生成 EMA (简单脚本)
# 编辑 invert.py 中的输入/输出路径,然后运行:
uv run python tongue_scripts/invert.py

# 批量处理: 编辑 invert.py 中的 base_dir 并取消注释批量处理代码块

# 渲染单个动画 (25 fps 用于 VSR)
uv run python tongue_scripts/tongue_animation/generate_tongue_animation.py

# 批量渲染视频 (25 fps)
uv run python tongue_scripts/test_tongue_grid_search_25fps.py

# 运行网格搜索查找最优参数
uv run python tongue_scripts/test_tongue_grid_search_25fps.py

# 分析下颌-舌头时序
uv run python tongue_scripts/jaw_tongue_sync_analysis.py --dataset-id 1_wayne_0_75_75

# 探测每音素延迟
uv run python tongue_scripts/phoneme_lag_probe.py --dataset-id 1_wayne_0_75_75 --clip-idx 63

# 交互式真实值编辑
uv run python tongue_scripts/ground_truth_tools/tongue_gt_editor.py --dataset-id 1_wayne_0_75_75

# 渲染带偏移的动画
uv run python tongue_scripts/jaw_tongue_sync_script/jaw_tongue_sync_render_shift.py \
    --shift-seconds 0.05
```

## 架构

### 数据流

```
音频 (.wav) ──► WavLM LoRA ──► EMA .npy (舌头运动)
                                      │
                                      ▼
BEAT JSON ◄── TextGrid ◄── Ground     │
     │                                 │
     ▼                                 ▼
process_beat_data()           load_ema_motion()
     │                                 │
     ▼                                 ▼
  face_seq[] ◄─────────────► ema_seq[T,4,3]
     │                                 │
     └──────────┬──────────────────────┘
                ▼
         FaceKitTongueRig.deform()
                │
                ▼
          Deformed mesh
                │
                ▼
         PyrenderRenderer ──► MP4 (25fps)
                │
                ▼
         AutoAVSR (VSR) ──► Transcript
                │
                ▼
            WER Score
```

### 核心类

#### `FaceKitTongueRig` (generate_tongue_animation.py)
混合样条-骨骼舌头变形绑定。

```python
rig = FaceKitTongueRig(
    vertices,           # 面部网格顶点
    faces,              # 面部网格面
    TONGUE_SLICE,       # slice(16611, 17039)
    ANCHOR_INDICES,     # [16661, 16696, 16755, 16758]
    BONE_INDICES,       # [16661, 16757]
    TONGUE_CONFIG       # 包含 rotation_deg, thickness, std_scalar 的字典
)

# 变形第 i 帧的舌头
deformed_verts, bone_mats, spline = rig.deform(ema_seq[i])
```

#### `TrimeshFaceModel` (face_model_io_trimesh.py)
面部模型容器,处理混合形状变形。

```python
face_model = load_face_model_trimesh(
    FACE_MODEL_DIR,
    load_identity=False,  # 身份模式可选
    load_expressions=True
)

# 使用混合形状权重变形
deformed_verts = face_model.deform({
    'jawOpen': 0.5,
    'mouthSmile_L': 0.3,
    'mouthSmile_R': 0.3,
    # ...
})
```

### 关键函数

```python
# 加载 BEAT 混合形状并重采样到目标 FPS
face_seq = process_beat_data(json_path, face_model, target_fps=25)

# 加载并反归一化 WavLM EMA 输出
ema_seq = load_ema_motion(npy_path, std_path, rig_anchors, std_scalar)

# 应用 jawOpen 最小值校正以实现嘴唇闭合
face_seq[:, jaw_idx] = np.maximum(0, face_seq[:, jaw_idx] - min_val)
```

## 配置

### 舌头绑定参数 (`TONGUE_CONFIG`)

| 参数 | 默认值 | 范围 | 效果 |
|------|--------|------|------|
| `rotation_deg` | 5 | 0-20 | 舌头俯仰角 |
| `thickness` | 1.2 | 1.0-4.0 | 垂直舌头厚度 |
| `std_scalar` | 0.20 | 0.1-0.4 | 运动幅度缩放 |
| `shift_y` | 0 | -5 到 +5 | 垂直偏移 |
| `shift_z` | 0 | -5 到 +5 | 前后偏移 |

### 顶点索引

| 区域 | 索引 | 说明 |
|------|------|------|
| 舌头网格 | 16611-17038 | `TONGUE_SLICE` |
| 舌头锚点 T4 (后部) | 16661 | 最靠后 |
| 舌头锚点 T3 (背部) | 16696 | 中后部 |
| 舌头锚点 T2 (叶部) | 16755 | 中前部 |
| 舌头锚点 T1 (尖部) | 16758 | 最靠前 |
| 牙龈 | 14062-17038 | 包围舌头 |

### EMA 输出格式

WavLM 输出形状为 `(T, 8)` 的 `.npy`:
- 重塑为 `(T, 4, 2)` - 4 个控制点 × 2D 增量
- 扩展为 `(T, 4, 3)`:
  - X = rig 锚点 X (固定)
  - Y = 锚点 Y + delta[:, 1] (垂直)
  - Z = 锚点 Z + delta[:, 0] (前后)

## 帧率 (CRITICAL)

**VSR 模型期望 25fps。** 使用 `speed_rate=2.0` 的 50fps 脚本会导致乱码输出。

| FPS | speed_rate | VSR 结果 |
|-----|------------|----------|
| 25 | 1.0 | ✅ 正确 |
| 50 | 2.0 | ❌ 乱码 (时间混叠) |

始终使用 `test_tongue_grid_search_25fps.py` 进行 VSR 评估。

## 延迟优化工作流

### 步骤 1: 分析当前时序

```bash
# 分析下颌-舌头相关性
uv run python tongue_scripts/jaw_tongue_sync_analysis.py \
    --dataset-id 1_wayne_0_75_75 \
    --segment-duration 10.0

# 输出: jaw_tongue_sync/1_wayne_0_75_75_jaw_tongue_sync.json
# 关键字段: corr_at_zero, best_lag_s, corr_at_best
```

### 步骤 2: 定义真实值 (可选)

```bash
# MRI 基础舌头位置的交互式编辑器
uv run python tongue_scripts/ground_truth_tools/tongue_gt_editor.py \
    --dataset-id 1_wayne_0_75_75

# 输出: *_tongue_gt.json,包含每音素锚点位置
```

### 步骤 3: 查找最优全局偏移

```bash
# 比较 EMA 与 GT,扫描全局偏移
uv run python tongue_scripts/ground_truth_tools/tongue_gt_compare.py \
    --gt-json jaw_tongue_sync/clip_tongue_gt.json \
    --max-shift-s 0.5

# 输出: best_global_shift_s (例如 +0.05s = 舌头延迟 50ms)
```

### 步骤 4: 渲染带偏移

```bash
# 应用偏移并渲染
uv run python tongue_scripts/jaw_tongue_sync_script/jaw_tongue_sync_render_shift.py \
    --dataset-id 1_wayne_0_75_75 \
    --shift-seconds 0.05 \
    --output batch_videos/shifted.mp4
```

### 步骤 5: 评估 WER

```bash
# 在渲染的视频上运行 VSR
cd ../ADFA_EVALUATION
uv run python Visual_Speech_Recognition_for_Multiple_Languages/infer.py \
    config_filename=configs/LRS3_V_WER19.1.ini \
    data_filename=tongue_scripts/batch_videos/shifted.mp4 \
    detector=mediapipe

# 与真实值计算 WER
uv run python compute_wer.py \
    --predicted-root transcripts/ \
    --textgrid-root Visual_Speech_Recognition_for_Multiple_Languages/data/beat_textgrids/
```

## 参数网格搜索

网格搜索脚本测试舌头参数组合:

```bash
uv run python tongue_scripts/test_tongue_grid_search_25fps.py
```

**参数网格 (27 种配置):**
- `rotation_deg`: [0, 10, 20]
- `thickness`: [1.0, 2.0, 4.0]
- `std_scalar`: [0.10, 0.25, 0.40]

**输出:** `grid/` 目录,包含:
- `rot00_thick1.0_std0.10/animation_with_audio.mp4`
- `rot00_thick1.0_std0.10/transcript.txt`
- `wer_report.csv` (按 WER 排序)

## 已知问题和修复

### 问题: VSR 输出重复乱码
**原因:** 50fps 渲染 + `speed_rate=2.0`
**修复:** 使用 25fps 脚本 (`test_tongue_grid_search_25fps.py`)

### 问题: 嘴唇不能完全闭合
**原因:** `jawOpen` 混合形状有非零最小值
**修复:** 在渲染脚本中应用最小值平移:
```python
face_seq[:, jaw_idx] = np.maximum(0, face_seq[:, jaw_idx] - np.min(face_seq[:, jaw_idx]))
```

### 问题: 舌头运动太小/太大
**原因:** `std_scalar` 未校准
**修复:** 运行网格搜索或手动调整 (典型范围 0.15-0.25)

### 问题: 全局相关性接近零
**原因:** 下颌和舌头有不同的发音功能
**修复:** 使用音素级分析 (`phoneme_lag_probe.py`) 而非全局相关性

## 文件依赖

### 输入文件 (每个片段)

| 文件 | 路径 | 来源 |
|------|------|------|
| EMA 运动 | `outputs/{clip_id}.npy` | WavLM 推理 |
| BEAT 混合形状 | `inputs/{clip_id}.json` | BEAT 数据集 |
| 音频 | `inputs/{clip_id}.wav` | BEAT 数据集 |
| 归一化 | `normalising_vectors/JW13_4points_std.npy` | 训练数据 |
| TextGrid | `../ADFA_EVALUATION/.../beat_textgrids/{clip_id}.TextGrid` | 音素级时间对齐 |

### 模型文件

| 文件 | 路径 | 大小 |
|------|------|------|
| 面部模型 | `../FaceXModel/` | ~100MB |
| VSR 模型 | `../ADFA_EVALUATION/.../LRS3_V_WER19.1/` | ~891MB |
| 语言模型 | `../ADFA_EVALUATION/.../lm_en_subword/` | ~191MB |

## 目录结构

```
tongue_scripts/
├── tongue_animation/              # 核心动画模块
│   ├── face_model_io_trimesh.py  # 面部模型 I/O (TrimeshFaceModel)
│   ├── generate_tongue_animation.py  # 舌头绑定 (FaceKitTongueRig)
│   └── render_face_animation_trimesh.py  # 渲染管线
├── ground_truth_tools/            # 真实值工具
│   ├── tongue_gt_editor.py       # 交互式编辑器 (matplotlib)
│   └── tongue_gt_compare.py      # EMA vs GT 比较
├── jaw_tongue_sync_script/        # 时序分析
│   ├── jaw_tongue_sync_analysis.py  # 相关性分析
│   └── jaw_tongue_sync_render_shift.py  # 偏移渲染
├── normalising_vectors/           # 归一化统计
├── batch_videos/                  # 批量输出
├── grid/                          # 网格搜索结果
├── inputs/                        # 输入数据 (BEAT + 音频)
├── outputs/                       # 输出数据 (EMA .npy)
├── inversion_checkpoints/         # WavLM LoRA 模型权重
├── invert.py                      # 音频→EMA 简单脚本 (单文件/批量)
├── wavlm_lora.py                  # WavLM 模型包装器
├── test_tongue_grid_search_25fps.py  # 网格搜索
├── phoneme_lag_probe.py           # 音素级延迟分析
├── same_phoneme_comparison.py     # 同音素比较
└── tongue_animation.py            # 独立舌头动画预览
```

## 音频反演 (EMA 生成)

### invert.py

简单的音频到 EMA 反演脚本。

**功能:**
- 加载预训练的 WavLM LoRA 模型 (`inversion_checkpoints/lora_multispeaker_*.pth`)
- 从音频文件生成 4 点舌头运动
- 应用 10Hz 低通滤波器 (50fps 采样率)
- 保存为 `.npy` 格式

**使用方法:**

1. 单文件处理 (默认):
```python
# 编辑 invert.py 中的路径:
data, sr = torchaudio.load("path/to/audio.wav")  # 输入音频
np.save("outputs/output.npy", ema)               # 输出 EMA

# 运行:
uv run python tongue_scripts/invert.py
```

2. 批量处理 (取消注释代码块):
```python
# 设置基础目录:
base_dir = "./26/"  # 包含 wav/ 和 npy_split/ 子目录

# 脚本会:
# - 将长音频分割为 160,000 采样段 (~10 秒 @ 16kHz)
# - 对每段运行反演
# - 保存到 npy_split/ 目录
```

**关键参数:**
- 模型: `lora_multispeaker_consistency_alpha_0.25_threshold_0_vctk_vvn_4tonguepoints`
- 滤波器: 5 阶 Butterworth 低通,10 Hz 截止频率
- 采样率: 50 fps (EMA 输出)

**输出格式:**
- 形状: `(T, 8)` - T 帧 × 8 个值 (4 点 × 2D)
- 可使用 `load_ema_motion()` 反归一化

## 成功指标

| WER | 质量 |
|-----|------|
| < 20% | 优秀 (接近人类唇读) |
| 20-40% | 良好 (可理解) |
| 40-60% | 中等 (可见问题) |
| > 60% | 差 (主要缺陷) |

**当前基线:** ~100% (乱码) 使用损坏的 50fps 管道
**目标:** <50% 使用校正的 25fps + 优化时序

## BEAT 数据集格式

BEAT JSON 格式:
```json
{
  "names": ["jawOpen", "mouthSmileLeft", ...],
  "frames": [
    [0.0, 0.5, 0.3, ...],  // 第 0 帧 (60 fps)
    [0.1, 0.6, 0.4, ...],  // 第 1 帧
    ...
  ]
}
```

## 音素类 (用于 GT 编辑器)

```python
PHONE_CLASSES = OrderedDict([
    ("VOWELS", ["IY", "IH", "EH", "AE", "AH", "UW", "UH", "AA", "AO", "OW", ...]),
    ("LIQUIDS", ["L", "R"]),
    ("NASALS", ["M", "N", "NG"]),
    ("STOPS", ["P", "B", "T", "D", "K", "G"]),
    ("FRICATIVES", ["F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"]),
    ("AFFRICATES", ["CH", "JH"]),
    ("GLIDES", ["W", "Y"]),
])
```
