# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ICT-FaceKit is USC ICT's morphable face model toolkit for 3D facial animation, with a focus on tongue articulation optimization for speech intelligibility. The project renders 3D facial animations driven by BEAT dataset blendshapes and WavLM-predicted tongue motion, then evaluates intelligibility using Visual Speech Recognition (AutoAVSR).

## Commands

### Environment Setup
```bash
# Install core dependencies (uses uv)
uv sync

# Install optional feature groups
uv sync --extra pyrender    # CPU rendering
uv sync --extra wavlm       # Speech inversion model
uv sync --extra pytorch3d   # PyTorch3D rendering
uv sync --extra all         # All features
```

### Running Scripts
```bash
# Run any script in the project
uv run python tongue_scripts/<script>.py

# Run VSR inference on a video
uv run python ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py \
    config_filename=configs/LRS3_V_WER19.1.ini \
    data_filename=path/to/video.mp4 \
    detector=mediapipe

# Compute WER between transcripts and ground truth
uv run python ADFA_EVALUATION/compute_wer.py \
    --predicted-root transcripts/ \
    --textgrid-root ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/data/beat_textgrids/ \
    --report-path wer_report.csv
```

### Testing
```bash
# Run tests (if available)
uv run pytest

# Linting
uv run flake8 tongue_scripts/
uv run black --check tongue_scripts/
```

## Architecture

### Directory Structure

```
ICT-FaceKit/
├── FaceXModel/           # 3D mesh assets
│   ├── generic_neutral_mesh.obj    # Base geometry (26719 vertices)
│   ├── *.obj                        # 53 expression blendshapes
│   └── vertex_indices.json          # Expression names & vertex mappings
├── tongue_scripts/       # Tongue animation & rendering pipeline
│   ├── tongue_animation.py          # Main rendering entry script
│   ├── wavlm_lora.py                # WavLM speech inversion model
│   ├── test_tongue_grid_search_25fps.py  # Parameter optimization
│   ├── phoneme_lag_probe.py         # Per-phoneme lag analysis
│   ├── same_phoneme_comparison.py   # Cross-render comparison utility
│   ├── invert.py                    # Audio inversion utility
│   ├── ground_truth_tools/
│   │   ├── tongue_gt_editor.py      # Ground-truth editor
│   │   └── tongue_gt_compare.py     # EMA vs GT comparison
│   └── tongue_animation/
│       ├── face_model_io_trimesh.py        # Face model loader (strict vertex order)
│       ├── render_face_animation_trimesh.py # Rendering utilities
│       └── generate_tongue_animation.py     # FaceKitTongueRig + EMA loading
├── ADFA_EVALUATION/      # VSR evaluation framework
│   ├── Visual_Speech_Recognition_for_Multiple_Languages/  # AutoAVSR model
│   ├── compute_wer.py    # WER computation
│   ├── english.py        # Text normalization
│   └── visualize_wer.py  # Result visualization
└── crucial_progress_report/  # Session notes and experiment progress logs
```

### Data Flow

```
BEAT JSON (blendshapes) → map_beat_to_ict_names() → process_beat_data() → FaceModel.deform()
Audio → WavLM → .npy (tongue EMA) → load_ema_motion() → FaceKitTongueRig.deform()
Deformed meshes → PyrenderRenderer → MP4 (25fps)
MP4 → AutoAVSR (VSR) → Transcript → WER computation → Quality metric
```

## Critical Conventions

### Frame Rate (CRITICAL)
- **VSR model expects 25fps**. Rendering at 50fps with `speed_rate=2.0` causes temporal aliasing and gibberish VSR output.
- Always render at 25fps with `speed_rate=1.0` for VSR evaluation.

### Blendshape Name Mapping
BEAT blendshape names differ from ICT. Use `map_beat_to_ict_names()` from `tongue_scripts/tongue_animation/render_face_animation_trimesh.py`:
- `jawOpen` stays `jawOpen`
- `jawLeft`, `jawRight`, `mouthLeft`, `mouthRight` map directly
- Suffix `Left/Right` becomes `_L/_R` (e.g., `browInnerUpLeft` → `browInnerUp_L`)
- `browInnerUp` and `cheekPuff` map to both `_L` and `_R` variants

### Tongue Configuration
- **Tongue anchor vertices**: `[16661, 16696, 16755, 16758]` (back to tip)
- **WavLM output**: 4 control points × 2D (Y=vertical, Z=forward/back), reshaped from `[T, 8]`
- **Rig parameters**: `rotation_deg`, `thickness`, `std_scalar` (optimized via grid search)

### Face Model Vertex Order
OBJ files must be loaded with strict parsing to preserve exact vertex order. Use `_load_obj_strict()` from `tongue_scripts/tongue_animation/face_model_io_trimesh.py`. Vertex order mismatch breaks blendshape animation.

## Key Files for Common Tasks

### Rendering Animation
- `tongue_scripts/tongue_animation.py` - Main tongue rendering pipeline
- `tongue_scripts/tongue_animation/generate_tongue_animation.py` - `FaceKitTongueRig` and EMA loading
- `tongue_scripts/tongue_animation/face_model_io_trimesh.py` - `TrimeshFaceModel` class

### VSR Evaluation
- `ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py` - VSR inference
- `ADFA_EVALUATION/compute_wer.py` - WER calculation
- `ADFA_EVALUATION/english.py` - Text normalization for WER

### Tongue Optimization
- `tongue_scripts/test_tongue_grid_search_25fps.py` - Parameter grid search
- `tongue_scripts/jaw_tongue_sync_script/jaw_tongue_sync_analysis.py` - Temporal alignment analysis
- `tongue_scripts/ground_truth_tools/tongue_gt_editor.py` - Ground truth editor for tongue positions

## Model Dependencies

### Pre-trained Models Required
- **VSR Model**: `LRS3_V_WER19.1/` (891MB) - Visual-only AutoAVSR
- **Language Model**: `lm_en_subword/` (191MB) - Optional, improves VSR accuracy
- **WavLM LoRA**: `inversion_checkpoints/*.pth` - Tongue EMA prediction

Download VSR models from the [Model Zoo](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages#model-zoo).

## Current Focus

The project is currently optimizing tongue articulation to improve VSR WER. Key findings:
- Global jaw-tongue correlation is near-zero (expected due to different articulatory roles)
- Phoneme-level analysis is required for alignment optimization
- Ground truth editor (`ground_truth_tools/tongue_gt_editor.py`) enables MRI-grounded tongue position definition

## Troubleshooting

### VSR Produces Gibberish
- Check frame rate: must be 25fps with `speed_rate=1.0`
- Verify mouth visibility in video
- Ensure MediaPipe landmarks are detected

### Blendshapes Not Applied Correctly
- Check vertex order preservation in OBJ loading
- Verify blendshape name mapping with `map_beat_to_ict_names()`

### Tongue Animation Issues
- Check EMA denormalization values in `normalising_vectors/`
- Verify tongue anchor vertex indices match rig expectations
- Adjust `std_scalar` for smoother deformation
