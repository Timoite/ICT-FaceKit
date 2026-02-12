# CLAUDE.md - tongue_scripts

This file provides guidance for Claude Code (claude.ai/code) when working with the tongue animation pipeline.

## Goal This Week

**Optimize and offset tongue animation lag to improve VSR/WER evaluation scores.**

The rendered facial animations should be reasonably intelligible (lower WER = better lip/tongue sync).

## Quick Reference

### Key Scripts by Task

| Task | Script | Description |
|------|--------|-------------|
| **Render animation** | `test.py` | Main tongue rig + face animation renderer |
| **Batch render** | `batch_render_corrected.py` | Render multiple videos with 25fps |
| **Grid search** | `test_tongue_grid_search_25fps.py` | Parameter sweep for optimal WER |
| **Analyze lag** | `jaw_tongue_sync_analysis.py` | Correlation analysis between jaw and tongue |
| **Phoneme probe** | `phoneme_lag_probe.py` | Per-phoneme lag estimation |
| **GT editor** | `ground_truth_tools/tongue_gt_editor.py` | Interactive ground truth tongue positions |
| **GT compare** | `ground_truth_tools/tongue_gt_compare.py` | Compare EMA vs ground truth, find optimal shift |
| **Render shifted** | `jaw_tongue_sync_render_shift.py` | Render with time-shifted tongue |

### Common Commands

```bash
# Render single animation
cd /home/timoite/Documents/ICT-FaceKit
uv run python tongue_scripts/test.py

# Batch render videos (25fps for VSR)
uv run python tongue_scripts/batch_render_corrected.py

# Run grid search to find optimal parameters
uv run python tongue_scripts/test_tongue_grid_search_25fps.py

# Analyze jaw-tongue timing on a clip
uv run python tongue_scripts/jaw_tongue_sync_analysis.py --dataset-id 1_wayne_0_75_75

# Probe per-phoneme lag
uv run python tongue_scripts/phoneme_lag_probe.py --dataset-id 1_wayne_0_75_75 --clip-idx 63

# Compare EMA against ground truth
uv run python tongue_scripts/ground_truth_tools/tongue_gt_compare.py --gt-json jaw_tongue_sync/clip_gt.json

# Render with shifted tongue
uv run python tongue_scripts/jaw_tongue_sync_render_shift.py --shift-seconds 0.05
```

## Architecture

### Data Flow

```
Audio (.wav) ──► WavLM LoRA ──► EMA .npy (tongue motion)
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

### Core Classes

#### `FaceKitTongueRig` (test.py)
Hybrid spline-bone tongue deformation rig.

```python
rig = FaceKitTongueRig(
    vertices,           # Face mesh vertices
    faces,              # Face mesh faces
    TONGUE_SLICE,       # slice(16611, 17039)
    ANCHOR_INDICES,     # [16661, 16696, 16755, 16758]
    BONE_INDICES,       # [16661, 16757]
    TONGUE_CONFIG       # Dict with rotation_deg, thickness, std_scalar
)

# Deform tongue for frame i
deformed_verts, bone_mats, spline = rig.deform(ema_seq[i])
```

#### Key Functions

```python
# Load BEAT blendshapes and resample to target FPS
face_seq = process_beat_data(json_path, face_model, target_fps=50)

# Load and denormalize WavLM EMA output
ema_seq = load_ema_motion(npy_path, std_path, rig_anchors, std_scalar)

# Apply jawOpen min correction for lip closure
face_seq[:, jaw_idx] = np.maximum(0, face_seq[:, jaw_idx] - min_val)
```

## Configuration

### Tongue Rig Parameters (`TONGUE_CONFIG`)

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `rotation_deg` | 5 | 0-20 | Tongue pitch angle |
| `thickness` | 1.2 | 1.0-4.0 | Vertical tongue thickness |
| `std_scalar` | 0.20 | 0.1-0.4 | Motion amplitude scaling |
| `shift_y` | 0 | -5 to +5 | Vertical offset |
| `shift_z` | 0 | -5 to +5 | Forward/back offset |

### Vertex Indices

| Region | Indices | Notes |
|--------|---------|-------|
| Tongue mesh | 16611-17038 | `TONGUE_SLICE` |
| Tongue anchor T4 (back) | 16661 | Most posterior |
| Tongue anchor T3 (dorsum) | 16696 | Middle-back |
| Tongue anchor T2 (blade) | 16755 | Middle-front |
| Tongue anchor T1 (tip) | 16758 | Most anterior |
| Gums | 14062-17038 | Surrounds tongue |

### EMA Output Format

WavLM outputs `.npy` with shape `(T, 8)`:
- Reshaped to `(T, 4, 2)` - 4 control points × 2D deltas
- Expanded to `(T, 4, 3)` with:
  - X = rig anchor X (fixed)
  - Y = anchor Y + delta[:, 1] (vertical)
  - Z = anchor Z + delta[:, 0] (forward/back)

## Frame Rate (CRITICAL)

**VSR model expects 25fps.** The 50fps scripts with `speed_rate=2.0` cause gibberish output.

| FPS | speed_rate | VSR Result |
|-----|------------|------------|
| 25 | 1.0 | ✅ Correct |
| 50 | 2.0 | ❌ Gibberish (temporal aliasing) |

Always use `batch_render_corrected.py` (25fps) or `test_tongue_grid_search_25fps.py` for VSR evaluation.

## Lag Optimization Workflow

### Step 1: Analyze Current Timing

```bash
# Analyze jaw-tongue correlation
uv run python tongue_scripts/jaw_tongue_sync_analysis.py \
    --dataset-id 1_wayne_0_75_75 \
    --segment-duration 10.0

# Output: jaw_tongue_sync/1_wayne_0_75_75_jaw_tongue_sync.json
# Key fields: corr_at_zero, best_lag_s, corr_at_best
```

### Step 2: Define Ground Truth (Optional)

```bash
# Interactive editor for MRI-grounded tongue positions
uv run python tongue_scripts/ground_truth_tools/tongue_gt_editor.py \
    --dataset-id 1_wayne_0_75_75

# Output: *_tongue_gt.json with per-phoneme anchor positions
```

### Step 3: Find Optimal Global Shift

```bash
# Compare EMA vs GT, sweep global shift
uv run python tongue_scripts/ground_truth_tools/tongue_gt_compare.py \
    --gt-json jaw_tongue_sync/clip_tongue_gt.json \
    --max-shift-s 0.5

# Output: best_global_shift_s (e.g., +0.05s = delay tongue by 50ms)
```

### Step 4: Render with Shift

```bash
# Apply shift and render
uv run python tongue_scripts/jaw_tongue_sync_render_shift.py \
    --dataset-id 1_wayne_0_75_75 \
    --shift-seconds 0.05 \
    --output batch_videos/shifted.mp4
```

### Step 5: Evaluate WER

```bash
# Run VSR on rendered video
uv run python ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py \
    config_filename=configs/LRS3_V_WER19.1.ini \
    data_filename=tongue_scripts/batch_videos/shifted.mp4 \
    detector=mediapipe

# Compute WER vs ground truth
uv run python ADFA_EVALUATION/compute_wer.py \
    --predicted-root transcripts/ \
    --textgrid-root ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/data/beat_textgrids/
```

## Grid Search for Parameters

The grid search script tests combinations of tongue parameters:

```bash
uv run python tongue_scripts/test_tongue_grid_search_25fps.py
```

**Parameter Grid (27 configs):**
- `rotation_deg`: [0, 10, 20]
- `thickness`: [1.0, 2.0, 4.0]
- `std_scalar`: [0.10, 0.25, 0.40]

**Output:** `tongue_param_tests_25fps/` with:
- `rot00_thick1.0_std0.10/animation_with_audio.mp4`
- `rot00_thick1.0_std0.10/transcript.txt`
- `wer_report.csv` (sorted by WER)

## Known Issues & Fixes

### Issue: VSR Produces Repetitive Gibberish
**Cause:** 50fps rendering with `speed_rate=2.0`
**Fix:** Use 25fps scripts (`batch_render_corrected.py`, `test_tongue_grid_search_25fps.py`)

### Issue: Lips Don't Close Fully
**Cause:** `jawOpen` blendshape has non-zero minimum
**Fix:** Apply min-shift correction in `batch_render_corrected.py`:
```python
face_seq[:, jaw_idx] = np.maximum(0, face_seq[:, jaw_idx] - np.min(face_seq[:, jaw_idx]))
```

### Issue: Tongue Motion Too Small/Large
**Cause:** `std_scalar` not calibrated
**Fix:** Run grid search or adjust manually (0.15-0.25 typical range)

### Issue: Global Correlation Near Zero
**Expected:** Jaw and tongue serve different articulatory roles
**Fix:** Use phoneme-level analysis (`phoneme_lag_probe.py`) instead of global correlation

## File Dependencies

### Input Files (per clip)

| File | Path | Source |
|------|------|--------|
| EMA motion | `outputs/{clip_id}.npy` | WavLM inference |
| BEAT blendshapes | `inputs/{clip_id}.json` | BEAT dataset |
| Audio | `inputs/{clip_id}.wav` | BEAT dataset |
| Normalization | `normalising_vectors/JW13_4points_std.npy` | Training data |

### Model Files

| File | Path | Size |
|------|------|------|
| Face model | `FaceXModel/` | ~100MB |
| VSR model | `ADFA_EVALUATION/.../LRS3_V_WER19.1/` | ~891MB |
| Language model | `ADFA_EVALUATION/.../lm_en_subword/` | ~191MB |

## Success Metrics

| WER | Quality |
|-----|---------|
| < 20% | Excellent (near human lip-reading) |
| 20-40% | Good (intelligible) |
| 40-60% | Moderate (visible issues) |
| > 60% | Poor (major artifacts) |

**Current baseline:** ~100% (gibberish) with broken 50fps pipeline
**Target:** <50% with corrected 25fps + optimized timing
