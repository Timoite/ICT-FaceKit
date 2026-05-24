# AGENTS.md - tongue_scripts

This file provides focused guidance for work inside `tongue_scripts/`.

## What this folder is for

`tongue_scripts/` is the research workspace for:

- WavLM-based articulatory inversion
- tongue rigging and mesh deformation
- active vs passive tongue rendering
- jaw/tongue lag analysis
- manual tongue ground-truth editing
- VSR-side evaluation helpers

The current practical goal is usually:

1. make or load a tongue `.npy`
2. render a 25 fps face video
3. compare active vs passive tongue behavior
4. improve transcript quality or alignment metrics

## Current scripts that matter

### Canonical folder layout

The canonical implementations now live in grouped subfolders:

- `inversion/`: audio -> articulatory motion and WavLM model code
- `rendering/`: active/passive tongue renderers and full-face render variants
- `analysis/`: lip aperture, phoneme lag, and geometry inspection tools
- `optimization/`: L-BFGS-B experiments, scalar sweeps, and optimizer helpers
- `pipelines/`: end-to-end dataset, speaker, and real-video workflows
- `timing/`: jaw/tongue synchronization analysis utilities
- `preview/`: standalone preview and visualization helpers

A small set of top-level scripts remain on purpose as convenient front-door entry points:

- `invert.py`
- `wavlm_lora.py`
- `render_dual_tongue_comparison.py`
- `run_render_dual_for_dataset.py`
- `multi_speaker_short_pipeline.py`
- `evaluate_vsr_ver.py`
- `render_fullface_matplotlib_combo.py`

Most other scripts should be used from their grouped folders directly.

### Core generation and rendering

- `invert.py`: audio -> articulatory `.npy`
- `wavlm_lora.py`: WavLM-large + LoRA articulatory regressor
- `tongue_animation/generate_tongue_animation.py`: shared rigging utilities
- `render_dual_tongue_comparison.py`: main 25 fps active/passive comparison renderer
- `run_render_dual_for_dataset.py`: thin CLI wrapper for per-dataset rendering

### Analysis and optimization

- `timing/jaw_tongue_sync_analysis.py`: global jaw/tongue timing correlation
- `analysis/phoneme_lag_probe.py`: lag probing around a specific phoneme clip
- `analysis/lip_aperture_textgrid_plot.py`: compare articulatory lip aperture to BEAT lip aperture
- `optimization/articulation_npy_optimizer.py`: phoneme-profile over-articulation of the `.npy`

### Ground-truth tools

- `ground_truth_tools/tongue_gt_editor.py`: interactive manual tongue target editor
- `ground_truth_tools/tongue_gt_compare.py`: sweep global shifts against manual GT

### Evaluation wrappers

- `evaluate_vsr_ver.py`: run VSR and compare VER / WER across videos
- `multi_speaker_short_pipeline.py`: orchestrates download -> invert -> lag estimate -> render -> evaluate

## Files that are not the main path

- `preview/tongue_animation.py` is a standalone separated-tongue preview script, not the production render/eval path.
- The old Wayne-only VER/WER runner has been removed; use `evaluate_vsr_ver.py` or `multi_speaker_short_pipeline.py` instead.
- Older references to `batch_render_corrected.py` or `test_tongue_grid_search_25fps.py` are stale in this checkout.

## Core mental model

### Articulatory output layout

The WavLM regressor emits a 16-D sequence.

- cols `0:8`: tongue control points as 4 x `(z, y)`
- cols `8:12`: lip coordinates used by lip-aperture analysis

The renderer only needs the first 8 columns.

### Rigging path

`tongue_animation/generate_tongue_animation.py` is the source of truth for:

- `process_beat_data()`
- `load_ema_motion()`
- `FaceKitTongueRig`
- `TONGUE_CONFIG`

`FaceKitTongueRig` uses:

- tongue slice: `slice(16611, 17039)`
- anchors: `[16661, 16696, 16755, 16758]`
- bone endpoints: `[16661, 16757]`

### Time bases

- BEAT JSON blendshapes: 60 fps source
- WavLM articulatory output: 50 fps
- VSR/render target: 25 fps

Positive tongue shift means delay.

## Most useful commands

### Audio -> `.npy`

```bash
uv run python tongue_scripts/invert.py \
  --wav /abs/path/clip.wav \
  --out /abs/path/clip.npy
```

### Render active vs passive tongue

```bash
uv run python tongue_scripts/run_render_dual_for_dataset.py \
  --dataset-id 1_wayne_0_75_75 \
  --speaker-id 1 \
  --beat-root /home/iite/ICT-FaceKit/data/beat_cache/beat_english_v0.2.1/beat_english_v0.2.1 \
  --motion-path /home/iite/ICT-FaceKit/tongue_scripts/outputs/1_wayne_0_75_75.npy \
  --output-dir /home/iite/ICT-FaceKit/tongue_scripts/outputs \
  --tongue-shift-seconds 0.12
```

### Timing analysis

```bash
uv run python tongue_scripts/timing/jaw_tongue_sync_analysis.py --dataset-id 1_wayne_0_75_75
uv run python tongue_scripts/analysis/phoneme_lag_probe.py --dataset-id 1_wayne_0_75_75 --clip-idx 63
```

### Manual GT workflow

```bash
uv run python tongue_scripts/ground_truth_tools/tongue_gt_editor.py --dataset-id 1_wayne_0_75_75
uv run python tongue_scripts/ground_truth_tools/tongue_gt_compare.py --gt-json /abs/path/clip_tongue_gt.json
```

### Multi-speaker benchmark

```bash
uv run python tongue_scripts/multi_speaker_short_pipeline.py \
  --num-speakers 5 \
  --infer-mode segmented \
  --vowel-mode grouped
```

## Render/eval rules worth preserving

- Keep the active/passive comparison renderer at 25 fps for VSR work.
- Keep `jawOpen` minimum-offset correction enabled unless the experiment explicitly studies the uncorrected baseline.
- `render_dual_tongue_comparison.py` is the current top-level convenience entry point for the grouped renderer and already bakes in both of those assumptions.
- `ffmpeg` is required for audio muxing and the VSR prep flow.

## Data paths you will touch most often

- BEAT cache: `data/beat_cache/beat_english_v0.2.1/beat_english_v0.2.1/<speaker>/`
- predicted motion: `tongue_scripts/outputs/<dataset>.npy`
- rendered videos: `tongue_scripts/outputs/*.mp4`
- multi-speaker results: `tongue_scripts/outputs/multi_speaker/`
- lag plots: `tongue_scripts/vis_output/` and `tongue_scripts/outputs/multi_speaker/`

## Common pitfalls

- Do not assume old script names from earlier notes still exist.
- Do not treat `preview/tongue_animation.py` as the main renderer.
- Do not forget that `infer.py` in ADFA uses Hydra `key=value`, while `infer_pipeline.py` uses normal `--flags`.
- Do not change mesh loading in a way that reorders vertices.
- Do not forget that positive shift delays the tongue.
