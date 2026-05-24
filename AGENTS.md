# AGENTS.md

This file provides guidance to Codex when working in this repository.

## Project snapshot

ICT-FaceKit in this checkout is centered on one practical research loop:

1. predict articulatory motion from audio with WavLM LoRA
2. inject the tongue motion into the ICT FaceKit mesh plus BEAT facial blendshapes
3. render comparison videos at 25 fps
4. run AutoAVSR / ADFA inference
5. score transcripts with WER and viseme-aware VER

The two directories that matter most for current work are:

- `tongue_scripts/`: tongue inversion, rigging, rendering, timing analysis, optimization
- `ADFA_EVALUATION/`: AutoAVSR inference wrappers, transcript normalization, WER utilities

## What is current in this repo

Some older docs mention scripts such as `batch_render_corrected.py` or `test_tongue_grid_search_25fps.py`. Those files are not present in this checkout. The current working entry points are:

- `tongue_scripts/invert.py`
- `tongue_scripts/render_dual_tongue_comparison.py`
- `tongue_scripts/run_render_dual_for_dataset.py`
- `tongue_scripts/multi_speaker_short_pipeline.py`
- `tongue_scripts/evaluate_vsr_ver.py`
- `tongue_scripts/timing/jaw_tongue_sync_analysis.py`
- `tongue_scripts/analysis/phoneme_lag_probe.py`
- `tongue_scripts/ground_truth_tools/tongue_gt_editor.py`
- `tongue_scripts/ground_truth_tools/tongue_gt_compare.py`
- `ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py`
- `ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer_pipeline.py`

## Key architecture

### Face model and blendshapes

- `FaceXModel/` contains the neutral mesh plus ICT expression morph targets.
- `tongue_scripts/tongue_animation/face_model_io_trimesh.py` uses a strict raw OBJ parser so vertex order is never reordered by Trimesh.
- Preserve that strict loading behavior. Blendshape deltas depend on 1:1 vertex correspondence.

### Tongue motion prediction

- `tongue_scripts/wavlm_lora.py` wraps `microsoft/wavlm-large`, swaps encoder layers to LoRA versions, and outputs a 16-D articulatory regression.
- `tongue_scripts/invert.py` is the real single-file entry point for inference from `.wav` to `.npy`.
- `invert.py` applies a low-pass filter and saves the raw model output sequence.
- The current tongue renderer uses columns `0:8` as 4 tongue anchors x `(z, y)`.
- Lip-analysis scripts also use columns `8:12` for upper/lower lip coordinates.

### Tongue rig and rendering

- `tongue_scripts/tongue_animation/generate_tongue_animation.py` contains the core reusable pieces:
  - `process_beat_data()`: maps BEAT JSON weights to ICT blendshape names and resamples to target fps
  - `load_ema_motion()`: denormalizes the WavLM output and converts it to 4 anchor points in 3D
  - `FaceKitTongueRig`: spline + bones + LBS hybrid tongue deformation
- `tongue_scripts/render_dual_tongue_comparison.py` is the current evaluation renderer:
  - renders a dynamic-tongue video and a passive-tongue control video
  - forces `FPS = 25`
  - applies `jawOpen` minimum-offset correction before rendering
  - optionally applies a global tongue delay via `TONGUE_SHIFT_SECONDS`
  - muxes audio back with `ffmpeg`
- `tongue_scripts/preview/tongue_animation.py` is a standalone preview / debugging script for the separated tongue mesh, not the main evaluation pipeline.

### ADFA / AutoAVSR inference

- `ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py` is the original Hydra entry point.
- `infer.py` must be called with Hydra-style overrides like:
  - `config_filename=configs/LRS3_V_WER19.1.ini`
  - `data_filename=/abs/path/video.mp4`
  - `detector=mediapipe`
- `ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer_pipeline.py` is the more robust wrapper for this project:
  - argparse CLI, not Hydra
  - enforces 25 fps with `ffmpeg`
  - tries to find a matching TextGrid
  - segments long clips on silence intervals from the `words` tier
  - targets about 8 second chunks by default
  - falls back to full-video inference if no TextGrid is found
  - writes transcript `.txt` files to the chosen output directory

### Evaluation

- `tongue_scripts/evaluate_vsr_ver.py` compares one or more videos against a ground-truth transcript and logs:
  - raw WER
  - normalized WER
  - viseme error rate via `evaluation_script/ver.py`
  - a composite comparison table in `tongue_scripts/outputs/vsr_composite_report.md`
- `ADFA_EVALUATION/compute_wer.py` computes word-level alignment directly from TextGrid words tiers.
- `ADFA_EVALUATION/jiwer_directory_wer.py` computes directory WER using `jiwer`.
- `ADFA_EVALUATION/normalize_transcripts.py` normalizes raw transcripts with `english.py`.

## Timing and alignment assumptions

These are important and show up repeatedly in code:

- BEAT facial JSON is treated as 60 fps in `process_beat_data()`.
- WavLM articulatory output is treated as 50 fps.
- VSR input should be 25 fps.
- Positive tongue shift means the tongue is delayed relative to jaw / face motion.
- `jawOpen` often has a non-zero floor; current renderers subtract the minimum to restore full lip closure.

### Critical indices and geometry

- Tongue slice: `slice(16611, 17039)`
- Anchor indices:
  - `16661` = T4 back
  - `16696` = T3 dorsum
  - `16755` = T2 blade
  - `16758` = T1 tip
- Bone endpoints: `[16661, 16757]`
- Gums and tongue region: vertices `14062:17039`

## Real workflows

### 1. Single clip: audio -> EMA

```bash
uv run python tongue_scripts/invert.py \
  --wav /abs/path/clip.wav \
  --out /abs/path/clip.npy
```

Notes:

- default checkpoint is `tongue_scripts/inversion_checkpoints/lora_multispeaker_consistency_alpha_0.25_threshold_0_vctk_vvn_4tonguepoints`
- this script expects Torch / Torchaudio / Transformers / LoRA dependencies to be installed

### 2. Single clip: render active vs passive tongue

```bash
uv run python tongue_scripts/run_render_dual_for_dataset.py \
  --dataset-id 1_wayne_0_75_75 \
  --speaker-id 1 \
  --beat-root /home/iite/ICT-FaceKit/data/beat_cache/beat_english_v0.2.1/beat_english_v0.2.1 \
  --motion-path /home/iite/ICT-FaceKit/tongue_scripts/outputs/1_wayne_0_75_75.npy \
  --output-dir /home/iite/ICT-FaceKit/tongue_scripts/outputs \
  --tongue-shift-seconds 0.12
```

This creates:

- `*_with_tongue.mp4`
- `*_passive_tongue.mp4`
- plus audio-muxed versions when audio is available

### 3. Single clip: VSR inference

Whole-video path:

```bash
cd ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages
uv run python infer.py \
  config_filename=configs/LRS3_V_WER19.1.ini \
  data_filename=/abs/path/video.mp4 \
  detector=mediapipe
```

Recommended segmented path:

```bash
cd ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages
uv run python infer_pipeline.py \
  --video-path /abs/path/video.mp4 \
  --textgrid-path /abs/path/video.TextGrid \
  --output-dir transcripts \
  --detector mediapipe \
  --target-fps 25 \
  --print-silence-stats
```

### 4. Compare active vs passive clips with VER + WER

```bash
uv run python tongue_scripts/evaluate_vsr_ver.py \
  --videos \
    /abs/path/with_tongue_with_audio.mp4 \
    /abs/path/passive_tongue_with_audio.mp4 \
  --ground-truth /abs/path/ground_truth.txt \
  --infer-mode segmented \
  --textgrid-path /abs/path/clip.TextGrid
```

### 5. Multi-speaker short benchmark

```bash
uv run python tongue_scripts/multi_speaker_short_pipeline.py \
  --num-speakers 5 \
  --infer-mode segmented \
  --vowel-mode grouped
```

This script orchestrates:

1. optional BEAT metadata download via `ADFA_EVALUATION/download_beat_data.py`
2. short-clip speaker selection
3. `invert.py`
4. lag estimation via `lip_aperture_textgrid_plot.py`
5. `run_render_dual_for_dataset.py`
6. `evaluate_vsr_ver.py`
7. lag summary CSV generation

### 6. Timing analysis and manual GT workflow

Use these when investigating jaw/tongue synchronization rather than raw VSR:

```bash
uv run python tongue_scripts/timing/jaw_tongue_sync_analysis.py --dataset-id 1_wayne_0_75_75
uv run python tongue_scripts/analysis/phoneme_lag_probe.py --dataset-id 1_wayne_0_75_75 --clip-idx 63
uv run python tongue_scripts/ground_truth_tools/tongue_gt_editor.py --dataset-id 1_wayne_0_75_75
uv run python tongue_scripts/ground_truth_tools/tongue_gt_compare.py --gt-json /abs/path/clip_tongue_gt.json
uv run python tongue_scripts/timing/jaw_tongue_sync_render_shift.py --dataset-id 1_wayne_0_75_75 --shift-seconds 0.05
```

## File and directory conventions

### BEAT data

The most common live path in this repo is:

- `data/beat_cache/beat_english_v0.2.1/beat_english_v0.2.1/<speaker_id>/`

Expected per-clip files:

- `<dataset_id>.wav`
- `<dataset_id>.json`
- `<dataset_id>.TextGrid`

### Tongue outputs

- predicted articulatory motion: `tongue_scripts/outputs/<dataset_id>.npy`
- rendered videos: `tongue_scripts/outputs/*.mp4`
- multi-speaker batch outputs: `tongue_scripts/outputs/multi_speaker/`
- lag analysis figures and CSVs: `tongue_scripts/outputs/multi_speaker/lag_summary.csv`, `tongue_scripts/vis_output/`, `tongue_scripts/jaw_tongue_sync/`

### Transcript outputs

- `infer_pipeline.py --video-path ...` writes `<output-dir>/<clip>.txt`
- `infer_pipeline.py --video-dir ...` mirrors the source tree under the output directory
- `compute_wer.py` expects transcripts under `speaker_*/videos/*.txt`

That last point matters: `compute_wer.py` is tied to the directory layout produced by directory-style evaluation, not arbitrary flat transcript folders.

## Important gotchas

- Do not trust older references to missing render scripts. Use the files listed in `What is current in this repo`.
- `infer.py` uses Hydra `key=value` arguments. `infer_pipeline.py` uses normal `--flags`.
- `infer_pipeline.py` depends on `ffmpeg` and `ffprobe` being available on PATH.
- The VSR pipeline is far more stable when clips are actually rendered or converted to 25 fps.
- `render_dual_tongue_comparison.py` already handles the 25 fps conversion on the rendering side.
- `infer_pipeline.py` will also enforce 25 fps before inference, which is a good safety net for external videos.
- `download_beat_data.py` only downloads JSON and TextGrid files, not the original audio.
- `compute_wer.py` builds ground-truth transcript text from the TextGrid `words` tier and writes mirrored files into `ground_truth_transcripts/`.
- `evaluation_script/ver.py` downloads NLTK resources on first use if they are missing.

## Good starting points when modifying code

- If the task is mesh deformation or render quality, start in `tongue_scripts/tongue_animation/generate_tongue_animation.py` and `tongue_scripts/render_dual_tongue_comparison.py`.
- If the task is timing or lag analysis, start in `tongue_scripts/timing/`, `tongue_scripts/analysis/phoneme_lag_probe.py`, and `tongue_scripts/analysis/lip_aperture_textgrid_plot.py`.
- If the task is transcript quality or inference behavior, start in `ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer_pipeline.py`.
- If the task is scoring, read `tongue_scripts/evaluate_vsr_ver.py`, `ADFA_EVALUATION/compute_wer.py`, and `evaluation_script/ver.py` together.

## Practical recommendations for future agents

- Prefer the 25 fps render/eval path unless you are explicitly studying resampling effects.
- Keep jaw correction enabled unless you are measuring the uncorrected baseline on purpose.
- Treat positive tongue shift as a delay.
- When adding new render or eval scripts, document whether they use:
  - 50 fps articulatory space
  - 60 fps BEAT source space
  - 25 fps VSR input space
- When touching mesh loaders, do not allow vertex reordering.
