# Real Video Pipeline Instructions

Use this note when asking Codex to run the real-video SMIRK/FaceKit pipeline so the technical defaults do not need to be restated each time.

## Default Goal

For each real talking-head video, run:

1. video/audio input
2. SMIRK FLAME tracking
3. FLAME-to-ARKit fitting
4. ARKit-to-ICT FaceKit motion export
5. WavLM tongue inversion from audio
6. lip-aperture timing analysis
7. active and passive tongue FaceKit renders
8. optional ADFA/AutoAVSR evaluation

The final render outputs should be active/passive video pairs with audio, plus the motion and timing-analysis artifacts needed to audit the run.

## Storage Rules

Large generated files should stay under:

```text
/research/milsrg1/user_workspace/ht467/smirk_task/outputs/
```

For the FADG0 dataset, use:

```text
/research/milsrg1/user_workspace/ht467/smirk_task/outputs/fadg0/<clip_id>/
```

Keep `tests/` lightweight by symlinking important outputs into it. Expected symlink names:

```text
tests/fadg0_<clip_id>_active_tongue_with_audio.mp4
tests/fadg0_<clip_id>_passive_tongue_with_audio.mp4
tests/fadg0_<clip_id>_arkit_face_motion.json
tests/fadg0_<clip_id>_lip_aperture_time_shift_analysis.json
tests/fadg0_<clip_id>_lip_aperture_time_shift_analysis.png
```

Do not copy large rendered videos into the repo unless explicitly requested.

## Current Main Entry Point

Use:

```bash
uv run python tongue_scripts/pipelines/run_fadg0_real_video_pipeline.py --video <video.mp4> --skip-existing
```

For the full FADG0 dataset:

```bash
uv run python tongue_scripts/pipelines/run_fadg0_real_video_pipeline.py --all --skip-existing
```

Default input directory:

```text
tongue_scripts/real_video/fadg0/mp4
```

Default output root:

```text
/research/milsrg1/user_workspace/ht467/smirk_task/outputs/fadg0
```

The script defaults to a smoke test when `--all` is omitted.

## Environment Rules

Use the repo `uv` environment for ICT-FaceKit, WavLM inversion, FaceKit rendering, and scoring unless there is a specific reason not to.

Use the SMIRK task environment for SMIRK fitting:

```text
/research/milsrg1/user_workspace/ht467/venvs/smirk-facekit/bin/python
```

Use the AutoAVSR/ADFA environment for ADFA inference:

```text
/research/milsrg1/user_workspace/ht467/tools/uv/adfa-vsr/bin/python
```

If dependencies are missing or incompatible, prefer creating a new task-specific `uv` venv under `/research/milsrg1/user_workspace/ht467/venvs/` instead of modifying existing working environments. Switch back to the normal ICT-FaceKit environment after the task.

## Timing and FPS Defaults

Real videos for FADG0 are 25 fps.

Use these defaults unless explicitly overridden:

```text
render fps: 25
face/ARKit fps: 25
WavLM tongue fps: 50
lip-aperture analysis fps: 50
max lip-aperture lag search: 0.5 seconds
```

Important: apply the lip-aperture timing shift to the raw 50 fps tongue `.npy` before rendering. The shifted file should be:

```text
tongue_motion_lipcorr_shifted.npy
```

This matters because a one-frame 50 fps shift can round to zero frames at 25 fps.

Positive shift follows the renderer convention: positive values delay the tongue sequence.

## Lip/Jaw Correction Policy

Do not apply the default jaw/lip closure preprocessing shift for SMIRK-converted real-video JSON.

Specifically:

- Do not call `apply_jawopen_offset_correction()` for this real-video pipeline.
- Do not subtract the `jawOpen` minimum as a lip-closure correction.
- Do apply lip-aperture correlation timing analysis by default.
- Do save the lip-aperture report and plot for every clip.

Expected timing-analysis artifacts:

```text
lip_aperture_time_shift_analysis.json
lip_aperture_time_shift_analysis.png
tongue_motion_lipcorr_shifted.npy
```

## Per-Clip Expected Outputs

For each `<clip_id>`, the organized output directory should include at least:

```text
audio_16k.wav
smirk_params.npz
smirk_flame_vertices.npz
arkit_coeffs.csv
arkit_fit_diagnostics.json
arkit_face_motion.json
ict_coeffs.npz
tongue_motion.npy
tongue_motion_lipcorr_shifted.npy
lip_aperture_time_shift_analysis.json
lip_aperture_time_shift_analysis.png
<clip_id>_active_tongue.mp4
<clip_id>_active_tongue_with_audio.mp4
<clip_id>_passive_tongue.mp4
<clip_id>_passive_tongue_with_audio.mp4
```

The active and passive `*_with_audio.mp4` files are the main deliverables for manual inspection and evaluation.

## Parallel Batch Runs

It is acceptable to process independent clips in parallel, but be conservative because SMIRK and rendering use GPU/OpenGL resources.

Recommended pattern:

- run 1-2 clips at a time on a single GPU
- use `--skip-existing` so completed SMIRK, tongue, and render artifacts can be reused
- verify output presence before launching a larger batch
- do not start many OpenGL renderers at once unless GPU memory and EGL stability have been checked

## ADFA / AutoAVSR Evaluation

Use the AutoAVSR env for inference:

```bash
cd ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages
/research/milsrg1/user_workspace/ht467/tools/uv/adfa-vsr/bin/python infer_pipeline.py \
  --video-dir <video_set_dir> \
  --output-dir <transcript_output_dir> \
  --detector mediapipe \
  --target-fps 25
```

For FADG0, ground-truth sentences are:

```text
tongue_scripts/real_video/fadg0/transcripts/<clip_id>.txt
```

When comparing rendered videos to originals, create organized video sets such as:

```text
<eval_root>/video_sets/original/<clip_id>.mp4
<eval_root>/video_sets/active/<clip_id>.mp4
<eval_root>/video_sets/passive/<clip_id>.mp4
```

Store transcripts and reports under:

```text
/research/milsrg1/user_workspace/ht467/smirk_task/outputs/fadg0_adfa_eval/
```

Use the ICT-FaceKit `uv` environment for VER/WER scoring if the AutoAVSR env lacks scoring dependencies such as `g2p_en`.

Note the decode settings in the ADFA log. Do not assume `ctc_weight=0.6`; the run must explicitly pass or log it. The previous default run used:

```text
beam_size=40, penalty=0.0, ctc_weight=0.1, lm_weight=0.0
```

## Direct SMIRK FLAME Rendering

The pre-ARKit FLAME motion files are:

```text
smirk_params.npz
smirk_flame_vertices.npz
```

SMIRK's own renderer imports PyTorch3D. If no current environment has `pytorch3d`, render the saved FLAME vertices directly with the repo's working `pyrender`/`trimesh` stack and clearly label the output as FLAME-direct pyrender.

Keep direct FLAME render outputs next to the clip artifacts and symlink inspection videos into `tests/`.

## Verification Checklist

Before reporting completion:

1. Confirm active/passive videos exist for every requested clip.
2. Confirm final videos have video and audio streams with `ffprobe`.
3. Confirm output fps is 25 for rendered/evaluation videos.
4. Confirm `lip_aperture_time_shift_analysis.json` exists.
5. Confirm `tongue_motion_lipcorr_shifted.npy` exists.
6. Confirm symlinks in `tests/` point to the large-storage outputs.
7. For generated comparison videos, decode one frame and visually inspect label placement.
8. Run focused tests when code changed:

```bash
uv run python -m pytest tests/test_real_video_pipeline.py -q
```

## What To Mention In Final Reports

Always include:

- exact output directory
- symlink paths in `tests/`
- whether lip-aperture shift was applied
- whether jaw/lip closure correction was skipped
- environment used for SMIRK and ADFA, if relevant
- any failed or unavailable dependency, such as missing PyTorch3D
- aggregate VER/WER numbers if evaluation was requested

