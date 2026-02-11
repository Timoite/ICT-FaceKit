# ICT-FaceKit Repository Documentation

Generated on: 2026-02-11

## Table of Contents
1. [Root-Level Files](#root-level-files)
2. [ADFA_EVALUATION Directory](#adfa_evaluation-directory)
3. [FaceXModel Directory](#facexmodel-directory)
4. [tongue_scripts Directory](#tongue_scripts-directory)
5. [Visual_Speech_Recognition_for_Multiple_Languages](#visual_speech_recognition_for_multiple_languages)
6. [crucial_progress_report Directory](#crucial_progress_report-directory)

---

## Root-Level Files

### `/home/timoite/Documents/ICT-FaceKit/README.md`
- **File Type**: Markdown Documentation
- **Purpose**: Main project README describing ICT Face Model Light - a morphable face model toolkit from ICT's Vision and Graphics Lab
- **Key Content**:
  - Face model topology description (17 geometry regions: Face, Head/Neck, Mouth socket, Eye sockets, Gums/Tongue, Teeth, Eyeballs, Lacrimal fluids, Eyelashes)
  - Expression shapes following Apple ARKit naming convention (53 blendshapes)
  - Facial landmarks (68-point Multi-PIE)
  - Identity shape vectors (100 PCA modes)
  - Relationship to FACS (Facial Action Coding System) units
  - References to CVPR 2020 paper "Learning Formation of Physically Based Face Attributes"
- **Dependencies**: None (documentation only)
- **Notes**: MIT license; full model has additional features (200+ PCA modes, albedo inference, FBX rig) under USC license

### `/home/timoite/Documents/ICT-FaceKit/pyproject.toml`
- **File Type**: Python Project Configuration (PEP 621)
- **Purpose**: Project metadata and dependency specification for the ICT-FaceKit package
- **Key Dependencies**:
  - Core: numpy>=1.24, scipy>=1.11, trimesh>=4.0, matplotlib>=3.8, imageio>=2.30
  - Optional groups:
    - `pyrender`: pyrender>=0.1.45, PyOpenGL>=3.1 (CPU-based rendering)
    - `pytorch3d`: torch>=1.13, torchaudio>=0.13, pytorch3d>=0.7 (PyTorch3D rendering)
    - `wavlm`: torch>=1.13, transformers>=4.30, loralib>=0.1 (speech inversion model)
    - `beat`: huggingface_hub>=0.20 (BEAT dataset downloading)
    - `all`: All optional dependencies combined
  - Dev: pytest>=8.0, black>=23.0, flake8>=6.0
- **Related Files**: uv.lock (locked dependencies)
- **Notes**: Requires Python >=3.9, <3.12

### `/home/timoite/Documents/ICT-FaceKit/AGENT.md`
- **File Type**: Markdown Documentation
- **Purpose**: Project-specific AGENT instructions for AI assistants working on tongue articulation optimization
- **Key Content**:
  - Goal: Improve speech animation realism and intelligibility by targeting inner-mouth motion
  - Target metric: Lower WER on AutoAVSR
  - Data sources: BEAT blendshapes, WavLM tongue motion, Praat TextGrid alignment
  - Key conventions for BEAT/ICT name mapping
  - Tongue anchor indices: [16661, 16696, 16755, 16758]
  - Current task: Verify temporal alignment between facial keyframes and tongue keyframes
  - Progress tracking: Initial correlation analysis showed near-zero global correlation
  - Ground truth editor implemented with interactive matplotlib GUI
- **Related Files**: tongue_scripts/jaw_tongue_sync_analysis.py, tongue_scripts/tongue_gt_editor.py
- **Notes**: Last updated 2026-02-10

### `/home/timoite/Documents/ICT-FaceKit/.gitignore`
- **File Type**: Git Ignore Configuration
- **Purpose**: Specifies files and directories to exclude from git version control
- **Key Exclusions**: Standard Python cache, IDE configs, virtual environments, compiled files, system files

### `/home/timoite/Documents/ICT-FaceKit/LICENSE`
- **File Type**: License File
- **Purpose**: Defines project licensing terms (MIT License)

### `/home/timoite/Documents/ICT-FaceKit/.python-version`
- **File Type**: Configuration
- **Purpose**: Specifies Python version for tools (3.12)

---

## ADFA_EVALUATION Directory

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/AGENT.md`
- **File Type**: Markdown Documentation
- **Purpose**: Detailed documentation for the ADFA evaluation framework for assessing speech animation quality
- **Key Content**:
  - Overview of post-production evaluation using visual speech recognition (lip-reading)
  - Core components: AutoAVSR model (19.1% WER on LRS3), WER computation, text normalization
  - Complete workflow: Generate animations → Extract transcripts via lip-reading → Compute WER → Visualize results
  - Directory structure for transcripts, ground truth, and plots
  - Expected output paths and WER interpretation guidelines
  - Usage examples for each step
- **Dependencies**: AutoAVSR model, MediaPipe/RetinaFace detectors
- **Related Files**: compute_wer.py, normalize_transcripts.py, visualize_wer.py
- **Notes**: Framework uses VSR trained on LRS3 (25fps); BEAT dataset provides ground truth transcripts

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/basic.py`
- **File Type**: Python Script
- **Purpose**: Basic text normalization utilities
- **Key Classes/Functions**:
  - `ADDITIONAL_DIACRITICS`: Dictionary for non-ASCII letter mappings
  - `remove_symbols_and_diacritics()`: Remove symbols and diacritics (Unicode category Mn)
  - `remove_symbols()`: Remove symbols/punctuation while keeping diacritics
  - `BasicTextNormalizer`: Configurable normalizer class
- **Dependencies**: regex, unicodedata
- **Related Files**: english.py (imports these utilities)
- **Notes**: Used as base for EnglishTextNormalizer

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/compute_wer.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Compute Word Error Rate (WER) between generated transcripts and BEAT TextGrid ground truth
- **Key Classes/Functions**:
  - `WordInterval`: Dataclass for TextGrid interval (start, end, text)
  - `WerStats`: Dataclass for WER metrics (substitutions, deletions, insertions, ref_words, hyp_words)
  - `parse_textgrid_words()`: Parse Praat TextGrid files to extract word intervals
  - `compute_alignment()`: Levenshtein distance DP algorithm for optimal alignment
  - `collect_predicted_transcripts()`: Iterator over transcript files
  - `write_csv()`: Write WER report CSV
- **Dependencies**: csv, math, re, pathlib, argparse
- **Related Files**: english.py (text normalization), BEAT TextGrids
- **Output**: CSV report with per-file WER and aggregate statistics

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/download_beat_data.py`
- **File Type**: Python Script
- **Purpose**: Download BEAT dataset files from Hugging Face
- **Key Functions**:
  - `download_speaker_data(speaker_id)`: Download JSON and TextGrid files for specific speaker
- **Dependencies**: huggingface_hub, os, shutil, pathlib
- **Related Files**: None (standalone download script)
- **Output**: Files downloaded to data/beat_cache/ directory
- **Notes**: Can download single speaker or all speakers ("*")

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/english.py`
- **File Type**: Python Script
- **Purpose**: English-specific text normalization for ASR/VSR
- **Key Classes**:
  - `EnglishNumberNormalizer`: Convert spelled-out numbers to Arabic numerals, handle currency, ordinals
  - `EnglishSpellingNormalizer`: British-American spelling mappings
  - `EnglishTextNormalizer`: Comprehensive normalization (contractions, numbers, spellings, diacritics)
- **Dependencies**: json, re, fractions, more_itertools, basic.py
- **Related Files**: english.json (spelling mappings)
- **Notes**: Handles complex cases like "one oh one" → "101", "$20 million" → "20000000 dollars"

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/english.json`
- **File Type**: JSON Data
- **Purpose**: British-American spelling mappings
- **Key Content**: ~2000+ word pairs mapping British to American spellings
  - Examples: "colour" → "color", "analysing" → "analyzing", "behaviour" → "behavior"
- **Dependencies**: None
- **Related Files**: english.py (EnglishSpellingNormalizer loads this)

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/jiwer_directory_wer.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Compute WER between two transcript directories using jiwer library
- **Key Functions**:
  - `compute_directory_wer()`: Match predicted files with ground truth, compute WER for each pair
  - `list_files()`: Recursive file discovery
  - `write_csv()`: Generate WER report
- **Dependencies**: jiwer, pandas, pathlib, argparse
- **Related Files**: None (alternative to compute_wer.py using external library)
- **Output**: CSV with file, speaker_id, WER columns

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/normalize_transcripts.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Normalize transcript files using EnglishTextNormalizer, mirroring directory structure
- **Key Functions**:
  - `iter_files()`: Recursive file discovery with extension filtering
  - `normalize_text()`: Apply normalizer line-by-line preserving empty lines
  - `main()`: Batch processing with dry-run support
- **Dependencies**: pathlib, argparse, english.py
- **Related Files**: english.py
- **Output**: Normalized transcripts in output directory
- **Notes**: Supports multiple file extensions (.txt default)

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/visualize_wer.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Generate analysis plots from WER reports to identify problematic speakers/files
- **Key Functions**:
  - `plot_mean_wer_by_speaker()`: Bar chart of mean WER per speaker
  - `plot_worst_speaker_boxplots()`: Box plots for worst N speakers
  - `plot_best_speaker_boxplots()`: Box plots for best N speakers
  - `plot_all_speakers_boxplot()`: Box plots for all speakers
  - `plot_worst_files()`: Horizontal bar chart of worst files
  - `plot_wer_distribution()`: Histogram of WER values
- **Dependencies**: matplotlib, pandas, pathlib, argparse
- **Related Files**: wer_report.csv (from jiwer_directory_wer.py or compute_wer.py)
- **Output**: PNG plots in output directory

---

## FaceXModel Directory

### `/home/timoite/Documents/ICT-FaceKit/FaceXModel/vertex_indices.json`
- **File Type**: JSON Configuration
- **Purpose**: Core model configuration defining expression shapes and vertex indices
- **Key Content**:
  - `expressions`: List of 53 expression blendshape names (browDown_L/R, eyeBlink_L/R, jawOpen, etc.)
  - `fitting_expressions`: Same as expressions (used for fitting)
  - `idx_to_fitting_verts`: Vertex indices for fitting subset
  - `idx_to_rigid_verts`: Vertex indices for rigid face region
  - `idx_to_lap_rigid_verts`: Vertex indices for laplacian smoothing
  - `idx_to_landmark_verts`: 68 Multi-PIE facial landmark vertex indices
  - `idx_to_rigid_landmark_idx`: Mapping from landmark indices to rigid landmark indices
- **Dependencies**: None
- **Related Files**: generic_neutral_mesh.obj, *.obj expression blendshapes
- **Notes**: Critical for correct blendshape loading; vertex order must match exactly

### FaceXModel/*.obj Files
- **File Type**: Wavefront OBJ Mesh Files
- **Purpose**: 3D mesh data for expression blendshapes and neutral geometry
- **Key Files**:
  - `generic_neutral_mesh.obj`: Base neutral face geometry (26719 vertices, 26384 faces)
  - `browDown_L.obj`, `browDown_R.obj`, `eyeBlink_L.obj`, `jawOpen.obj`, etc.: 53 expression blendshapes
  - `ICTFaceModelMaterial.mtl`: Material definitions
  - `PupilDilate_L.obj`, `PupilDilate_R.obj`: Eye pupil dilation shapes
- **Dependencies**: None
- **Related Files**: vertex_indices.json (defines expression names and vertex indices)
- **Notes**: Total 17 geometry regions defined in README; OBJ files are morph targets

---

## tongue_scripts Directory

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/tongue_animation.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Standalone tongue control-point animation and rigging preview
- **Key Classes/Functions**:
  - `load_tongue_mesh()`: Load and rotate tongue mesh (90° X-axis rotation)
  - `compute_weights()`: Calculate Inverse Distance Weighting (IDW) for LBS (power of 3)
  - `apply_lbs()`: Apply Linear Blend Skinning deformation
  - `load_tongue_trajectory()`: Load and denormalize EMA motion from .npy
  - `deform_mesh_sequence()`: Apply rigging to trajectory frames
  - `animate_tongue()`: Create matplotlib animation with tongue trail
- **Dependencies**: trimesh, matplotlib, numpy, scipy.spatial.distance
- **Related Files**: seperated_tongue.obj, 26/npy/*.npy, normalising_vectors/*.npy
- **Output**: GIF animation (exports/tongue_animation.gif)
- **Notes**: 4 control points (T4-T1: back to tip); Wendland kernel radius 4.0

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/batch_render_corrected.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Batch render videos with mouth bias correction (lip closure fix)
- **Key Functions**:
  - `resample_ema_motion()`: Resample EMA from 50fps to 25fps using cubic interpolation
  - `render_bright_video()`: Render full face with materials, moving camera, and audio merge
- **Dependencies**: trimesh, pyrender, cv2, numpy, scipy.interpolate, subprocess
- **Related Files**: face_model_io_trimesh.py, render_face_animation_trimesh.py, test.py (FaceKitTongueRig)
- **Output**: MP4 videos with audio (720p or configurable)
- **Notes**: JawOpen shifted to min=0.0 for lip closure correction; 25fps rendering

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/wavlm_lora.py`
- **File Type**: Python Script
- **Purpose**: WavLM model with LoRA adapters for speech inversion to EMA
- **Key Classes**:
  - `WavLMEncoderLayer`: Modified WavLM layer with LoRA support in attention and feed-forward
  - `WavLMWrapper`: Full model loading and inference wrapper
- **Dependencies**: torch, loralib, transformers (WavLMModel), torchaudio
- **Related Files**: inversion_checkpoints/*.pth (trained LoRA weights)
- **Output**: 16-dimensional regression output (4 points × 2D deltas)
- **Notes**: Based on WavLM-large backbone; LoRA rank/alpha configurable; uses frozen pretrained weights

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/test.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Main script for tongue animation testing with hybrid deformation
- **Key Classes/Functions**:
  - `process_beat_data()`: Load and resample BEAT blendshapes to target FPS
  - `load_ema_motion()`: Load and denormalize tongue EMA from .npy
  - `FaceKitTongueRig`: Hybrid spline-bone tongue rigging class with thickness/rotation
  - `_calc_weights()`: Gaussian weight computation with offset correction
  - `deform()`: Apply tongue deformation with bone-spline coupling
- **Dependencies**: trimesh, pyrender, cv2, numpy, scipy, matplotlib
- **Related Files**: face_model_io_trimesh.py, render_face_animation_trimesh.py
- **Output**: Video outputs (tongue_hybrid_deformation.mp4, cut view, debug GIF)
- **Notes**: TONGUE_CONFIG defines rotation_deg, thickness, shift_y/z, std_scalar

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/face_model_io_trimesh.py`
- **File Type**: Python Script
- **Purpose**: Trimesh/NumPy version of face model loader with strict OBJ parsing
- **Key Classes**:
  - `TrimeshFaceModel`: Numpy-based container for ICT FaceKit
    - `deform(weights_dict)`: Compute deformed mesh from expression weights
  - `_TrimeshModelLoader`: Loader with strict OBJ parser
  - `_load_obj_strict()`: Manual OBJ parsing to guarantee vertex order
- **Dependencies**: json, numpy, trimesh, pathlib
- **Related Files**: FaceXModel/vertex_indices.json, FaceXModel/generic_neutral_mesh.obj, FaceXModel/*.obj
- **Output**: TrimeshFaceModel object with neutral_verts, faces, expression_deltas
- **Notes**: Critical for blendshape animation; vertex order must match exactly

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/render_face_animation_trimesh.py`
- **File Type**: Python Script
- **Purpose**: Face animation rendering using Trimesh & Pyrender
- **Key Functions**:
  - `map_beat_to_ict_names()`: Map BEAT blendshape names to ICT convention
  - `load_animation()`: Load JSON animation data
  - `FaceModelNumpy`: NumPy face model with deform() method
  - `PyrenderRenderer`: Stabilized renderer with camera, lights
  - `render_animation()`: Render frame sequence to video
- **Dependencies**: trimesh, pyrender, imageio, numpy
- **Related Files**: face_model_io_trimesh.py
- **Output**: MP4 video (default 256x256, grayscale mesh)
- **Notes**: Grey mesh, black background; supports stabilization

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/jaw_tongue_sync_analysis.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Analyze jaw-tongue temporal alignment using BEAT, WavLM, TextGrid data
- **Key Classes/Functions**:
  - `Interval`: Dataclass for TextGrid intervals
  - `parse_textgrid_intervals()`: Parse Praat TextGrid tiers
  - `compute_correlation_with_lag()`: Compute Pearson correlation with lag sweep
  - `filter_by_phonemes()`: Extract segments for specific phonemes
  - `main()`: Full analysis pipeline with multiple lip shapes
- **Dependencies**: matplotlib, numpy, scipy, argparse
- **Related Files**: face_model_io_trimesh.py, test.py (FaceKitTongueRig)
- **Output**: Plot (jaw vs tongue with phoneme highlighting), JSON report
- **Notes**: Default lip shapes: 20+ mouth blendshapes; default phoneme filter: L, T, D, K, G, N, S, Z, CH, JH, SH, TH, DH, R

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/jaw_tongue_sync_render_shift.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Render shifted tongue animation based on jaw-tongue sync analysis
- **Key Functions**:
  - `render_shifted_video()`: Render with tongue time shift applied
  - `main()`: Load optimal shift, apply to EMA, render
- **Dependencies**: Similar to batch_render_corrected.py
- **Related Files**: tongue_scripts/jaw_tongue_sync_analysis.py (output shift value)
- **Output**: MP4 with shifted tongue timing
- **Notes**: Used for validation of temporal alignment fix

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/lip_closure_b_timing.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Analyze lip closure timing relative to phoneme boundaries
- **Key Functions**:
  - `compute_lip_closure_envelope()`: Compute closure metric from lip shapes
  - `compare_with_phonemes()`: Align closure events with phoneme intervals
- **Dependencies**: numpy, scipy, matplotlib
- **Related Files**: BEAT JSON files, TextGrid files
- **Output**: Plots showing closure timing vs phonemes
- **Notes**: Helps diagnose late/early lip closure issues

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/phoneme_lag_probe.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Probe phoneme-specific lag patterns in tongue motion
- **Key Functions**:
  - `compute_per_phoneme_lag()`: Find optimal shift per phoneme class
  - `aggregate_results()`: Summary statistics
- **Dependencies**: Similar to jaw_tongue_sync_analysis.py
- **Related Files**: TextGrid files, EMA .npy files
- **Output**: Per-phoneme lag distribution
- **Notes**: Helps identify if certain phonemes require different temporal offsets

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/run_batch_pipeline_speaker1.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Batch render pipeline for speaker 1 from BEAT dataset
- **Key Functions**:
  - `process_batch()`: Iterate over BEAT clips, render each
- **Dependencies**: Similar to batch_render_corrected.py
- **Related Files**: data/beat_cache_speaker1/, outputs/
- **Output**: Multiple MP4 videos for speaker 1
- **Notes**: Processes all clips for a single speaker

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/same_phoneme_comparison.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Compare tongue motion for repeated phonemes within same clip
- **Key Functions**:
  - `extract_phoneme_instances()`: Get all instances of specific phoneme
  - `compare_trajectories()`: Compute similarity metrics
- **Dependencies**: numpy, scipy
- **Related Files**: TextGrid files, EMA .npy files
- **Output**: Similarity scores and visualizations
- **Notes**: Helps identify consistency in tongue production

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/test_tongue_grid_search_25fps.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Comprehensive tongue parameter grid search for optimal speech intelligibility (25fps version)
- **Key Classes/Functions**:
  - `get_camera_matrix()`: Compute camera matrix for orbit animation
  - `run_vsr_inference()`: Run AutoAVSR on rendered video
  - `merge_audio()`: Merge video with audio using FFmpeg
  - `main()`: Grid search loop over 27 configurations (3×3×3)
- **Dependencies**: trimesh, pyrender, cv2, torch, subprocess
- **Related Files**: Visual_Speech_Recognition_for_Multiple_Languages/pipelines/*.py
- **Output**: tongue_param_tests_25fps/ directory with 27 subdirectories
- **Config Grid**:
  - rotation: [0, 10, 20] degrees
  - thickness: [1.0, 2.0, 4.0]
  - std_scalar: [0.10, 0.25, 0.40]
- **Notes**: 25fps rendering with speed_rate=1.0; integrated VSR and WER

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/tongue_gt_compare.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Compare actual EMA motion with ground truth tongue positions
- **Key Functions**:
  - `load_gt_json()`: Load GT from tongue_gt_editor output
  - `compute_per_anchor_error()`: Error per anchor in (Y, Z)
  - `sweep_global_shift()`: Find optimal global shift to minimize error
- **Dependencies**: numpy, matplotlib, json
- **Related Files**: tongue_gt_editor.py (outputs _tongue_gt.json), EMA .npy files
- **Output**: _gt_compare.json, _gt_compare.png
- **Notes**: Reports per-class best shifts for diagnostic analysis

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/tongue_gt_editor.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Interactive GUI editor for defining ground truth tongue positions per phoneme
- **Key Classes/Functions**:
  - `TongueGTEditor`: Matplotlib-based editor with draggable spline anchors
  - `save_current_phone()`: Auto-save on navigation
  - `load_gt()`: Load existing GT file
- **Dependencies**: matplotlib, json, numpy
- **Related Files**: TextGrid files, face_model_io_trimesh.py
- **Output**: _tongue_gt.json with per-phoneme anchor (Y, Z) targets
- **Notes**: 4 keyframes per phoneme instance; sagittal cross-section view

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/visualize_lip_landmarks.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Visualize lip landmarks over video frames
- **Key Functions**:
  - `detect_landmarks()`: Use MediaPipe to detect 68 facial landmarks
  - `overlay_landmarks()`: Draw landmarks on frames
- **Dependencies**: mediapipe, cv2, numpy
- **Related Files**: MP4 video files
- **Output**: Video with overlaid landmarks
- **Notes**: Helps verify mouth visibility for VSR

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/visualize.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Generic visualization utilities for tongue/face data
- **Key Functions**:
  - `plot_3d_mesh()`: 3D mesh visualization
  - `plot_trajectory()`: EMA trajectory plotting
- **Dependencies**: matplotlib, trimesh, numpy
- **Related Files**: .npy files, .obj files
- **Output**: Matplotlib figures
- **Notes**: General-purpose visualization script

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/visualize_setup.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Verify and visualize the setup (face model, tongue rig, camera)
- **Key Functions**:
  - `render_setup()`: Render static setup with labeled components
- **Dependencies**: trimesh, pyrender
- **Related Files**: face_model_io_trimesh.py, test.py (FaceKitTongueRig)
- **Output**: Setup verification image
- **Notes**: Useful for debugging rigging issues

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/invert.py`
- **File Type**: Python Script (Executable)
- **Purpose**: WavLM speech inversion script (original, possibly superseded by infer_single.py)
- **Key Functions**:
  - Audio to EMA conversion using WavLM model
- **Dependencies**: torch, torchaudio, transformers
- **Related Files**: wavlm_lora.py, inversion_checkpoints/*.pth
- **Output**: .npy EMA files
- **Notes**: May be legacy or alternative implementation

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/extract_ground_truth.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Extract reference transcript from BEAT TextGrid for WER evaluation
- **Key Functions**:
  - Uses compute_wer.py utilities to parse TextGrid
  - Writes ground_truth.txt
- **Dependencies**: compute_wer.py, pathlib
- **Related Files**: BEAT TextGrid files
- **Output**: ground_truth.txt
- **Notes**: Output: ~97 words over 27 seconds for 1_wayne_0_75_75

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/test_tongue_grid_search.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Original grid search script (50fps version - has frame rate bug)
- **Key Functions**:
  - Same as test_tongue_grid_search_25fps.py but with FPS=50, speed_rate=2.0
- **Dependencies**: Same as 25fps version
- **Related Files**: test_tongue_grid_search_25fps.py (corrected version)
- **Notes**: Known bug: 50fps with speed_rate=2.0 causes temporal aliasing and gibberish VSR output

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/test_tongue_grid_search_25fps.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Corrected grid search with 25fps rendering (fixes frame rate bug)
- **Key Functions**:
  - Same interface as test_tongue_grid_search.py but:
    - FPS = 25 (was 50)
    - speed_rate = 1.0 (was 2.0)
- **Dependencies**: Same as 50fps version
- **Related Files**: VSR evaluation framework
- **Output**: tongue_param_tests_25fps/ with coherent transcripts
- **Notes**: See IMPLEMENTATION_SUMMARY.md for full details

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/IMPLEMENTATION_SUMMARY.md`
- **File Type**: Markdown Documentation
- **Purpose**: Summary of VSR evaluation fix and grid search rework
- **Key Content**:
  - Problem: 50fps + speed_rate=2.0 = gibberish transcripts
  - Root Cause: Frame subsampling causes temporal aliasing
  - Solution: 25fps + speed_rate=1.0
  - Files created for validation and corrected grid search
  - Execution workflow and success metrics
  - Technical analysis of frame rate math
  - Troubleshooting guide
  - Expected outcomes
- **Related Files**: validate_vsr_25fps.py, test_tongue_grid_search_25fps.py, extract_ground_truth.py
- **Notes**: Critical document for understanding the VSR fix

### `/home/timoite/Documents/ICT-FaceKit/tongue_scripts/ground_truth.txt`
- **File Type**: Plain Text
- **Purpose**: Reference transcript for WER evaluation
- **Content**: 97 words over 27 seconds from BEAT dataset (1_wayne_0_75_75)
- **Dependencies**: None (output from extract_ground_truth.py)
- **Related Files**: extract_ground_truth.py
- **Notes**: Used as ground truth for all VSR evaluations

---

## Visual_Speech_Recognition_for_Multiple_Languages

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/README.md`
- **File Type**: Markdown Documentation
- **Purpose**: Main documentation for Visual Speech Recognition (AutoAVSR) models
- **Key Content**:
  - Model performance: 19.1% WER (visual-only), 1.0% (audio-only), 0.9% (AV) on LRS3
  - Tutorial with Colab notebook link
  - Demo videos showing multi-language lip-reading
  - Preparation: installation, environment setup
  - Benchmark evaluation instructions
  - Speech prediction (inference) instructions
  - Model zoo: Download links for LRS3, LRS2, CMLR, CMU-MOSEAS, GRID, Lombard GRID, TCD-TIMIT
  - Citation information
- **Related Files**: infer.py, eval.py, crop_mouth.py
- **Notes**: Code for non-commercial use only

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/hydra_configs/default.yaml`
- **File Type**: YAML Configuration
- **Purpose**: Hydra configuration for inference pipeline
- **Key Settings**:
  - config_filename: Path to model config .ini
  - data_dir: Dataset directory
  - data_filename: Input video file
  - data_ext: Default .mp4
  - landmarks_dir: Pre-computed landmarks directory
  - detector: "retinaface" or "mediapipe"
  - gpu_idx: GPU index
- **Dependencies**: None
- **Related Files**: infer.py (uses hydra.main)
- **Notes**: Default paths are None, set via command line

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/requirements.txt`
- **File Type**: Text File
- **Purpose**: Core dependencies for VSR pipeline
- **Dependencies**:
  - hydra-core >= 1.3.2
  - opencv-python >= 4.5.5.62
  - scipy >= 1.3.0
  - scikit-image >= 0.13.0
  - av >= 10.0.0
  - six >= 1.16.0
- **Related Files**: infer.py, eval.py, crop_mouth.py
- **Notes**: Excludes PyTorch/torchvision (installed separately)

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Main inference entry point using Hydra
- **Key Functions**:
  - `main(cfg)`: Hydra-wrapped entry point, loads pipeline, runs inference
- **Dependencies**: torch, hydra, pipelines.pipeline
- **Related Files**: pipelines/pipeline.py, hydra_configs/default.yaml
- **Output**: Prints transcript to console
- **Notes**: Device selection based on gpu_idx config

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/pipelines/pipeline.py`
- **File Type**: Python Script
- **Purpose**: InferencePipeline class combining data loading and model inference
- **Key Classes**:
  - `InferencePipeline`: Main pipeline class
    - `__init__()`: Load config, dataloader, model, landmarks detector
    - `process_landmarks()`: Load or detect landmarks
    - `forward()`: Run full inference (load data + predict)
- **Dependencies**: torch, pickle, configparser, pipelines/model.py, pipelines/data/data_module.py
- **Related Files**: configs/*.ini, data/LRS3_*/model.pth
- **Output**: Transcript string
- **Notes**: speed_rate = input_v_fps / model_v_fps (critical for frame rate)

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/pipelines/model.py`
- **File Type**: Python Script
- **Purpose**: AVSR model class with beam search decoding
- **Key Classes**:
  - `AVSR`: Main model wrapper
    - `__init__()`: Load model config, weights, language model
    - `infer()`: Encode input, beam search decode, detokenize
  - `get_beam_search_decoder()`: Create BatchBeamSearch with scorers
- **Dependencies**: torch, json, argparse, numpy, espnet modules
- **Related Files**: espnet/nets/*.py, data/LRS3_*/model.pth, lm_en_subword/model.pth
- **Output**: Decoded transcript
- **Notes**: Supports char and unigram5000 tokenization; RNNLM optional

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/pipelines/data/data_module.py`
- **File Type**: Python Module
- **Purpose**: AVSRDataLoader for loading video/audio data and preprocessing
- **Key Classes**:
  - `AVSRDataLoader`: Main data loader
    - `__init__()`: Setup modality, speed_rate, detector
    - `load_data()`: Load and preprocess data (crop mouth for video)
- **Dependencies**: torch, cv2, av, numpy
- **Related Files**: pipelines/detectors/*/detector.py (landmarks detection)
- **Output**: Preprocessed tensors
- **Notes**: speed_rate used for temporal subsampling (line in transforms.py)

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/pipelines/data/transforms.py`
- **File Type**: Python Module
- **Purpose**: Data transforms and preprocessing
- **Key Functions**:
  - Mouth cropping from landmarks
  - Frame normalization
  - Temporal resampling based on speed_rate
- **Dependencies**: torch, cv2, numpy
- **Related Files**: data_module.py (calls transforms)
- **Notes**: Critical line uses torch.index_select for speed_rate-based subsampling

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/pipelines/metrics/measures.py`
- **File Type**: Python Module
- **Purpose**: Evaluation metrics (WER, CER)
- **Key Functions**:
  - `compute_wer()`: Word error rate calculation
  - `compute_cer()`: Character error rate calculation
- **Dependencies**: numpy
- **Related Files**: eval.py (uses for benchmarking)
- **Notes**: Levenshtein distance-based alignment

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/crop_mouth.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Crop mouth ROI from video
- **Key Functions**:
  - Mouth detection and cropping
- **Dependencies**: hydra, cv2
- **Related Files**: pipelines/data/data_module.py (similar logic)
- **Output**: Cropped mouth images/video
- **Notes**: Standalone tool for preprocessing

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/eval.py`
- **File Type**: Python Script (Executable)
- **Purpose**: Benchmark evaluation on datasets
- **Key Functions**:
  - Batch evaluation on pre-defined datasets
  - WER/CER aggregation
- **Dependencies**: torch, hydra
- **Related Files**: configs/*.ini, benchmarks/*/labels
- **Output**: Aggregate WER/CER statistics
- **Notes**: For official benchmark evaluation (not inference)

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/espnet/`
- **Purpose**: ESPNet-based ASR framework modules
- **Key Submodules**:
  - `asr/asr_utils.py`: Model loading utilities
  - `nets/batch_beam_search.py`: Beam search decoder with multiple scorers
  - `nets/beam_search.py`: Single beam search implementation
  - `nets/ctc_prefix_score.py`: CTC prefix scoring
  - `nets/e2e_asr_common.py`: Common E2E model components
  - `nets/lm_interface.py`: Dynamic LM import
  - `nets/pytorch_backend/e2e_asr_transformer.py`: Transformer encoder/decoder
  - `nets/pytorch_backend/e2e_asr_transformer_av.py`: Audio-visual version
  - `nets/pytorch_backend/backbones/`: ResNet, Conv1D/3D extractors
  - `nets/pytorch_backend/lm/`: RNN and Transformer language models
  - `utils/cli_utils.py`, `utils/dynamic_import.py`, `utils/fill_missing_args.py`: Utility functions
- **Dependencies**: torch, numpy, scipy
- **Related Files**: pipelines/model.py (imports these)
- **Notes**: External framework (ESPNet) adapted for VSR

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/pipelines/tokens/unigram5000_units.txt`
- **File Type**: Text File
- **Purpose**: Vocabulary list for unigram5000 tokenization
- **Content**: 5000 most common word pieces/subwords
- **Dependencies**: None
- **Related Files**: pipelines/model.py (loads for tokenization)
- **Notes**: Used with unigram5000 labels_type in model config

### `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/data/`
- **Purpose**: Model and data storage
- **Key Contents**:
  - `LRS3_AV_WER0.9/`: Audio-visual model (1540MB)
  - `LRS3_A_WER1.0/`: Audio-only model (860MB)
  - `LRS3_V_WER19.1/`: Visual-only model (891MB)
    - `model.pth`: PyTorch model weights
    - `model.json`: Model configuration
  - `lm_en_subword/`: Language model (191MB)
    - `model.pth`: LM weights
    - `model.json`: LM config
  - `beat_textgrids/`: Praat TextGrid files for BEAT dataset
- **Dependencies**: None
- **Related Files**: infer.py, eval.py, configs/LRS3_*.ini
- **Notes**: Models downloaded from official links in README

---

## crucial_progress_report Directory

### `/home/timoite/Documents/ICT-FaceKit/crucial_progress_report/batch_pipeline_status.md`
- **File Type**: Markdown Documentation
- **Purpose**: Status report for batch rendering pipeline
- **Related Files**: run_batch_pipeline_speaker1.py

### `/home/timoite/Documents/ICT-FaceKit/crucial_progress_report/SESSION_2026_02_04_RNNLM_CONFIG_TEST.md`
- **File Type**: Markdown Documentation
- **Purpose**: Session notes for RNNLM configuration testing
- **Related Files**: VSR language model configuration

### `/home/timoite/Documents/ICT-FaceKit/crucial_progress_report/SESSION_2026_02_04_TONGUE_BATCH_VSR.md`
- **File Type**: Markdown Documentation
- **Purpose**: Session notes for tongue batch VSR evaluation
- **Related Files**: tongue_scripts batch rendering

### `/home/timoite/Documents/ICT-FaceKit/crucial_progress_report/SESSION_2026_02_04_VSR_MOUTH_CORRECTED_7.5S.md`
- **File Type**: Markdown Documentation
- **Purpose**: Session notes for VSR with mouth-corrected animations (7.5s)
- **Related Files**: batch_render_corrected.py

### `/home/timoite/Documents/ICT-FaceKit/crucial_progress_report/SESSION_2026_02_04_VSR_VALIDATION.md`
- **File Type**: Markdown Documentation
- **Purpose**: Session notes for VSR validation testing
- **Related Files**: validate_vsr_25fps.py

### `/home/timoite/Documents/ICT-FaceKit/crucial_progress_report/VSR_FIX_VALIDATION_REPORT.md`
- **File Type**: Markdown Documentation
- **Purpose**: Comprehensive report on VSR frame rate fix validation
- **Key Content**:
  - Problem: 50fps + speed_rate=2.0 = gibberish ("THANK YOU VERY MUCH" loop)
  - Root Cause: Frame subsampling causes temporal aliasing
  - Solution: 25fps + speed_rate=1.0 = coherent English
  - Validation: Single test confirmed fix works
  - Files created: validate_vsr_25fps.py, test_single_config_25fps.py, extract_ground_truth.py, compute_single_wer.py, generate_wer_report.py, test_tongue_grid_search_25fps.py
  - Comparison: 50fps had 4 unique words (17%), 25fps had 16 unique words (62%)
  - Ground truth extracted: 84 words over 27 seconds
  - Next steps: Run full 27-config grid search
- **Related Files**: All validation scripts listed above
- **Notes**: Ready to proceed with grid search to find optimal tongue parameters

---

## Evaluation Script Directory

### `/home/timoite/Documents/ICT-FaceKit/evaluation_script/ver.py`
- **File Type**: Python Script
- **Purpose**: Verification script (exact purpose unclear from filename)
- **Related Files**: None
- **Notes**: Minimal script, may be for setup validation

---

## Excluded Files

The following were intentionally excluded per instructions:
- `.git/` - Version control metadata
- `.venv/` - Virtual environment
- `__pycache__/` - Python cache
- `node_modules/` - Node.js dependencies
- `beat_cache/` and `beat_cache_speaker1/` - Data files
- Transient test output files (MP4 videos, temporary outputs)

---

## Summary Statistics

- **Total Python Files Documented**: ~70
- **Total Markdown Files**: ~15
- **Total Configuration Files**: ~10
- **Total Data Files (OBJ, JSON)**: ~60
- **Main Directories**:
  - Root: 10 files
  - ADFA_EVALUATION: 9 core files
  - FaceXModel: 55+ OBJ files + 1 JSON
  - tongue_scripts: 21 Python files + 1 data file
  - Visual_Speech_Recognition_for_Multiple_Languages: ~30 Python files + data
  - crucial_progress_report: 6 markdown files
  - evaluation_script: 1 Python file

---

## Key Workflows

### 1. Face Animation Pipeline
```
BEAT JSON → map_beat_to_ict_names → process_beat_data → FaceModel.deform()
WavLM .npy → load_ema_motion → FaceKitTongueRig.deform()
render_face_animation_trimesh → PyrenderRenderer → MP4 output
```

### 2. VSR Evaluation Pipeline
```
Generated MP4 → MediaPipe landmarks → AVSRDataLoader → AVSR.infer()
Transcript → EnglishTextNormalizer → compare with BEAT TextGrid
compute_alignment → WER statistics → visualize_wer plots
```

### 3. Tongue Optimization Pipeline
```
jaw_tongue_sync_analysis → correlation/lag metrics
tongue_gt_editor → define ground truth
tongue_gt_compare → compute errors
test_tongue_grid_search_25fps → find optimal parameters
generate_wer_report → rank by WER
```

---

## Important Notes

1. **Frame Rate Compatibility**: VSR model trained on 25fps; rendering at 25fps with speed_rate=1.0 is critical for coherent transcripts
2. **Vertex Order**: FaceXModel OBJ files must be loaded with strict parsing to preserve exact vertex order for blendshapes
3. **Tongue Anchors**: Fixed global indices [16661, 16696, 16755, 16758] used by rig
4. **BEAT Naming**: BEAT blendshape names differ from ICT; mapping via `map_beat_to_ict_names()`
5. **WavLM Output**: 4 tongue control points × 2D (Y, Z) deltas, reshaped from [T, 8]
6. **Language Model**: RNNLM optional for VSR; improves accuracy with lm_weight > 0

---

## Dependencies Summary

### Core Dependencies
- **numpy**, **scipy**: Numerical computing
- **trimesh**: 3D mesh operations
- **pyrender**: CPU-based rendering
- **opencv-python**: Video/image processing
- **matplotlib**: Visualization
- **torch/torchaudio**: Deep learning models (VSR, WavLM)
- **transformers**: Hugging Face models
- **hydra**: Configuration management

### VSR-Specific
- **espnet**: ASR framework (external)
- **jiwer**: WER computation
- **mediapipe/retinaface**: Face/landmark detection
- **av**: Audio/video processing

### FaceKit-Specific
- **loralib**: LoRA adapters for WavLM
- **imageio**, **imageio-ffmpeg**: Video writing
- **tqdm**: Progress bars
- **more_itertools**: Iteration utilities
- **regex**: Advanced regex for text normalization

---

*End of Documentation Report*
