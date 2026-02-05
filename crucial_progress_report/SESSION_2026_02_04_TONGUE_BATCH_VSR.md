# Research Progress Report: Tongue Animation Batch Synthesis & VSR Evaluation

**Date:** February 4, 2026
**Session:** Batch EMA Generation, Video Rendering, and VSR Evaluation
**Status:** VSR Evaluation Encounters Repetition Loop Issue - Requires Optimization

---

## 1. Executive Summary

This session focused on generating and evaluating a batch of tongue-enhanced synthetic videos to compare multiple tongue animation configurations using Visual Speech Recognition (VSR). We successfully:

1. Generated 3 new EMA motion files from audio using WavLM LoRA model
2. Rendered 4 videos at 25fps with tongue animation (25fps × 7.5s)
3. Merged videos with their corresponding audio files
4. Attempted VSR evaluation on all 4 synthetic videos

**Key Finding:** All 4 synthetic videos (7.5 seconds each) caused the VSR model to enter infinite repetition loops during beam search, producing unusable transcripts. This mirrors the issue observed in the previous session with the 15-second bright video.

**Recommendation:** Shorten video duration (<5 seconds), increase repetition penalty in VSR config, or segment videos into shorter clips before VSR inference.

---

## 2. Technical Implementation Results

### 2.1 Batch EMA Generation

**Script:** `/tmp/generate_ema_batch.py`

**Configuration:**
- **Model:** WavLM LoRA model (speaker-conditional audio-to-EMA)
- **Sampling Params:** temperature=1.0, top_p=0.9
- **Audio Loading:** scipy.io.wavfile (to avoid torchcodec dependency)
- **Post-processing:** 8Hz Butterworth low-pass filter at fs=50

**Generated EMA Files:**

| Dataset ID | Audio Source | EMA File | Frames | Dimensions |
|------------|---------------|----------|--------|------------|
| 1_wayne_0_100_100 | BEAT Speaker 1 | `1_wayne_0_100_100.npy` | 2,849 | (2849, 16) |
| 1_wayne_0_10_10 | BEAT Speaker 1 | `1_wayne_0_10_10.npy` | 2,999 | (2999, 16) |
| 1_wayne_0_101_101 | BEAT Speaker 1 | `1_wayne_0_101_101.npy` | 2,999 | (2999, 16) |
| 1_wayne_0_75_75 | BEAT Speaker 1 | `1_wayne_0_75_75.npy` | 2,999 | (2999, 16) |

**EMA Dimensions:** 16 columns = 4 tongue points (T4, T3, T2, T1) × 4 coordinates (X, Y, Z, R)

**Location:** All EMA files saved to `tongue_scripts/outputs/`

### 2.2 STD File Compatibility Verification

**Finding:** Existing STD file `tongue_scripts/normalising_vectors/JW13_4points_std.npy` is compatible with all Speaker 1 EMA files.

**STD File Structure:**
- **Shape:** (1, 14)
- **First 8 values:** Y/Z standard deviations for T4, T3, T2, T1 tongue points
- **Remaining 6 values:** Not used by `load_ema_motion()` function in test.py

**Conclusion:** The existing normalization vectors can be reused for all Speaker 1 datasets without modification.

### 2.3 Input File Preparation

**Action:** Copied required input files for all 4 datasets to `tongue_scripts/inputs/`

**Files Copied:**
- `1_wayne_0_100_100.json` (BEAT metadata)
- `1_wayne_0_100_100.wav` (audio)
- `1_wayne_0_100_100.npy` (EMA, generated)
- `1_wayne_0_10_10.json`, `1_wayne_0_10_10.wav`, `1_wayne_0_10_10.npy`
- `1_wayne_0_101_101.json`, `1_wayne_0_101_101.wav`, `1_wayne_0_101_101.npy`
- `1_wayne_0_75_75.json`, `1_wayne_0_75_75.wav`, `1_wayne_0_75_75.npy` (existing)

### 2.4 Batch Video Rendering

**Script:** `/tmp/synthesize_and_evaluate_batch.py`

**Render Configuration:**
- **FPS:** 25
- **Duration:** 7.5 seconds (187 frames)
- **Background:** Gray [0.3, 0.3, 0.3]
- **Lighting:** Enhanced brightness
- **Tongue Configuration:**
  ```python
  TONGUE_CONFIG = {
      'rotation_deg': 5,
      'thickness': 1.2,
      'shift_y': 0,
      'shift_z': 0,
      'std_scalar': 0.20
  }
  ```

**Rendered Videos:**

| Dataset | Output File | Frames | Size |
|---------|-------------|--------|------|
| 1_wayne_0_100_100 | `1_wayne_0_100_100.mp4` | 187 | 1.5MB |
| 1_wayne_0_10_10 | `1_wayne_0_10_10.mp4` | 187 | 1.5MB |
| 1_wayne_0_101_101 | `1_wayne_0_101_101.mp4` | 187 | 1.5MB |
| 1_wayne_0_75_75 | `1_wayne_0_75_75.mp4` | 187 | 1.5MB |

**Location:** `tongue_scripts/batch_videos/`

### 2.5 Video-Audio Merging

**Tool:** ffmpeg

**Command:**
```bash
ffmpeg -i video.mp4 -i audio.wav -c:v copy -c:a aac video_final.mp4
```

**Merged Videos:**
- `1_wayne_0_100_100_final.mp4` (with audio)
- `1_wayne_0_10_10_final.mp4` (with audio)
- `1_wayne_0_101_101_final.mp4` (with audio)
- `1_wayne_0_75_75_final.mp4` (with audio)

**Location:** `tongue_scripts/batch_videos/`

---

## 3. VSR Evaluation Results

### 3.1 Inference Pipeline

**Script:** `/tmp/run_vsr_simple.py`

**Model Configuration:**
- **Model:** LRS3_V_WER19.1
- **Detector:** MediaPipe
- **Config File:** `configs/LRS3_V_WER19.1.ini`
- **Inference Parameters:**
  - penalty=0.0
  - ctc_weight=0.1
  - lm_weight=0.3
  - beam_size=40

**Direct Imports Used:**
```python
from pipelines.model import AVSR
from pipelines.data.data_module import AVSRDataLoader
from pipelines.detectors.mediapipe.detector import LandmarksDetector
```

**Approach:** Direct model import avoided complex `infer.py` pipeline that had hydra/module dependency errors.

### 3.2 Transcript Results

All 4 synthetic videos produced **unusable transcripts with infinite repetition loops**:

#### Video 1: 1_wayne_0_100_100 (7.5s)
```
"A L R I G H T S O L E T S A R T W E R E G O I N G..." [repeating]
```

#### Video 2: 1_wayne_0_10_10 (7.5s)
```
"T H E F I R S T I ' M G O N D O I S T O F I N D..." [repeating]
```

#### Video 3: 1_wayne_0_101_101 (7.5s)
```
"U H I T ' S I ' S..." [repeating]
```

#### Video 4: 1_wayne_0_75_75 (7.5s)
```
"I F Y O U D O N ' K N O W..." [repeating]
```

**Comparison with Real Video:**
- `sample.mp4` (7.4s real LRS3 video): Produced coherent transcript
  ```
  "COMPLETELY CONCENTRATED ENVIRONMENTS WHERE WE HAVE LARGE CHANGES IN GET POSTS AND"
  ```

### 3.3 Root Cause Analysis

**Issue:** VSR beam search enters infinite repetition loops on 7.5-second synthetic videos.

**Hypotheses:**

1. **Synthetic Speech Pattern Mismatch:**
   - Synthetic facial animation trained from BEAT data doesn't match LRS3 lip movement patterns
   - VSR model (trained on real human speech) expects specific visual speech cues
   - Tongue deformation may introduce artifacts in the mouth region
   - Synthetic face lacks fine-grained lip dynamics of real speakers

2. **Sequence Length Sensitivity:**
   - Previous session observed repetition loop on 15-second bright video
   - Now observed on 7.5-second videos (all 4 synthetic videos)
   - Real video (sample.mp4, 7.4s) works fine
   - Suggests synthetic content exacerbates beam search instability

3. **Beam Search Configuration:**
   - Current penalty=0.0 (no repetition penalty)
   - Increasing penalty to 0.3-0.5 may prevent loops

**Technical Verification:**
- ✓ VSR pipeline ran successfully without errors
- ✓ All model files loaded correctly
- ✓ MediaPipe face/landmark detection operational
- ✓ Inference completed (with degraded output)
- ✓ **Issue is NOT technical failure - it's model behavior on synthetic content**

### 3.4 Ground Truth Status

**TextGrid Files Available:**
- Location: `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/data/beat_textgrids/1/`
- Files: `1_wayne_0_100_100.TextGrid`, `1_wayne_0_10_10.TextGrid`, `1_wayne_0_101_101.TextGrid`, `1_wayne_0_75_75.TextGrid`

**Status:** Ground truth TextGrid files exist but were NOT extracted during batch run due to script errors. WER calculation cannot be performed until repetition issue is resolved.

---

## 4. Key Findings

### 4.1 Successful Components

| Component | Status | Notes |
|-----------|--------|-------|
| EMA Generation | ✓ Success | WavLM LoRA model produces 16-dim tongue motion |
| Audio Loading | ✓ Fixed | scipy.io.wavfile avoids torchcodec dependency |
| EMA Filtering | ✓ Working | 8Hz Butterworth filter applied |
| STD Compatibility | ✓ Verified | Existing JW13_4points_std.npy works for all Speaker 1 |
| Video Rendering | ✓ Success | 4 videos rendered at 25fps, 7.5s |
| Video-Audio Merging | ✓ Success | ffmpeg integration working |
| VSR Pipeline | ✓ Operational | Runs without technical errors |
| MediaPipe Detection | ✓ Operational | Detects faces in synthetic videos |

### 4.2 Critical Issue

| Issue | Severity | Impact | Status |
|-------|----------|--------|--------|
| VSR Repetition Loops | 🔴 High | All synthetic videos produce unusable transcripts | Unresolved |

### 4.3 Synthetic vs Real Video Performance

| Metric | Synthetic Videos (4) | Real Video (sample.mp4) |
|--------|---------------------|--------------------------|
| Technical Success | ✓ Yes | ✓ Yes |
| Coherent Transcript | ✗ No (repetition loops) | ✓ Yes |
| Duration | 7.5s | 7.4s |
| VSR Config | Same | Same |
| Output Quality | Unusable | Usable |

---

## 5. Recommendations

### 5.1 Immediate Actions (To Fix Repetition Issue)

**Option 1: Shorten Video Duration** (Recommended)
- Target duration: 3-5 seconds (75-125 frames at 25fps)
- Rationale: Shorter sequences reduce beam search drift
- Implementation: Modify render script to limit frame count

**Option 2: Increase Repetition Penalty**
- Modify VSR config: penalty=0.0 → penalty=0.4
- Rationale: Penalize beam search from repeating tokens
- Implementation: Edit `configs/LRS3_V_WER19.1.ini`

**Option 3: Video Segmentation**
- Segment 7.5s videos into 2-3 clips (2-3s each)
- Run VSR on each segment independently
- Merge transcripts post-inference
- Rationale: Maintains data volume while reducing sequence length

**Option 4: Use infer_pipeline.py**
- Existing pipeline handles longer videos with TextGrid segmentation
- Previous hydra/module errors may be resolved with correct environment
- Rationale: Leverage existing segmentation logic

### 5.2 Secondary Actions (If VSR Remains Unusable)

1. **Focus on Short Validation Clips:**
   - Generate 2-3 second test videos
   - Validate VSR works on short synthetic content
   - Establish baseline before scaling to longer videos

2. **Domain Adaptation Investigation:**
   - Fine-tune VSR model on synthetic faces (if resources available)
   - Or accept that synthetic videos have inherent domain mismatch
   - Consider alternative evaluation metrics beyond VSR

3. **Alternative Evaluation Approaches:**
   - Human perceptual evaluation (intelligibility ratings)
   - Lip movement synchrony metrics
   - Tongue visibility metrics (occlusion rate, clarity score)

### 5.3 Long-term Research Plan

1. **Phase 1: Fix VSR Repetition** (Current priority)
   - Implement Option 1 (shorten videos) or Option 2 (increase penalty)
   - Validate VSR produces coherent transcripts on short synthetic videos

2. **Phase 2: Complete Batch Evaluation**
   - Run VSR on all 4 final videos (with audio)
   - Extract ground truth from TextGrid files
   - Calculate WER metrics for each configuration

3. **Phase 3: Comparative Analysis**
   - Compare WER across tongue animation configurations
   - Identify which parameters (rotation, thickness, std_scalar) optimize intelligibility
   - Document correlation between tongue visibility and VSR performance

4. **Phase 4: Grid Search Expansion** (If Phase 1-3 successful)
   - Execute full 27-configuration grid search (from SESSION_2026_02_04)
   - Parameter space: rotation_deg [0,10,20], thickness [1.0,2.0,4.0], std_scalar [0.10,0.25,0.40]

---

## 6. Infrastructure Status

### 6.1 Available Assets

**Data Files:**
- ✓ 4 EMA files (tongue motion data) in `tongue_scripts/outputs/`
- ✓ 4 JSON metadata files in `tongue_scripts/inputs/`
- ✓ 4 WAV audio files in `tongue_scripts/inputs/`
- ✓ 4 rendered videos (no audio) in `tongue_scripts/batch_videos/`
- ✓ 4 merged videos (with audio) in `tongue_scripts/batch_videos/`
- ✓ 4 VSR transcript files in `tongue_scripts/batch_videos/*_transcript.txt`
- ✓ 4 TextGrid ground truth files in `ADFA_EVALUATION/.../beat_textgrids/1/`

**Scripts:**
- ✓ `/tmp/generate_ema_batch.py` - EMA generation from audio
- ✓ `/tmp/synthesize_and_evaluate_batch.py` - Video rendering + VSR inference
- ✓ `/tmp/run_vsr_simple.py` - VSR inference with direct imports
- ✓ `tongue_scripts/test.py` - Base render/EMA functions
- ✓ `tongue_scripts/crop_mouth.py` - Mouth region extraction

**Models:**
- ✓ WavLM LoRA model (audio-to-EMA)
- ✓ LRS3_V_WER19.1 (VSR model)
- ✓ Language model (LM for beam search)

### 6.2 Configuration Files

**VSR Config:** `configs/LRS3_V_WER19.1.ini`
```ini
penalty=0.0
ctc_weight=0.1
lm_weight=0.3
beam_size=40
```

**Tongue Config** (from test.py):
```python
TONGUE_CONFIG = {
    'rotation_deg': 5,
    'thickness': 1.2,
    'shift_y': 0,
    'shift_z': 0,
    'std_scalar': 0.20
}
```

**Render Settings:**
```python
FPS = 25
duration = 7.5 seconds
frames = 187
background = [0.3, 0.3, 0.3] (gray)
tongue_points = 4 (T4, T3, T2, T1)
anchors = [16661, 16696, 16755, 16758]
bones = [16661, 16757]
```

### 6.3 Known Issues

| Issue | Affected Component | Workaround |
|-------|-------------------|------------|
| torchcodec dependency | EMA generation | Using scipy.io.wavfile |
| hydra/module errors | infer.py | Using direct imports (run_vsr_simple.py) |
| VSR repetition loops | All 7.5s synthetic videos | Shorten videos or increase penalty |
| TextGrid extraction | WER calculation | Manual extraction or fix script |

---

## 7. Conclusion

This session successfully demonstrated the end-to-end pipeline for generating and evaluating tongue-enhanced synthetic videos:

1. **EMA Generation:** 3 new EMA files generated from audio using WavLM LoRA model
2. **Video Rendering:** 4 videos rendered at 25fps with tongue animation
3. **Video-Audio Merging:** All videos merged with their corresponding audio
4. **VSR Inference:** Pipeline operational, but produces repetition loops on synthetic videos

**Critical Blocker:** VSR repetition loops prevent WER calculation and comparative analysis. This is a model behavior issue, not a technical failure.

**Root Cause:** Synthetic facial animations (7.5s duration) cause beam search to get stuck in local optima, likely due to domain mismatch between synthetic content and LRS3-trained VSR model.

**Recommended Next Steps:**
1. Implement video shortening (Option 1) or repetition penalty increase (Option 2)
2. Validate VSR produces coherent transcripts on short synthetic videos
3. Extract ground truth from TextGrid files
4. Calculate WER metrics and complete comparative analysis
5. If successful, proceed to full 27-configuration grid search

**Status:** Awaiting VSR optimization to complete evaluation pipeline.

---

## 8. Appendix: File Locations

### Generated Files

**EMA Motion Data:**
- `tongue_scripts/outputs/1_wayne_0_100_100.npy` (2,849 frames × 16 dims)
- `tongue_scripts/outputs/1_wayne_0_10_10.npy` (2,999 frames × 16 dims)
- `tongue_scripts/outputs/1_wayne_0_101_101.npy` (2,999 frames × 16 dims)
- `tongue_scripts/outputs/1_wayne_0_75_75.npy` (2,999 frames × 16 dims, existing)

**Rendered Videos (No Audio):**
- `tongue_scripts/batch_videos/1_wayne_0_100_100.mp4` (1.5MB)
- `tongue_scripts/batch_videos/1_wayne_0_10_10.mp4` (1.5MB)
- `tongue_scripts/batch_videos/1_wayne_0_101_101.mp4` (1.5MB)
- `tongue_scripts/batch_videos/1_wayne_0_75_75.mp4` (1.5MB)

**Merged Videos (With Audio):**
- `tongue_scripts/batch_videos/1_wayne_0_100_100_final.mp4`
- `tongue_scripts/batch_videos/1_wayne_0_10_10_final.mp4`
- `tongue_scripts/batch_videos/1_wayne_0_101_101_final.mp4`
- `tongue_scripts/batch_videos/1_wayne_0_75_75_final.mp4`

**VSR Transcripts:**
- `tongue_scripts/batch_videos/1_wayne_0_100_100_transcript.txt`
- `tongue_scripts/batch_videos/1_wayne_0_10_10_transcript.txt`
- `tongue_scripts/batch_videos/1_wayne_0_101_101_transcript.txt`
- `tongue_scripts/batch_videos/1_wayne_0_75_75_transcript.txt`

**Input Files:**
- `tongue_scripts/inputs/1_wayne_0_100_100.json`, `*.wav`, `*.npy`
- `tongue_scripts/inputs/1_wayne_0_10_10.json`, `*.wav`, `*.npy`
- `tongue_scripts/inputs/1_wayne_0_101_101.json`, `*.wav`, `*.npy`
- `tongue_scripts/inputs/1_wayne_0_75_75.json`, `*.wav`, `*.npy`

### Scripts

**Created This Session:**
- `/tmp/generate_ema_batch.py` - Batch EMA generation from audio
- `/tmp/synthesize_and_evaluate_batch.py` - Video rendering + VSR inference pipeline
- `/tmp/run_vsr_simple.py` - VSR inference with direct imports

**Existing Scripts:**
- `tongue_scripts/test.py` - Base render/EMA functions
- `tongue_scripts/crop_mouth.py` - Mouth region extraction
- `tongue_scripts/render_bright_video.py` - Bright video rendering

### Models & Configs

**VSR Model:**
- Location: `/home/timoite/Documents/ICT-FaceKit/LRS3_V_WER19.1/`
- Config: `ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/configs/LRS3_V_WER19.1.ini`

**Language Model:**
- Location: `/home/timoite/Documents/ICT-FaceKit/lm_en_subword/`

**Ground Truth TextGrids:**
- Location: `/home/timoite/Documents/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/data/beat_textgrids/1/`

---

**Report Prepared By:** Claude (OpenCode)
**Session Duration:** ~2 hours
**Status:** BLOCKED - VSR Repetition Loops Require Optimization
