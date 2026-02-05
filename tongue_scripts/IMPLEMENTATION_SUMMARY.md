# VSR Evaluation Fix & Grid Search Rework - Implementation Summary

## Overview

All required scripts have been created to fix the VSR evaluation system and implement a corrected 25fps grid search pipeline.

## Problem Identified

**Root Cause**: Frame rate mismatch
- Original: Rendered 50fps videos with `speed_rate=2.0` for a 25fps-trained VSR model
- Effect: Destructive frame subsampling → temporal aliasing → gibberish transcripts
- Evidence: "THE ANALYST AND THE ANALYST AND THE ANALYST..." repeated across all 27 configs

## Solution Implemented

### Critical Changes
1. **FPS**: 50 → 25 (render at VSR model's native frame rate)
2. **speed_rate**: 2.0 → 1.0 (no frame rate conversion)
3. **Output directory**: `tongue_param_tests_25fps/` (separate from broken results)

## Files Created

### Phase 1: VSR Validation
- **`validate_vsr_25fps.py`**: Tests VSR on `sample_25fps.mp4` with correct configuration
  - Expected output: "COMPLETELY CONCENTRATED ENVIRONMENTS WHERE WE HAVE..."
  - Validates no repetitive patterns
  - Checks transcript quality

### Phase 2: Ground Truth Extraction
- **`extract_ground_truth.py`**: Parses BEAT TextGrid to extract reference transcript
  - Uses existing `compute_wer.py` utilities
  - Outputs: `ground_truth.txt` (97 words over 27 seconds)

### Phase 3: Corrected Grid Search
- **`test_tongue_grid_search_25fps.py`**: Fixed grid search with 25fps rendering
  - **Line 49**: `FPS = 25` (changed from 50)
  - **Line 391**: `speed_rate=1.0` (changed from 2.0)
  - **Line 402**: `lm_weight=0.3` (language model enabled)
  - Integrated WER calculation for each configuration
  - Tests 27 configurations (3×3×3 grid)

### Phase 4: WER Evaluation
- **`compute_single_wer.py`**: Calculate WER for single transcript
  - Usage: `python compute_single_wer.py <transcript.txt> [<textgrid.TextGrid>]`
  - Returns detailed WER statistics

- **`generate_wer_report.py`**: Batch WER calculation for all results
  - Output: `wer_report.csv` sorted by WER
  - Shows top 5 configurations
  - Provides WER statistics (mean, median, min, max)

### Documentation
- **`VSR_VALIDATION_GUIDE.md`**: Expected results and troubleshooting
- **`EXECUTION_GUIDE.md`**: Complete execution script (bash)
- **`IMPLEMENTATION_SUMMARY.md`**: This file

## Execution Workflow

### Quick Start (Single Command)
```bash
cd /home/timoite/Documents/ICT-FaceKit
uv run python tongue_scripts/validate_vsr_25fps.py && \
uv run python tongue_scripts/extract_ground_truth.py && \
uv run python tongue_scripts/test_tongue_grid_search_25fps.py && \
uv run python tongue_scripts/generate_wer_report.py
```

### Step-by-Step Execution

#### Step 1: Validate VSR
```bash
cd /home/timoite/Documents/ICT-FaceKit
uv run python tongue_scripts/validate_vsr_25fps.py
```

**Expected output**:
```
COMPLETELY CONCENTRATED ENVIRONMENTS WHERE WE HAVE LARGE CHANGES IN GET POSTS AND
```

**Success criteria**:
- ✓ No repetitive patterns
- ✓ Transcript length > 50 characters
- ✓ Recognizable English words

#### Step 2: Extract Ground Truth
```bash
uv run python tongue_scripts/extract_ground_truth.py
```

**Output**: `ground_truth.txt` (97 words)

#### Step 3: Run Grid Search
```bash
uv run python tongue_scripts/test_tongue_grid_search_25fps.py
```

**Configuration**:
- 27 parameter combinations
- 10-second clips per configuration
- Estimated time: ~18 minutes

**Monitor progress**:
```bash
tail -f tongue_scripts/tongue_param_tests_25fps/progress.json
```

#### Step 4: Generate WER Report
```bash
uv run python tongue_scripts/generate_wer_report.py
```

**Output**: `tongue_param_tests_25fps/wer_report.csv`

**View results**:
```bash
head -10 tongue_scripts/tongue_param_tests_25fps/wer_report.csv
```

## Success Metrics

### Primary Metrics

| Metric | Current (Broken) | Target | Stretch |
|--------|------------------|--------|---------|
| **WER** | ~100% (gibberish) | <50% | <30% |
| **Transcript Quality** | Repetitive gibberish | Coherent English | Comprehensible |
| **Parameter Discovery** | N/A | Best config identified | WER improvement >20% |

### Validation Gates

**Gate 1** - VSR Validation:
- ✅ Pass: Coherent transcript from `sample_25fps.mp4`
- ❌ Fail: Debug model paths, dependencies, MediaPipe

**Gate 2** - Grid Search:
- ✅ Pass: ≥5/27 configs produce readable transcripts
- ❌ Fail: Re-examine VSR configuration

**Gate 3** - WER Evaluation:
- ✅ Pass: Best WER <50%
- ❌ Fail: Expand parameter grid

## Parameter Grid

### Coarse Grid (First Pass)
```python
ROTATION_RANGE = [0, 10, 20]      # degrees
THICKNESS_RANGE = [1.0, 2.0, 4.0]
STD_SCALAR_RANGE = [0.10, 0.25, 0.40]
# Total: 27 configurations
```

### Fine Grid (If Results Promising)
```python
ROTATION_RANGE = [0, 5, 10, 15, 20]           # 5 values
THICKNESS_RANGE = [1.0, 1.5, 2.0, 3.0, 4.0]   # 5 values
STD_SCALAR_RANGE = [0.10, 0.18, 0.25, 0.32, 0.40]  # 5 values
# Total: 125 configurations
```

## Technical Details

### Frame Rate Math

**Incorrect** (current broken):
```python
FPS = 50  # Render at 50fps
speed_rate = 50.0 / 25.0  # = 2.0
# → Subsamples every other frame → temporal aliasing → gibberish
```

**Correct** (proposed fix):
```python
FPS = 25  # Render at 25fps
speed_rate = 1.0  # No conversion
# → Native frame rate → clean temporal coherence → intelligible
```

### Why 25fps Over 50fps?

| Factor | 25fps | 50fps |
|--------|-------|-------|
| VSR Compatibility | ✅ Native | ❌ Requires subsampling |
| Rendering Speed | ✅ 2× faster | ❌ Slower |
| Temporal Quality | ✅ Clean | ⚠️ Aliasing risk |
| VSR Performance | ✅ Optimal | ❌ Unproven |

## Troubleshooting

### Issue 1: VSR Still Fails on 25fps
**Symptoms**: Still getting gibberish or errors

**Solutions**:
1. Verify model file integrity:
   ```bash
   ls -lh LRS3_V_WER19.1/model.pth
   ls -lh lm_en_subword/model.pth
   ```
2. Test with official inference script
3. Check MediaPipe detection:
   ```python
   from pipelines.detectors.mediapipe.detector import LandmarksDetector
   detector = LandmarksDetector()
   landmarks = detector("sample_25fps.mp4")
   print(f"Detected {len(landmarks)} frames")
   ```

### Issue 2: All Configs Still Poor (WER ≥50%)
**Symptoms**: WER doesn't improve across configurations

**Solutions**:
1. Verify video renders correctly (check frame rate)
2. Check tongue rig implementation
3. Test individual parameters in isolation
4. Expand parameter ranges

### Issue 3: WER Calculation Errors
**Symptoms**: WER = infinity or NaN

**Solutions**:
1. Ensure ground truth has words
2. Check transcript normalization
3. Verify TextGrid parsing

## Expected Outcomes

### If Successful (WER <50%)
1. Identify best tongue configuration
2. Validate on full 27-second video
3. Test on additional BEAT samples
4. Expand to fine parameter grid (125 configs)

### If Unsuccessful (WER ≥50%)
1. Investigate tongue rig implementation bugs
2. Test with known-good video (real speaker footage)
3. Consider alternative VSR models
4. Analyze failure patterns

## File Structure

```
tongue_scripts/
├── validate_vsr_25fps.py              # VSR validation script
├── extract_ground_truth.py            # Ground truth extraction
├── test_tongue_grid_search_25fps.py   # Corrected grid search
├── compute_single_wer.py              # Single-file WER calculator
├── generate_wer_report.py             # Batch WER report generator
├── VSR_VALIDATION_GUIDE.md            # Validation guide
├── EXECUTION_GUIDE.md                 # Execution script
└── IMPLEMENTATION_SUMMARY.md          # This file

tongue_param_tests_25fps/              # Results directory (created during execution)
├── rot00_thick1.0_std0.10/            # Example configuration
│   ├── config.json                    # Parameter metadata
│   ├── animation_with_audio.mp4       # Rendered video
│   └── transcript.txt                 # VSR transcript
├── ...                                 # Other 26 configurations
├── progress.json                      # Real-time progress
├── all_results.json                   # Complete results
└── wer_report.csv                     # Sorted WER rankings
```

## Next Steps

### Immediate
1. Run `validate_vsr_25fps.py` to confirm VSR works
2. Execute full grid search
3. Review WER report

### Post-Grid Search
1. **Best config validation**: Run on full 27-second video
2. **Parameter analysis**: Identify trends in rotation, thickness, std_scalar
3. **Qualitative review**: Compare top 5 transcripts manually
4. **Expand grid**: If results promising, run fine 125-config grid

## References

- **BEAT TextGrid**: `ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/data/beat_textgrids/1/1_wayne_0_75_75.TextGrid`
- **VSR Config**: `ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/configs/LRS3_V_WER19.1.ini`
- **Original Grid Search**: `tongue_scripts/test_tongue_grid_search.py` (has FPS/speed_rate bug)
- **WER Utilities**: `ADFA_EVALUATION/compute_wer.py`

## Contact

For issues or questions, refer to:
- VSR validation guide: `VSR_VALIDATION_GUIDE.md`
- Execution script: `EXECUTION_GUIDE.md`
