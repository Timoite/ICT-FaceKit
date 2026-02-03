# Tongue Parameter Grid Search - Complete Pipeline

## Overview

This pipeline systematically tests different tongue configuration parameters to identify optimal settings for speech intelligibility in full-face animations.

## Problem Statement

The VSR (Visual Speech Recognition) transcript extracted from our generated animation doesn't match the expected ground truth. We need to determine if:
1. Different tongue configurations improve lip intelligibility
2. The issue is with animation quality vs. VSR model accuracy
3. Certain parameter ranges work better than others

## Grid Search Parameters

### 1. Rotation (degrees)
- **Purpose**: Adjusts tongue's initial pitch/angle
- **Range**: 0° to 20°
- **Values**: 0, 5, 10, 15, 20

### 2. Thickness
- **Purpose**: Scales tongue thickness in Y-direction
- **Range**: 1.0 to 4.0
- **Values**: 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0

### 3. Standard Deviation Scalar
- **Purpose**: Controls amplitude of EMA-driven motion
- **Range**: 0.10 to 0.40
- **Values**: 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40

**Total Combinations**: 5 × 7 × 7 = **245 configurations**

## Files Created

### Main Scripts
1. **`test_tongue_grid_search.py`** - Main grid search engine
   - Renders videos for each configuration
   - Runs VSR inference
   - Saves transcripts and metadata

2. **`analyze_grid_results.py`** - Results analyzer
   - Calculates WER for each configuration
   - Identifies best performing parameters
   - Statistical analysis by parameter

3. **`quick_test.sh`** - Quick test runner (27 configs)
4. **`run_grid_search.sh`** - Full grid search runner (245 configs)

### Output Structure
```
tongue_scripts/
├── test_tongue_grid_search.py
├── analyze_grid_results.py
├── run_grid_search.sh
├── quick_test.sh
└── tongue_param_tests/
    ├── README.md
    ├── rot00_thick1.0_std0.10/
    │   ├── config.json
    │   ├── animation_with_audio.mp4
    │   └── transcript.txt
    ├── rot00_thick1.0_std0.15/
    │   └── ...
    ├── progress.json (live updates)
    ├── all_results.json (complete metadata)
    └── analysis_summary.json (statistics)
```

## Usage

### Option 1: Quick Test (Recommended First)
**Time**: 30-45 minutes
**Configs**: 27 combinations

```bash
cd /home/timoite/Documents/ICT-FaceKit/tongue_scripts
./quick_test.sh
```

### Option 2: Full Grid Search
**Time**: 4-6 hours
**Configs**: 245 combinations

```bash
cd /home/timoite/Documents/ICT-FaceKit/tongue_scripts
./run_grid_search.sh
```

### Option 3: Custom Parameters
Edit `test_tongue_grid_search.py` lines 32-34:
```python
ROTATION_RANGE = [0, 5, 10, 15, 20]
THICKNESS_RANGE = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
STD_SCALAR_RANGE = [0.10, 0.15, 0.20, 0.25, 0.30]
```

Then run: `uv run python tongue_scripts/test_tongue_grid_search.py`

## Analyzing Results

After tests complete:

```bash
python tongue_scripts/analyze_grid_results.py
```

**Output shows**:
- Top 10 best configurations (by WER)
- Configurations containing key words ("disneyland", "childhood", "angry")
- Average WER by parameter value
- Best overall configuration

## Expected Outcomes

### Best Case Scenario
- Some configurations achieve lower WER (better transcript match)
- Clear pattern emerges (e.g., higher std_scalar helps)
- Optimal parameter range identified

### Worst Case Scenario
- All configurations have similar high WER
- No clear pattern in results
- **Conclusion**: Issue may be:
  - EMA data doesn't match audio
  - VSR model struggles with synthetic animations
  - Need different evaluation approach

## Evaluation Metrics

### Primary Metric
- **WER** (Word Error Rate): Lower is better
  - Calculated against ground truth transcript
  - Range: 0% (perfect) to 100% (completely wrong)

### Secondary Metrics
- **Keyword Detection**: Does transcript contain key words?
  - "disneyland"
  - "childhood"  
  - "angry"
- **Transcript Length**: Very short/long transcripts indicate issues

### Parameter Analysis
- Average WER by rotation angle
- Average WER by thickness setting
- Average WER by std_scalar value

## Live Progress Monitoring

During grid search:
```bash
# Check progress
cat tongue_scripts/tongue_param_tests/progress.json

# Watch latest results
tail -f tongue_scripts/grid_search.log

# Check generated videos
ls tongue_scripts/tongue_param_tests/*/*.mp4 | wc -l
```

## Resource Requirements

### Full Grid Search (245 configs)
- **Disk Space**: ~2-3 GB (videos + transcripts)
- **Time**: 4-6 hours
- **CPU**: High utilization during rendering
- **RAM**: 4-8 GB recommended

### Quick Test (27 configs)
- **Disk Space**: ~300-500 MB
- **Time**: 30-45 minutes
- **CPU**: Moderate

## Next Steps After Testing

1. **Review Results**: Run `analyze_grid_results.py`
2. **Select Best Config**: Identify parameters with lowest WER
3. **Visual Inspection**: Watch top 3 videos to assess quality
4. **Update Default Config**: Apply best parameters to `test.py`
5. **Document Findings**: Note optimal parameter ranges

## Troubleshooting

### Out of Memory
- Reduce MAX_SECONDS in script
- Run fewer configurations
- Close other applications

### VSR Errors
- Check model files exist in `ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/data/`
- Verify mediapipe installation

### Rendering Too Slow
- Lower resolution in script (change W, H variables)
- Reduce MAX_SECONDS
- Test smaller parameter subset

## Summary

This systematic grid search will definitively answer:
- ✅ Which tongue parameters produce most intelligible speech
- ✅ Whether parameter tuning improves VSR accuracy
- ✅ Optimal configuration for future animations
- ✅ If the issue is animation quality or VSR model limitations
