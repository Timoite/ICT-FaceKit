#!/usr/bin/env python3
"""
Test VSR inference WITH language model on already generated videos.
"""
import sys
from pathlib import Path
import json

VSR_DIR = Path(__file__).parent.parent / "ADFA_EVALUATION" / "Visual_Speech_Recognition_for_Multiple_Languages"
TEST_OUTPUT_DIR = Path(__file__).parent / "tongue_param_tests"

sys.path.insert(0, str(VSR_DIR))

import torch
from pipelines.model import AVSR
from pipelines.data.data_module import AVSRDataLoader
from pipelines.detectors.mediapipe.detector import LandmarksDetector

# Model paths - using newly downloaded models
MODEL_PATH = str(Path(__file__).parent.parent / "LRS3_V_WER19.1" / "model.pth")
MODEL_CONF = str(Path(__file__).parent.parent / "LRS3_V_WER19.1" / "model.json")
LM_PATH = str(Path(__file__).parent.parent / "lm_en_subword" / "model.pth")
LM_CONF = str(Path(__file__).parent.parent / "lm_en_subword" / "model.json")

print("="*60)
print("VSR WITH LANGUAGE MODEL TEST")
print("="*60)
print(f"Model: {MODEL_PATH}")
print(f"Config: {MODEL_CONF}")
print(f"Language Model: {LM_PATH}")
print(f"LM Config: {LM_CONF}")
print()

# Get list of completed videos
completed_dirs = sorted([d for d in TEST_OUTPUT_DIR.iterdir() if d.is_dir() and (d / "transcript.txt").exists()])

print(f"Found {len(completed_dirs)} completed configurations")
print()

device = torch.device("cpu")

# Test each one
results = []
for config_dir in completed_dirs[:5]:  # Test first 5
    config_name = config_dir.name
    video_path = config_dir / "animation_with_audio.mp4"
    
    if not video_path.exists():
        print(f"✗ {config_name}: No video found")
        continue
    
    print(f"\n[{config_name}]")
    print(f"  Video: {video_path}")
    
    try:
        # Initialize dataloader
        dataloader = AVSRDataLoader(modality="video", speed_rate=50.0/25.0, detector="mediapipe")
        
        # Load models
        print(f"  Loading models...")
        model = AVSR(
            modality="video",
            model_path=MODEL_PATH,
            model_conf=MODEL_CONF,
            rnnlm=LM_PATH,
            rnnlm_conf=LM_CONF,
            penalty=0.0,
            ctc_weight=0.1,
            lm_weight=0.0,  # Start with no LM weight, can try 0.1-0.3
            beam_size=40,
            device=device
        )
        
        # Detect landmarks
        print(f"  Detecting landmarks...")
        detector = LandmarksDetector()
        landmarks = detector(str(video_path))
        
        # Load video data
        print(f"  Loading video data...")
        data = dataloader.load_data(str(video_path), landmarks)
        
        # Run inference
        print(f"  Running inference...")
        transcript = model.infer(data)
        
        # Clean transcript
        transcript = transcript.replace("▁", " ").strip()
        
        print(f"  ✓ Transcript: {transcript[:150]}...")
        
        # Save new transcript
        output_file = config_dir / "transcript_with_lm.txt"
        with open(output_file, "w") as f:
            f.write(transcript)
        
        results.append({
            "config": config_name,
            "transcript": transcript,
            "length": len(transcript)
        })
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Successfully processed: {len(results)}")

for r in results:
    print(f"\n{r['config']}:")
    print(f"  Length: {r['length']} chars")
    print(f"  Preview: {r['transcript'][:100]}...")

print(f"\nResults saved to: */transcript_with_lm.txt")
