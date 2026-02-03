#!/usr/bin/env python3
"""
Test VSR on the provided demo video to see if it works at all.
"""
import sys
from pathlib import Path

VSR_DIR = Path(__file__).parent.parent / "ADFA_EVALUATION" / "Visual_Speech_Recognition_for_Multiple_Languages"
sys.path.insert(0, str(VSR_DIR))

import torch
from infer_pipeline import InferencePipeline

# Test on their demo video
demo_video = str(VSR_DIR / "data" / "26_reamey_0_112_112.mp4")

print("Testing VSR on demo video...")
print(f"Video: {demo_video}")

device = torch.device("cpu")

try:
    pipeline = InferencePipeline(
        config_filename=str(VSR_DIR / "configs" / "LRS3_V_WER32.3.ini"),
        detector="mediapipe",
        face_track=True,
        device=device
    )
    
    transcript = pipeline(demo_video)
    print(f"\nTranscript: {transcript}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
