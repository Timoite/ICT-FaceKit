#!/usr/bin/env python3
"""
Test what the audio track actually contains using audio ASR.
"""
import sys
import os
from pathlib import Path

# Add VSR module to path
VSR_DIR = Path(__file__).parent.parent / "ADFA_EVALUATION" / "Visual_Speech_Recognition_for_Multiple_Languages"
sys.path.insert(0, str(VSR_DIR))

import torch
from infer_pipeline import InferencePipeline

# Configuration
VIDEO_PATH = str(Path(__file__).parent / "outputs" / "tongue_hybrid_deformation.mp4")

def main():
    print("="*60)
    print("Audio-Only ASR (to check actual audio content)")
    print("="*60)
    print(f"Video: {VIDEO_PATH}")
    print()

    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video file not found: {VIDEO_PATH}")
        sys.exit(1)

    # Setup device (use CPU to avoid CUDA issues)
    device = torch.device("cpu")

    # Initialize pipeline with audio-only modality
    print("Initializing Audio-Only ASR pipeline...")
    print("Note: This may take a moment to load the model")

    try:
        pipeline = InferencePipeline(
            modality="audio",  # Audio-only (not visual)
            model_path=str(VSR_DIR / "data" / "LRS3_A_WER1.0" / "model.pth"),
            model_conf=str(VSR_DIR / "data" / "LRS3_A_WER1.0" / "model.json"),
            detector=None,
            face_track=False,
            device=device
        )

        print("\nRunning audio-only ASR...")
        transcript = pipeline(VIDEO_PATH)

        print("\n" + "="*60)
        print("AUDIO TRANSCRIPT:")
        print("="*60)
        print(transcript)
        print("="*60)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
