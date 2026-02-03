#!/usr/bin/env python3
"""
Simple VSR inference using the VSR pipeline directly.
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
VSR_CONFIG = str(VSR_DIR / "configs" / "LRS3_V_WER19.1_50fps.ini")

def main():
    print("="*60)
    print("VSR Inference on Generated Animation")
    print("="*60)
    print(f"Video: {VIDEO_PATH}")
    print(f"Config: {VSR_CONFIG}")
    print()

    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video file not found: {VIDEO_PATH}")
        sys.exit(1)

    # Setup device (use CPU to avoid CUDA issues)
    device = torch.device("cpu")

    # Initialize pipeline
    print("Initializing VSR pipeline...")
    print("Note: This may take a moment to load the model")

    try:
        pipeline = InferencePipeline(
            modality="video",  # Visual-only
            model_path=str(VSR_DIR / "data" / "LRS3_V_WER19.1" / "model.pth"),
            model_conf=str(VSR_DIR / "data" / "LRS3_V_WER19.1" / "model.json"),
            detector="mediapipe",
            face_track=True,
            device=device
        )

        print("\nRunning inference...")
        transcript = pipeline(VIDEO_PATH)

        print("\n" + "="*60)
        print("EXTRACTED TRANSCRIPT:")
        print("="*60)
        print(transcript)
        print("="*60)

        # Save transcript
        output_file = Path(__file__).parent / "outputs" / "transcript.txt"
        with open(output_file, "w") as f:
            f.write(transcript)
        print(f"\nTranscript saved to: {output_file}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
