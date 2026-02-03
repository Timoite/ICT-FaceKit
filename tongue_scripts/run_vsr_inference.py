#!/usr/bin/env python3
"""
Simple script to extract transcript from generated video using VSR model.
"""
import sys
import os
from pathlib import Path

# Add VSR module to path
VSR_DIR = Path(__file__).parent.parent / "ADFA_EVALUATION" / "Visual_Speech_Recognition_for_Multiple_Languages"
sys.path.insert(0, str(VSR_DIR))

import torch
from pipelines.model import AVSR
from pipelines.data.data_module import AVSRDataLoader
from pipelines.detectors.mediapipe.detector import LandmarksDetector

# Configuration
VIDEO_PATH = Path(__file__).parent / "outputs" / "tongue_hybrid_deformation.mp4"
MODEL_CONF = str(VSR_DIR / "configs" / "LRS3_V_WER19.1.ini")
MODEL_PATH = str(VSR_DIR / "data" / "LRS3_V_WER19.1" / "model.pth")

def main():
    print("="*60)
    print("VSR Inference on Generated Animation")
    print("="*60)
    print(f"Video: {VIDEO_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Config: {MODEL_CONF}")
    print()

    if not VIDEO_PATH.exists():
        print(f"ERROR: Video file not found: {VIDEO_PATH}")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        print("Please download the model first.")
        sys.exit(1)

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    print("\nLoading VSR model...")
    modality = "video"  # Visual-only speech recognition
    dataloader = AVSRDataLoader(modality, detector="mediapipe")
    model = AVSR(
        modality,
        MODEL_PATH,
        MODEL_CONF,
        rnnlm=None,
        rnnlm_conf=None,
        penalty=0.0,
        ctc_weight=0.1,
        lm_weight=0.0,
        beam_size=40,
        device=device
    )

    # Detect landmarks and load data
    print("Detecting facial landmarks...")
    detector = LandmarksDetector()
    landmarks = detector(str(VIDEO_PATH))

    print("Loading video data...")
    data = dataloader.load_data(str(VIDEO_PATH), landmarks)

    if data is None:
        print("ERROR: Failed to load video data")
        sys.exit(1)

    # Run inference
    print("Running VSR inference...")
    transcript = model.infer(data)

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

if __name__ == "__main__":
    main()
