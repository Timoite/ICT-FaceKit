#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "torchaudio",
#     "scipy",
#     "numpy",
#     "trimesh",
#     "pyrender",
#     "opencv-python",
#     "tqdm",
#     "transformers",
#     "loralib",
#     "matplotlib",
# ]
# ///
"""
Batch Pipeline for Speaker 1
----------------------------
1. Inference: Audio -> Tongue Animation (.npy) using WavLM Lora.
2. Rendering: Face + Tongue -> Video (.mp4) at 25fps.
   - Includes "Humane" skin tone.
   - Includes "Moving Camera" visualization.
   - Includes "Lip Correction" (Bias to mouthClose).
"""

import sys
import os
import torch
import torchaudio
import scipy.signal as sig
import numpy as np
import trimesh
import pyrender
import cv2
import json
import subprocess
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import interp1d

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

# Import existing modules
try:
    from wavlm_lora import WavLMWrapper
    from face_model_io_trimesh import load_face_model_trimesh
    from test import process_beat_data, load_ema_motion, FaceKitTongueRig, TONGUE_CONFIG
except ImportError:
    # Try parent dir
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    from tongue_scripts.wavlm_lora import WavLMWrapper
    from face_model_io_trimesh import load_face_model_trimesh
    from tongue_scripts.test import process_beat_data, load_ema_motion, FaceKitTongueRig, TONGUE_CONFIG

# Add batch_render_corrected
try:
    from tongue_scripts.batch_render_corrected import render_bright_video
except ImportError:
    sys.path.insert(0, str(SCRIPT_DIR))
    from batch_render_corrected import render_bright_video

# Global Config
PROJECT_ROOT = SCRIPT_DIR.parent
FACE_MODEL_DIR = str(PROJECT_ROOT / "FaceXModel")
# Update Input Dir to Full Dataset
INPUT_DIR = PROJECT_ROOT / "ADFA_EVALUATION" / "data" / "beat_cache_speaker1" / "beat_english_v0.2.1" / "beat_english_v0.2.1" / "1"
OUTPUT_NPY_DIR = SCRIPT_DIR / "outputs"
BATCH_OUTPUT_DIR = SCRIPT_DIR / "batch_outputs"
CHECKPOINT_PATH = SCRIPT_DIR / "inversion_checkpoints" / "lora_multispeaker_consistency_alpha_0.25_threshold_0_vctk_vvn_4tonguepoints"
STD_PATH = SCRIPT_DIR / "normalising_vectors" / "JW13_4points_std.npy"

# Render Config (Only for Inference Truncation)
MAX_DURATION = 7.5 # Seconds
# Filter for inference
b, a = sig.butter(5, 10, "low", fs=50)

class BatchPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load Inference Model
        print("Loading WavLM Model...")
        self.inv_model = WavLMWrapper().to(self.device)
        state_dict = torch.load(CHECKPOINT_PATH, map_location=self.device)
        self.inv_model.load_state_dict(state_dict)
        self.inv_model.eval()
        
        # Directories
        OUTPUT_NPY_DIR.mkdir(exist_ok=True)
        BATCH_OUTPUT_DIR.mkdir(exist_ok=True)

    def infer_tongue(self, wav_path, output_name):
        """Generate .npy from .wav"""
        print(f"  Inferring tongue for {wav_path.name}...")
        # Custom loading using scipy
        import scipy.io.wavfile as wavfile
        sr, d = wavfile.read(str(wav_path))
        
        d = d.astype(np.float32)
        # Normalize if integer type
        if d.max() > 1.0 or d.min() < -1.0:
             d = d / 32768.0
             
        data = torch.from_numpy(d).float()
        if data.ndim == 1: 
            data = data.unsqueeze(0) # (T,) -> (1, T)
        else:
            data = data.t() # (T, C) -> (C, T)

        # Truncate to MAX_DURATION
        max_samples = int(MAX_DURATION * sr)
        if data.shape[1] > max_samples:
            data = data[:, :max_samples]
            
        with torch.no_grad():
            if self.device.type == 'cuda':
                data = data.to(self.device)
            if data.ndim == 1:
                data = data.unsqueeze(0)
            ema = self.inv_model(data).squeeze().detach().cpu().numpy()
            
        # Filter
        ema = sig.filtfilt(b, a, ema, axis=0)
        
        out_path = OUTPUT_NPY_DIR / f"{output_name}.npy"
        np.save(str(out_path), ema)
        return out_path

    def run(self):
        print("Starting Batch Pipeline...")
        # Find 1_*.wav
        if not INPUT_DIR.exists():
            print(f"CRITICAL: Input directory {INPUT_DIR} does not exist.")
            return

        wav_files = sorted(list(INPUT_DIR.glob("1_*.wav")))
        print(f"Found {len(wav_files)} audio files in {INPUT_DIR}.")
        
        for wav_path in tqdm(wav_files):
            dataset_id = wav_path.stem # e.g. "1_wayne_0_75_75"
            json_path = INPUT_DIR / f"{dataset_id}.json"
            output_vid = BATCH_OUTPUT_DIR / f"{dataset_id}_corrected.mp4"
            
            if not json_path.exists():
                print(f"Skipping {dataset_id}: JSON not found.")
                continue

            try:
                # 1. Inference
                npy_path = self.infer_tongue(wav_path, dataset_id)
                
                # 2. Render (Delegated to batch_render_corrected)
                render_bright_video(
                    dataset_id=dataset_id,
                    wav_path=str(wav_path),
                    json_path=str(json_path),
                    npy_path=str(npy_path),
                    output_path=str(output_vid)
                )
                
            except Exception as e:
                print(f"❌ Error processing {dataset_id}: {e}")

if __name__ == "__main__":
    pipeline = BatchPipeline()
    pipeline.run()
