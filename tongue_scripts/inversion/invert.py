#!/usr/bin/env python3
"""Generate tongue EMA motion (.npy) from WAV using the WavLM inversion model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.signal as sig
from scipy.io import wavfile
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tongue_scripts.inversion.wavlm_lora import WavLMWrapper

# 8Hz low-pass filter on 50fps trajectory
B, A = sig.butter(5, 10, "low", fs=50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer tongue motion .npy from a wav file.")
    parser.add_argument("--wav", required=True, help="Input wav path")
    parser.add_argument("--out", required=True, help="Output npy path")
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "tongue_scripts" / "inversion_checkpoints" / "lora_multispeaker_consistency_alpha_0.25_threshold_0_vctk_vvn_4tonguepoints"),
        help="Model checkpoint path",
    )
    return parser.parse_args()


def infer_ema(wav_path: Path, checkpoint_path: Path) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WavLMWrapper().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    try:
        wav, _ = torchaudio.load(str(wav_path))
    except Exception:
        sr, raw = wavfile.read(str(wav_path))
        if raw.ndim == 1:
            raw = raw[None, :]
        else:
            raw = raw.T
        if raw.dtype.kind in {"i", "u"}:
            max_abs = np.iinfo(raw.dtype).max
            raw = raw.astype(np.float32) / float(max_abs)
        else:
            raw = raw.astype(np.float32)
        wav = torch.from_numpy(raw)

    with torch.no_grad():
        ema = model(wav.to(device)).squeeze().detach().cpu().numpy()

    ema = sig.filtfilt(B, A, ema, axis=0)
    return ema


def main() -> None:
    args = parse_args()
    wav_path = Path(args.wav)
    out_path = Path(args.out)
    checkpoint_path = Path(args.checkpoint)

    if not wav_path.is_file():
        raise SystemExit(f"Missing wav file: {wav_path}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"Missing checkpoint: {checkpoint_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ema = infer_ema(wav_path, checkpoint_path)
    np.save(str(out_path), ema)
    print(f"Saved: {out_path} shape={ema.shape}")


if __name__ == "__main__":
    main()
