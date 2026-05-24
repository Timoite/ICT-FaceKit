#!/usr/bin/env python3
"""Generate tongue EMA motion (.npy) from WAV using the WavLM inversion model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.signal as sig
import torch
import torchaudio
from scipy.io import wavfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tongue_scripts.inversion.wavlm_lora import WavLMWrapper

# 8Hz low-pass filter on 50fps trajectory
B, A = sig.butter(5, 10, "low", fs=50)
DEFAULT_MAX_SECONDS = 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer tongue motion .npy from a wav file."
    )
    parser.add_argument("--wav", required=True, help="Input wav path")
    parser.add_argument("--out", required=True, help="Output npy path")
    parser.add_argument(
        "--checkpoint",
        default=str(
            PROJECT_ROOT
            / "tongue_scripts"
            / "inversion_checkpoints"
            / "lora_multispeaker_consistency_alpha_0.25_threshold_0_vctk_vvn_4tonguepoints"
        ),
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
        help="Only use the first N seconds of audio. Use 0 or a negative value to disable truncation.",
    )
    return parser.parse_args()


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    checkpoint_path: Path, device: torch.device | None = None
) -> WavLMWrapper:
    """Load the WavLM LoRA inversion model once for single-clip or batch inference."""
    if device is None:
        device = default_device()

    model = WavLMWrapper().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def truncate_wav(
    wav: torch.Tensor, sample_rate: int, max_seconds: float | None
) -> torch.Tensor:
    if max_seconds is None or max_seconds <= 0:
        return wav

    max_samples = int(round(float(max_seconds) * float(sample_rate)))
    if max_samples <= 0 or wav.shape[-1] <= max_samples:
        return wav
    return wav[..., :max_samples]


def load_wav(
    wav_path: Path, max_seconds: float | None = DEFAULT_MAX_SECONDS
) -> torch.Tensor:
    try:
        wav, sample_rate = torchaudio.load(str(wav_path))
    except Exception:
        sample_rate, raw = wavfile.read(str(wav_path))
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
    return truncate_wav(wav, int(sample_rate), max_seconds)


def infer_ema(
    wav_path: Path,
    checkpoint_path: Path | None = None,
    *,
    model: WavLMWrapper | None = None,
    device: torch.device | None = None,
    max_seconds: float | None = DEFAULT_MAX_SECONDS,
) -> np.ndarray:
    """Infer and low-pass filter tongue EMA motion for one wav file.

    Pass a preloaded ``model`` for batch inference to avoid reloading the checkpoint
    for every clip. The old ``infer_ema(wav_path, checkpoint_path)`` call style is
    still supported.
    """
    if device is None:
        device = default_device()
    if model is None:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required when model is not provided")
        model = load_model(checkpoint_path, device)

    wav = load_wav(wav_path, max_seconds=max_seconds)
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
    ema = infer_ema(wav_path, checkpoint_path, max_seconds=args.max_seconds)
    np.save(str(out_path), ema)
    print(f"Saved: {out_path} shape={ema.shape}")


if __name__ == "__main__":
    main()
