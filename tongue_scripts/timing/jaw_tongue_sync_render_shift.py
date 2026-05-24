#!/usr/bin/env python3
"""
Apply a global time shift to a WavLM tongue .npy.
Positive shift delays tongue (pushes motion later).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
TONGUE_SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TONGUE_SCRIPTS_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shift tongue .npy.")
    parser.add_argument("--dataset-id", default="1_wayne_0_10_10", help="BEAT clip id")
    parser.add_argument(
        "--tongue-npy-dir",
        default=str(TONGUE_SCRIPTS_DIR / "outputs"),
        help="Directory containing WavLM tongue .npy files",
    )
    parser.add_argument(
        "--output-npy-dir",
        default=str(TONGUE_SCRIPTS_DIR / "outputs_shifted"),
        help="Directory to write shifted .npy files",
    )
    parser.add_argument("--tongue-fps", type=float, default=50.0, help="FPS of WavLM tongue output")
    parser.add_argument("--shift-seconds", type=float, default=0.0, help="Shift seconds (positive delays tongue)")
    parser.add_argument(
        "--pad-mode",
        default="edge",
        choices=["edge", "zero"],
        help="Padding mode when shifting",
    )
    return parser.parse_args()


def shift_sequence(seq: np.ndarray, shift_frames: int, pad_mode: str) -> np.ndarray:
    if shift_frames == 0:
        return seq

    n = len(seq)
    if pad_mode == "zero":
        pad_value_start = np.zeros_like(seq[0])
        pad_value_end = np.zeros_like(seq[0])
    else:
        pad_value_start = seq[0]
        pad_value_end = seq[-1]

    if shift_frames > 0:
        pad = np.repeat(pad_value_start[None, ...], shift_frames, axis=0)
        shifted = np.concatenate([pad, seq], axis=0)[:n]
    else:
        shift_frames = abs(shift_frames)
        pad = np.repeat(pad_value_end[None, ...], shift_frames, axis=0)
        shifted = np.concatenate([seq[shift_frames:], pad], axis=0)
    return shifted


def main() -> None:
    args = parse_args()
    dataset_id = args.dataset_id

    npy_path = Path(args.tongue_npy_dir) / f"{dataset_id}.npy"
    output_npy_dir = Path(args.output_npy_dir)
    output_npy_dir.mkdir(parents=True, exist_ok=True)

    if not npy_path.exists():
        raise FileNotFoundError(f"Missing tongue .npy: {npy_path}")

    shift_frames = int(round(args.shift_seconds * args.tongue_fps))
    raw = np.load(npy_path)
    shifted = shift_sequence(raw, shift_frames, args.pad_mode)

    shift_ms = int(round(args.shift_seconds * 1000))
    out_name = f"{dataset_id}_shift_{shift_ms}ms.npy"
    out_path = output_npy_dir / out_name
    np.save(out_path, shifted)

    print(f"Shifted .npy saved: {out_path}")


if __name__ == "__main__":
    main()
