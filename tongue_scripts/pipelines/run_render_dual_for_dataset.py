#!/usr/bin/env python3
"""Run render_dual_tongue_comparison.py for a specified dataset id/speaker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TONGUE_SCRIPTS_DIR = PROJECT_ROOT / "tongue_scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tongue_scripts.rendering import render_dual_tongue_comparison as rd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render active/passive tongue videos for a dataset instance.")
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--speaker-id", required=True)
    p.add_argument(
        "--beat-root",
        default=str(PROJECT_ROOT / "data" / "beat_cache" / "beat_english_v0.2.1" / "beat_english_v0.2.1"),
        help="Root containing per-speaker folders",
    )
    p.add_argument("--motion-path", required=True, help="Path to EMA .npy")
    p.add_argument(
        "--output-dir",
        default=str(TONGUE_SCRIPTS_DIR / "outputs" / "multi_speaker"),
    )
    p.add_argument("--tongue-shift-seconds", type=float, default=0.120)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    speaker_root = Path(args.beat_root) / str(args.speaker_id)
    dataset_id = args.dataset_id

    rd.DATASET_ID = dataset_id
    rd.BEAT_DATA_ROOT = speaker_root
    rd.BEAT_JSON_PATH = str(speaker_root / f"{dataset_id}.json")
    rd.AUDIO_PATH = str(speaker_root / f"{dataset_id}.wav")
    rd.MOTION_PATH = str(Path(args.motion_path))
    rd.TONGUE_SHIFT_SECONDS = float(args.tongue_shift_seconds)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rd.OUTPUT_DIR = out_dir
    rd.OUTPUT_VIDEO_WITH_TONGUE = str(out_dir / f"{dataset_id}_with_tongue.mp4")
    rd.OUTPUT_VIDEO_PASSIVE_TONGUE = str(out_dir / f"{dataset_id}_passive_tongue.mp4")

    rd.main()


if __name__ == "__main__":
    main()
