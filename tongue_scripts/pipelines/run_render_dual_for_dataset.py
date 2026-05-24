#!/usr/bin/env python3
"""Run render_dual_tongue_comparison.py for a specified dataset id/speaker."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TONGUE_SCRIPTS_DIR = PROJECT_ROOT / "tongue_scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def clip_instance_id(dataset_id: str) -> str:
    """Return the BEAT clip instance id used for organized render outputs."""
    for token in reversed(dataset_id.split("_")):
        if token.isdigit():
            return str(int(token))
    return dataset_id


def clip_output_dir(base_output_dir: Path, speaker_id: str, dataset_id: str) -> Path:
    """Build outputs/<speaker>/<instance> for one clip."""
    return base_output_dir / str(speaker_id) / clip_instance_id(dataset_id)


def video_output_dir(base_output_dir: Path, speaker_id: str, dataset_id: str) -> Path:
    """Build outputs/<speaker>/<instance>/videos for rendered face videos."""
    return clip_output_dir(base_output_dir, speaker_id, dataset_id) / "videos"


def tongue_motion_output_dir(
    base_output_dir: Path, speaker_id: str, dataset_id: str
) -> Path:
    """Build outputs/<speaker>/<instance>/tongue_motion for inversion .npy files."""
    return clip_output_dir(base_output_dir, speaker_id, dataset_id) / "tongue_motion"


# Backwards-compatible name used by older callers in this checkout.
face_animation_output_dir = video_output_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render active/passive tongue videos for a dataset instance."
    )
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--speaker-id", required=True)
    p.add_argument(
        "--beat-root",
        default=str(
            PROJECT_ROOT
            / "data"
            / "beat_cache"
            / "beat_english_v0.2.1"
            / "beat_english_v0.2.1"
        ),
        help="Root containing per-speaker folders",
    )
    p.add_argument("--motion-path", required=True, help="Path to EMA .npy")
    p.add_argument(
        "--output-dir",
        default=str(TONGUE_SCRIPTS_DIR / "outputs"),
        help="Base output directory. Rendered videos are written under <speaker>/<instance>/videos.",
    )
    p.add_argument("--tongue-shift-seconds", type=float, default=0.120)
    p.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use pyrender's EGL/OpenGL GPU backend for offscreen rendering.",
    )
    return p.parse_args()


def configure_render_backend(use_gpu: bool) -> None:
    """Configure the OpenGL backend before importing pyrender-backed renderer code."""
    if not use_gpu:
        return

    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"


def main() -> None:
    args = parse_args()
    configure_render_backend(args.use_gpu)

    from tongue_scripts.rendering import render_dual_tongue_comparison as rd

    speaker_root = Path(args.beat_root) / str(args.speaker_id)
    dataset_id = args.dataset_id

    rd.DATASET_ID = dataset_id
    rd.BEAT_DATA_ROOT = speaker_root
    rd.BEAT_JSON_PATH = str(speaker_root / f"{dataset_id}.json")
    rd.AUDIO_PATH = str(speaker_root / f"{dataset_id}.wav")
    rd.MOTION_PATH = str(Path(args.motion_path))
    rd.TONGUE_SHIFT_SECONDS = float(args.tongue_shift_seconds)

    out_dir = video_output_dir(Path(args.output_dir), str(args.speaker_id), dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    rd.OUTPUT_DIR = out_dir
    rd.OUTPUT_VIDEO_WITH_TONGUE = str(out_dir / f"{dataset_id}_with_tongue.mp4")
    rd.OUTPUT_VIDEO_PASSIVE_TONGUE = str(out_dir / f"{dataset_id}_passive_tongue.mp4")

    rd.main()


if __name__ == "__main__":
    main()
