#!/usr/bin/env python3
"""Batch-render active/passive tongue videos for all datasets of one speaker."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TONGUE_SCRIPTS_DIR = PROJECT_ROOT / "tongue_scripts"
SCRIPT_PATH = Path(__file__).resolve()
RENDER_DATASET_SCRIPT = SCRIPT_PATH.parent / "run_render_dual_for_dataset.py"
INVERT_SCRIPT = TONGUE_SCRIPTS_DIR / "inversion" / "invert.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render all available datasets for one speaker using the dual tongue renderer."
    )
    parser.add_argument("--speaker-id", default="1", help="BEAT speaker id")
    parser.add_argument(
        "--beat-root",
        default=str(PROJECT_ROOT / "data" / "beat_cache" / "beat_english_v0.2.1" / "beat_english_v0.2.1"),
        help="Root containing per-speaker BEAT folders",
    )
    parser.add_argument(
        "--motion-dir",
        default=str(TONGUE_SCRIPTS_DIR / "outputs"),
        help="Directory containing or receiving EMA .npy files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(TONGUE_SCRIPTS_DIR / "outputs" / "speaker_1_wayne"),
        help="Directory for rendered videos",
    )
    parser.add_argument(
        "--tongue-shift-seconds",
        type=float,
        default=0.12,
        help="Global tongue delay in seconds",
    )
    parser.add_argument(
        "--generate-missing-motion",
        action="store_true",
        help="Run invert.py when a dataset has a wav file but no .npy yet",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip datasets whose active/passive with-audio outputs already exist",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for debugging",
    )
    return parser.parse_args()


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def list_dataset_ids(speaker_root: Path) -> list[str]:
    dataset_ids = sorted(path.stem for path in speaker_root.glob("*.json"))
    return dataset_ids


def main() -> None:
    args = parse_args()

    beat_root = Path(args.beat_root)
    speaker_root = beat_root / str(args.speaker_id)
    motion_dir = Path(args.motion_dir)
    output_dir = Path(args.output_dir)

    if not speaker_root.is_dir():
        raise SystemExit(f"Speaker directory not found: {speaker_root}")
    if not RENDER_DATASET_SCRIPT.is_file():
        raise SystemExit(f"Missing render wrapper: {RENDER_DATASET_SCRIPT}")
    if args.generate_missing_motion and not INVERT_SCRIPT.is_file():
        raise SystemExit(f"Missing invert.py: {INVERT_SCRIPT}")

    motion_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_ids = list_dataset_ids(speaker_root)
    if args.limit is not None:
        dataset_ids = dataset_ids[: args.limit]

    if not dataset_ids:
        raise SystemExit(f"No dataset JSON files found in {speaker_root}")

    print(f"Speaker {args.speaker_id}: discovered {len(dataset_ids)} dataset ids")

    rendered = 0
    skipped_existing = 0
    skipped_missing_wav = 0
    skipped_missing_motion = 0
    failed: list[str] = []

    for dataset_id in dataset_ids:
        wav_path = speaker_root / f"{dataset_id}.wav"
        motion_path = motion_dir / f"{dataset_id}.npy"
        active_with_audio = output_dir / f"{dataset_id}_with_tongue_with_audio.mp4"
        passive_with_audio = output_dir / f"{dataset_id}_passive_tongue_with_audio.mp4"

        if args.skip_existing and active_with_audio.exists() and passive_with_audio.exists():
            print(f"[SKIP existing] {dataset_id}")
            skipped_existing += 1
            continue

        if not motion_path.exists():
            if not wav_path.exists():
                print(f"[SKIP missing wav] {dataset_id}")
                skipped_missing_wav += 1
                continue

            if not args.generate_missing_motion:
                print(f"[SKIP missing motion] {dataset_id}")
                skipped_missing_motion += 1
                continue

            try:
                run(
                    [
                        sys.executable,
                        str(INVERT_SCRIPT),
                        "--wav",
                        str(wav_path),
                        "--out",
                        str(motion_path),
                    ],
                    cwd=SCRIPT_PATH.parent,
                )
            except subprocess.CalledProcessError:
                failed.append(dataset_id)
                continue

        try:
            run(
                [
                    sys.executable,
                    str(RENDER_DATASET_SCRIPT),
                    "--dataset-id",
                    dataset_id,
                    "--speaker-id",
                    str(args.speaker_id),
                    "--beat-root",
                    str(beat_root),
                    "--motion-path",
                    str(motion_path),
                    "--output-dir",
                    str(output_dir),
                    "--tongue-shift-seconds",
                    str(args.tongue_shift_seconds),
                ],
                cwd=SCRIPT_PATH.parent,
            )
            rendered += 1
        except subprocess.CalledProcessError:
            failed.append(dataset_id)

    print("\n=== Summary ===")
    print(f"Rendered: {rendered}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Skipped missing wav: {skipped_missing_wav}")
    print(f"Skipped missing motion: {skipped_missing_motion}")
    print(f"Failed: {len(failed)}")
    if failed:
        print("Failed dataset ids:")
        for dataset_id in failed:
            print(f"  - {dataset_id}")


if __name__ == "__main__":
    main()
