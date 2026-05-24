#!/usr/bin/env python3
"""Batch-generate tongue motion .npy files for a BEAT-style dataset.

The output layout is intentionally compatible with the render wrapper helpers:

    <output-dir>/<speaker_id>/<clip_instance>/tongue_motion/<dataset_id>.npy

A CSV manifest is written beside the outputs so downstream experiments can consume
all generated paths without re-scanning the dataset tree.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TONGUE_SCRIPTS_DIR = PROJECT_ROOT / "tongue_scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tongue_scripts.inversion.invert import (
    DEFAULT_MAX_SECONDS,
    default_device,
    infer_ema,
    load_model,
)
from tongue_scripts.pipelines.run_render_dual_for_dataset import (
    tongue_motion_output_dir,
)

DEFAULT_BEAT_ROOT = PROJECT_ROOT / "data" / "beat_cache" / "beat_english_v0.2.1"
FALLBACK_BEAT_ROOTS = [
    PROJECT_ROOT / "data" / "beat_cache" / "beat_english_v0.2.1",
    PROJECT_ROOT / "ADFA_EVALUATION" / "data" / "beat_cache" / "beat_english_v0.2.1",
]
DEFAULT_OUTPUT_DIR = TONGUE_SCRIPTS_DIR / "outputs"
DEFAULT_CHECKPOINT = (
    TONGUE_SCRIPTS_DIR
    / "inversion_checkpoints"
    / "lora_multispeaker_consistency_alpha_0.25_threshold_0_vctk_vvn_4tonguepoints"
)


@dataclass(frozen=True)
class ClipJob:
    speaker_id: str
    dataset_id: str
    wav_path: Path
    out_path: Path
    has_json: bool
    has_textgrid: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Infer tongue motion .npy files for every wav in a BEAT-style dataset."
    )
    p.add_argument(
        "--beat-root",
        default=str(DEFAULT_BEAT_ROOT),
        help=(
            "Dataset root containing per-speaker folders with .wav files. "
            "Both the outer beat_english_v0.2.1 folder and the inner "
            "beat_english_v0.2.1/beat_english_v0.2.1 folder are accepted. "
            "Default uses the project-level data/beat_cache symlink."
        ),
    )
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Base output directory. Files are written under "
            "<output-dir>/<speaker>/<clip-instance>/tongue_motion/."
        ),
    )
    p.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="WavLM LoRA checkpoint path used by invert.py.",
    )
    p.add_argument(
        "--speaker-ids",
        nargs="+",
        default=None,
        help="Optional speaker ids to process. Default: all speaker directories.",
    )
    p.add_argument(
        "--require-json",
        action="store_true",
        help="Only process clips that also have a sibling BEAT .json file.",
    )
    p.add_argument(
        "--require-textgrid",
        action="store_true",
        help="Only process clips that also have a sibling .TextGrid file.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate .npy files even if the output already exists.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of clips to process, useful for smoke tests.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs and write no .npy files.",
    )
    p.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first failed clip instead of recording the error and continuing.",
    )
    p.add_argument(
        "--manifest-name",
        default="manifest.csv",
        help="CSV manifest filename under --output-dir.",
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
        help="Only use the first N seconds of each wav. Use 0 or a negative value to disable truncation.",
    )
    return p.parse_args()


def has_wavs(path: Path) -> bool:
    return any(path.glob("*/*.wav"))


def candidate_beat_roots(path: Path) -> list[Path]:
    requested = path.expanduser()
    candidates = [requested]
    candidates.extend(FALLBACK_BEAT_ROOTS)

    expanded: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        for option in (resolved, resolved / resolved.name):
            if option not in seen:
                seen.add(option)
                expanded.append(option)
    return expanded


def resolve_beat_root(path: Path) -> Path:
    """Return the directory that directly contains BEAT speaker folders with wavs."""
    candidates = candidate_beat_roots(path)
    if not candidates:
        raise SystemExit(f"Missing BEAT root: {path.expanduser()}")

    for candidate in candidates:
        if candidate.is_dir() and has_wavs(candidate):
            return candidate

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise SystemExit(f"Missing BEAT root: {path.expanduser()}")


def speaker_directories(beat_root: Path, speaker_ids: list[str] | None) -> list[Path]:
    if speaker_ids:
        return [beat_root / str(sid) for sid in speaker_ids]
    return sorted(path for path in beat_root.iterdir() if path.is_dir())


def discover_jobs(args: argparse.Namespace) -> tuple[Path, list[ClipJob]]:
    beat_root = resolve_beat_root(Path(args.beat_root))
    output_dir = Path(args.output_dir).expanduser().resolve()

    jobs: list[ClipJob] = []
    for speaker_dir in speaker_directories(beat_root, args.speaker_ids):
        if not speaker_dir.is_dir():
            print(f"[WARN] missing speaker directory: {speaker_dir}")
            continue

        speaker_id = speaker_dir.name
        for wav_path in sorted(speaker_dir.glob("*.wav")):
            dataset_id = wav_path.stem
            json_path = speaker_dir / f"{dataset_id}.json"
            textgrid_path = speaker_dir / f"{dataset_id}.TextGrid"
            has_json = json_path.is_file()
            has_textgrid = textgrid_path.is_file()

            if args.require_json and not has_json:
                continue
            if args.require_textgrid and not has_textgrid:
                continue

            out_dir = tongue_motion_output_dir(output_dir, speaker_id, dataset_id)
            jobs.append(
                ClipJob(
                    speaker_id=speaker_id,
                    dataset_id=dataset_id,
                    wav_path=wav_path,
                    out_path=out_dir / f"{dataset_id}.npy",
                    has_json=has_json,
                    has_textgrid=has_textgrid,
                )
            )

    if args.limit is not None:
        jobs = jobs[: max(0, int(args.limit))]
    return beat_root, jobs


def write_manifest(
    output_dir: Path, manifest_name: str, rows: list[dict[str, object]]
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / manifest_name
    fieldnames = [
        "speaker_id",
        "dataset_id",
        "wav_path",
        "npy_path",
        "status",
        "shape",
        "has_json",
        "has_textgrid",
        "seconds",
        "error",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    beat_root, jobs = discover_jobs(args)
    print(f"Using BEAT root: {beat_root}")
    print(f"Found {len(jobs)} wav clips.")
    print(f"Output root: {output_dir}")

    if not jobs:
        raise SystemExit(
            "No wav clips matched the requested filters. "
            "This inversion step requires original .wav files alongside the BEAT "
            ".json/.TextGrid files in each speaker directory."
        )

    if args.dry_run:
        for job in jobs[:20]:
            print(f"[DRY] {job.speaker_id}/{job.dataset_id} -> {job.out_path}")
        if len(jobs) > 20:
            print(f"[DRY] ... {len(jobs) - 20} more")
        return

    if not checkpoint_path.is_file():
        raise SystemExit(f"Missing checkpoint: {checkpoint_path}")

    device = default_device()
    print(f"Loading checkpoint on {device}: {checkpoint_path}")
    model = load_model(checkpoint_path, device)

    rows: list[dict[str, object]] = []
    processed = 0
    skipped = 0
    failed = 0

    for index, job in enumerate(jobs, start=1):
        row: dict[str, object] = {
            "speaker_id": job.speaker_id,
            "dataset_id": job.dataset_id,
            "wav_path": str(job.wav_path),
            "npy_path": str(job.out_path),
            "status": "",
            "shape": "",
            "has_json": int(job.has_json),
            "has_textgrid": int(job.has_textgrid),
            "seconds": "",
            "error": "",
        }

        if job.out_path.exists() and not args.overwrite:
            try:
                shape = np.load(job.out_path, mmap_mode="r").shape
                row["shape"] = "x".join(str(dim) for dim in shape)
            except Exception:
                row["shape"] = "unknown"
            row["status"] = "skipped_exists"
            rows.append(row)
            skipped += 1
            print(f"[{index}/{len(jobs)}] skip existing {job.out_path}")
            continue

        start = time.perf_counter()
        try:
            job.out_path.parent.mkdir(parents=True, exist_ok=True)
            ema = infer_ema(
                job.wav_path,
                model=model,
                device=device,
                max_seconds=args.max_seconds,
            )
            np.save(str(job.out_path), ema)
            elapsed = time.perf_counter() - start
            row["status"] = "ok"
            row["shape"] = "x".join(str(dim) for dim in ema.shape)
            row["seconds"] = f"{elapsed:.3f}"
            processed += 1
            print(
                f"[{index}/{len(jobs)}] saved {job.out_path} "
                f"shape={ema.shape} time={elapsed:.1f}s"
            )
        except Exception as exc:
            failed += 1
            row["status"] = "error"
            row["error"] = repr(exc)
            print(f"[{index}/{len(jobs)}] ERROR {job.wav_path}: {exc}")
            if args.stop_on_error:
                rows.append(row)
                write_manifest(output_dir, args.manifest_name, rows)
                raise

        rows.append(row)

    manifest_path = write_manifest(output_dir, args.manifest_name, rows)
    print(
        "Done: "
        f"processed={processed} skipped={skipped} failed={failed} "
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
