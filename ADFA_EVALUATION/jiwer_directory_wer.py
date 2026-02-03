#!/usr/bin/env python3
"""Compute WER between two transcript directories using jiwer."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

from jiwer import wer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the word error rate (WER) between generated transcripts and "
            "ground truth files using jiwer."
        )
    )
    parser.add_argument(
        "--predicted-dir",
        type=Path,
        default=Path("Visual_Speech_Recognition_for_Multiple_Languages/transcript-normalized"),
        help="Directory that contains predicted transcripts",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=Path("ground_truth_transcripts"),
        help="Directory that contains ground truth transcripts",
    )
    parser.add_argument(
        "--glob-pattern",
        default="*.txt",
        help="Glob pattern to select transcript files (default: *.txt)",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("wer_directory_report.csv"),
        help="Optional CSV file to store per-file WER values",
    )
    return parser.parse_args()


def list_files(root: Path, pattern: str) -> Sequence[Path]:
    return sorted(path for path in root.rglob(pattern) if path.is_file())


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def get_speaker_id(relative_path: Path) -> str:
    # Directory names already encode speaker IDs (e.g. speaker_1/videos/file.txt).
    return relative_path.parts[0] if relative_path.parts else "unknown"


def write_csv(report_path: Path, rows: Sequence[Tuple[Path, str, float]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("file,speaker_id,wer\n")
        for rel_path, speaker_id, score in rows:
            # only include id without prefix 'speaker_'
            if speaker_id.startswith("speaker_"):
                speaker_id = speaker_id[len("speaker_") :]
            handle.write(f"{rel_path},{speaker_id},{score:.6f}\n")


def compute_directory_wer(predicted_root: Path, ground_truth_root: Path, glob_pattern: str, report_path: Path) -> None:
    if not predicted_root.is_dir():
        raise SystemExit(f"Predicted directory not found: {predicted_root}")
    if not ground_truth_root.is_dir():
        raise SystemExit(f"Ground truth directory not found: {ground_truth_root}")

    predicted_files = list_files(predicted_root, glob_pattern)
    if not predicted_files:
        raise SystemExit("No predicted transcripts found with the provided glob pattern")

    matched_results: List[Tuple[Path, str, float]] = []
    missing_ground_truth: List[Path] = []

    ground_truth_index = {path.relative_to(ground_truth_root): path for path in list_files(ground_truth_root, glob_pattern)}

    for predicted_file in predicted_files:
        relative_path = predicted_file.relative_to(predicted_root)
        ground_truth_file = ground_truth_root / relative_path
        if not ground_truth_file.is_file():
            missing_ground_truth.append(relative_path)
            continue

        reference = read_text(ground_truth_file)
        hypothesis = read_text(predicted_file)
        speaker_id = get_speaker_id(relative_path)
        if not reference:
            score = 0.0 if not hypothesis else float("inf")
        else:
            score = wer(reference, hypothesis)
        matched_results.append((relative_path, speaker_id, score))
        ground_truth_index.pop(relative_path, None)

    if not matched_results:
        raise SystemExit("No matching transcript pairs were found")

    avg_wer = sum(score for _, _, score in matched_results) / len(matched_results)
    best_file, _, best_wer = min(matched_results, key=lambda item: item[2])
    worst_file, _, worst_wer = max(matched_results, key=lambda item: item[2])

    print(f"Matched transcript pairs: {len(matched_results)}")
    print(f"Average WER: {avg_wer:.4f}")
    print(f"Best WER:  {best_wer:.4f}  ({best_file})")
    print(f"Worst WER: {worst_wer:.4f} ({worst_file})")

    write_csv(report_path, matched_results)

    if missing_ground_truth:
        print(f"Skipped {len(missing_ground_truth)} predicted files without ground truth matches")
    if ground_truth_index:
        print(f"Ground truth files without predictions: {len(ground_truth_index)}")


def main() -> None:
    args = parse_args()
    compute_directory_wer(
        args.predicted_dir.resolve(),
        args.ground_truth_dir.resolve(),
        args.glob_pattern,
        args.report_path.resolve(),
    )


if __name__ == "__main__":
    main()
