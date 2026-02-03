#!/usr/bin/env python3
"""Compute WER for generated transcripts against BEAT TextGrid ground truth."""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_WORD_RE = re.compile(r"[a-zA-Z0-9']+")


@dataclass
class WordInterval:
    start: float
    end: float
    text: str


@dataclass
class WerStats:
    substitutions: int
    deletions: int
    insertions: int
    ref_words: int
    hyp_words: int

    @property
    def total_errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        if self.ref_words == 0:
            return math.inf if self.total_errors else 0.0
        return self.total_errors / self.ref_words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute WER between generated transcripts and TextGrid ground truth")
    parser.add_argument("--predicted-root", default="transcripts", help="Directory that contains generated transcripts")
    parser.add_argument(
        "--textgrid-root",
        default="Visual_Speech_Recognition_for_Multiple_Languages/data/beat_textgrids",
        help="Directory that contains TextGrid files organised by speaker ID",
    )
    parser.add_argument(
        "--ground-truth-root",
        default="ground_truth_transcripts",
        help="Directory to write the washed TextGrid transcripts",
    )
    parser.add_argument("--report-path", default="wer_report.csv", help="CSV file to store per-file WER results")
    parser.add_argument("--tier-name", default="words", help="TextGrid tier name to read")
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit for debugging")
    return parser.parse_args()


def parse_textgrid_words(textgrid_path: Path, tier_name: str) -> List[WordInterval]:
    intervals: List[WordInterval] = []
    in_tier = False
    current: Dict[str, str] = {}

    with textgrid_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("item ["):
                in_tier = False
                continue
            if line.startswith('name = "'):
                tier = line.split("=", 1)[1].strip().strip('"')
                in_tier = tier == tier_name
                continue
            if not in_tier:
                continue
            if line.startswith("intervals ["):
                current = {}
                continue
            if line.startswith("xmin ="):
                current["start"] = line.split("=", 1)[1].strip()
                continue
            if line.startswith("xmax ="):
                current["end"] = line.split("=", 1)[1].strip()
                continue
            if line.startswith("text ="):
                text_value = line.split("=", 1)[1].strip()
                if text_value.startswith('"') and text_value.endswith('"'):
                    text_value = text_value[1:-1]
                current["text"] = text_value
                if {"start", "end", "text"} <= current.keys():
                    try:
                        start = float(current["start"])
                        end = float(current["end"])
                    except ValueError:
                        start, end = 0.0, 0.0
                    intervals.append(WordInterval(start=start, end=end, text=current["text"]))
    return intervals


def intervals_to_tokens(intervals: Sequence[WordInterval]) -> List[str]:
    return [interval.text.strip() for interval in intervals if interval.text.strip()]


def normalize_tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def compute_alignment(ref_tokens: Sequence[str], hyp_tokens: Sequence[str]) -> WerStats:
    n = len(ref_tokens)
    m = len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                substitution = dp[i - 1][j - 1] + 1
                insertion = dp[i][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                dp[i][j] = min(substitution, insertion, deletion)

    substitutions = deletions = insertions = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i - 1] == hyp_tokens[j - 1]:
            i -= 1
            j -= 1
            continue

        choices: List[Tuple[int, str]] = []
        if i > 0 and j > 0:
            choices.append((dp[i - 1][j - 1], "sub"))
        if j > 0:
            choices.append((dp[i][j - 1], "ins"))
        if i > 0:
            choices.append((dp[i - 1][j], "del"))
        prev_cost, op = min(choices, key=lambda item: item[0])

        if op == "sub":
            substitutions += 1
            i -= 1
            j -= 1
        elif op == "ins":
            insertions += 1
            j -= 1
        elif op == "del":
            deletions += 1
            i -= 1

    return WerStats(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        ref_words=n,
        hyp_words=m,
    )


def ensure_ground_truth_file(
    gt_tokens: Sequence[str],
    predicted_file: Path,
    predicted_root: Path,
    output_root: Path,
) -> Path:
    rel_path = predicted_file.relative_to(predicted_root)
    output_path = output_root / rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = " ".join(gt_tokens).strip()
    if text:
        text += "\n"
    output_path.write_text(text, encoding="utf-8")
    return output_path


def collect_predicted_transcripts(predicted_root: Path) -> Iterable[Path]:
    for speaker_dir in sorted(predicted_root.glob("speaker_*")):
        videos_dir = speaker_dir / "videos"
        if not videos_dir.is_dir():
            continue
        for transcript in sorted(videos_dir.glob("*.txt")):
            yield transcript


def extract_speaker_id(transcript_path: Path) -> Optional[str]:
    try:
        speaker_segment = transcript_path.parents[1].name  # speaker_X
    except IndexError:
        return None
    if speaker_segment.startswith("speaker_"):
        return speaker_segment.split("_", 1)[1]
    return None


def compute_wer_for_file(
    transcript_path: Path,
    predicted_root: Path,
    textgrid_root: Path,
    ground_truth_root: Path,
    tier_name: str,
) -> Optional[Tuple[str, str, WerStats]]:
    speaker_id = extract_speaker_id(transcript_path)
    if not speaker_id:
        print(f"Skipping {transcript_path}: unable to infer speaker id")
        return None

    textgrid_path = textgrid_root / speaker_id / f"{transcript_path.stem}.TextGrid"
    if not textgrid_path.is_file():
        print(f"Missing TextGrid for {transcript_path}")
        return None

    intervals = parse_textgrid_words(textgrid_path, tier_name)
    gt_tokens = intervals_to_tokens(intervals)
    ensure_ground_truth_file(gt_tokens, transcript_path, predicted_root, ground_truth_root)

    ground_truth_tokens = normalize_tokens(" ".join(gt_tokens))
    hypothesis_tokens = normalize_tokens(transcript_path.read_text(encoding="utf-8", errors="ignore"))
    stats = compute_alignment(ground_truth_tokens, hypothesis_tokens)
    rel_path = str(transcript_path.relative_to(predicted_root))
    return rel_path, speaker_id, stats


def write_csv(report_path: Path, rows: List[Tuple[str, str, WerStats]], summary: WerStats) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "file",
            "speaker_id",
            "ref_words",
            "hyp_words",
            "substitutions",
            "deletions",
            "insertions",
            "wer",
        ])
        for rel_path, speaker_id, stats in rows:
            writer.writerow(
                [
                    rel_path,
                    speaker_id,
                    stats.ref_words,
                    stats.hyp_words,
                    stats.substitutions,
                    stats.deletions,
                    stats.insertions,
                    f"{stats.wer:.4f}" if math.isfinite(stats.wer) else "inf",
                ]
            )
        writer.writerow(
            [
                "OVERALL",
                "-",
                summary.ref_words,
                summary.hyp_words,
                summary.substitutions,
                summary.deletions,
                summary.insertions,
                f"{summary.wer:.4f}" if math.isfinite(summary.wer) else "inf",
            ]
        )


def aggregate_stats(rows: List[Tuple[str, str, WerStats]]) -> WerStats:
    total_sub = sum(stats.substitutions for _, _, stats in rows)
    total_del = sum(stats.deletions for _, _, stats in rows)
    total_ins = sum(stats.insertions for _, _, stats in rows)
    total_ref = sum(stats.ref_words for _, _, stats in rows)
    total_hyp = sum(stats.hyp_words for _, _, stats in rows)
    return WerStats(
        substitutions=total_sub,
        deletions=total_del,
        insertions=total_ins,
        ref_words=total_ref,
        hyp_words=total_hyp,
    )


def main() -> None:
    args = parse_args()
    predicted_root = Path(args.predicted_root).resolve()
    textgrid_root = Path(args.textgrid_root).resolve()
    ground_truth_root = Path(args.ground_truth_root).resolve()
    report_path = Path(args.report_path).resolve()

    if not predicted_root.is_dir():
        raise SystemExit(f"Predicted transcripts directory not found: {predicted_root}")
    if not textgrid_root.is_dir():
        raise SystemExit(f"TextGrid directory not found: {textgrid_root}")

    processed_rows: List[Tuple[str, str, WerStats]] = []
    for idx, transcript_path in enumerate(collect_predicted_transcripts(predicted_root), start=1):
        if args.max_files and idx > args.max_files:
            break
        result = compute_wer_for_file(
            transcript_path,
            predicted_root,
            textgrid_root,
            ground_truth_root,
            args.tier_name,
        )
        if result is None:
            continue
        processed_rows.append(result)

    if not processed_rows:
        raise SystemExit("No transcript/TextGrid pairs processed. Check paths and file naming.")

    summary_stats = aggregate_stats(processed_rows)
    write_csv(report_path, processed_rows, summary_stats)

    print(f"Processed {len(processed_rows)} transcript/TextGrid pairs")
    print(f"Overall WER: {summary_stats.wer:.4%}" if math.isfinite(summary_stats.wer) else "Overall WER: inf")
    print(f"Report written to {report_path}")
    print(f"Ground truth transcripts stored in {ground_truth_root}")


if __name__ == "__main__":
    main()
