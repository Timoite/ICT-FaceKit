#!/usr/bin/env python3
"""Summarize cached lip-aperture time-shift estimates for research reporting.

Reads the SQLite database produced by estimate_lip_aperture_shifts.py and writes:
- a Markdown report with whole-dataset and per-speaker tables
- a per-speaker CSV table for downstream plotting/spreadsheets
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = (
    PROJECT_ROOT
    / "tongue_scripts"
    / "outputs"
    / "time_shifts"
    / "lip_aperture_time_shifts.sqlite3"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "tongue_scripts" / "outputs" / "time_shifts"


def fmt_ms(seconds: float, digits: int = 1) -> str:
    return f"{seconds * 1000:.{digits}f}"


def safe_stdev(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of empty values")
    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * p
    low = int(idx)
    high = min(low + 1, len(sorted_values) - 1)
    frac = idx - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def speaker_sort_key(speaker_id: str) -> tuple[int, int | str]:
    return (0, int(speaker_id)) if speaker_id.isdigit() else (1, speaker_id)


def load_rows(db_path: Path) -> list[sqlite3.Row]:
    if not db_path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            """
            SELECT *
            FROM lip_aperture_time_shifts
            WHERE status = 'ok'
              AND render_shift_seconds IS NOT NULL
            ORDER BY CAST(speaker_id AS INTEGER), dataset_id
            """
        ).fetchall()
    finally:
        conn.close()


def summarize_whole(rows: list[sqlite3.Row]) -> dict[str, float | int]:
    shifts = [float(r["render_shift_seconds"]) for r in rows]
    frames = [float(r["best_lag_frames"]) for r in rows]
    best_corr = [float(r["best_correlation"]) for r in rows]
    zero_corr = [float(r["zero_lag_correlation"]) for r in rows]
    gains = [b - z for b, z in zip(best_corr, zero_corr)]
    return {
        "clips": len(rows),
        "speakers": len({str(r["speaker_id"]) for r in rows}),
        "mean_shift_s": mean(shifts),
        "median_shift_s": median(shifts),
        "sd_shift_s": safe_stdev(shifts),
        "q1_shift_s": percentile(shifts, 0.25),
        "q3_shift_s": percentile(shifts, 0.75),
        "min_shift_s": min(shifts),
        "max_shift_s": max(shifts),
        "mean_frames": mean(frames),
        "median_frames": median(frames),
        "mean_best_corr": mean(best_corr),
        "median_best_corr": median(best_corr),
        "mean_zero_corr": mean(zero_corr),
        "median_zero_corr": median(zero_corr),
        "mean_corr_gain": mean(gains),
        "median_corr_gain": median(gains),
    }


def summarize_speakers(rows: list[sqlite3.Row]) -> list[dict[str, str | int | float]]:
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[str(row["speaker_id"])].append(row)

    summaries: list[dict[str, str | int | float]] = []
    for speaker_id in sorted(grouped, key=speaker_sort_key):
        sr = grouped[speaker_id]
        shifts = [float(r["render_shift_seconds"]) for r in sr]
        frames = [float(r["best_lag_frames"]) for r in sr]
        best_corr = [float(r["best_correlation"]) for r in sr]
        zero_corr = [float(r["zero_lag_correlation"]) for r in sr]
        gains = [b - z for b, z in zip(best_corr, zero_corr)]
        summaries.append(
            {
                "speaker_id": speaker_id,
                "clips": len(sr),
                "mean_delay_ms": mean(shifts) * 1000,
                "median_delay_ms": median(shifts) * 1000,
                "sd_delay_ms": safe_stdev(shifts) * 1000,
                "min_delay_ms": min(shifts) * 1000,
                "max_delay_ms": max(shifts) * 1000,
                "mean_lag_frames": mean(frames),
                "min_lag_frames": min(frames),
                "max_lag_frames": max(frames),
                "mean_best_corr": mean(best_corr),
                "mean_zero_lag_corr": mean(zero_corr),
                "mean_corr_gain": mean(gains),
            }
        )
    return summaries


def boundary_rows(rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
    return [
        r
        for r in rows
        if abs(float(r["render_shift_seconds"])) >= float(r["max_lag_seconds"])
    ]


def write_speaker_csv(
    path: Path, speaker_rows: list[dict[str, str | int | float]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(speaker_rows[0].keys()) if speaker_rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(speaker_rows)


def markdown_report(
    *,
    db_path: Path,
    rows: list[sqlite3.Row],
    whole: dict[str, float | int],
    clean_whole: dict[str, float | int],
    speaker_rows: list[dict[str, str | int | float]],
    boundary: list[sqlite3.Row],
) -> str:
    status_counts = Counter(str(r["status"]) for r in rows)
    speaker_means = [float(r["mean_delay_ms"]) for r in speaker_rows]

    lines: list[str] = []
    lines.append("# Lip-aperture time-shift summary")
    lines.append("")
    lines.append(f"Source database: `{db_path}`")
    lines.append("")
    lines.append(
        "Positive delay means the articulatory/WavLM-derived lip aperture leads BEAT lip motion; rendering should delay articulatory/tongue motion by this amount."
    )
    lines.append("")
    lines.append("## Whole dataset")
    lines.append("")
    lines.append(
        "| Scope | Clips | Speakers | Mean delay (ms) | Median (ms) | SD (ms) | Q1–Q3 (ms) | Range (ms) | Mean best corr. | Mean zero-lag corr. | Mean corr. gain |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        "| All successful rows "
        f"| {whole['clips']} | {whole['speakers']} "
        f"| {fmt_ms(float(whole['mean_shift_s']))} "
        f"| {fmt_ms(float(whole['median_shift_s']), 0)} "
        f"| {fmt_ms(float(whole['sd_shift_s']))} "
        f"| {fmt_ms(float(whole['q1_shift_s']), 0)}–{fmt_ms(float(whole['q3_shift_s']), 0)} "
        f"| {fmt_ms(float(whole['min_shift_s']), 0)}–{fmt_ms(float(whole['max_shift_s']), 0)} "
        f"| {float(whole['mean_best_corr']):.3f} "
        f"| {float(whole['mean_zero_corr']):.3f} "
        f"| {float(whole['mean_corr_gain']):+.3f} |"
    )
    lines.append(
        "| Excluding ±max-lag boundary rows "
        f"| {clean_whole['clips']} | {clean_whole['speakers']} "
        f"| {fmt_ms(float(clean_whole['mean_shift_s']))} "
        f"| {fmt_ms(float(clean_whole['median_shift_s']), 0)} "
        f"| {fmt_ms(float(clean_whole['sd_shift_s']))} "
        f"| {fmt_ms(float(clean_whole['q1_shift_s']), 0)}–{fmt_ms(float(clean_whole['q3_shift_s']), 0)} "
        f"| {fmt_ms(float(clean_whole['min_shift_s']), 0)}–{fmt_ms(float(clean_whole['max_shift_s']), 0)} "
        f"| {float(clean_whole['mean_best_corr']):.3f} "
        f"| {float(clean_whole['mean_zero_corr']):.3f} "
        f"| {float(clean_whole['mean_corr_gain']):+.3f} |"
    )
    lines.append("")
    lines.append("## Speaker-level aggregate")
    lines.append("")
    lines.append(f"Unweighted mean of speaker means: **{mean(speaker_means):.1f} ms**")
    lines.append(
        f"Range of speaker means: **{min(speaker_means):.1f}–{max(speaker_means):.1f} ms**"
    )
    lines.append("")
    lines.append("## Per-speaker table")
    lines.append("")
    lines.append(
        "| Speaker | Clips | Mean delay (ms) | Median (ms) | SD (ms) | Range (ms) | Mean lag frames | Frame range | Best corr. | Zero-lag corr. | Corr. gain |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in speaker_rows:
        lines.append(
            f"| {r['speaker_id']} "
            f"| {r['clips']} "
            f"| {float(r['mean_delay_ms']):.1f} "
            f"| {float(r['median_delay_ms']):.0f} "
            f"| {float(r['sd_delay_ms']):.1f} "
            f"| {float(r['min_delay_ms']):.0f}–{float(r['max_delay_ms']):.0f} "
            f"| {float(r['mean_lag_frames']):.2f} "
            f"| {float(r['min_lag_frames']):.0f}–{float(r['max_lag_frames']):.0f} "
            f"| {float(r['mean_best_corr']):.3f} "
            f"| {float(r['mean_zero_lag_corr']):.3f} "
            f"| {float(r['mean_corr_gain']):+.3f} |"
        )
    lines.append("")
    lines.append("## Boundary-hit rows")
    lines.append("")
    lines.append(f"Rows at ±max lag: **{len(boundary)}**")
    lines.append("")
    if boundary:
        lines.append("| Speaker | Dataset | Delay (ms) | Best corr. | Zero-lag corr. |")
        lines.append("|---:|---|---:|---:|---:|")
        for r in boundary:
            lines.append(
                f"| {r['speaker_id']} | `{r['dataset_id']}` "
                f"| {float(r['render_shift_seconds']) * 1000:.0f} "
                f"| {float(r['best_correlation']):.3f} "
                f"| {float(r['zero_lag_correlation']):.3f} |"
            )
    lines.append("")
    lines.append("## Reproducibility checks")
    lines.append("")
    lines.append(f"Status counts among loaded rows: `{dict(status_counts)}`")
    lines.append(f"Loaded speakers: `{whole['speakers']}`")
    lines.append(f"Loaded clips: `{whole['clips']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rows = load_rows(args.db)
    if not rows:
        raise SystemExit("No successful time-shift rows found")

    boundary = boundary_rows(rows)
    clean_rows = [r for r in rows if r not in boundary]

    whole = summarize_whole(rows)
    clean_whole = summarize_whole(clean_rows)
    speaker_rows = summarize_speakers(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.out_dir / "lip_aperture_time_shift_summary.md"
    csv_path = args.out_dir / "lip_aperture_time_shift_speaker_summary.csv"

    write_speaker_csv(csv_path, speaker_rows)
    md_path.write_text(
        markdown_report(
            db_path=args.db,
            rows=rows,
            whole=whole,
            clean_whole=clean_whole,
            speaker_rows=speaker_rows,
            boundary=boundary,
        ),
        encoding="utf-8",
    )

    print(f"Loaded {whole['clips']} successful clips from {whole['speakers']} speakers")
    print(f"Boundary-hit rows: {len(boundary)}")
    print(f"Whole-dataset median delay: {fmt_ms(float(whole['median_shift_s']), 0)} ms")
    print(f"Whole-dataset mean delay: {fmt_ms(float(whole['mean_shift_s']))} ms")
    print(f"Markdown report: {md_path}")
    print(f"Speaker CSV:     {csv_path}")


if __name__ == "__main__":
    main()
