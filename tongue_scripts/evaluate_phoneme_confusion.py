#!/usr/bin/env python3
"""Batch active/passive tongue VSR evaluation with phone confusion matrices."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from jiwer import wer as jiwer_wer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_script.phone_confusion import (  # noqa: E402
    DELETE_TOKEN,
    FOCUSED_TONGUE_PHONES,
    INSERT_TOKEN,
    KNOWN_PHONES,
    AlignmentStep,
    PhoneConfusionResult,
    PhoneConfusionStats,
    analysis_category_for_phone,
    assert_known_phone_category_coverage,
    evaluate_phone_confusion,
    parse_textgrid_intervals,
    parse_textgrid_phones,
    visual_category_for_phone,
)
from evaluation_script.ver import calculate_ver  # noqa: E402


ACTIVE_SUFFIX = "_with_tongue_with_audio"
PASSIVE_SUFFIX = "_passive_tongue_with_audio"
DEFAULT_EXPERIMENT_NAME = "active_vs_passive_phoneme_confusion"


@dataclass(frozen=True)
class VideoPair:
    dataset_id: str
    active_video: Path
    passive_video: Path
    textgrid_path: Path | None


@dataclass(frozen=True)
class ConditionEvaluation:
    dataset_id: str
    condition: str
    video_path: Path
    transcript_path: Path
    transcript: str
    result: PhoneConfusionResult
    wer_norm: float | None
    ver: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build phone-level active/passive tongue VSR confusion matrices."
    )
    parser.add_argument("--dataset-id", default=None, help="Optional single dataset id filter.")
    parser.add_argument(
        "--video-roots",
        nargs="+",
        default=[
            str(PROJECT_ROOT / "tongue_scripts" / "outputs"),
            str(PROJECT_ROOT / "tongue_scripts" / "outputs" / "multi_speaker"),
        ],
        help="Directories searched recursively for active/passive rendered videos.",
    )
    parser.add_argument(
        "--textgrid-roots",
        nargs="+",
        default=[
            str(
                PROJECT_ROOT
                / "ADFA_EVALUATION"
                / "data"
                / "beat_cache"
                / "beat_english_v0.2.1"
                / "beat_english_v0.2.1"
            ),
            str(
                PROJECT_ROOT
                / "ADFA_EVALUATION"
                / "Visual_Speech_Recognition_for_Multiple_Languages"
                / "data"
                / "beat_textgrids"
            ),
            str(PROJECT_ROOT / "tongue_scripts" / "inputs"),
        ],
        help="TextGrid roots containing either <dataset>.TextGrid or <speaker>/<dataset>.TextGrid.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "tongue_scripts" / "outputs" / "phoneme_confusion"),
        help="Root output directory for timestamped experiment artifacts.",
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--timestamp", default=None, help="Optional output run directory timestamp.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional limit for debugging.")
    parser.add_argument("--dry-run", action="store_true", help="Only show discovered pairs and TextGrid status.")
    parser.add_argument("--force-infer", action="store_true", help="Rerun VSR even when cached transcripts exist.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--infer-pipeline-script",
        default=str(
            PROJECT_ROOT
            / "ADFA_EVALUATION"
            / "Visual_Speech_Recognition_for_Multiple_Languages"
            / "infer_pipeline.py"
        ),
    )
    parser.add_argument("--detector", default="mediapipe")
    parser.add_argument("--target-fps", type=int, default=25)
    parser.add_argument("--seg-target-seconds", type=float, default=8.0)
    parser.add_argument("--seg-min-seconds", type=float, default=4.0)
    parser.add_argument("--seg-max-seconds", type=float, default=12.0)
    parser.add_argument("--seg-min-silence", type=float, default=0.0)
    return parser.parse_args()


def normalize_text_for_wer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def finite_float(value: float | None) -> str:
    if value is None:
        return ""
    if math.isfinite(value):
        return f"{value:.6f}"
    return "inf"


def dataset_id_from_stem(stem: str, suffix: str) -> str | None:
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return None


def resolve_textgrid(dataset_id: str, textgrid_roots: Sequence[Path]) -> Path | None:
    speaker_id = dataset_id.split("_", 1)[0]
    candidates: list[Path] = []
    for root in textgrid_roots:
        candidates.append(root / f"{dataset_id}.TextGrid")
        candidates.append(root / speaker_id / f"{dataset_id}.TextGrid")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def discover_pairs(
    video_roots: Sequence[Path],
    textgrid_roots: Sequence[Path],
    dataset_filter: str | None,
) -> list[VideoPair]:
    active: dict[str, Path] = {}
    passive: dict[str, Path] = {}

    for root in video_roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.mp4")):
            active_id = dataset_id_from_stem(path.stem, ACTIVE_SUFFIX)
            if active_id and (dataset_filter is None or active_id == dataset_filter):
                active.setdefault(active_id, path)
                continue
            passive_id = dataset_id_from_stem(path.stem, PASSIVE_SUFFIX)
            if passive_id and (dataset_filter is None or passive_id == dataset_filter):
                passive.setdefault(passive_id, path)

    pairs: list[VideoPair] = []
    for dataset_id in sorted(set(active) & set(passive)):
        pairs.append(
            VideoPair(
                dataset_id=dataset_id,
                active_video=active[dataset_id],
                passive_video=passive[dataset_id],
                textgrid_path=resolve_textgrid(dataset_id, textgrid_roots),
            )
        )
    return pairs


def print_dry_run(pairs: Sequence[VideoPair]) -> None:
    print(f"Discovered active/passive pairs: {len(pairs)}")
    for pair in pairs:
        tg = str(pair.textgrid_path) if pair.textgrid_path else "MISSING"
        print(f"- {pair.dataset_id}")
        print(f"  active : {pair.active_video}")
        print(f"  passive: {pair.passive_video}")
        print(f"  textgrid: {tg}")


def run_inference_segmented(
    python_bin: str,
    infer_pipeline_script: Path,
    video_path: Path,
    textgrid_path: Path,
    detector: str,
    target_fps: int,
    seg_target_seconds: float,
    seg_min_seconds: float,
    seg_max_seconds: float,
    seg_min_silence: float,
) -> str:
    with tempfile.TemporaryDirectory(prefix="phone_confusion_vsr_") as tmp_dir:
        cmd = [
            python_bin,
            str(infer_pipeline_script),
            "--video-path",
            str(video_path),
            "--output-dir",
            tmp_dir,
            "--textgrid-path",
            str(textgrid_path),
            "--target-fps",
            str(target_fps),
            "--target-segment-seconds",
            str(seg_target_seconds),
            "--min-segment-seconds",
            str(seg_min_seconds),
            "--max-segment-seconds",
            str(seg_max_seconds),
            "--min-silence-seconds",
            str(seg_min_silence),
            "--detector",
            detector,
        ]
        proc = subprocess.run(
            cmd,
            cwd=infer_pipeline_script.parent,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            merged = (proc.stdout or "") + "\n" + (proc.stderr or "")
            raise RuntimeError(f"Inference failed for {video_path.name}:\n{merged[-1600:]}")

        out_txt = Path(tmp_dir) / f"{video_path.stem}.txt"
        if not out_txt.is_file():
            merged = (proc.stdout or "") + "\n" + (proc.stderr or "")
            raise RuntimeError(f"Inference transcript missing for {video_path.name}:\n{merged[-1600:]}")
        return out_txt.read_text(encoding="utf-8", errors="ignore").strip()


def get_or_create_transcript(
    args: argparse.Namespace,
    output_dir: Path,
    dataset_id: str,
    condition: str,
    video_path: Path,
    textgrid_path: Path,
) -> tuple[str, Path]:
    transcript_dir = output_dir / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_dir / f"{dataset_id}_{condition}.txt"

    if transcript_path.is_file() and not args.force_infer:
        return transcript_path.read_text(encoding="utf-8", errors="ignore").strip(), transcript_path

    transcript = run_inference_segmented(
        python_bin=args.python_bin,
        infer_pipeline_script=Path(args.infer_pipeline_script),
        video_path=video_path,
        textgrid_path=textgrid_path,
        detector=args.detector,
        target_fps=args.target_fps,
        seg_target_seconds=args.seg_target_seconds,
        seg_min_seconds=args.seg_min_seconds,
        seg_max_seconds=args.seg_max_seconds,
        seg_min_silence=args.seg_min_silence,
    )
    transcript_path.write_text(transcript + "\n", encoding="utf-8")
    return transcript, transcript_path


def textgrid_words_text(textgrid_path: Path) -> str:
    intervals = parse_textgrid_intervals(textgrid_path, "words")
    return " ".join(interval.text.strip() for interval in intervals if interval.text.strip())


def evaluate_condition(
    args: argparse.Namespace,
    output_dir: Path,
    pair: VideoPair,
    condition: str,
    video_path: Path,
    ref_phones: Sequence[str],
    reference_text: str,
) -> ConditionEvaluation:
    if pair.textgrid_path is None:
        raise ValueError(f"Missing TextGrid for {pair.dataset_id}")

    transcript, transcript_path = get_or_create_transcript(
        args=args,
        output_dir=output_dir,
        dataset_id=pair.dataset_id,
        condition=condition,
        video_path=video_path,
        textgrid_path=pair.textgrid_path,
    )
    result = evaluate_phone_confusion(ref_phones, transcript)

    wer_norm: float | None = None
    ver: float | None = None
    if reference_text.strip():
        wer_norm = jiwer_wer(normalize_text_for_wer(reference_text), normalize_text_for_wer(transcript))
        ver = calculate_ver(reference_text, transcript.lower(), vowel_mode="grouped")[0]

    return ConditionEvaluation(
        dataset_id=pair.dataset_id,
        condition=condition,
        video_path=video_path,
        transcript_path=transcript_path,
        transcript=transcript,
        result=result,
        wer_norm=wer_norm,
        ver=ver,
    )


def write_confusion_csv(path: Path, stats: PhoneConfusionStats) -> None:
    phones = sorted(KNOWN_PHONES)
    row_labels = phones + [INSERT_TOKEN]
    col_labels = phones + [DELETE_TOKEN]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ref\\hyp", *col_labels])
        for row_label in row_labels:
            writer.writerow(
                [row_label, *[stats.confusion.get((row_label, col_label), 0) for col_label in col_labels]]
            )


def plot_confusion_matrix(path: Path, stats: PhoneConfusionStats, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    phones = sorted(KNOWN_PHONES)
    row_labels = phones + [INSERT_TOKEN]
    col_labels = phones + [DELETE_TOKEN]
    matrix = np.array(
        [[stats.confusion.get((row_label, col_label), 0) for col_label in col_labels] for row_label in row_labels],
        dtype=float,
    )

    fig_width = max(12, len(col_labels) * 0.34)
    fig_height = max(10, len(row_labels) * 0.30)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, interpolation="nearest", aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Hypothesis phone")
    ax.set_ylabel("Reference phone")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(row_labels, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def phone_delta_rows(
    active_stats: PhoneConfusionStats,
    passive_stats: PhoneConfusionStats,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for phone in sorted(KNOWN_PHONES):
        active_total = active_stats.per_ref_total.get(phone, 0)
        passive_total = passive_stats.per_ref_total.get(phone, 0)
        active_correct = active_stats.per_ref_correct.get(phone, 0)
        passive_correct = passive_stats.per_ref_correct.get(phone, 0)
        active_recall = active_stats.recall_for(phone)
        passive_recall = passive_stats.recall_for(phone)
        rows.append(
            {
                "phone": phone,
                "analysis_category": analysis_category_for_phone(phone),
                "visual_category": visual_category_for_phone(phone),
                "active_total": str(active_total),
                "active_correct": str(active_correct),
                "active_recall": f"{active_recall:.6f}",
                "passive_total": str(passive_total),
                "passive_correct": str(passive_correct),
                "passive_recall": f"{passive_recall:.6f}",
                "active_minus_passive_recall": f"{active_recall - passive_recall:.6f}",
            }
        )
    return rows


def write_per_phone_delta(
    path: Path,
    active_stats: PhoneConfusionStats,
    passive_stats: PhoneConfusionStats,
) -> list[dict[str, str]]:
    rows = phone_delta_rows(active_stats, passive_stats)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_tongue_sensitive_delta(path: Path, delta_rows: Sequence[dict[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_phone = {row["phone"]: row for row in delta_rows}
    phones = [phone for phone in FOCUSED_TONGUE_PHONES if phone in by_phone]
    deltas = [float(by_phone[phone]["active_minus_passive_recall"]) for phone in phones]
    colors = ["#2f8f46" if delta >= 0 else "#b94343" for delta in deltas]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(phones, deltas, color=colors)
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.set_ylabel("Active recall - passive recall")
    ax.set_title("Tongue-sensitive phone recall delta")
    ax.set_ylim(min(-0.1, min(deltas, default=0.0) - 0.05), max(0.1, max(deltas, default=0.0) + 0.05))
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def metric_row(evaluation: ConditionEvaluation, textgrid_path: Path) -> dict[str, str]:
    stats = evaluation.result.stats
    tongue_total = sum(
        stats.per_ref_total.get(phone, 0)
        for phone in KNOWN_PHONES
        if analysis_category_for_phone(phone).startswith("TONGUE_")
    )
    tongue_correct = sum(
        stats.per_ref_correct.get(phone, 0)
        for phone in KNOWN_PHONES
        if analysis_category_for_phone(phone).startswith("TONGUE_")
    )
    return {
        "dataset_id": evaluation.dataset_id,
        "condition": evaluation.condition,
        "video_path": str(evaluation.video_path),
        "textgrid_path": str(textgrid_path),
        "transcript_path": str(evaluation.transcript_path),
        "ref_phone_count": str(stats.ref_count),
        "hyp_phone_count": str(stats.hyp_count),
        "correct": str(stats.correct),
        "substitutions": str(stats.substitutions),
        "deletions": str(stats.deletions),
        "insertions": str(stats.insertions),
        "phone_error_rate": finite_float(stats.phone_error_rate),
        "phone_accuracy": finite_float(stats.phone_accuracy),
        "tongue_ref_count": str(tongue_total),
        "tongue_correct": str(tongue_correct),
        "tongue_recall": finite_float(stats.tongue_sensitive_recall()),
        "wer_norm": finite_float(evaluation.wer_norm),
        "ver_grouped": finite_float(evaluation.ver),
    }


def write_per_dataset_metrics(path: Path, rows: Sequence[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sample_alignment_steps(alignment: Sequence[AlignmentStep], op: str, limit: int = 20) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for step in alignment:
        if step.op != op:
            continue
        phone = step.ref or step.hyp or ""
        samples.append(
            {
                "op": step.op,
                "ref": step.ref,
                "hyp": step.hyp,
                "ref_index": step.ref_index,
                "hyp_index": step.hyp_index,
                "analysis_category": analysis_category_for_phone(phone),
                "visual_category": visual_category_for_phone(phone),
            }
        )
        if len(samples) >= limit:
            break
    return samples


def write_alignment_samples(path: Path, evaluations: Sequence[ConditionEvaluation]) -> None:
    payload: dict[str, dict[str, dict[str, list[dict[str, object]]]]] = {}
    for evaluation in evaluations:
        payload.setdefault(evaluation.dataset_id, {})[evaluation.condition] = {
            "substitutions": sample_alignment_steps(evaluation.result.alignment, "sub"),
            "deletions": sample_alignment_steps(evaluation.result.alignment, "del"),
            "insertions": sample_alignment_steps(evaluation.result.alignment, "ins"),
        }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def aggregate_stats(evaluations: Sequence[ConditionEvaluation], condition: str) -> PhoneConfusionStats:
    stats = PhoneConfusionStats()
    for evaluation in evaluations:
        if evaluation.condition == condition:
            stats.merge(evaluation.result.stats)
    return stats


def summary_condition_line(label: str, stats: PhoneConfusionStats) -> str:
    return (
        f"| {label} | {stats.ref_count} | {stats.hyp_count} | {stats.phone_error_rate:.4f} | "
        f"{stats.phone_accuracy:.4f} | {stats.tongue_sensitive_recall():.4f} | "
        f"{stats.substitutions} | {stats.deletions} | {stats.insertions} |"
    )


def write_summary(
    path: Path,
    args: argparse.Namespace,
    pairs: Sequence[VideoPair],
    evaluations: Sequence[ConditionEvaluation],
    active_stats: PhoneConfusionStats,
    passive_stats: PhoneConfusionStats,
    delta_rows: Sequence[dict[str, str]],
) -> None:
    top_improved = sorted(
        delta_rows,
        key=lambda row: float(row["active_minus_passive_recall"]),
        reverse=True,
    )[:10]
    focused = [row for row in delta_rows if row["phone"] in FOCUSED_TONGUE_PHONES]

    lines: list[str] = []
    lines.append(f"# {args.experiment_name}")
    lines.append("")
    lines.append(f"- generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- evaluated pairs: {len({ev.dataset_id for ev in evaluations})}")
    lines.append(f"- discovered pairs: {len(pairs)}")
    lines.append(f"- detector: `{args.detector}`")
    lines.append(f"- inference: segmented LRS3 VSR via `{args.infer_pipeline_script}`")
    lines.append("")
    lines.append("## Aggregate Phone Metrics")
    lines.append("")
    lines.append("| Condition | Ref phones | Hyp phones | PER | Accuracy | Tongue-sensitive recall | Subs | Dels | Ins |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(summary_condition_line("active", active_stats))
    lines.append(summary_condition_line("passive", passive_stats))
    lines.append("")
    lines.append(
        f"- active - passive accuracy delta: **{active_stats.phone_accuracy - passive_stats.phone_accuracy:.4f}**"
    )
    lines.append(
        "- active - passive tongue-sensitive recall delta: "
        f"**{active_stats.tongue_sensitive_recall() - passive_stats.tongue_sensitive_recall():.4f}**"
    )
    lines.append("")
    lines.append("## Top Per-Phone Recall Improvements")
    lines.append("")
    lines.append("| Phone | Category | Active recall | Passive recall | Delta |")
    lines.append("|---|---|---:|---:|---:|")
    for row in top_improved:
        lines.append(
            f"| {row['phone']} | {row['analysis_category']} | {float(row['active_recall']):.4f} | "
            f"{float(row['passive_recall']):.4f} | {float(row['active_minus_passive_recall']):.4f} |"
        )
    lines.append("")
    lines.append("## Focused Tongue Phones")
    lines.append("")
    lines.append("| Phone | Active recall | Passive recall | Delta |")
    lines.append("|---|---:|---:|---:|")
    for row in focused:
        lines.append(
            f"| {row['phone']} | {float(row['active_recall']):.4f} | "
            f"{float(row['passive_recall']):.4f} | {float(row['active_minus_passive_recall']):.4f} |"
        )
    lines.append("")
    lines.append("## Datasets")
    lines.append("")
    lines.append("| Dataset | Active video | Passive video | TextGrid |")
    lines.append("|---|---|---|---|")
    for pair in pairs:
        if not any(ev.dataset_id == pair.dataset_id for ev in evaluations):
            continue
        lines.append(
            f"| `{pair.dataset_id}` | `{pair.active_video.name}` | `{pair.passive_video.name}` | "
            f"`{pair.textgrid_path}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    assert_known_phone_category_coverage()

    video_roots = [Path(item) for item in args.video_roots]
    textgrid_roots = [Path(item) for item in args.textgrid_roots]
    pairs = discover_pairs(video_roots, textgrid_roots, args.dataset_id)
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]

    if args.dry_run:
        print_dry_run(pairs)
        return

    timestamp = args.timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path(args.output_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluations: list[ConditionEvaluation] = []
    metric_rows: list[dict[str, str]] = []
    successful_pairs: list[VideoPair] = []
    for pair in pairs:
        if pair.textgrid_path is None:
            print(f"[SKIP] {pair.dataset_id}: missing TextGrid")
            continue
        ref_intervals = parse_textgrid_phones(pair.textgrid_path)
        ref_phones = [interval.text for interval in ref_intervals]
        if not ref_phones:
            print(f"[SKIP] {pair.dataset_id}: no reference phones")
            continue
        reference_text = textgrid_words_text(pair.textgrid_path)

        print(f"[EVAL] {pair.dataset_id}")
        active_eval = evaluate_condition(
            args=args,
            output_dir=output_dir,
            pair=pair,
            condition="active",
            video_path=pair.active_video,
            ref_phones=ref_phones,
            reference_text=reference_text,
        )
        passive_eval = evaluate_condition(
            args=args,
            output_dir=output_dir,
            pair=pair,
            condition="passive",
            video_path=pair.passive_video,
            ref_phones=ref_phones,
            reference_text=reference_text,
        )
        evaluations.extend([active_eval, passive_eval])
        metric_rows.extend(
            [
                metric_row(active_eval, pair.textgrid_path),
                metric_row(passive_eval, pair.textgrid_path),
            ]
        )
        successful_pairs.append(pair)
        print(
            "       active PER="
            f"{active_eval.result.stats.phone_error_rate:.4f} passive PER="
            f"{passive_eval.result.stats.phone_error_rate:.4f}"
        )

    if not evaluations:
        raise SystemExit("No successful active/passive phoneme evaluations.")

    active_stats = aggregate_stats(evaluations, "active")
    passive_stats = aggregate_stats(evaluations, "passive")

    write_per_dataset_metrics(output_dir / "per_dataset_metrics.csv", metric_rows)
    delta_rows = write_per_phone_delta(output_dir / "per_phone_delta.csv", active_stats, passive_stats)
    write_confusion_csv(output_dir / "active_confusion.csv", active_stats)
    write_confusion_csv(output_dir / "passive_confusion.csv", passive_stats)
    plot_confusion_matrix(output_dir / "active_confusion.png", active_stats, "Active tongue phone confusion")
    plot_confusion_matrix(output_dir / "passive_confusion.png", passive_stats, "Passive tongue phone confusion")
    plot_tongue_sensitive_delta(output_dir / "tongue_sensitive_delta.png", delta_rows)
    write_alignment_samples(output_dir / "alignment_samples.json", evaluations)
    write_summary(
        path=output_dir / "summary.md",
        args=args,
        pairs=successful_pairs,
        evaluations=evaluations,
        active_stats=active_stats,
        passive_stats=passive_stats,
        delta_rows=delta_rows,
    )
    print(f"Phoneme confusion outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
