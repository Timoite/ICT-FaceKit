#!/usr/bin/env python3
"""Evaluate VOCASets active-best videos against passive controls."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tongue_scripts.evaluation.evaluate_vsr_ver import evaluate_videos  # noqa: E402
from tongue_scripts.pipelines.render_vocaset_active_best_worker import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT as DEFAULT_ACTIVE_ROOT,
    EXPERIMENT_NAME,
)


@dataclass(frozen=True)
class BatchEvaluationPaths:
    active_root: Path
    passive_root: Path
    transcript_root: Path


@dataclass(frozen=True)
class EvalJob:
    clip_id: str
    speaker: str
    sentence: str
    active_video: Path
    passive_video: Path
    ground_truth: str


DEFAULT_PASSIVE_ROOT = (
    Path("/research/milsrg1/user_workspace/ht467/smirk_task/outputs") / "vocasets_passive"
)
DEFAULT_REPORT_ROOT = (
    Path("/research/milsrg1/user_workspace/ht467/smirk_task/outputs")
    / "vocasets_active_passive_eval"
    / EXPERIMENT_NAME
)
DEFAULT_REPORT_LINK_DIR = (
    PROJECT_ROOT
    / "tests"
    / "vocaset_outputs"
    / "comparisons"
    / EXPERIMENT_NAME
)


def sentence_index(sentence: str) -> int:
    match = re.fullmatch(r"sentence(\d+)", sentence)
    if not match:
        raise ValueError(f"Invalid sentence id: {sentence}")
    return int(match.group(1))


def sentence_ground_truth(transcript_path: Path, sentence: str) -> str:
    lines = [
        line.strip()
        for line in transcript_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]
    index = sentence_index(sentence)
    if not 1 <= index <= len(lines):
        raise ValueError(f"{sentence} is outside transcript range for {transcript_path}")
    return lines[index - 1]


def find_active_video(active_dir: Path) -> Path | None:
    matches = sorted(active_dir.glob("*_active_tongue_with_audio.mp4"))
    return matches[0] if matches else None


def build_eval_jobs(paths: BatchEvaluationPaths) -> list[EvalJob]:
    jobs: list[EvalJob] = []
    for active_dir in sorted(paths.active_root.glob("*/*")):
        if not active_dir.is_dir():
            continue
        speaker = active_dir.parent.name
        sentence = active_dir.name
        active_video = find_active_video(active_dir)
        passive_video = (
            paths.passive_root
            / speaker
            / sentence
            / f"{speaker}_{sentence}_passive_tongue_with_audio.mp4"
        )
        transcript_path = paths.transcript_root / f"{speaker}.txt"
        if active_video is None or not passive_video.is_file() or not transcript_path.is_file():
            continue
        jobs.append(
            EvalJob(
                clip_id=f"{speaker}_{sentence}",
                speaker=speaker,
                sentence=sentence,
                active_video=active_video,
                passive_video=passive_video,
                ground_truth=sentence_ground_truth(transcript_path, sentence),
            )
        )
    return jobs


def split_jobs(jobs: Sequence[EvalJob], workers: int) -> list[list[EvalJob]]:
    return [list(jobs[index::workers]) for index in range(workers)]


def metric_path(report_root: Path, clip_id: str, condition: str) -> Path:
    return report_root / "metrics" / f"{clip_id}_{condition}_metrics.json"


def save_metric(path: Path, row: dict, job: EvalJob, condition: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(row)
    payload.update(
        {
            "clip_id": job.clip_id,
            "speaker": job.speaker,
            "sentence": job.sentence,
            "condition": condition,
            "ground_truth": job.ground_truth,
        }
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_metric(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_one_video(args: argparse.Namespace, job: EvalJob, video: Path, condition: str) -> dict:
    metric = metric_path(Path(args.report_root), job.clip_id, condition)
    if metric.is_file() and not args.force_eval:
        print(f"[SKIP] existing metric {metric}", flush=True)
        return load_metric(metric) or {}

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        prefix=f"{job.clip_id}_gt_",
        suffix=".txt",
        delete=False,
    ) as f:
        f.write(job.ground_truth + "\n")
        gt_path = Path(f.name)
    try:
        eval_args = argparse.Namespace(
            videos=[str(video)],
            ground_truth=str(gt_path),
            infer_script=str(
                PROJECT_ROOT
                / "ADFA_EVALUATION"
                / "Visual_Speech_Recognition_for_Multiple_Languages"
                / "infer.py"
            ),
            infer_pipeline_script=str(
                PROJECT_ROOT
                / "ADFA_EVALUATION"
                / "Visual_Speech_Recognition_for_Multiple_Languages"
                / "infer_pipeline.py"
            ),
            infer_mode=args.infer_mode,
            textgrid_path=None,
            seg_target_seconds=8.0,
            seg_min_seconds=4.0,
            seg_max_seconds=12.0,
            seg_min_silence=0.0,
            config_filename=args.config_filename,
            vowel_mode=args.vowel_mode,
            detector=args.detector,
            python_bin=args.python_bin,
            experiment_name=f"{job.clip_id}_{condition}_{EXPERIMENT_NAME}",
            report_path=str(Path(args.report_root) / "adfa_pair_reports.md"),
            report_mode="append",
            dataset_id=job.clip_id,
            speaker_id=job.speaker,
            hypothesis="active best-setting tongue render should improve VSR over passive",
        )
        _, rows = evaluate_videos(eval_args)
        row = rows[0]
        save_metric(metric, row, job, condition)
        return load_metric(metric) or row
    finally:
        gt_path.unlink(missing_ok=True)


def evaluate_jobs(args: argparse.Namespace, jobs: list[EvalJob]) -> list[dict]:
    rows: list[dict] = []
    for job in jobs:
        print(f"[EVAL] {job.clip_id}", flush=True)
        active = evaluate_one_video(args, job, job.active_video, "active")
        passive = evaluate_one_video(args, job, job.passive_video, "passive")
        rows.extend([active, passive])
        if should_write_incremental_summary(args):
            write_comparison_artifacts(Path(args.report_root), rows)
    return rows


def should_write_incremental_summary(args: argparse.Namespace) -> bool:
    return not getattr(args, "metrics_only", False) and not getattr(args, "summarize_only", False)


def comparison_rows(rows: list[dict]) -> list[dict]:
    by_clip: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_clip.setdefault(row["clip_id"], {})[row["condition"]] = row

    comparisons: list[dict] = []
    for clip_id, pair in sorted(by_clip.items()):
        if "active" not in pair or "passive" not in pair:
            continue
        active = pair["active"]
        passive = pair["passive"]
        comparisons.append(
            {
                "clip_id": clip_id,
                "speaker": active["speaker"],
                "sentence": active["sentence"],
                "active_ver": active["ver"],
                "passive_ver": passive["ver"],
                "delta_ver_active_minus_passive": active["ver"] - passive["ver"],
                "active_wer_norm": active["wer_norm"],
                "passive_wer_norm": passive["wer_norm"],
                "delta_wer_norm_active_minus_passive": active["wer_norm"] - passive["wer_norm"],
                "active_composite": active["composite_index"],
                "passive_composite": passive["composite_index"],
                "active_hypothesis": active["hypothesis"],
                "passive_hypothesis": passive["hypothesis"],
                "ground_truth": active["ground_truth"],
                "active_video": active["video"],
                "passive_video": passive["video"],
            }
        )
    return comparisons


def write_comparison_artifacts(report_root: Path, rows: list[dict]) -> None:
    comparisons = comparison_rows(rows)
    report_root.mkdir(parents=True, exist_ok=True)

    metrics_csv = report_root / "vocaset_active_passive_metrics_long.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "clip_id",
            "speaker",
            "sentence",
            "condition",
            "ver",
            "wer_norm",
            "wer_raw",
            "composite_index",
            "viseme_accuracy",
            "word_accuracy",
            "hypothesis",
            "ground_truth",
            "video",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["clip_id"], r["condition"])):
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    comparison_csv = report_root / "vocaset_active_passive_comparison.csv"
    with comparison_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "clip_id",
            "speaker",
            "sentence",
            "active_ver",
            "passive_ver",
            "delta_ver_active_minus_passive",
            "active_wer_norm",
            "passive_wer_norm",
            "delta_wer_norm_active_minus_passive",
            "active_composite",
            "passive_composite",
            "active_hypothesis",
            "passive_hypothesis",
            "ground_truth",
            "active_video",
            "passive_video",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparisons)

    write_plots(report_root, comparisons)


def write_plots(report_root: Path, comparisons: list[dict]) -> None:
    if not comparisons:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(comparisons, key=lambda row: row["delta_ver_active_minus_passive"])
    x = list(range(len(ordered)))
    active_ver = [row["active_ver"] for row in ordered]
    passive_ver = [row["passive_ver"] for row in ordered]
    deltas = [row["delta_ver_active_minus_passive"] for row in ordered]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, active_ver, linewidth=1.2, label="active VER")
    ax.plot(x, passive_ver, linewidth=1.2, label="passive VER")
    ax.set_title("VOCASets Active Best vs Passive VER by Clip")
    ax.set_xlabel("Clip sorted by active-passive VER delta")
    ax.set_ylabel("VER (lower is better)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_root / "vocaset_active_passive_ver_lines.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#2f7d32" if delta < 0 else "#9b1c1c" for delta in deltas]
    ax.bar(x, deltas, color=colors, width=0.9)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("VOCASets VER Delta: Active Minus Passive")
    ax.set_xlabel("Clip sorted by delta")
    ax.set_ylabel("VER delta (negative means active better)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(report_root / "vocaset_active_passive_ver_delta.png", dpi=180)
    plt.close(fig)


def replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    tmp_link = link.with_name(f".{link.name}.{os.getpid()}.tmp")
    tmp_link.unlink(missing_ok=True)
    tmp_link.symlink_to(target)
    os.replace(tmp_link, link)


def link_report_artifacts(report_root: Path, link_dir: Path) -> None:
    for name in (
        "vocaset_active_passive_metrics_long.csv",
        "vocaset_active_passive_comparison.csv",
        "vocaset_active_passive_ver_lines.png",
        "vocaset_active_passive_ver_delta.png",
        "adfa_pair_reports.md",
    ):
        target = report_root / name
        if target.exists():
            replace_symlink(link_dir / name, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-root", default=str(DEFAULT_ACTIVE_ROOT))
    parser.add_argument("--passive-root", default=str(DEFAULT_PASSIVE_ROOT))
    parser.add_argument(
        "--transcript-root",
        default=str(PROJECT_ROOT / "tests" / "vocasets" / "sentencestext"),
    )
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--link-dir", default=str(DEFAULT_REPORT_LINK_DIR))
    parser.add_argument(
        "--python-bin",
        default="/research/milsrg1/user_workspace/ht467/tools/uv/adfa-vsr/bin/python",
    )
    parser.add_argument("--infer-mode", choices=["full", "segmented"], default="full")
    parser.add_argument("--config-filename", default="configs/LRS3_V_WER19.1.ini")
    parser.add_argument("--vowel-mode", choices=["grouped", "exact"], default="grouped")
    parser.add_argument("--detector", default="mediapipe")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--worker-index", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only write per-video metric JSON files; use --summarize-only afterwards.",
    )
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_root = Path(args.report_root)
    paths = BatchEvaluationPaths(
        active_root=Path(args.active_root),
        passive_root=Path(args.passive_root),
        transcript_root=Path(args.transcript_root),
    )
    jobs = build_eval_jobs(paths)
    if args.workers is not None and args.worker_index is not None:
        jobs = split_jobs(jobs, args.workers)[args.worker_index]
    print(f"[EVAL-BATCH] jobs={len(jobs)}", flush=True)

    if not args.summarize_only:
        evaluate_jobs(args, jobs)

    if args.metrics_only:
        return

    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((report_root / "metrics").glob("*_metrics.json"))
    ]
    write_comparison_artifacts(report_root, rows)
    link_report_artifacts(report_root, Path(args.link_dir))


if __name__ == "__main__":
    main()
