#!/usr/bin/env python3
"""
Wayne-specific active vs passive VER/WER runner.

Creates one markdown log per evaluation run and maintains a lightweight index
for the Wayne same-speaker experiment.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "tongue_scripts" / "outputs"
WAYNE_LOG_DIR = OUTPUT_ROOT / "wayne_logs"
WAYNE_INDEX_PATH = WAYNE_LOG_DIR / "INDEX.md"
WAYNE_DATASET_ID = "1_wayne_0_75_75"
WAYNE_SPEAKER_ID = "1"
DEFAULT_EXPERIMENT_NAME = "wayne_same_speaker_active_vs_passive"

VIDEO_CANDIDATES = {
    "active": [
        OUTPUT_ROOT / f"{WAYNE_DATASET_ID}_with_tongue_with_audio.mp4",
        OUTPUT_ROOT / "multi_speaker" / f"{WAYNE_DATASET_ID}_with_tongue_with_audio.mp4",
        PROJECT_ROOT / "tongue_scripts" / "batch_videos" / f"{WAYNE_DATASET_ID}_final.mp4",
    ],
    "passive": [
        OUTPUT_ROOT / f"{WAYNE_DATASET_ID}_passive_tongue_with_audio.mp4",
        OUTPUT_ROOT / "multi_speaker" / f"{WAYNE_DATASET_ID}_passive_tongue_with_audio.mp4",
    ],
}

GROUND_TRUTH_CANDIDATES = [
    PROJECT_ROOT / "tongue_scripts" / "ground_truth.txt",
]

TEXTGRID_CANDIDATES = [
    PROJECT_ROOT / "tongue_scripts" / "inputs" / f"{WAYNE_DATASET_ID}.TextGrid",
    PROJECT_ROOT / "ADFA_EVALUATION" / "data" / "beat_cache" / "beat_english_v0.2.1" / "beat_english_v0.2.1" / WAYNE_SPEAKER_ID / f"{WAYNE_DATASET_ID}.TextGrid",
]

DEFAULT_HYPOTHESIS = "active tongue should outperform passive tongue on same-speaker Wayne data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Wayne same-speaker VER/WER evaluation.")
    parser.add_argument("--dataset-id", default=WAYNE_DATASET_ID)
    parser.add_argument("--speaker-id", default=WAYNE_SPEAKER_ID)
    parser.add_argument("--infer-mode", choices=["full", "segmented"], default="full")
    parser.add_argument("--vowel-mode", choices=["grouped", "exact"], default="grouped")
    parser.add_argument("--config-filename", default="configs/LRS3_V_WER19.1.ini")
    parser.add_argument("--detector", default="mediapipe")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--log-dir", default=str(WAYNE_LOG_DIR))
    parser.add_argument("--ground-truth", default=None, help="Override Wayne transcript source.")
    parser.add_argument("--textgrid-path", default=None, help="Override segmented inference TextGrid path.")
    parser.add_argument(
        "--hypothesis",
        default=DEFAULT_HYPOTHESIS,
        help="Hypothesis statement included in each markdown log.",
    )
    parser.add_argument(
        "--fallback-wer-threshold",
        type=float,
        default=0.95,
        help="Trigger segmented fallback when the best full-run normalized WER meets or exceeds this threshold.",
    )
    parser.add_argument(
        "--fallback-repetition-threshold",
        type=float,
        default=0.35,
        help="Trigger segmented fallback when repeated 3-word phrases exceed this fraction of all 3-word windows.",
    )
    parser.add_argument(
        "--disable-segmented-fallback",
        action="store_true",
        help="Skip the segmented retry even if the full-video result looks unusable.",
    )
    return parser.parse_args()


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def resolve_video(label: str) -> Path:
    resolved = first_existing(VIDEO_CANDIDATES[label])
    if resolved is None:
        checked = "\n".join(f"  - {path}" for path in VIDEO_CANDIDATES[label])
        raise SystemExit(f"Missing Wayne {label} video. Checked:\n{checked}")
    return resolved


def resolve_ground_truth(override: str | None) -> Path:
    if override:
        path = Path(override)
        if not path.is_file():
            raise SystemExit(f"Ground truth file not found: {path}")
        return path
    resolved = first_existing(GROUND_TRUTH_CANDIDATES)
    if resolved is None:
        checked = "\n".join(f"  - {path}" for path in GROUND_TRUTH_CANDIDATES)
        raise SystemExit(f"Missing Wayne ground truth file. Checked:\n{checked}")
    return resolved


def resolve_textgrid(override: str | None) -> Path | None:
    if override:
        path = Path(override)
        if not path.is_file():
            raise SystemExit(f"TextGrid file not found: {path}")
        return path
    return first_existing(TEXTGRID_CANDIDATES)


def build_log_path(log_dir: Path, dataset_id: str, infer_mode: str, vowel_mode: str, suffix: str = "") -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    suffix_part = f"_{suffix}" if suffix else ""
    base_name = f"{timestamp}_{dataset_id}_{infer_mode}_{vowel_mode}{suffix_part}"
    candidate = log_dir / f"{base_name}.md"
    counter = 2
    while candidate.exists():
        candidate = log_dir / f"{base_name}_run{counter:02d}.md"
        counter += 1
    return candidate


def create_eval_args(
    run_args: argparse.Namespace,
    active_video: Path,
    passive_video: Path,
    ground_truth: Path,
    report_path: Path,
    *,
    infer_mode: str,
    textgrid_path: Path | None,
    experiment_name: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        videos=[str(active_video), str(passive_video)],
        ground_truth=str(ground_truth),
        infer_script=str(PROJECT_ROOT / "ADFA_EVALUATION" / "Visual_Speech_Recognition_for_Multiple_Languages" / "infer.py"),
        infer_pipeline_script=str(PROJECT_ROOT / "ADFA_EVALUATION" / "Visual_Speech_Recognition_for_Multiple_Languages" / "infer_pipeline.py"),
        infer_mode=infer_mode,
        textgrid_path=str(textgrid_path) if textgrid_path else None,
        seg_target_seconds=8.0,
        seg_min_seconds=4.0,
        seg_max_seconds=12.0,
        seg_min_silence=0.0,
        config_filename=run_args.config_filename,
        vowel_mode=run_args.vowel_mode,
        detector=run_args.detector,
        python_bin=sys.executable,
        experiment_name=experiment_name,
        report_path=str(report_path),
        report_mode="write",
        dataset_id=run_args.dataset_id,
        speaker_id=run_args.speaker_id,
        hypothesis=run_args.hypothesis,
    )


def trigram_repetition_ratio(text: str) -> float:
    words = normalize_text(text).split()
    if len(words) < 3:
        return 0.0
    trigrams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 0.0
    repeated = len(trigrams) - len(set(trigrams))
    return repeated / float(len(trigrams))


def run_quality_needs_fallback(rows_sorted: list[dict], wer_threshold: float, repetition_threshold: float) -> bool:
    best_wer = min(row["wer_norm"] for row in rows_sorted)
    max_repetition = max(trigram_repetition_ratio(row["hypothesis"]) for row in rows_sorted)
    return best_wer >= wer_threshold or max_repetition >= repetition_threshold


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def write_wayne_index(log_dir: Path) -> None:
    logs = sorted(path for path in log_dir.glob("*.md") if path.name != WAYNE_INDEX_PATH.name)
    lines = [
        "# Wayne Same-Speaker Evaluation Logs",
        "",
        "| Log | Dataset | Mode | Vowel Mode | Timestamp |",
        "|---|---|---|---|---|",
    ]

    pattern = re.compile(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{6})_(?P<dataset>.+?)_(?P<mode>full|segmented)_(?P<vowel>grouped|exact)(?:_.+)?\.md$"
    )
    for log_path in logs:
        match = pattern.match(log_path.name)
        if match:
            timestamp = match.group("timestamp")
            dataset = match.group("dataset")
            mode = match.group("mode")
            vowel = match.group("vowel")
        else:
            timestamp = "-"
            dataset = WAYNE_DATASET_ID
            mode = "-"
            vowel = "-"
        rel_path = log_path.relative_to(PROJECT_ROOT)
        lines.append(f"| `{rel_path}` | `{dataset}` | `{mode}` | `{vowel}` | `{timestamp}` |")

    lines.append("")
    WAYNE_INDEX_PATH.write_text("\n".join(lines), encoding="utf-8")


def execute_run(eval_args: SimpleNamespace) -> list[dict]:
    import evaluate_vsr_ver as evv

    ground_truth, rows_sorted = evv.evaluate_videos(eval_args)
    report_block = evv.build_report_block(
        experiment_name=eval_args.experiment_name,
        ground_truth=ground_truth,
        rows_sorted=rows_sorted,
        args=eval_args,
    )
    evv.write_report(Path(eval_args.report_path), report_block, eval_args.report_mode)
    return rows_sorted


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    active_video = resolve_video("active")
    passive_video = resolve_video("passive")
    ground_truth = resolve_ground_truth(args.ground_truth)
    textgrid_path = resolve_textgrid(args.textgrid_path)

    full_log_path = build_log_path(log_dir, args.dataset_id, args.infer_mode, args.vowel_mode)
    full_args = create_eval_args(
        args,
        active_video,
        passive_video,
        ground_truth,
        full_log_path,
        infer_mode=args.infer_mode,
        textgrid_path=textgrid_path if args.infer_mode == "segmented" else None,
        experiment_name=args.experiment_name,
    )

    rows_sorted = execute_run(full_args)

    if (
        args.infer_mode == "full"
        and not args.disable_segmented_fallback
        and run_quality_needs_fallback(
            rows_sorted,
            wer_threshold=args.fallback_wer_threshold,
            repetition_threshold=args.fallback_repetition_threshold,
        )
    ):
        if textgrid_path is None:
            print("[WARN] Full-video run looks unusable, but no Wayne TextGrid was found for segmented fallback.")
        else:
            fallback_log_path = build_log_path(log_dir, args.dataset_id, "segmented", args.vowel_mode, suffix="fallback")
            fallback_args = create_eval_args(
                args,
                active_video,
                passive_video,
                ground_truth,
                fallback_log_path,
                infer_mode="segmented",
                textgrid_path=textgrid_path,
                experiment_name=f"{args.experiment_name}_segmented_fallback",
            )
            execute_run(fallback_args)

    write_wayne_index(log_dir)
    print(f"Wayne logs index updated: {WAYNE_INDEX_PATH}")


if __name__ == "__main__":
    main()
