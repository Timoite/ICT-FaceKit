#!/usr/bin/env python3
"""
Run VSR inference and compute VER/Viseme accuracy against a fixed ground truth.

Default usage is aligned with the current ICT-FaceKit layout and ground truth file.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from jiwer import wer as jiwer_wer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_script.ver import calculate_ver


DEFAULT_HYPOTHESIS = "active tongue should outperform passive tongue on same-speaker Wayne data"


def normalize_for_wer(text: str) -> str:
    """Lowercase + remove punctuation + collapse spaces for stable WER."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VSR outputs with VER + WER and write a composite report.")
    parser.add_argument(
        "--videos",
        nargs="+",
        default=[
            str(PROJECT_ROOT / "tongue_scripts" / "outputs" / "1_wayne_0_75_75_with_tongue_with_audio.mp4"),
            str(PROJECT_ROOT / "tongue_scripts" / "outputs" / "1_wayne_0_75_75_passive_tongue_with_audio.mp4"),
        ],
        help="One or more video files to evaluate.",
    )
    parser.add_argument(
        "--ground-truth",
        default=str(PROJECT_ROOT / "tongue_scripts" / "ground_truth.txt"),
        help="Ground-truth transcript text file.",
    )
    parser.add_argument(
        "--infer-script",
        default=str(PROJECT_ROOT / "ADFA_EVALUATION" / "Visual_Speech_Recognition_for_Multiple_Languages" / "infer.py"),
        help="Path to infer.py.",
    )
    parser.add_argument(
        "--infer-pipeline-script",
        default=str(PROJECT_ROOT / "ADFA_EVALUATION" / "Visual_Speech_Recognition_for_Multiple_Languages" / "infer_pipeline.py"),
        help="Path to infer_pipeline.py for segmented inference.",
    )
    parser.add_argument(
        "--infer-mode",
        choices=["full", "segmented"],
        default="full",
        help="Inference mode: direct full-video infer.py or TextGrid-segmented infer_pipeline.py.",
    )
    parser.add_argument(
        "--textgrid-path",
        default=None,
        help="Explicit TextGrid path used when --infer-mode segmented.",
    )
    parser.add_argument("--seg-target-seconds", type=float, default=8.0)
    parser.add_argument("--seg-min-seconds", type=float, default=4.0)
    parser.add_argument("--seg-max-seconds", type=float, default=12.0)
    parser.add_argument("--seg-min-silence", type=float, default=0.0)
    parser.add_argument(
        "--config-filename",
        default="configs/LRS3_V_WER19.1.ini",
        help="Config argument passed to infer.py.",
    )
    parser.add_argument(
        "--vowel-mode",
        choices=["grouped", "exact"],
        default="grouped",
        help="VER vowel mapping mode: grouped viseme vowels or exact vowel labels.",
    )
    parser.add_argument("--detector", default="mediapipe", help="Detector argument passed to infer.py.")
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to run inference.",
    )
    parser.add_argument(
        "--experiment-name",
        default="active_vs_passive_default",
        help="Experiment tag written into the report.",
    )
    parser.add_argument(
        "--report-path",
        default=str(PROJECT_ROOT / "tongue_scripts" / "outputs" / "vsr_composite_report.md"),
        help="Markdown report path.",
    )
    parser.add_argument(
        "--report-mode",
        choices=["append", "write"],
        default="append",
        help="Whether to append to or overwrite the report file.",
    )
    parser.add_argument("--dataset-id", default=None, help="Dataset identifier recorded in the report metadata.")
    parser.add_argument("--speaker-id", default=None, help="Speaker identifier recorded in the report metadata.")
    parser.add_argument(
        "--hypothesis",
        default=DEFAULT_HYPOTHESIS,
        help="Research hypothesis statement recorded in the report metadata.",
    )
    return parser.parse_args()


def build_report_block(
    experiment_name: str,
    ground_truth: str,
    rows_sorted: list[dict],
    args: argparse.Namespace,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_id = args.dataset_id or Path(rows_sorted[0]["video"]).name.split("_with_tongue")[0].split("_passive_tongue")[0]

    lines: list[str] = []
    lines.append(f"## Run: {experiment_name} | {now}")
    lines.append("")
    lines.append("### Settings")
    lines.append(f"- dataset id: `{dataset_id}`")
    if args.speaker_id:
        lines.append(f"- speaker id: `{args.speaker_id}`")
    lines.append(f"- config: `{args.config_filename}`")
    lines.append(f"- infer mode: `{args.infer_mode}`")
    lines.append(f"- vowel mode: `{args.vowel_mode}`")
    lines.append(f"- detector: `{args.detector}`")
    lines.append(f"- infer script: `{args.infer_script}`")
    lines.append(f"- ground truth source: `{args.ground_truth}`")
    lines.append(f"- report mode: `{args.report_mode}`")
    lines.append("")
    lines.append("### Experiment Metadata")
    lines.append(f"- hypothesis: {args.hypothesis}")
    for r in rows_sorted:
        lines.append(f"- video: `{r['video']}`")
    lines.append("")
    lines.append("### VER Summary")
    lines.append("| Video | VER | WER(norm) | WER(raw) | Composite (0.5*VER + 0.5*WER_norm) | Viseme Accuracy | Word Accuracy(norm) | HYP words |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows_sorted:
        lines.append(
            f"| {Path(r['video']).name} | {r['ver']:.4f} | {r['wer_norm']:.4f} | {r['wer_raw']:.4f} | {r['composite_index']:.4f} | {r['viseme_accuracy']:.2f}% | {r['word_accuracy']:.2f}% | {r['hyp_words']} |"
        )
    lines.append("")
    if len(rows_sorted) >= 2:
        best = rows_sorted[0]
        worst = rows_sorted[-1]
        delta_ver = worst["ver"] - best["ver"]
        delta_wer = worst["wer_norm"] - best["wer_norm"]
        delta_composite = worst["composite_index"] - best["composite_index"]
        lines.append(
            f"- Best (by composite): **{Path(best['video']).name}** (VER={best['ver']:.4f}, WER_norm={best['wer_norm']:.4f}, Composite={best['composite_index']:.4f})"
        )
        lines.append(
            f"- Worst (by composite): **{Path(worst['video']).name}** (VER={worst['ver']:.4f}, WER_norm={worst['wer_norm']:.4f}, Composite={worst['composite_index']:.4f})"
        )
        lines.append(f"- VER gap (worst - best): **{delta_ver:.4f}**")
        lines.append(f"- WER gap (worst - best): **{delta_wer:.4f}**")
        lines.append(f"- Composite gap (worst - best): **{delta_composite:.4f}**")
        lines.append("")

    lines.append("### Ground Truth")
    lines.append(ground_truth)
    lines.append("")

    lines.append("### Hypotheses")
    for r in rows_sorted:
        lines.append(f"#### {Path(r['video']).name}")
        lines.append(f"- VER: {r['ver']:.4f}")
        lines.append(f"- WER(norm): {r['wer_norm']:.4f}")
        lines.append(f"- WER(raw): {r['wer_raw']:.4f}")
        lines.append(f"- Composite Index: {r['composite_index']:.4f}")
        lines.append(f"- Viseme Accuracy: {r['viseme_accuracy']:.2f}%")
        lines.append(f"- Word Accuracy(norm): {r['word_accuracy']:.2f}%")
        lines.append(f"- HYP: {r['hypothesis']}")
        lines.append("")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def run_inference(python_bin: str, infer_script: Path, config_filename: str, video_path: Path, detector: str) -> str:
    cmd = [
        python_bin,
        str(infer_script),
        f"config_filename={config_filename}",
        f"data_filename={video_path}",
        f"detector={detector}",
    ]
    proc = subprocess.run(cmd, cwd=infer_script.parent, capture_output=True, text=True)
    merged = (proc.stdout or "") + "\n" + (proc.stderr or "")
    match = re.search(r"hyp:\s*(.*)", merged)
    if not match:
        tail = merged[-500:].strip()
        raise RuntimeError(f"Failed to parse hypothesis for {video_path.name}. Inference tail:\n{tail}")
    return match.group(1).strip()


def run_inference_segmented(
    python_bin: str,
    infer_pipeline_script: Path,
    config_filename: str,
    video_path: Path,
    detector: str,
    textgrid_path: Path,
    seg_target_seconds: float,
    seg_min_seconds: float,
    seg_max_seconds: float,
    seg_min_silence: float,
) -> str:
    with tempfile.TemporaryDirectory(prefix="vsr_segmented_") as tmp_dir:
        cmd = [
            python_bin,
            str(infer_pipeline_script),
            "--video-path",
            str(video_path),
            "--output-dir",
            tmp_dir,
            "--decode-config",
            config_filename,
            "--textgrid-path",
            str(textgrid_path),
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
        proc = subprocess.run(cmd, cwd=infer_pipeline_script.parent, capture_output=True, text=True)
        if proc.returncode != 0:
            merged = (proc.stdout or "") + "\n" + (proc.stderr or "")
            raise RuntimeError(f"Segmented inference failed for {video_path.name}:\n{merged[-1200:]}")

        out_txt = Path(tmp_dir) / f"{video_path.stem}.txt"
        if not out_txt.exists():
            raise RuntimeError(f"Segmented transcript missing: {out_txt}")
        return out_txt.read_text(encoding="utf-8", errors="ignore").strip()


def validate_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, list[Path], Path | None]:
    gt_path = Path(args.ground_truth)
    infer_script = Path(args.infer_script)
    infer_pipeline_script = Path(args.infer_pipeline_script)
    video_paths = [Path(v) for v in args.videos]

    if not gt_path.is_file():
        raise SystemExit(f"Ground truth file not found: {gt_path}")
    if not infer_script.is_file():
        raise SystemExit(f"infer.py not found: {infer_script}")
    if args.infer_mode == "segmented" and not infer_pipeline_script.is_file():
        raise SystemExit(f"infer_pipeline.py not found: {infer_pipeline_script}")

    textgrid_path = Path(args.textgrid_path) if args.textgrid_path else None
    if args.infer_mode == "segmented":
        if textgrid_path is None or not textgrid_path.is_file():
            raise SystemExit("--textgrid-path is required and must exist when --infer-mode segmented")

    return gt_path, infer_script, infer_pipeline_script, video_paths, textgrid_path


def evaluate_videos(args: argparse.Namespace) -> tuple[str, list[dict]]:
    gt_path, infer_script, infer_pipeline_script, video_paths, textgrid_path = validate_inputs(args)

    ground_truth = gt_path.read_text(encoding="utf-8", errors="ignore").strip().lower()

    rows: list[dict] = []
    print("=" * 88)
    print("VSR Composite Evaluation (VER + WER)")
    print("=" * 88)
    print(f"Ground truth: {gt_path}")
    print(f"Infer script: {infer_script}")
    print(f"Config: {args.config_filename}")
    print(f"Vowel mode: {args.vowel_mode}")
    print("-" * 88)

    for video_path in video_paths:
        if not video_path.is_file():
            print(f"[SKIP] Missing video: {video_path}")
            continue

        try:
            if args.infer_mode == "segmented":
                hyp = run_inference_segmented(
                    python_bin=args.python_bin,
                    infer_pipeline_script=infer_pipeline_script,
                    config_filename=args.config_filename,
                    video_path=video_path,
                    detector=args.detector,
                    textgrid_path=textgrid_path,
                    seg_target_seconds=args.seg_target_seconds,
                    seg_min_seconds=args.seg_min_seconds,
                    seg_max_seconds=args.seg_max_seconds,
                    seg_min_silence=args.seg_min_silence,
                )
            else:
                hyp = run_inference(
                    python_bin=args.python_bin,
                    infer_script=infer_script,
                    config_filename=args.config_filename,
                    video_path=video_path,
                    detector=args.detector,
                )
            ver, ref_visemes, hyp_visemes = calculate_ver(
                ground_truth,
                hyp.lower(),
                vowel_mode=args.vowel_mode,
            )
            hyp_lower = hyp.lower()
            word_error_raw = jiwer_wer(ground_truth, hyp_lower)
            word_error_norm = jiwer_wer(normalize_for_wer(ground_truth), normalize_for_wer(hyp_lower))
            composite_index = 0.5 * ver + 0.5 * word_error_norm
            row = {
                "video": str(video_path),
                "ver": ver,
                "wer_raw": word_error_raw,
                "wer_norm": word_error_norm,
                "composite_index": composite_index,
                "viseme_accuracy": (1.0 - ver) * 100.0,
                "word_accuracy": (1.0 - word_error_norm) * 100.0,
                "hypothesis": hyp,
                "hyp_words": len(hyp.split()),
                "ref_visemes": ref_visemes,
                "hyp_visemes": hyp_visemes,
            }
            rows.append(row)
            print(f"[OK] {video_path.name}")
            print(
                f"     VER={row['ver']:.4f}  WER(norm)={row['wer_norm']:.4f}  WER(raw)={row['wer_raw']:.4f}  Composite={row['composite_index']:.4f}"
            )
            print(
                f"     VisemeAcc={row['viseme_accuracy']:.2f}%  WordAcc(norm)={row['word_accuracy']:.2f}%"
            )
            print(f"     hyp_words={row['hyp_words']}")
        except Exception as exc:
            print(f"[FAIL] {video_path.name}: {exc}")

    if not rows:
        raise SystemExit("No successful evaluations.")

    rows_sorted = sorted(rows, key=lambda r: r["composite_index"])
    print("-" * 88)
    for r in rows_sorted:
        name = Path(r["video"]).name
        print(
            f"  {name:50s}  VER={r['ver']:.4f}  WER(norm)={r['wer_norm']:.4f}  Composite={r['composite_index']:.4f}"
        )

    return ground_truth, rows_sorted


def write_report(report_path: Path, report_block: str, report_mode: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if report_mode == "append" else "w"
    with report_path.open(file_mode, encoding="utf-8") as f:
        f.write(report_block)
    print(f"Report {'appended' if report_mode == 'append' else 'written'}: {report_path}")


def main() -> None:
    args = parse_args()

    ground_truth, rows_sorted = evaluate_videos(args)

    report_path = Path(args.report_path)
    report_block = build_report_block(
        experiment_name=args.experiment_name,
        ground_truth=ground_truth,
        rows_sorted=rows_sorted,
        args=args,
    )
    write_report(report_path, report_block, args.report_mode)


if __name__ == "__main__":
    main()
