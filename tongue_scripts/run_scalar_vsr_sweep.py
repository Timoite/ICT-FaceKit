#!/usr/bin/env python3
"""
Run a low-dimensional scalar sweep over tongue std scaling and rank VSR results.

This experiment intentionally avoids phoneme-wise high-dimensional optimization.
It answers a simpler question first:

    "If we only change the global tongue motion amplitude, can we beat the
    current unoptimized baseline on VSR?"

Rendering is delegated to render_fullface_shift_compare.py so the geometry path
stays aligned with the existing full-face renderer.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tongue_scripts import evaluate_vsr_ver as ev


@dataclass
class EvalMetrics:
    video: str
    scalar: float
    ver: float
    wer_raw: float
    wer_norm: float
    composite: float
    viseme_accuracy: float
    word_accuracy: float
    hypothesis: str


def parse_textgrid_words(textgrid_path: Path, tier_name: str = "words") -> list[str]:
    words: list[str] = []
    in_tier = False
    current: dict[str, str] = {}
    with textgrid_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
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
                txt = line.split("=", 1)[1].strip().strip('"')
                current["text"] = txt
                if {"start", "end", "text"} <= current.keys():
                    token = txt.strip().lower()
                    if token and token != "sp":
                        words.append(token)
    return words


def ensure_ground_truth(textgrid_path: Path, ground_truth_path: Path) -> str:
    if ground_truth_path.is_file():
        return ground_truth_path.read_text(encoding="utf-8", errors="ignore").strip().lower()
    words = parse_textgrid_words(textgrid_path, tier_name="words")
    text = " ".join(words).strip().lower()
    if not text:
        raise RuntimeError(f"No words extracted from TextGrid words tier: {textgrid_path}")
    ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_path.write_text(text + "\n", encoding="utf-8")
    return text


def run_cmd(cmd: list[str], cwd: Optional[Path] = None) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def shift_tag(shift_seconds: float) -> str:
    return f"shift{int(round(float(shift_seconds) * 1000.0)):03d}ms"


def scalar_tag(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    safe = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    return f"scalar_{safe}"


def render_fullface_video(
    *,
    python_bin: str,
    dataset_id: str,
    speaker_id: str,
    beat_root: Path,
    motion_path: Path,
    output_dir: Path,
    std_scalar: float,
    render_shift_seconds: float,
    fps: int,
    max_seconds: Optional[float],
    skip_existing: bool,
) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin,
        str(SCRIPT_DIR / "render_fullface_shift_compare.py"),
        "--dataset-id",
        dataset_id,
        "--speaker-id",
        str(speaker_id),
        "--beat-root",
        str(beat_root.resolve()),
        "--motion-path",
        str(motion_path.resolve()),
        "--std-scalar",
        str(std_scalar),
        "--fps",
        str(int(fps)),
        "--shift-seconds",
        str(render_shift_seconds),
        "--output-dir",
        str(output_dir),
    ]
    if max_seconds is not None:
        cmd.extend(["--max-seconds", str(float(max_seconds))])
    if skip_existing:
        cmd.append("--skip-existing")
    else:
        cmd.append("--no-skip-existing")
    run_cmd(cmd, cwd=SCRIPT_DIR)

    video = output_dir / dataset_id / f"{dataset_id}_FULL_FACE_{shift_tag(render_shift_seconds)}_with_audio.mp4"
    if not video.is_file():
        fallback = output_dir / dataset_id / f"{dataset_id}_FULL_FACE_{shift_tag(render_shift_seconds)}.mp4"
        if fallback.is_file():
            return fallback
        raise RuntimeError(f"Rendered video not found: {video}")
    return video


def evaluate_video(
    *,
    python_bin: str,
    scalar: float,
    video_path: Path,
    ground_truth: str,
    infer_mode: str,
    textgrid_path: Path,
    detector: str,
    vowel_mode: str,
    config_filename: str,
    seg_target_seconds: float,
    seg_min_seconds: float,
    seg_max_seconds: float,
    seg_min_silence: float,
) -> EvalMetrics:
    video_path = video_path.resolve()
    textgrid_path = textgrid_path.resolve()
    infer_script = PROJECT_ROOT / "ADFA_EVALUATION" / "Visual_Speech_Recognition_for_Multiple_Languages" / "infer.py"
    infer_pipeline_script = (
        PROJECT_ROOT / "ADFA_EVALUATION" / "Visual_Speech_Recognition_for_Multiple_Languages" / "infer_pipeline.py"
    )
    if infer_mode == "segmented":
        hyp = ev.run_inference_segmented(
            python_bin=python_bin,
            infer_pipeline_script=infer_pipeline_script,
            video_path=video_path,
            detector=detector,
            textgrid_path=textgrid_path,
            seg_target_seconds=seg_target_seconds,
            seg_min_seconds=seg_min_seconds,
            seg_max_seconds=seg_max_seconds,
            seg_min_silence=seg_min_silence,
        )
    else:
        hyp = ev.run_inference(
            python_bin=python_bin,
            infer_script=infer_script,
            config_filename=config_filename,
            video_path=video_path,
            detector=detector,
        )

    hyp_lower = hyp.lower()
    ver, _, _ = ev.calculate_ver(ground_truth, hyp_lower, vowel_mode=vowel_mode)
    wer_raw = ev.jiwer_wer(ground_truth, hyp_lower)
    wer_norm = ev.jiwer_wer(ev.normalize_for_wer(ground_truth), ev.normalize_for_wer(hyp_lower))
    composite = 0.5 * ver + 0.5 * wer_norm
    return EvalMetrics(
        video=str(video_path),
        scalar=float(scalar),
        ver=float(ver),
        wer_raw=float(wer_raw),
        wer_norm=float(wer_norm),
        composite=float(composite),
        viseme_accuracy=float((1.0 - ver) * 100.0),
        word_accuracy=float((1.0 - wer_norm) * 100.0),
        hypothesis=hyp.strip(),
    )


def relative_improvement(baseline: float, candidate: float) -> float:
    if baseline <= 1e-12:
        return 0.0
    return (baseline - candidate) / baseline


def build_markdown(
    *,
    dataset_id: str,
    motion_path: Path,
    render_shift_seconds: float,
    baseline_scalar: float,
    rows: list[dict],
    best_row: dict,
) -> str:
    lines: list[str] = []
    lines.append(f"# Scalar VSR Sweep: {dataset_id}")
    lines.append("")
    lines.append(f"- Motion path: `{motion_path}`")
    lines.append(f"- Render shift seconds: `{render_shift_seconds:+.4f}`")
    lines.append(f"- Baseline scalar: `{baseline_scalar:.4f}`")
    lines.append("")
    lines.append("| Scalar | VER | WER(norm) | WER(raw) | Composite | ΔComposite | ΔWER(norm) | Rel Composite | Rank |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| {row['scalar']:.4f} | {m['ver']:.4f} | {m['wer_norm']:.4f} | {m['wer_raw']:.4f} | "
            f"{m['composite']:.4f} | {row['delta_composite']:+.4f} | {row['delta_wer_norm']:+.4f} | "
            f"{row['rel_composite']:+.4f} | {row['rank']} |"
        )
    lines.append("")
    lines.append(
        f"- Best scalar: `{best_row['scalar']:.4f}` with Composite={best_row['metrics']['composite']:.4f}, "
        f"WER(norm)={best_row['metrics']['wer_norm']:.4f}, VER={best_row['metrics']['ver']:.4f}"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep global tongue std scaling and compare VSR against an unoptimized baseline."
    )
    parser.add_argument("--dataset-id", default="1_wayne_0_75_75")
    parser.add_argument("--speaker-id", default="1")
    parser.add_argument(
        "--beat-root",
        default=str(PROJECT_ROOT / "data" / "beat_cache" / "beat_english_v0.2.1" / "beat_english_v0.2.1"),
    )
    parser.add_argument(
        "--motion-path",
        default=str(SCRIPT_DIR / "outputs" / "1_wayne_0_75_75_pre_shift012.npy"),
        help=(
            "Input motion .npy. Use an already globally aligned file here if you have one; "
            "the default is a pre-shifted example to avoid mixing scalar sweep with extra timing changes."
        ),
    )
    parser.add_argument("--textgrid-path", default=None)
    parser.add_argument("--ground-truth-path", default=None)
    parser.add_argument("--run-root", default=str(SCRIPT_DIR / "outputs" / "scalar_vsr_runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--python-bin", default=sys.executable)

    parser.add_argument(
        "--scalars",
        nargs="+",
        type=float,
        default=[0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26],
        help="Global std scaling values to sweep.",
    )
    parser.add_argument("--baseline-scalar", type=float, default=0.20)
    parser.add_argument(
        "--render-shift-seconds",
        type=float,
        default=0.0,
        help=(
            "Extra render-time shift applied on top of motion-path. "
            "Default is 0.0 so pre-aligned motion is not shifted twice."
        ),
    )
    parser.add_argument("--render-fps", type=int, default=25)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--skip-existing-renders", action="store_true", default=True)
    parser.add_argument("--no-skip-existing-renders", dest="skip_existing_renders", action="store_false")

    parser.add_argument("--infer-mode", choices=["full", "segmented"], default="segmented")
    parser.add_argument("--detector", default="mediapipe")
    parser.add_argument("--vowel-mode", choices=["grouped", "exact"], default="grouped")
    parser.add_argument("--config-filename", default="configs/LRS3_V_WER19.1.ini")
    parser.add_argument("--seg-target-seconds", type=float, default=8.0)
    parser.add_argument("--seg-min-seconds", type=float, default=4.0)
    parser.add_argument("--seg-max-seconds", type=float, default=12.0)
    parser.add_argument("--seg-min-silence", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    beat_root = Path(args.beat_root)
    speaker_dir = beat_root / str(args.speaker_id)
    motion_path = Path(args.motion_path)
    textgrid_path = Path(args.textgrid_path) if args.textgrid_path else (speaker_dir / f"{args.dataset_id}.TextGrid")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name if args.run_name else f"{stamp}_{args.dataset_id}"
    run_dir = Path(args.run_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_path = (
        Path(args.ground_truth_path)
        if args.ground_truth_path
        else run_dir / f"{args.dataset_id}_ground_truth.txt"
    )

    if not motion_path.is_file():
        raise SystemExit(f"Motion file not found: {motion_path}")
    if not textgrid_path.is_file():
        raise SystemExit(f"TextGrid not found: {textgrid_path}")

    scalars = [float(s) for s in args.scalars]
    if float(args.baseline_scalar) not in scalars:
        scalars.append(float(args.baseline_scalar))
    scalars = sorted(set(scalars))

    print("=" * 88)
    print("SCALAR VSR SWEEP")
    print("=" * 88)
    print(f"dataset_id:          {args.dataset_id}")
    print(f"motion_path:         {motion_path}")
    print(f"render_shift_secs:   {float(args.render_shift_seconds):+.4f}")
    print(f"baseline_scalar:     {float(args.baseline_scalar):.4f}")
    print(f"scalars:             {', '.join(f'{s:.4f}' for s in scalars)}")
    print(f"run_dir:             {run_dir}")
    print("=" * 88)

    ground_truth = ensure_ground_truth(textgrid_path, ground_truth_path)

    rows: list[dict] = []
    for scalar in scalars:
        scalar_dir = run_dir / scalar_tag(scalar)
        render_dir = scalar_dir / "render"
        transcript_dir = scalar_dir / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)

        print("-" * 88)
        print(f"[SCALAR {scalar:.4f}] rendering + VSR")
        video_path = render_fullface_video(
            python_bin=args.python_bin,
            dataset_id=args.dataset_id,
            speaker_id=str(args.speaker_id),
            beat_root=beat_root,
            motion_path=motion_path,
            output_dir=render_dir,
            std_scalar=float(scalar),
            render_shift_seconds=float(args.render_shift_seconds),
            fps=int(args.render_fps),
            max_seconds=args.max_seconds,
            skip_existing=bool(args.skip_existing_renders),
        )
        metrics = evaluate_video(
            python_bin=args.python_bin,
            scalar=float(scalar),
            video_path=video_path,
            ground_truth=ground_truth,
            infer_mode=args.infer_mode,
            textgrid_path=textgrid_path,
            detector=args.detector,
            vowel_mode=args.vowel_mode,
            config_filename=args.config_filename,
            seg_target_seconds=args.seg_target_seconds,
            seg_min_seconds=args.seg_min_seconds,
            seg_max_seconds=args.seg_max_seconds,
            seg_min_silence=args.seg_min_silence,
        )
        row = {
            "scalar": float(scalar),
            "video_path": str(video_path),
            "metrics": metrics.__dict__,
        }
        rows.append(row)
        print(
            f"[RESULT {scalar:.4f}] VER={metrics.ver:.4f} "
            f"WER(norm)={metrics.wer_norm:.4f} Composite={metrics.composite:.4f}"
        )

    baseline_row = next((row for row in rows if abs(row["scalar"] - float(args.baseline_scalar)) < 1e-9), None)
    if baseline_row is None:
        raise RuntimeError(f"Baseline scalar row missing: {args.baseline_scalar}")

    baseline_metrics = baseline_row["metrics"]
    rows_sorted = sorted(rows, key=lambda row: (row["metrics"]["composite"], row["metrics"]["wer_norm"]))
    for rank, row in enumerate(rows_sorted, start=1):
        metrics = row["metrics"]
        row["rank"] = int(rank)
        row["delta_composite"] = float(baseline_metrics["composite"] - metrics["composite"])
        row["delta_wer_norm"] = float(baseline_metrics["wer_norm"] - metrics["wer_norm"])
        row["rel_composite"] = float(relative_improvement(baseline_metrics["composite"], metrics["composite"]))
        row["rel_wer_norm"] = float(relative_improvement(baseline_metrics["wer_norm"], metrics["wer_norm"]))

    best_row = rows_sorted[0]

    summary = {
        "dataset_id": args.dataset_id,
        "speaker_id": str(args.speaker_id),
        "motion_path": str(motion_path.resolve()),
        "textgrid_path": str(textgrid_path.resolve()),
        "ground_truth_path": str(ground_truth_path.resolve()),
        "render_shift_seconds": float(args.render_shift_seconds),
        "baseline_scalar": float(args.baseline_scalar),
        "scalars": scalars,
        "rows_sorted": rows_sorted,
        "best_row": best_row,
        "settings": {
            "infer_mode": args.infer_mode,
            "detector": args.detector,
            "vowel_mode": args.vowel_mode,
            "config_filename": args.config_filename,
            "render_fps": int(args.render_fps),
            "max_seconds": args.max_seconds,
            "skip_existing_renders": bool(args.skip_existing_renders),
        },
    }

    summary_json = run_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_csv = run_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "scalar",
                "ver",
                "wer_norm",
                "wer_raw",
                "composite",
                "delta_composite",
                "delta_wer_norm",
                "rel_composite",
                "rel_wer_norm",
                "video_path",
            ],
        )
        writer.writeheader()
        for row in rows_sorted:
            metrics = row["metrics"]
            writer.writerow(
                {
                    "rank": row["rank"],
                    "scalar": row["scalar"],
                    "ver": metrics["ver"],
                    "wer_norm": metrics["wer_norm"],
                    "wer_raw": metrics["wer_raw"],
                    "composite": metrics["composite"],
                    "delta_composite": row["delta_composite"],
                    "delta_wer_norm": row["delta_wer_norm"],
                    "rel_composite": row["rel_composite"],
                    "rel_wer_norm": row["rel_wer_norm"],
                    "video_path": row["video_path"],
                }
            )

    summary_md = run_dir / "summary.md"
    summary_md.write_text(
        build_markdown(
            dataset_id=args.dataset_id,
            motion_path=motion_path.resolve(),
            render_shift_seconds=float(args.render_shift_seconds),
            baseline_scalar=float(args.baseline_scalar),
            rows=rows_sorted,
            best_row=best_row,
        ),
        encoding="utf-8",
    )

    print("=" * 88)
    print("Scalar sweep complete.")
    print(f"Best scalar:   {best_row['scalar']:.4f}")
    print(f"Summary JSON:  {summary_json}")
    print(f"Summary CSV:   {summary_csv}")
    print(f"Summary MD:    {summary_md}")
    print("=" * 88)


if __name__ == "__main__":
    main()
