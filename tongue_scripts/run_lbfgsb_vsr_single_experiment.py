#!/usr/bin/env python3
"""
Single-sample L-BFGS-B experiment driver:
baseline -> 3 candidates -> pick best -> reproducibility rerun.
"""

from __future__ import annotations

import argparse
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


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    tau_alveolar_mm: float
    tau_interdental_mm: float
    lambda_contact: float
    tip_delta_bounds_mm: float
    contact_window_fraction: float


@dataclass
class EvalMetrics:
    video: str
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


def render_dynamic_video(
    python_bin: str,
    dataset_id: str,
    speaker_id: str,
    beat_root: Path,
    motion_path: Path,
    output_dir: Path,
    tongue_shift_seconds: float,
) -> Path:
    output_dir = output_dir.resolve()
    motion_path = motion_path.resolve()
    beat_root = beat_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin,
        str(SCRIPT_DIR / "run_render_dual_for_dataset.py"),
        "--dataset-id",
        dataset_id,
        "--speaker-id",
        str(speaker_id),
        "--beat-root",
        str(beat_root),
        "--motion-path",
        str(motion_path),
        "--output-dir",
        str(output_dir),
        "--tongue-shift-seconds",
        str(tongue_shift_seconds),
    ]
    run_cmd(cmd, cwd=SCRIPT_DIR)
    dynamic_with_audio = output_dir / f"{dataset_id}_with_tongue_with_audio.mp4"
    if not dynamic_with_audio.is_file():
        raise RuntimeError(f"Rendered dynamic video not found: {dynamic_with_audio}")
    return dynamic_with_audio


def evaluate_video(
    *,
    python_bin: str,
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


def clear_improvement(
    baseline: EvalMetrics,
    candidate: EvalMetrics,
    abs_threshold: float,
    rel_threshold: float,
) -> bool:
    d_comp = baseline.composite - candidate.composite
    d_wer = baseline.wer_norm - candidate.wer_norm
    r_comp = relative_improvement(baseline.composite, candidate.composite)
    r_wer = relative_improvement(baseline.wer_norm, candidate.wer_norm)
    abs_rule = (d_comp >= abs_threshold and d_wer >= 0.0) or (d_wer >= abs_threshold and d_comp >= 0.0)
    rel_rule = r_comp >= rel_threshold and r_wer >= rel_threshold
    return bool(abs_rule or rel_rule)


def build_candidates() -> list[CandidateConfig]:
    return [
        CandidateConfig(
            name="A",
            tau_alveolar_mm=1.5,
            tau_interdental_mm=1.5,
            lambda_contact=8.0,
            tip_delta_bounds_mm=10.0,
            contact_window_fraction=0.60,
        ),
        CandidateConfig(
            name="B",
            tau_alveolar_mm=1.0,
            tau_interdental_mm=1.2,
            lambda_contact=12.0,
            tip_delta_bounds_mm=12.0,
            contact_window_fraction=0.60,
        ),
        CandidateConfig(
            name="C",
            tau_alveolar_mm=0.8,
            tau_interdental_mm=1.0,
            lambda_contact=16.0,
            tip_delta_bounds_mm=14.0,
            contact_window_fraction=0.70,
        ),
    ]


def run_optimizer_candidate(
    *,
    python_bin: str,
    dataset_id: str,
    speaker_id: str,
    beat_root: Path,
    motion_path: Path,
    textgrid_path: Path,
    output_motion_path: Path,
    candidate: CandidateConfig,
    maxiter: int,
    scalar: float,
    lambda_data: float,
    lambda_smooth: float,
    lambda_prior: float,
) -> Path:
    beat_root = beat_root.resolve()
    motion_path = motion_path.resolve()
    textgrid_path = textgrid_path.resolve()
    output_motion_path = output_motion_path.resolve()
    cmd = [
        python_bin,
        str(SCRIPT_DIR / "run_phoneme_lbfgsb_for_dataset.py"),
        "--dataset-id",
        dataset_id,
        "--speaker-id",
        str(speaker_id),
        "--beat-root",
        str(beat_root),
        "--motion-path",
        str(motion_path),
        "--textgrid-path",
        str(textgrid_path),
        "--output-path",
        str(output_motion_path),
        "--maxiter",
        str(maxiter),
        "--scalar",
        str(scalar),
        "--lambda-data",
        str(lambda_data),
        "--lambda-smooth",
        str(lambda_smooth),
        "--lambda-prior",
        str(lambda_prior),
        "--lambda-contact-alveolar",
        str(candidate.lambda_contact),
        "--lambda-contact-interdental",
        str(candidate.lambda_contact),
        "--tau-alveolar-mm",
        str(candidate.tau_alveolar_mm),
        "--tau-interdental-mm",
        str(candidate.tau_interdental_mm),
        "--tip-delta-bounds-mm",
        str(candidate.tip_delta_bounds_mm),
        "--contact-window-fraction",
        str(candidate.contact_window_fraction),
    ]
    run_cmd(cmd, cwd=SCRIPT_DIR)
    if not output_motion_path.is_file():
        raise RuntimeError(f"Optimizer output missing: {output_motion_path}")
    return output_motion_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline + 3 L-BFGS-B candidates + reproducibility rerun for one dataset."
    )
    parser.add_argument("--dataset-id", default="1_wayne_0_75_75")
    parser.add_argument("--speaker-id", default="1")
    parser.add_argument(
        "--beat-root",
        default=str(PROJECT_ROOT / "data" / "beat_cache" / "beat_english_v0.2.1" / "beat_english_v0.2.1"),
    )
    parser.add_argument("--motion-path", default=None)
    parser.add_argument("--textgrid-path", default=None)
    parser.add_argument("--ground-truth-path", default=None)
    parser.add_argument("--run-root", default=str(SCRIPT_DIR / "outputs" / "lbfgsb_runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--python-bin", default=sys.executable)

    parser.add_argument("--tongue-shift-seconds", type=float, default=0.12)
    parser.add_argument("--infer-mode", choices=["full", "segmented"], default="segmented")
    parser.add_argument("--detector", default="mediapipe")
    parser.add_argument("--vowel-mode", choices=["grouped", "exact"], default="grouped")
    parser.add_argument("--config-filename", default="configs/LRS3_V_WER19.1.ini")
    parser.add_argument("--seg-target-seconds", type=float, default=8.0)
    parser.add_argument("--seg-min-seconds", type=float, default=4.0)
    parser.add_argument("--seg-max-seconds", type=float, default=12.0)
    parser.add_argument("--seg-min-silence", type=float, default=0.0)

    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--scalar", type=float, default=0.20)
    parser.add_argument("--lambda-data", type=float, default=0.1)
    parser.add_argument("--lambda-smooth", type=float, default=0.05)
    parser.add_argument("--lambda-prior", type=float, default=0.01)

    parser.add_argument("--abs-improvement-threshold", type=float, default=0.02)
    parser.add_argument("--rel-improvement-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    beat_root = Path(args.beat_root)
    speaker_dir = beat_root / str(args.speaker_id)
    motion_path = Path(args.motion_path) if args.motion_path else (SCRIPT_DIR / "outputs" / f"{args.dataset_id}.npy")
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

    print("=" * 88)
    print("L-BFGS-B SINGLE SAMPLE EXPERIMENT")
    print("=" * 88)
    print(f"dataset_id: {args.dataset_id}")
    print(f"motion:     {motion_path}")
    print(f"textgrid:   {textgrid_path}")
    print(f"run_dir:    {run_dir}")
    print("=" * 88)

    ground_truth = ensure_ground_truth(textgrid_path, ground_truth_path)

    # M1 baseline
    baseline_render_dir = run_dir / "baseline_render"
    baseline_video = render_dynamic_video(
        python_bin=args.python_bin,
        dataset_id=args.dataset_id,
        speaker_id=str(args.speaker_id),
        beat_root=beat_root,
        motion_path=motion_path,
        output_dir=baseline_render_dir,
        tongue_shift_seconds=float(args.tongue_shift_seconds),
    )
    baseline_eval = evaluate_video(
        python_bin=args.python_bin,
        video_path=baseline_video,
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
    print(
        f"[BASELINE] VER={baseline_eval.ver:.4f}  WER(norm)={baseline_eval.wer_norm:.4f}  Composite={baseline_eval.composite:.4f}"
    )

    candidate_rows: list[dict] = []
    for candidate in build_candidates():
        print("-" * 88)
        print(f"[CANDIDATE {candidate.name}] tau=({candidate.tau_alveolar_mm},{candidate.tau_interdental_mm}) "
              f"lambda_contact={candidate.lambda_contact} tip_bound={candidate.tip_delta_bounds_mm} "
              f"window={candidate.contact_window_fraction}")
        cand_dir = run_dir / f"candidate_{candidate.name}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        cand_motion = cand_dir / f"{args.dataset_id}_lbfgsb.npy"
        run_optimizer_candidate(
            python_bin=args.python_bin,
            dataset_id=args.dataset_id,
            speaker_id=str(args.speaker_id),
            beat_root=beat_root,
            motion_path=motion_path,
            textgrid_path=textgrid_path,
            output_motion_path=cand_motion,
            candidate=candidate,
            maxiter=int(args.maxiter),
            scalar=float(args.scalar),
            lambda_data=float(args.lambda_data),
            lambda_smooth=float(args.lambda_smooth),
            lambda_prior=float(args.lambda_prior),
        )
        cand_video = render_dynamic_video(
            python_bin=args.python_bin,
            dataset_id=args.dataset_id,
            speaker_id=str(args.speaker_id),
            beat_root=beat_root,
            motion_path=cand_motion,
            output_dir=cand_dir / "render",
            tongue_shift_seconds=float(args.tongue_shift_seconds),
        )
        cand_eval = evaluate_video(
            python_bin=args.python_bin,
            video_path=cand_video,
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
        d_comp = baseline_eval.composite - cand_eval.composite
        d_wer = baseline_eval.wer_norm - cand_eval.wer_norm
        row = {
            "name": candidate.name,
            "config": candidate.__dict__,
            "motion_path": str(cand_motion),
            "video_path": str(cand_video),
            "metrics": cand_eval.__dict__,
            "delta_composite": float(d_comp),
            "delta_wer_norm": float(d_wer),
            "rel_composite": float(relative_improvement(baseline_eval.composite, cand_eval.composite)),
            "rel_wer_norm": float(relative_improvement(baseline_eval.wer_norm, cand_eval.wer_norm)),
        }
        row["clear_improvement"] = clear_improvement(
            baseline=baseline_eval,
            candidate=cand_eval,
            abs_threshold=float(args.abs_improvement_threshold),
            rel_threshold=float(args.rel_improvement_threshold),
        )
        candidate_rows.append(row)
        print(
            f"[RESULT {candidate.name}] VER={cand_eval.ver:.4f} WER(norm)={cand_eval.wer_norm:.4f} "
            f"Composite={cand_eval.composite:.4f}  "
            f"Δcomp={row['delta_composite']:+.4f}  Δwer={row['delta_wer_norm']:+.4f}  "
            f"clear={row['clear_improvement']}"
        )

    improving = [r for r in candidate_rows if r["clear_improvement"]]
    pool = improving if improving else candidate_rows
    best = min(pool, key=lambda r: (r["metrics"]["composite"], r["metrics"]["wer_norm"]))

    print("=" * 88)
    print(f"Best candidate: {best['name']} (clear improvement pool size: {len(improving)})")
    print("=" * 88)

    # M5 reproducibility rerun for best config.
    best_cfg = CandidateConfig(**best["config"])
    rerun_dir = run_dir / f"best_{best_cfg.name}_rerun"
    rerun_dir.mkdir(parents=True, exist_ok=True)
    rerun_motion = rerun_dir / f"{args.dataset_id}_lbfgsb.npy"
    run_optimizer_candidate(
        python_bin=args.python_bin,
        dataset_id=args.dataset_id,
        speaker_id=str(args.speaker_id),
        beat_root=beat_root,
        motion_path=motion_path,
        textgrid_path=textgrid_path,
        output_motion_path=rerun_motion,
        candidate=best_cfg,
        maxiter=int(args.maxiter),
        scalar=float(args.scalar),
        lambda_data=float(args.lambda_data),
        lambda_smooth=float(args.lambda_smooth),
        lambda_prior=float(args.lambda_prior),
    )
    rerun_video = render_dynamic_video(
        python_bin=args.python_bin,
        dataset_id=args.dataset_id,
        speaker_id=str(args.speaker_id),
        beat_root=beat_root,
        motion_path=rerun_motion,
        output_dir=rerun_dir / "render",
        tongue_shift_seconds=float(args.tongue_shift_seconds),
    )
    rerun_eval = evaluate_video(
        python_bin=args.python_bin,
        video_path=rerun_video,
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

    summary = {
        "dataset_id": args.dataset_id,
        "speaker_id": str(args.speaker_id),
        "run_dir": str(run_dir.resolve()),
        "baseline": baseline_eval.__dict__,
        "candidates": candidate_rows,
        "best_candidate": best,
        "rerun_best": rerun_eval.__dict__,
        "thresholds": {
            "abs_improvement_threshold": float(args.abs_improvement_threshold),
            "rel_improvement_threshold": float(args.rel_improvement_threshold),
        },
        "paths": {
            "ground_truth": str(ground_truth_path.resolve()),
            "textgrid": str(textgrid_path.resolve()),
            "input_motion": str(motion_path.resolve()),
        },
    }
    summary_json = run_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines: list[str] = []
    md_lines.append(f"# L-BFGS-B Single Experiment: {args.dataset_id}")
    md_lines.append("")
    md_lines.append(f"- Run dir: `{run_dir}`")
    md_lines.append(f"- Ground truth: `{ground_truth_path}`")
    md_lines.append(f"- Infer mode: `{args.infer_mode}`")
    md_lines.append("")
    md_lines.append("## Baseline")
    md_lines.append(
        f"- VER={baseline_eval.ver:.4f}, WER(norm)={baseline_eval.wer_norm:.4f}, "
        f"WER(raw)={baseline_eval.wer_raw:.4f}, Composite={baseline_eval.composite:.4f}"
    )
    md_lines.append("")
    md_lines.append("## Candidates")
    md_lines.append("| Name | VER | WER(norm) | Composite | ΔComposite | ΔWER(norm) | Clear |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in candidate_rows:
        m = row["metrics"]
        md_lines.append(
            f"| {row['name']} | {m['ver']:.4f} | {m['wer_norm']:.4f} | {m['composite']:.4f} | "
            f"{row['delta_composite']:+.4f} | {row['delta_wer_norm']:+.4f} | {int(row['clear_improvement'])} |"
        )
    md_lines.append("")
    md_lines.append("## Best + Rerun")
    md_lines.append(f"- Best candidate: `{best['name']}`")
    md_lines.append(
        f"- Best metrics: VER={best['metrics']['ver']:.4f}, "
        f"WER(norm)={best['metrics']['wer_norm']:.4f}, Composite={best['metrics']['composite']:.4f}"
    )
    md_lines.append(
        f"- Rerun metrics: VER={rerun_eval.ver:.4f}, "
        f"WER(norm)={rerun_eval.wer_norm:.4f}, Composite={rerun_eval.composite:.4f}"
    )
    md_lines.append("")
    summary_md = run_dir / "summary.md"
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("=" * 88)
    print("Experiment complete.")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary MD:   {summary_md}")
    print("=" * 88)


if __name__ == "__main__":
    main()
