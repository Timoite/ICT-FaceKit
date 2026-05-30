#!/usr/bin/env python3
"""Dataset-wide VOCASets std_scalar sweep with thickness=1.4, rotation=0, no shifts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

EXPERIMENT_NAME = "std_0p100_0p300_step0p025_th1p400_rot0"
DEFAULT_OUTPUT_ROOT = (
    Path("/research/milsrg1/user_workspace/ht467/smirk_task/outputs")
    / "vocasets_std_sweep"
    / EXPERIMENT_NAME
)
DEFAULT_LINK_ROOT = PROJECT_ROOT / "tests" / "vocaset_outputs" / "grid_search" / EXPERIMENT_NAME
DEFAULT_ACTIVE_BEST_ROOT = (
    Path("/research/milsrg1/user_workspace/ht467/smirk_task/outputs")
    / "vocasets_active_best"
    / "std0p27_z0p00_rot5"
)


@dataclass(frozen=True)
class StdSweepParams:
    std_scalar: float
    shift_z: float = 0.0
    rotation_deg: float = 0.0
    thickness: float = 1.4
    shift_y: float = 0.0


@dataclass(frozen=True)
class StdSweepJob:
    speaker: str
    sentence: str
    clip_id: str
    json_path: Path
    wav_path: Path
    transcript_path: Path
    ground_truth: str
    params: StdSweepParams


@dataclass(frozen=True)
class StdSweepOutputPaths:
    clip_id: str
    out_dir: Path
    motion_path: Path
    raw_video: Path
    audio_video: Path
    link: Path
    metric_json: Path


@dataclass(frozen=True)
class StdSweepMetric:
    clip_id: str
    speaker: str
    sentence: str
    std_scalar: float
    ver: float
    wer_norm: float
    composite_index: float
    hypothesis: str
    ground_truth: str
    video: str


def _format_float(value: float) -> str:
    sign = "m" if value < 0 else ""
    return sign + f"{abs(value):.3f}".replace(".", "p")


def std_slug(value: float) -> str:
    return f"std{_format_float(value)}"


def params_slug(params: StdSweepParams) -> str:
    return (
        f"{std_slug(params.std_scalar)}"
        f"_th{_format_float(params.thickness)}"
        f"_rot{_format_float(params.rotation_deg)}"
    )


def build_std_values(start: float = 0.1, stop: float = 0.3, step: float = 0.025) -> list[float]:
    values: list[float] = []
    current = start
    epsilon = step / 10.0
    while current <= stop + epsilon:
        values.append(round(current, 3))
        current += step
    return values


def sentence_index(sentence: str) -> int | None:
    match = re.fullmatch(r"sentence(\d+)", sentence)
    if not match:
        return None
    return int(match.group(1))


def sentence_ground_truth(transcript_path: Path, sentence: str) -> str | None:
    index = sentence_index(sentence)
    if index is None or not transcript_path.is_file():
        return None
    lines = [
        line.strip()
        for line in transcript_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]
    if not 1 <= index <= len(lines):
        return None
    return lines[index - 1]


def collect_base_clips(json_root: Path, wav_root: Path, transcript_root: Path) -> list[dict]:
    clips: list[dict] = []
    for json_path in sorted(json_root.glob("*/*.json")):
        speaker = json_path.parent.name
        sentence = json_path.stem
        clip_id = f"{speaker}_{sentence}"
        wav_path = wav_root / f"{clip_id}.wav"
        transcript_path = transcript_root / f"{speaker}.txt"
        ground_truth = sentence_ground_truth(transcript_path, sentence)
        if not wav_path.is_file() or ground_truth is None:
            continue
        clips.append(
            {
                "speaker": speaker,
                "sentence": sentence,
                "clip_id": clip_id,
                "json_path": json_path,
                "wav_path": wav_path,
                "transcript_path": transcript_path,
                "ground_truth": ground_truth,
            }
        )
    return clips


def collect_std_sweep_jobs(
    json_root: Path,
    wav_root: Path,
    transcript_root: Path,
    std_values: Sequence[float],
    *,
    thickness: float = 1.4,
    rotation_deg: float = 0.0,
    shift_z: float = 0.0,
    shift_y: float = 0.0,
) -> list[StdSweepJob]:
    jobs: list[StdSweepJob] = []
    for clip in collect_base_clips(json_root, wav_root, transcript_root):
        for std_value in std_values:
            jobs.append(
                StdSweepJob(
                    speaker=clip["speaker"],
                    sentence=clip["sentence"],
                    clip_id=clip["clip_id"],
                    json_path=clip["json_path"],
                    wav_path=clip["wav_path"],
                    transcript_path=clip["transcript_path"],
                    ground_truth=clip["ground_truth"],
                    params=StdSweepParams(
                        std_scalar=std_value,
                        thickness=thickness,
                        rotation_deg=rotation_deg,
                        shift_z=shift_z,
                        shift_y=shift_y,
                    ),
                )
            )
    return jobs


def active_output_paths(
    output_root: Path,
    link_root: Path,
    speaker: str,
    sentence: str,
    params: StdSweepParams,
) -> StdSweepOutputPaths:
    clip_id = f"{speaker}_{sentence}"
    slug = params_slug(params)
    std_dir = std_slug(params.std_scalar)
    out_dir = output_root / std_dir / speaker / sentence
    filename = f"{clip_id}_{slug}_active_tongue"
    return StdSweepOutputPaths(
        clip_id=clip_id,
        out_dir=out_dir,
        motion_path=output_root / "_motion" / speaker / sentence / "tongue_motion.npy",
        raw_video=out_dir / f"{filename}.mp4",
        audio_video=out_dir / f"{filename}_with_audio.mp4",
        link=link_root / "videos" / std_dir / f"vocaset_{filename}_with_audio.mp4",
        metric_json=output_root / "metrics" / std_dir / f"{clip_id}_metrics.json",
    )


def split_jobs(jobs: Sequence[StdSweepJob], workers: int) -> list[list[StdSweepJob]]:
    return [list(jobs[index::workers]) for index in range(workers)]


def claim_lock(lock_root: Path, key: str) -> Path | None:
    lock_path = lock_root / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()}.lock"
    try:
        lock_path.mkdir(parents=True)
    except FileExistsError:
        return None
    (lock_path / "key.txt").write_text(key + "\n", encoding="utf-8")
    return lock_path


def release_lock(lock_path: Path | None) -> None:
    if lock_path is None:
        return
    try:
        (lock_path / "key.txt").unlink(missing_ok=True)
        lock_path.rmdir()
    except OSError:
        pass


def replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    tmp_link = link.with_name(f".{link.name}.{os.getpid()}.tmp")
    tmp_link.unlink(missing_ok=True)
    tmp_link.symlink_to(target)
    os.replace(tmp_link, link)


def existing_active_best_motion(active_best_root: Path, speaker: str, sentence: str) -> Path | None:
    path = active_best_root / speaker / sentence / "tongue_motion" / "tongue_motion.npy"
    return path if path.is_file() else None


def ensure_motion(
    job: StdSweepJob,
    motion_path: Path,
    active_best_root: Path,
    loaded_model_state: dict,
    *,
    force: bool,
) -> Path:
    if not force and motion_path.is_file():
        return motion_path
    cached = existing_active_best_motion(active_best_root, job.speaker, job.sentence)
    if cached is not None and not force:
        return cached

    from tongue_scripts.inversion.invert import infer_ema, load_model
    import numpy as np

    motion_path.parent.mkdir(parents=True, exist_ok=True)
    if "model" not in loaded_model_state:
        loaded_model_state["model"] = load_model(Path(loaded_model_state["checkpoint"]))
    ema = infer_ema(
        job.wav_path,
        model=loaded_model_state["model"],
        max_seconds=loaded_model_state["max_seconds"],
    )
    np.save(str(motion_path), ema)
    print(f"[MOTION] {motion_path} shape={ema.shape}", flush=True)
    return motion_path


def render_one_job(
    job: StdSweepJob,
    paths: StdSweepOutputPaths,
    *,
    force: bool,
    face_model,
    loaded_model_state: dict,
    active_best_root: Path,
    fps: int,
    source_fps: float,
) -> None:
    from tongue_scripts.pipelines.grid_search_vocaset_active_tongue import (
        resample_ema_to_face_frames,
    )
    from tongue_scripts.rendering.render_dual_tongue_comparison import (
        ANCHOR_INDICES,
        BONE_INDICES,
        STD_PATH,
        TONGUE_CONFIG,
        TONGUE_SLICE,
        apply_jawopen_offset_correction,
        merge_audio,
        render_video_with_dynamic_tongue,
    )
    from tongue_scripts.tongue_animation.generate_tongue_animation import (
        FaceKitTongueRig,
        load_blendshape_json_sequence,
        load_ema_motion,
    )

    if not force and paths.audio_video.is_file():
        replace_symlink(paths.link, paths.audio_video)
        print(f"[SKIP] existing render {paths.audio_video}", flush=True)
        return

    paths.out_dir.mkdir(parents=True, exist_ok=True)
    motion_path = ensure_motion(
        job,
        paths.motion_path,
        active_best_root,
        loaded_model_state,
        force=force,
    )
    config = dict(TONGUE_CONFIG)
    config.update(
        {
            "std_scalar": job.params.std_scalar,
            "shift_z": job.params.shift_z,
            "rotation_deg": job.params.rotation_deg,
            "thickness": job.params.thickness,
            "shift_y": job.params.shift_y,
        }
    )
    tongue_rig = FaceKitTongueRig(
        face_model.neutral_verts,
        face_model.faces,
        TONGUE_SLICE,
        ANCHOR_INDICES,
        BONE_INDICES,
        config,
    )
    face_seq = load_blendshape_json_sequence(
        job.json_path,
        face_model,
        source_fps=source_fps,
        target_fps=fps,
    )
    face_seq = apply_jawopen_offset_correction(face_seq, face_model)
    ema_seq = load_ema_motion(motion_path, STD_PATH, tongue_rig.anchors, config["std_scalar"])
    ema_seq = resample_ema_to_face_frames(ema_seq, len(face_seq))
    print(f"[RENDER] {job.clip_id} {std_slug(job.params.std_scalar)} -> {paths.audio_video}", flush=True)
    render_video_with_dynamic_tongue(
        face_model,
        face_seq,
        tongue_rig,
        ema_seq,
        str(paths.raw_video),
        fps=fps,
    )
    merge_audio(str(paths.raw_video), str(job.wav_path), str(paths.audio_video))
    replace_symlink(paths.link, paths.audio_video)


def metric_to_sweep_metric(payload: dict) -> StdSweepMetric:
    return StdSweepMetric(
        clip_id=payload["clip_id"],
        speaker=payload["speaker"],
        sentence=payload["sentence"],
        std_scalar=float(payload["std_scalar"]),
        ver=float(payload["ver"]),
        wer_norm=float(payload["wer_norm"]),
        composite_index=float(payload["composite_index"]),
        hypothesis=payload.get("hypothesis", ""),
        ground_truth=payload.get("ground_truth", ""),
        video=payload.get("video", ""),
    )


def save_metric(path: Path, row: dict, job: StdSweepJob) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(row)
    payload.update(
        {
            "clip_id": job.clip_id,
            "speaker": job.speaker,
            "sentence": job.sentence,
            "std_scalar": job.params.std_scalar,
            "shift_z": job.params.shift_z,
            "rotation_deg": job.params.rotation_deg,
            "thickness": job.params.thickness,
            "shift_y": job.params.shift_y,
            "ground_truth": job.ground_truth,
        }
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def evaluate_one_job(args: argparse.Namespace, job: StdSweepJob, paths: StdSweepOutputPaths) -> None:
    if paths.metric_json.is_file() and not args.force_eval:
        print(f"[SKIP] existing metric {paths.metric_json}", flush=True)
        return
    if not paths.audio_video.is_file():
        print(f"[MISSING] render not found: {paths.audio_video}", flush=True)
        return

    from tongue_scripts.evaluation.evaluate_vsr_ver import evaluate_videos

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
            videos=[str(paths.audio_video)],
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
            experiment_name=f"{job.clip_id}_{std_slug(job.params.std_scalar)}_{EXPERIMENT_NAME}",
            report_path=str(Path(args.output_root) / "adfa_std_sweep_report.md"),
            report_mode="append",
            dataset_id=job.clip_id,
            speaker_id=job.speaker,
            hypothesis="VOCASets std_scalar sweep for active tongue VSR",
        )
        _, rows = evaluate_videos(eval_args)
        save_metric(paths.metric_json, rows[0], job)
    finally:
        gt_path.unlink(missing_ok=True)


def load_metrics(metric_root: Path) -> list[StdSweepMetric]:
    rows: list[StdSweepMetric] = []
    for path in sorted(metric_root.glob("std*/*_metrics.json")):
        rows.append(metric_to_sweep_metric(json.loads(path.read_text(encoding="utf-8"))))
    return rows


def comparison_rows_by_std(rows: Sequence[StdSweepMetric]) -> list[dict]:
    by_std: dict[float, list[StdSweepMetric]] = {}
    for row in rows:
        by_std.setdefault(row.std_scalar, []).append(row)

    summary: list[dict] = []
    for std_value, std_rows in sorted(by_std.items()):
        vers = [row.ver for row in std_rows]
        wers = [row.wer_norm for row in std_rows]
        composites = [row.composite_index for row in std_rows]
        summary.append(
            {
                "std_scalar": std_value,
                "clip_count": len(std_rows),
                "mean_ver": mean(vers),
                "median_ver": median(vers),
                "mean_wer_norm": mean(wers),
                "median_wer_norm": median(wers),
                "mean_composite": mean(composites),
                "median_composite": median(composites),
            }
        )
    return summary


def best_vote_summary(rows: Sequence[StdSweepMetric]) -> list[dict]:
    summary_by_std = {row["std_scalar"]: dict(row, best_vote_count=0) for row in comparison_rows_by_std(rows)}
    by_clip: dict[str, list[StdSweepMetric]] = {}
    for row in rows:
        by_clip.setdefault(row.clip_id, []).append(row)
    for clip_rows in by_clip.values():
        best = min(clip_rows, key=lambda row: (row.ver, row.wer_norm, row.composite_index, row.std_scalar))
        summary_by_std[best.std_scalar]["best_vote_count"] += 1
    return sorted(
        summary_by_std.values(),
        key=lambda row: (-row["best_vote_count"], row["mean_ver"], row["std_scalar"]),
    )


def write_long_csv(path: Path, rows: Sequence[StdSweepMetric]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "speaker",
        "sentence",
        "std_scalar",
        "ver",
        "wer_norm",
        "composite_index",
        "hypothesis",
        "ground_truth",
        "video",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.clip_id, item.std_scalar)):
            writer.writerow({name: getattr(row, name) for name in fieldnames})


def write_summary_csv(path: Path, summary: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "std_scalar",
        "best_vote_count",
        "clip_count",
        "mean_ver",
        "median_ver",
        "mean_wer_norm",
        "median_wer_norm",
        "mean_composite",
        "median_composite",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(summary, start=1):
            out = dict(row)
            out["rank"] = rank
            writer.writerow(out)


def write_plot(path: Path, summary: Sequence[dict]) -> None:
    if not summary:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(summary, key=lambda row: row["std_scalar"])
    xs = [row["std_scalar"] for row in ordered]
    mean_ver = [row["mean_ver"] for row in ordered]
    median_ver = [row["median_ver"] for row in ordered]
    votes = [row["best_vote_count"] for row in ordered]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(xs, mean_ver, marker="o", linewidth=2.0, label="mean VER")
    ax1.plot(xs, median_ver, marker="s", linewidth=1.6, label="median VER")
    ax1.set_xlabel("std_scalar")
    ax1.set_ylabel("VER (lower is better)")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.bar(xs, votes, width=0.010, color="#7a7a7a", alpha=0.25, label="best votes")
    ax2.set_ylabel("Best-clip vote count")

    best = sorted(summary, key=lambda row: (-row["best_vote_count"], row["mean_ver"], row["std_scalar"]))[0]
    ax1.axvline(best["std_scalar"], color="black", linestyle="--", linewidth=1.0)
    ax1.set_title(
        f"VOCASets std_scalar sweep, th=1.4 rot=0: best vote std={best['std_scalar']:.3f}"
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_best_by_clip_csv(path: Path, rows: Sequence[StdSweepMetric]) -> None:
    by_clip: dict[str, list[StdSweepMetric]] = {}
    for row in rows:
        by_clip.setdefault(row.clip_id, []).append(row)
    fieldnames = [
        "clip_id",
        "speaker",
        "sentence",
        "best_std_scalar",
        "best_ver",
        "best_wer_norm",
        "best_composite",
        "hypothesis",
        "ground_truth",
        "video",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for clip_id, clip_rows in sorted(by_clip.items()):
            best = min(clip_rows, key=lambda row: (row.ver, row.wer_norm, row.composite_index, row.std_scalar))
            writer.writerow(
                {
                    "clip_id": clip_id,
                    "speaker": best.speaker,
                    "sentence": best.sentence,
                    "best_std_scalar": best.std_scalar,
                    "best_ver": best.ver,
                    "best_wer_norm": best.wer_norm,
                    "best_composite": best.composite_index,
                    "hypothesis": best.hypothesis,
                    "ground_truth": best.ground_truth,
                    "video": best.video,
                }
            )


def link_artifacts(output_root: Path, link_root: Path) -> None:
    for name in (
        "vocaset_std_sweep_metrics_long.csv",
        "vocaset_std_sweep_summary_by_std.csv",
        "vocaset_std_sweep_best_by_clip.csv",
        "vocaset_std_sweep_ver_curve.png",
        "adfa_std_sweep_report.md",
    ):
        target = output_root / name
        if target.exists():
            replace_symlink(link_root / "reports" / name, target)


def summarize(output_root: Path, link_root: Path) -> list[dict]:
    rows = load_metrics(output_root / "metrics")
    write_long_csv(output_root / "vocaset_std_sweep_metrics_long.csv", rows)
    write_best_by_clip_csv(output_root / "vocaset_std_sweep_best_by_clip.csv", rows)
    summary = best_vote_summary(rows)
    write_summary_csv(output_root / "vocaset_std_sweep_summary_by_std.csv", summary)
    write_plot(output_root / "vocaset_std_sweep_ver_curve.png", summary)
    link_artifacts(output_root, link_root)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["render", "eval", "summarize"], required=True)
    parser.add_argument(
        "--json-root",
        default=str(PROJECT_ROOT / "tests" / "vocasets" / "blendshape_json"),
    )
    parser.add_argument("--wav-root", default=str(PROJECT_ROOT / "tests" / "vocasets" / "wav"))
    parser.add_argument(
        "--transcript-root",
        default=str(PROJECT_ROOT / "tests" / "vocasets" / "sentencestext"),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--link-root", default=str(DEFAULT_LINK_ROOT))
    parser.add_argument("--active-best-root", default=str(DEFAULT_ACTIVE_BEST_ROOT))
    parser.add_argument(
        "--checkpoint",
        default=str(
            PROJECT_ROOT
            / "tongue_scripts"
            / "inversion_checkpoints"
            / "lora_multispeaker_consistency_alpha_0.25_threshold_0_vctk_vvn_4tonguepoints"
        ),
    )
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--source-fps", type=float, default=60.0)
    parser.add_argument("--std-start", type=float, default=0.1)
    parser.add_argument("--std-stop", type=float, default=0.3)
    parser.add_argument("--std-step", type=float, default=0.025)
    parser.add_argument("--std-values", nargs="+", type=float, default=None)
    parser.add_argument("--thickness", type=float, default=1.4)
    parser.add_argument("--rotation-deg", type=float, default=0.0)
    parser.add_argument("--shift-z", type=float, default=0.0)
    parser.add_argument("--shift-y", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--worker-index", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--python-bin",
        default="/research/milsrg1/user_workspace/ht467/tools/uv/adfa-vsr/bin/python",
    )
    parser.add_argument("--infer-mode", choices=["full", "segmented"], default="full")
    parser.add_argument("--config-filename", default="configs/LRS3_V_WER19.1.ini")
    parser.add_argument("--vowel-mode", choices=["grouped", "exact"], default="grouped")
    parser.add_argument("--detector", default="mediapipe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    link_root = Path(args.link_root)
    std_values = (
        build_std_values(args.std_start, args.std_stop, args.std_step)
        if args.std_values is None
        else [round(value, 3) for value in args.std_values]
    )

    if args.phase == "summarize":
        summary = summarize(output_root, link_root)
        print(f"[SUMMARY] metric_rows={len(load_metrics(output_root / 'metrics'))}", flush=True)
        if summary:
            best = summary[0]
            print(
                "[SUMMARY] "
                f"best_vote_std={best['std_scalar']:.3f} "
                f"votes={best['best_vote_count']} "
                f"mean_ver={best['mean_ver']:.6f}",
                flush=True,
            )
        print(f"[SUMMARY] reports={link_root / 'reports'}", flush=True)
        return

    jobs = collect_std_sweep_jobs(
        Path(args.json_root),
        Path(args.wav_root),
        Path(args.transcript_root),
        std_values,
        thickness=args.thickness,
        rotation_deg=args.rotation_deg,
        shift_z=args.shift_z,
        shift_y=args.shift_y,
    )
    if args.workers is not None and args.worker_index is not None:
        jobs = split_jobs(jobs, args.workers)[args.worker_index]
    print(f"[{args.phase.upper()}] jobs={len(jobs)} std_values={std_values}", flush=True)

    if args.phase == "render":
        from tongue_scripts.tongue_animation.face_model_io_trimesh import load_face_model_trimesh

        face_model = load_face_model_trimesh(PROJECT_ROOT / "FaceXModel")
        loaded_model_state = {
            "checkpoint": args.checkpoint,
            "max_seconds": args.max_seconds,
        }
        lock_root = output_root / "_locks" / "render"
        for job in jobs:
            paths = active_output_paths(output_root, link_root, job.speaker, job.sentence, job.params)
            lock_key = f"{job.clip_id}_{std_slug(job.params.std_scalar)}"
            lock_path = claim_lock(lock_root, lock_key)
            if lock_path is None:
                print(f"[SKIP] locked render {lock_key}", flush=True)
                continue
            try:
                render_one_job(
                    job,
                    paths,
                    force=args.force,
                    face_model=face_model,
                    loaded_model_state=loaded_model_state,
                    active_best_root=Path(args.active_best_root),
                    fps=args.fps,
                    source_fps=args.source_fps,
                )
            finally:
                release_lock(lock_path)
        return

    lock_root = output_root / "_locks" / "eval"
    for job in jobs:
        paths = active_output_paths(output_root, link_root, job.speaker, job.sentence, job.params)
        lock_key = f"{job.clip_id}_{std_slug(job.params.std_scalar)}"
        lock_path = claim_lock(lock_root, lock_key)
        if lock_path is None:
            print(f"[SKIP] locked eval {lock_key}", flush=True)
            continue
        try:
            evaluate_one_job(args, job, paths)
        finally:
            release_lock(lock_path)


if __name__ == "__main__":
    main()
