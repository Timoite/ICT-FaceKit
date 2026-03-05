#!/usr/bin/env python3
"""Download a BEAT subset, pick short clips from multiple speakers, render/evaluate active vs passive tongue."""

from __future__ import annotations

import argparse
import contextlib
import csv
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ADFA_EVALUATION.download_beat_data import download_speaker_data

BEAT_ROOT = PROJECT_ROOT / "data" / "beat_cache" / "beat_english_v0.2.1" / "beat_english_v0.2.1"
OUTPUT_ROOT = PROJECT_ROOT / "tongue_scripts" / "outputs" / "multi_speaker"
GT_DIR = OUTPUT_ROOT / "ground_truth"
LAG_PLOT_DIR = OUTPUT_ROOT / "lag_plots"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch multi-speaker active/passive tongue evaluation.")
    p.add_argument("--num-speakers", type=int, default=5, help="How many speakers to process (5-10 recommended).")
    p.add_argument(
        "--speaker-pool",
        nargs="+",
        default=["2", "4", "5", "6", "7", "16", "19", "22", "25", "26"],
        help="Candidate speaker ids to sample from.",
    )
    p.add_argument("--max-duration", type=float, default=56.0, help="Max clip duration (seconds) for short-list.")
    p.add_argument("--vowel-mode", choices=["grouped", "exact"], default="grouped")
    p.add_argument("--infer-mode", choices=["full", "segmented"], default="full")
    p.add_argument("--report-path", default=str(PROJECT_ROOT / "tongue_scripts" / "outputs" / "vsr_composite_report.md"))
    p.add_argument("--skip-download", action="store_true", help="Skip subset download stage.")
    p.add_argument("--lag-window-seconds", type=float, default=5.0, help="Diagnostic lag-plot window duration in seconds.")
    p.add_argument("--lag-max-seconds", type=float, default=0.5, help="Max lag search range in seconds.")
    p.add_argument("--target-fps", type=float, default=50.0, help="Target FPS used for lag analysis.")
    p.add_argument("--tongue-fps", type=float, default=50.0, help="Native tongue npy fps.")
    p.add_argument("--beat-fps", type=float, default=60.0, help="Native BEAT json fps.")
    p.add_argument("--smooth-frames", type=int, default=5, help="Moving average window for lag analysis.")
    p.add_argument("--articulatory-scalar", type=float, default=0.20, help="Articulatory displacement scalar used for denormalization.")
    return p.parse_args()


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def wav_duration_seconds(path: Path) -> float:
    with contextlib.closing(wave.open(str(path), "r")) as wf:
        return wf.getnframes() / float(wf.getframerate())


def textgrid_to_transcript(textgrid_path: Path) -> str:
    import sys

    vsr_dir = PROJECT_ROOT / "ADFA_EVALUATION" / "Visual_Speech_Recognition_for_Multiple_Languages"
    if str(vsr_dir) not in sys.path:
        sys.path.append(str(vsr_dir))

    import infer_pipeline as ip

    intervals = ip.parse_textgrid_words(str(textgrid_path))
    words = [w.text.strip().lower() for w in intervals if w.text and w.text.strip() and w.text.strip() != "sp"]
    return " ".join(words)


def choose_short_instances(speaker_pool: list[str], max_duration: float, num_speakers: int) -> list[tuple[str, str, float]]:
    chosen: list[tuple[str, str, float]] = []

    for sid in speaker_pool:
        spk_dir = BEAT_ROOT / sid
        if not spk_dir.is_dir():
            continue

        local_best: tuple[str, float] | None = None
        for wav in spk_dir.glob("*.wav"):
            stem = wav.stem
            if not (spk_dir / f"{stem}.json").exists() or not (spk_dir / f"{stem}.TextGrid").exists():
                continue
            try:
                d = wav_duration_seconds(wav)
            except Exception:
                continue
            if d > max_duration:
                continue
            if local_best is None or d < local_best[1]:
                local_best = (stem, d)

        if local_best is not None:
            chosen.append((sid, local_best[0], local_best[1]))

    chosen.sort(key=lambda x: x[2])
    return chosen[:num_speakers]


def estimate_lag_and_plot(
    dataset_id: str,
    beat_root: Path,
    npy_path: Path,
    args: argparse.Namespace,
) -> dict:
    import lip_aperture_textgrid_plot as lap

    wav_path = beat_root / f"{dataset_id}.wav"
    tg_path = beat_root / f"{dataset_id}.TextGrid"
    json_path = beat_root / f"{dataset_id}.json"
    face_model_dir = PROJECT_ROOT / "FaceXModel"
    mu_path = PROJECT_ROOT / "tongue_scripts" / "normalising_vectors" / "JW13_4points_mu.npy"
    std_path = PROJECT_ROOT / "tongue_scripts" / "normalising_vectors" / "JW13_4points_std.npy"

    raw_motion = np.load(npy_path)
    mu = np.load(mu_path).astype(np.float32).reshape(-1)
    std = np.load(std_path).astype(np.float32).reshape(-1)

    denorm = (
        raw_motion[:, :lap.NORM_VECTOR_COLS].astype(np.float32)
        * std[:lap.NORM_VECTOR_COLS]
        * float(args.articulatory_scalar)
        + mu[:lap.NORM_VECTOR_COLS]
    )
    lip_motion = denorm[:, lap.TONGUE_COORD_COLS:lap.TONGUE_COORD_COLS + lap.LIP_COORD_COLS]
    upper_point = lip_motion[:, 0:2]
    lower_point = lip_motion[:, 2:4]
    lip_aperture_art = np.linalg.norm(upper_point - lower_point, axis=1).reshape(-1, 1)
    lip_aperture_art = lap.resample_matrix(lip_aperture_art, source_fps=args.tongue_fps, target_fps=args.target_fps).squeeze()

    lip_aperture_bs = lap.load_blendshape_lip_aperture(
        json_path,
        face_model_dir,
        beat_fps=args.beat_fps,
        target_fps=args.target_fps,
    )

    lip_aperture_art = lap.moving_average(lip_aperture_art, args.smooth_frames)
    lip_aperture_bs = lap.moving_average(lip_aperture_bs, args.smooth_frames)

    t_art = np.arange(len(lip_aperture_art), dtype=np.float32) / float(args.target_fps)
    t_bs = np.arange(len(lip_aperture_bs), dtype=np.float32) / float(args.target_fps)
    max_t = min(float(t_art[-1]) if len(t_art) else 0.0, float(t_bs[-1]) if len(t_bs) else 0.0)

    intervals_all = lap.parse_textgrid_intervals(tg_path, "phones")
    non_empty = [iv for iv in intervals_all if iv.text.strip()]
    if non_empty:
        window_start = float(min(iv.start for iv in non_empty))
    else:
        window_start = 0.0
    window_end = min(max_t, window_start + float(args.lag_window_seconds))
    if window_end <= window_start:
        window_start = max(0.0, max_t - float(args.lag_window_seconds))
        window_end = max_t

    art_mask = (t_art >= window_start) & (t_art <= window_end)
    bs_mask = (t_bs >= window_start) & (t_bs <= window_end)
    art_windowed = lip_aperture_art[art_mask]
    bs_windowed = lip_aperture_bs[bs_mask]

    max_lag_frames = int(round(float(args.lag_max_seconds) * float(args.target_fps)))
    _, best_lag_frames, best_corr = lap.compute_lag_correlation(art_windowed, bs_windowed, max_lag_frames)
    best_lag_seconds = float(best_lag_frames) / float(args.target_fps)

    plot_path = LAG_PLOT_DIR / f"{dataset_id}_lag_window5s.png"
    py = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    run(
        [
            py,
            str(PROJECT_ROOT / "tongue_scripts" / "lip_aperture_textgrid_plot.py"),
            "--dataset-id", dataset_id,
            "--beat-root", str(beat_root),
            "--tongue-npy-dir", str(npy_path.parent),
            "--target-fps", str(args.target_fps),
            "--tongue-fps", str(args.tongue_fps),
            "--beat-fps", str(args.beat_fps),
            "--smooth-frames", str(args.smooth_frames),
            "--articulatory-scalar", str(args.articulatory_scalar),
            "--window-start", f"{window_start:.4f}",
            "--window-end", f"{window_end:.4f}",
            "--max-lag-seconds", str(args.lag_max_seconds),
            "--output-path", str(plot_path),
        ],
        cwd=PROJECT_ROOT / "tongue_scripts",
    )

    return {
        "dataset_id": dataset_id,
        "window_start": window_start,
        "window_end": window_end,
        "best_lag_frames": best_lag_frames,
        "best_lag_seconds": best_lag_seconds,
        "best_correlation": float(best_corr),
        "plot_path": str(plot_path),
    }


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)
    LAG_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Download subset through project download script
    if not args.skip_download:
        print("=== Downloading subset metadata (JSON/TextGrid) ===")
        for sid in args.speaker_pool:
            download_speaker_data(sid)
    else:
        print("=== Skipping download stage ===")

    # 2) Pick short instances (one per speaker)
    chosen = choose_short_instances(args.speaker_pool, args.max_duration, args.num_speakers)
    if len(chosen) < args.num_speakers:
        print(f"[WARN] only found {len(chosen)} speakers <= {args.max_duration}s")
    if not chosen:
        raise SystemExit("No candidate instances found.")

    print("=== Chosen instances ===")
    for sid, dataset_id, dur in chosen:
        print(f"speaker={sid}  dataset={dataset_id}  duration={dur:.2f}s")

    py = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    lag_rows: list[dict] = []

    # 3) Per instance: invert -> render -> evaluatea
    for sid, dataset_id, dur in chosen:
        spk_dir = BEAT_ROOT / sid
        wav_path = spk_dir / f"{dataset_id}.wav"
        tg_path = spk_dir / f"{dataset_id}.TextGrid"
        npy_path = OUTPUT_ROOT / f"{dataset_id}.npy"

        gt_text = textgrid_to_transcript(tg_path)
        gt_path = GT_DIR / f"{dataset_id}.txt"
        gt_path.write_text(gt_text, encoding="utf-8")

        if not npy_path.exists():
            run(
                [
                    py,
                    str(PROJECT_ROOT / "tongue_scripts" / "invert.py"),
                    "--wav",
                    str(wav_path),
                    "--out",
                    str(npy_path),
                ],
                cwd=PROJECT_ROOT / "tongue_scripts",
            )

        lag_info = estimate_lag_and_plot(dataset_id, spk_dir, npy_path, args)
        lag_rows.append({
            "speaker_id": sid,
            "dataset_id": dataset_id,
            "duration_sec": dur,
            **lag_info,
        })
        print(
            f"[LAG] {dataset_id}: best={lag_info['best_lag_seconds']:+.4f}s "
            f"({lag_info['best_lag_frames']} frames @ {args.target_fps:.1f}fps), "
            f"corr={lag_info['best_correlation']:.3f}"
        )

        run(
            [
                py,
                str(PROJECT_ROOT / "tongue_scripts" / "run_render_dual_for_dataset.py"),
                "--dataset-id",
                dataset_id,
                "--speaker-id",
                sid,
                "--beat-root",
                str(BEAT_ROOT),
                "--motion-path",
                str(npy_path),
                "--output-dir",
                str(OUTPUT_ROOT),
                "--tongue-shift-seconds",
                f"{lag_info['best_lag_seconds']:.4f}",
            ],
            cwd=PROJECT_ROOT / "tongue_scripts",
        )

        active_video = OUTPUT_ROOT / f"{dataset_id}_with_tongue_with_audio.mp4"
        passive_video = OUTPUT_ROOT / f"{dataset_id}_passive_tongue_with_audio.mp4"

        run(
            [
                py,
                str(PROJECT_ROOT / "tongue_scripts" / "evaluate_vsr_ver.py"),
                "--videos",
                str(active_video),
                str(passive_video),
                "--ground-truth",
                str(gt_path),
                "--infer-mode",
                args.infer_mode,
                "--textgrid-path",
                str(tg_path),
                "--vowel-mode",
                args.vowel_mode,
                "--experiment-name",
                f"multi_speaker_{sid}_{dataset_id}_{dur:.1f}s_{args.vowel_mode}_{args.infer_mode}",
                "--report-path",
                args.report_path,
            ],
            cwd=PROJECT_ROOT,
        )

    lag_csv = OUTPUT_ROOT / "lag_summary.csv"
    with lag_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "speaker_id",
                "dataset_id",
                "duration_sec",
                "window_start",
                "window_end",
                "best_lag_frames",
                "best_lag_seconds",
                "best_correlation",
                "plot_path",
            ],
        )
        writer.writeheader()
        writer.writerows(lag_rows)

    print("=== Lag summary saved:", lag_csv)
    print("=== Lag plots directory:", LAG_PLOT_DIR)

    print("=== Done. Report updated at:", args.report_path)


if __name__ == "__main__":
    main()
