#!/usr/bin/env python3
"""
Estimate jaw vs tongue lag around a specific phoneme clip using the manifest timing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
TONGUE_SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TONGUE_SCRIPTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tongue_scripts.tongue_animation.face_model_io_trimesh import load_face_model_trimesh
from tongue_scripts.tongue_animation.generate_tongue_animation import (
    process_beat_data,
    load_ema_motion,
    FaceKitTongueRig,
    TONGUE_CONFIG,
)

TONGUE_SLICE = slice(16611, 17039)
ANCHOR_INDICES = [16661, 16696, 16755, 16758]
BONE_INDICES = [16661, 16757]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate jaw-tongue lag for a phoneme clip from manifest timing."
    )
    parser.add_argument("--dataset-id", default="1_wayne_0_75_75", help="BEAT clip id")
    parser.add_argument(
        "--beat-root",
        default=str(
            PROJECT_ROOT
            / "ADFA_EVALUATION"
            / "data"
            / "beat_cache_speaker1"
            / "beat_english_v0.2.1"
            / "beat_english_v0.2.1"
            / "1"
        ),
        help="Root folder containing BEAT JSON/TextGrid files",
    )
    parser.add_argument(
        "--tongue-npy-dir",
        default=str(TONGUE_SCRIPTS_DIR / "outputs"),
        help="Directory containing WavLM tongue .npy files",
    )
    parser.add_argument(
        "--std-path",
        default=str(TONGUE_SCRIPTS_DIR / "normalising_vectors" / "JW13_4points_std.npy"),
        help="Tongue normalization std .npy",
    )
    parser.add_argument(
        "--face-model-dir",
        default=str(PROJECT_ROOT / "FaceXModel"),
        help="Face model directory",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Phoneme manifest JSON (defaults to outputs/phoneme_comparision_videos)",
    )
    parser.add_argument("--clip-idx", type=int, default=63, help="Manifest phoneme idx")
    parser.add_argument(
        "--clip-name",
        default=None,
        help="Clip filename to match (e.g., idx63_6.76-7.29.mp4)",
    )
    parser.add_argument("--analysis-fps", type=float, default=50.0, help="Target FPS")
    parser.add_argument("--tongue-fps", type=float, default=50.0, help="Tongue FPS")
    parser.add_argument("--beat-fps", type=float, default=50.0, help="BEAT FPS")
    parser.add_argument("--pad-seconds", type=float, default=0.25, help="Padding around clip")
    parser.add_argument("--max-lag", type=float, default=0.5, help="Max lag in seconds")
    parser.add_argument("--tongue-anchor-idx", type=int, default=3, help="Tongue anchor index")
    parser.add_argument("--tongue-axis", choices=["x", "y", "z"], default="y")
    parser.add_argument(
        "--no-jaw-correction",
        action="store_true",
        help="Disable jawOpen min-shift correction",
    )
    parser.add_argument(
        "--shift-output",
        default=None,
        help="Optional path to write shifted .npy using the best lag",
    )
    parser.add_argument(
        "--pad-mode",
        choices=["edge", "zero"],
        default="edge",
        help="Padding mode for shifted .npy",
    )
    parser.add_argument(
        "--plot-path",
        default=str(TONGUE_SCRIPTS_DIR / "outputs" / "phoneme_lag_plot.png"),
        help="Optional output PNG for signal/correlation plot",
    )
    parser.add_argument(
        "--invert-tongue",
        action="store_true",
        default=True,
        help="Invert tongue signal polarity for analysis (default: enabled)",
    )
    parser.add_argument(
        "--no-invert-tongue",
        dest="invert_tongue",
        action="store_false",
        help="Disable tongue signal polarity inversion",
    )
    return parser.parse_args()


def resample_series(values: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    if np.isclose(source_fps, target_fps):
        return values
    duration = len(values) / source_fps
    n_target = max(1, int(duration * target_fps))
    x_source = np.linspace(0.0, duration, len(values))
    x_target = np.linspace(0.0, duration, n_target)
    return np.interp(x_target, x_source, values)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def zscore(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0.0:
        return values - mean
    return (values - mean) / std


def find_local_maxima(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return np.array([], dtype=int)
    dv = np.diff(values)
    peaks = []
    for i in range(1, len(dv)):
        if dv[i - 1] > 0 and dv[i] <= 0:
            peaks.append(i)
    return np.array(peaks, dtype=int)


def select_top_peaks(values: np.ndarray, peak_idx: np.ndarray, max_peaks: int = 3) -> np.ndarray:
    if len(peak_idx) == 0:
        return peak_idx
    order = np.argsort(values[peak_idx])[::-1]
    top = peak_idx[order[:max_peaks]]
    return np.sort(top)


def lag_sweep(
    a: np.ndarray, b: np.ndarray, max_lag_frames: int
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    lags = np.arange(-max_lag_frames, max_lag_frames + 1)
    corrs: List[float] = []
    for lag in lags:
        if lag < 0:
            corr = safe_corr(a[-lag:], b[: len(b) + lag])
        elif lag > 0:
            corr = safe_corr(a[:-lag], b[lag:])
        else:
            corr = safe_corr(a, b)
        corrs.append(corr)
    corrs_arr = np.array(corrs, dtype=np.float32)
    best_idx = int(np.argmax(corrs_arr))
    best_lag = int(lags[best_idx])
    best_corr = float(corrs_arr[best_idx])
    return lags, corrs_arr, best_lag, best_corr


def shift_sequence(seq: np.ndarray, shift_frames: int, pad_mode: str) -> np.ndarray:
    if shift_frames == 0:
        return seq
    n = len(seq)
    if pad_mode == "zero":
        pad_start = np.zeros_like(seq[:1])
        pad_end = np.zeros_like(seq[:1])
    else:
        pad_start = seq[:1]
        pad_end = seq[-1:]

    if shift_frames > 0:
        pad = np.repeat(pad_start, shift_frames, axis=0)
        shifted = np.concatenate([pad, seq], axis=0)[:n]
    else:
        shift_frames = abs(shift_frames)
        pad = np.repeat(pad_end, shift_frames, axis=0)
        shifted = np.concatenate([seq[shift_frames:], pad], axis=0)
    return shifted


def load_manifest(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_manifest_entry(
    manifest: Dict[str, List[Dict[str, Any]]],
    idx: int | None,
    clip_name: str | None,
) -> Dict[str, Any]:
    target_name = clip_name
    for items in manifest.values():
        for item in items:
            if idx is not None and int(item.get("idx", -1)) == idx:
                return item
            if target_name:
                clip_path = Path(item.get("clip", ""))
                if clip_path.name == target_name:
                    return item
    raise ValueError("Could not find matching entry in manifest.")


def build_signals(
    dataset_id: str,
    beat_root: Path,
    npy_dir: Path,
    std_path: Path,
    face_model_dir: Path,
    beat_fps: float,
    tongue_fps: float,
    analysis_fps: float,
    tongue_anchor_idx: int,
    tongue_axis: str,
    apply_jaw_correction: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    json_path = beat_root / f"{dataset_id}.json"
    npy_path = npy_dir / f"{dataset_id}.npy"

    if not json_path.exists():
        raise FileNotFoundError(f"Missing BEAT JSON: {json_path}")
    if not npy_path.exists():
        raise FileNotFoundError(f"Missing tongue .npy: {npy_path}")

    face_model = load_face_model_trimesh(str(face_model_dir))
    face_seq = process_beat_data(str(json_path), face_model, target_fps=beat_fps)

    if "jawOpen" not in face_model.expression_names:
        raise RuntimeError("jawOpen not found in expression names")

    jaw_idx = face_model.expression_names.index("jawOpen")
    jaw_signal = face_seq[:, jaw_idx].astype(np.float32)

    if apply_jaw_correction:
        min_val = float(np.min(jaw_signal))
        jaw_signal = np.maximum(0.0, jaw_signal - min_val)

    tongue_rig = FaceKitTongueRig(
        face_model.neutral_verts,
        face_model.faces,
        TONGUE_SLICE,
        ANCHOR_INDICES,
        BONE_INDICES,
        TONGUE_CONFIG,
    )
    ema_seq = load_ema_motion(str(npy_path), str(std_path), tongue_rig.anchors, TONGUE_CONFIG["std_scalar"])

    anchor_idx = max(0, min(int(tongue_anchor_idx), ema_seq.shape[1] - 1))
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_idx = axis_map[tongue_axis]
    tongue_signal = ema_seq[:, anchor_idx, axis_idx].astype(np.float32)

    jaw_signal = resample_series(jaw_signal, beat_fps, analysis_fps)
    tongue_signal = resample_series(tongue_signal, tongue_fps, analysis_fps)

    min_len = min(len(jaw_signal), len(tongue_signal))
    return jaw_signal[:min_len], tongue_signal[:min_len]


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest_path) if args.manifest_path else (
        SCRIPT_DIR
        / "outputs"
        / "phoneme_comparision_videos"
        / f"{args.dataset_id}_manifest.json"
    )

    manifest = load_manifest(manifest_path)
    entry = find_manifest_entry(manifest, args.clip_idx, args.clip_name)

    start = float(entry["start"])
    end = float(entry["end"])
    label = entry.get("phoneme", "")
    clip_path = Path(entry.get("clip", ""))

    seg_start = max(0.0, start - args.pad_seconds)
    seg_end = max(seg_start, end + args.pad_seconds)

    jaw_signal, tongue_signal = build_signals(
        dataset_id=args.dataset_id,
        beat_root=Path(args.beat_root),
        npy_dir=Path(args.tongue_npy_dir),
        std_path=Path(args.std_path),
        face_model_dir=Path(args.face_model_dir),
        beat_fps=args.beat_fps,
        tongue_fps=args.tongue_fps,
        analysis_fps=args.analysis_fps,
        tongue_anchor_idx=args.tongue_anchor_idx,
        tongue_axis=args.tongue_axis,
        apply_jaw_correction=not args.no_jaw_correction,
    )

    start_idx = int(seg_start * args.analysis_fps)
    end_idx = int(seg_end * args.analysis_fps)
    end_idx = min(end_idx, len(jaw_signal))

    jaw_seg = jaw_signal[start_idx:end_idx]
    tongue_seg_base = tongue_signal[start_idx:end_idx]

    def _run_analysis(
        tongue_seg: np.ndarray, label_suffix: str, plot_path: Path | None
    ) -> Tuple[int, float, float]:
        max_lag_frames = max(1, int(args.max_lag * args.analysis_fps))
        lags, corrs, best_lag, best_corr = lag_sweep(jaw_seg, tongue_seg, max_lag_frames)
        zero_corr = safe_corr(jaw_seg, tongue_seg)

        best_lag_s = best_lag / args.analysis_fps
        recommended_shift_s = -best_lag_s

        print(f"{label_suffix}Correlation at zero lag: {zero_corr:.4f}")
        print(
            f"{label_suffix}Best lag: {best_lag} frames ({best_lag_s:+.3f}s), corr={best_corr:.4f}"
        )
        print(
            f"{label_suffix}Recommended tongue shift to align with jaw: "
            f"{recommended_shift_s:+.3f}s (positive delays tongue)"
        )

        times = np.arange(len(jaw_seg)) / args.analysis_fps + seg_start
        jaw_z = zscore(jaw_seg)
        tongue_z = zscore(tongue_seg)

        jaw_peaks = select_top_peaks(jaw_seg, find_local_maxima(jaw_seg))
        tongue_peaks = select_top_peaks(tongue_seg, find_local_maxima(tongue_seg))
        peak_pairs: List[Tuple[int, int]] = []

        if len(jaw_peaks) and len(tongue_peaks):
            print(f"{label_suffix}Jaw peak times: {times[jaw_peaks]}")
            print(f"{label_suffix}Tongue peak times: {times[tongue_peaks]}")
            pair_lags = []
            for jp in jaw_peaks:
                nearest = tongue_peaks[np.argmin(np.abs(tongue_peaks - jp))]
                peak_pairs.append((jp, int(nearest)))
                pair_lags.append(float(times[nearest] - times[jp]))
            print(f"{label_suffix}Peak lag (tongue - jaw): {np.array(pair_lags)}")

        if plot_path:
            fig, (ax_signal, ax_corr) = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
            ax_signal.plot(times, jaw_z, label="jawOpen (z)", color="#1f77b4")
            ax_signal.plot(times, tongue_z, label="tongue (z)", color="#d62728")
            ax_signal.axvspan(start, end, color="#f2c94c", alpha=0.25, label="clip window")
            if len(jaw_peaks):
                ax_signal.scatter(
                    times[jaw_peaks],
                    jaw_z[jaw_peaks],
                    s=50,
                    c="#1f77b4",
                    marker="v",
                    label="jaw peaks",
                )
            if len(tongue_peaks):
                ax_signal.scatter(
                    times[tongue_peaks],
                    tongue_z[tongue_peaks],
                    s=50,
                    c="#d62728",
                    marker="^",
                    label="tongue peaks",
                )
            if peak_pairs:
                for jp, tp in peak_pairs:
                    t_jaw = times[jp]
                    t_tongue = times[tp]
                    dt = t_tongue - t_jaw
                    ax_signal.plot(
                        [t_jaw, t_tongue],
                        [jaw_z[jp], tongue_z[tp]],
                        color="#666666",
                        linewidth=1.0,
                        alpha=0.8,
                    )
                    ax_signal.text(
                        (t_jaw + t_tongue) / 2.0,
                        max(jaw_z[jp], tongue_z[tp]) + 0.1,
                        f"{dt:+.3f}s",
                        fontsize=8,
                        color="#333333",
                        ha="center",
                    )
            ax_signal.set_title(f"Jaw vs tongue signals{label_suffix.strip()}")
            ax_signal.set_xlabel("Time (s)")
            ax_signal.set_ylabel("Z-score")
            ax_signal.legend(loc="upper right")

            ax_corr.plot(lags / args.analysis_fps, corrs, color="#555555")
            ax_corr.axvline(best_lag / args.analysis_fps, color="#d62728", linestyle="--")
            ax_corr.set_title("Correlation vs lag (jaw vs tongue)")
            ax_corr.set_xlabel("Lag (s) [+] tongue delayed")
            ax_corr.set_ylabel("Correlation")

            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)
            print(f"Saved plot: {plot_path}")

        return best_lag, best_corr, recommended_shift_s

    print("Clip:", clip_path.name)
    print(f"Phoneme: {label} | idx={entry.get('idx')} | {start:.3f}-{end:.3f}s")
    print(f"Segment window: {seg_start:.3f}-{seg_end:.3f}s (pad={args.pad_seconds:.3f}s)")

    base_plot = Path(args.plot_path) if args.plot_path else None
    tongue_seg = -tongue_seg_base if args.invert_tongue else tongue_seg_base
    _, _, recommended_shift_s = _run_analysis(
        tongue_seg,
        "[Inverted] " if args.invert_tongue else "",
        base_plot,
    )

    if args.shift_output:
        shift_frames = int(round(recommended_shift_s * args.tongue_fps))
        src_path = Path(args.tongue_npy_dir) / f"{args.dataset_id}.npy"
        raw = np.load(src_path)
        if len(raw) != len(tongue_signal):
            print("Warning: shift output uses full raw sequence length.")
        shifted_full = shift_sequence(raw, shift_frames, args.pad_mode)
        out_path = Path(args.shift_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, shifted_full)
        print(f"Shifted .npy saved: {out_path}")


if __name__ == "__main__":
    main()
