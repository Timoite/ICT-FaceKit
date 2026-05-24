#!/usr/bin/env python3
"""
Analyze jaw-tongue temporal alignment using BEAT blendshapes, WavLM tongue .npy,
and TextGrid phoneme intervals. Produces a plot and a small JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

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

DEFAULT_LIP_SHAPES = [
    "mouthClose",
    "mouthFunnel",
    "mouthPucker",
    "mouthShrugUpper",
    "mouthShrugLower",
    "mouthRollUpper",
    "mouthRollLower",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthLeft",
    "mouthRight",
]

DEFAULT_PHONE_FILTER = [
    "L",
    "T",
    "D",
    "K",
    "G",
    "N",
    "S",
    "Z",
    "CH",
    "JH",
    "SH",
    "TH",
    "DH",
    "R",
]


@dataclass
class Interval:
    start: float
    end: float
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze jaw-tongue timing on a short segment.")
    parser.add_argument("--dataset-id", default="1_wayne_0_10_10", help="BEAT clip id")
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
        "--output-dir",
        default=str(SCRIPT_DIR / "jaw_tongue_sync"),
        help="Directory to write plots and reports",
    )
    parser.add_argument("--analysis-fps", type=float, default=50.0, help="Target FPS for analysis")
    parser.add_argument("--beat-fps", type=float, default=50.0, help="FPS used for BEAT resampling")
    parser.add_argument("--tongue-fps", type=float, default=50.0, help="FPS of WavLM tongue output")
    parser.add_argument("--segment-start", type=float, default=0.0, help="Segment start time (s)")
    parser.add_argument("--segment-duration", type=float, default=10.0, help="Segment duration (s)")
    parser.add_argument("--max-lag", type=float, default=0.5, help="Max lag in seconds for sweep")
    parser.add_argument("--tongue-anchor-idx", type=int, default=3, help="Tongue anchor index (0-3)")
    parser.add_argument("--tongue-axis", default="y", choices=["x", "y", "z"], help="Axis for tongue motion")
    parser.add_argument("--phone-tier", default="phones", help="TextGrid tier name")
    parser.add_argument(
        "--phone-filter",
        default=",".join(DEFAULT_PHONE_FILTER),
        help="Comma-separated phone labels to highlight (stress digits ignored)",
    )
    parser.add_argument(
        "--lip-shapes",
        default=",".join(DEFAULT_LIP_SHAPES),
        help="Comma-separated lip blendshape names to average",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    return parser.parse_args()


def parse_textgrid_intervals(textgrid_path: Path, tier_name: str) -> List[Interval]:
    intervals: List[Interval] = []
    in_tier = False
    current = {}

    with textgrid_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
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
                text_value = line.split("=", 1)[1].strip()
                if text_value.startswith('"') and text_value.endswith('"'):
                    text_value = text_value[1:-1]
                current["text"] = text_value
                if {"start", "end", "text"} <= current.keys():
                    try:
                        start = float(current["start"])
                        end = float(current["end"])
                    except ValueError:
                        start, end = 0.0, 0.0
                    intervals.append(Interval(start=start, end=end, text=current["text"]))
    return intervals


def normalize_phone(label: str) -> str:
    return re.sub(r"[0-9]", "", label.upper())


def filter_intervals(intervals: Sequence[Interval], allowed: Sequence[str]) -> List[Interval]:
    allowed_set = {normalize_phone(item) for item in allowed if item}
    filtered = []
    for interval in intervals:
        if not interval.text:
            continue
        if normalize_phone(interval.text) in allowed_set:
            filtered.append(interval)
    return filtered


def resample_series(values: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    if math.isclose(source_fps, target_fps):
        return values
    duration = len(values) / source_fps
    n_target = max(1, int(duration * target_fps))
    x_source = np.linspace(0.0, duration, len(values))
    x_target = np.linspace(0.0, duration, n_target)
    return np.interp(x_target, x_source, values)


def zscore(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0.0:
        return values - mean
    return (values - mean) / std


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def lag_sweep(a: np.ndarray, b: np.ndarray, max_lag_frames: int) -> Tuple[np.ndarray, np.ndarray, int, float]:
    lags = np.arange(-max_lag_frames, max_lag_frames + 1)
    corrs = []
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


def main() -> None:
    args = parse_args()
    beat_root = Path(args.beat_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = args.dataset_id
    json_path = beat_root / f"{dataset_id}.json"
    textgrid_path = beat_root / f"{dataset_id}.TextGrid"
    npy_path = Path(args.tongue_npy_dir) / f"{dataset_id}.npy"

    if not json_path.exists():
        raise FileNotFoundError(f"Missing BEAT JSON: {json_path}")
    if not textgrid_path.exists():
        raise FileNotFoundError(f"Missing TextGrid: {textgrid_path}")
    if not npy_path.exists():
        raise FileNotFoundError(f"Missing tongue .npy: {npy_path}")

    face_model = load_face_model_trimesh(args.face_model_dir)
    face_seq = process_beat_data(str(json_path), face_model, target_fps=args.beat_fps)

    jaw_idx = None
    if "jawOpen" in face_model.expression_names:
        jaw_idx = face_model.expression_names.index("jawOpen")

    if jaw_idx is None:
        raise RuntimeError("jawOpen not found in expression names")

    jaw_signal = face_seq[:, jaw_idx]

    lip_shape_names = [name.strip() for name in args.lip_shapes.split(",") if name.strip()]
    lip_indices = [
        face_model.expression_names.index(name)
        for name in lip_shape_names
        if name in face_model.expression_names
    ]
    if lip_indices:
        lip_signal = np.mean(face_seq[:, lip_indices], axis=1)
    else:
        lip_signal = np.zeros_like(jaw_signal)

    tongue_rig = FaceKitTongueRig(
        face_model.neutral_verts,
        face_model.faces,
        TONGUE_SLICE,
        ANCHOR_INDICES,
        BONE_INDICES,
        TONGUE_CONFIG,
    )
    ema_seq = load_ema_motion(
        str(npy_path),
        args.std_path,
        tongue_rig.anchors,
        TONGUE_CONFIG["std_scalar"],
    )

    anchor_idx = max(0, min(int(args.tongue_anchor_idx), ema_seq.shape[1] - 1))
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_idx = axis_map[args.tongue_axis]
    tongue_signal = ema_seq[:, anchor_idx, axis_idx]

    jaw_signal = resample_series(jaw_signal, args.beat_fps, args.analysis_fps)
    lip_signal = resample_series(lip_signal, args.beat_fps, args.analysis_fps)
    tongue_signal = resample_series(tongue_signal, args.tongue_fps, args.analysis_fps)

    min_len = min(len(jaw_signal), len(tongue_signal))
    jaw_signal = jaw_signal[:min_len]
    lip_signal = lip_signal[:min_len]
    tongue_signal = tongue_signal[:min_len]

    seg_start = max(0.0, args.segment_start)
    seg_end = seg_start + max(0.0, args.segment_duration)

    start_idx = int(seg_start * args.analysis_fps)
    end_idx = int(seg_end * args.analysis_fps)
    end_idx = min(end_idx, min_len)

    jaw_seg = jaw_signal[start_idx:end_idx]
    lip_seg = lip_signal[start_idx:end_idx]
    tongue_seg = tongue_signal[start_idx:end_idx]

    times = np.arange(len(jaw_seg)) / args.analysis_fps + seg_start

    intervals = parse_textgrid_intervals(textgrid_path, args.phone_tier)
    phone_filter = [item.strip() for item in args.phone_filter.split(",") if item.strip()]
    phone_intervals = filter_intervals(intervals, phone_filter)

    max_lag_frames = max(1, int(args.max_lag * args.analysis_fps))
    lags, corrs, best_lag, best_corr = lag_sweep(jaw_seg, tongue_seg, max_lag_frames)
    zero_corr = safe_corr(jaw_seg, tongue_seg)

    report = {
        "dataset_id": dataset_id,
        "analysis_fps": args.analysis_fps,
        "segment_start": seg_start,
        "segment_duration": seg_end - seg_start,
        "jaw_index": jaw_idx,
        "lip_shapes_used": [lip_shape_names[i] for i in range(len(lip_shape_names)) if lip_shape_names[i] in face_model.expression_names],
        "tongue_anchor_idx": anchor_idx,
        "tongue_axis": args.tongue_axis,
        "max_lag_s": args.max_lag,
        "best_lag_frames": int(best_lag),
        "best_lag_s": float(best_lag / args.analysis_fps),
        "corr_at_zero": float(zero_corr),
        "corr_at_best": float(best_corr),
        "phone_filter": phone_filter,
        "n_phone_intervals": len(phone_intervals),
    }

    report_path = output_dir / f"{dataset_id}_jaw_tongue_sync.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.no_plot:
        fig, (ax_signal, ax_corr) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

        jaw_z = zscore(jaw_seg)
        tongue_z = zscore(tongue_seg)
        lip_z = zscore(lip_seg)

        ax_signal.plot(times, jaw_z, label="jawOpen (z)", color="#1f77b4")
        ax_signal.plot(times, tongue_z, label="tongue (z)", color="#d62728")
        ax_signal.plot(times, lip_z, label="lip avg (z)", color="#2ca02c", linestyle="--")

        for interval in phone_intervals:
            if interval.end < seg_start or interval.start > seg_end:
                continue
            start = max(interval.start, seg_start)
            end = min(interval.end, seg_end)
            ax_signal.axvspan(start, end, color="#f2c94c", alpha=0.25)
            mid = (start + end) / 2.0
            ax_signal.text(mid, ax_signal.get_ylim()[1], interval.text, fontsize=8, ha="center", va="top")

        ax_signal.set_title("Jaw vs Tongue (z-scored) with phoneme spans")
        ax_signal.set_xlabel("Time (s)")
        ax_signal.set_ylabel("Z-score")
        ax_signal.legend(loc="upper right")

        ax_corr.plot(lags / args.analysis_fps, corrs, color="#555555")
        ax_corr.axvline(0.0, color="#888888", linestyle="--", linewidth=1)
        ax_corr.axvline(best_lag / args.analysis_fps, color="#d62728", linestyle="--", linewidth=1)
        ax_corr.set_title("Jaw-Tongue correlation vs lag")
        ax_corr.set_xlabel("Lag (s) [+] tongue delayed")
        ax_corr.set_ylabel("Pearson r")

        plot_path = output_dir / f"{dataset_id}_jaw_tongue_sync.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    print("Jaw-tongue sync analysis complete.")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
