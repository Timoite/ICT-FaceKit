#!/usr/bin/env python3
"""
Measure lip-closure timing at B phoneme times from TextGrid using BEAT blendshapes.
Outputs per-B timing stats to confirm if lip closure lags the labeled B.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from face_model_io_trimesh import load_face_model_trimesh
    from test import process_beat_data
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT))
    from tongue_scripts.face_model_io_trimesh import load_face_model_trimesh
    from tongue_scripts.test import process_beat_data


@dataclass
class Interval:
    start: float
    end: float
    text: str


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report lip-closure timing around B phonemes from TextGrid."
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
        "--face-model-dir",
        default=str(PROJECT_ROOT / "FaceXModel"),
        help="Face model directory",
    )
    parser.add_argument("--analysis-fps", type=float, default=50.0, help="Target FPS")
    parser.add_argument("--phone-tier", default="phones", help="TextGrid tier name")
    parser.add_argument(
        "--window",
        type=float,
        default=0.2,
        help="Seconds around B midpoint to search for min closure signal",
    )
    parser.add_argument(
        "--signal-mode",
        choices=["gap", "jaw", "lip"],
        default="gap",
        help="Blendshape signal: gap=jawOpen-cue, jaw=jawOpen, lip=lip-closure avg",
    )
    parser.add_argument(
        "--lip-shapes",
        default=",".join(DEFAULT_LIP_SHAPES),
        help="Comma-separated lip blendshape names to average",
    )
    parser.add_argument(
        "--no-jaw-correction",
        action="store_true",
        help="Disable jawOpen min-shift correction",
    )
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


def compute_lip_gap_from_blendshapes(
    face_model_dir: Path,
    beat_root: Path,
    dataset_id: str,
    analysis_fps: float,
    lip_shapes: Sequence[str],
    signal_mode: str,
    apply_jaw_correction: bool,
) -> Tuple[np.ndarray, np.ndarray, str]:
    json_path = beat_root / f"{dataset_id}.json"
    face_model = load_face_model_trimesh(str(face_model_dir))

    face_seq = process_beat_data(str(json_path), face_model, target_fps=analysis_fps)
    if face_seq.size == 0:
        raise RuntimeError("Empty face sequence from BEAT data")

    expr_names = face_model.expression_names

    jaw_signal = np.zeros(face_seq.shape[0], dtype=np.float32)
    if "jawOpen" in expr_names:
        jaw_idx = expr_names.index("jawOpen")
        jaw_signal = face_seq[:, jaw_idx].astype(np.float32)
        if apply_jaw_correction:
            min_val = float(np.min(jaw_signal))
            jaw_signal = np.maximum(0.0, jaw_signal - min_val)
    else:
        print("Warning: jawOpen not found in expression names.")

    lip_indices = [expr_names.index(name) for name in lip_shapes if name in expr_names]
    if lip_indices:
        lip_signal = np.mean(face_seq[:, lip_indices], axis=1).astype(np.float32)
    else:
        lip_signal = np.zeros(face_seq.shape[0], dtype=np.float32)
        print("Warning: no requested lip blendshapes found in expression names.")

    if signal_mode == "jaw":
        signal = jaw_signal
        label = "jawOpen"
    elif signal_mode == "lip":
        signal = lip_signal
        label = "lip-avg"
    else:
        signal = jaw_signal - lip_signal
        label = "jawOpen - lip-avg"

    times = np.arange(len(signal)) / analysis_fps
    return times, signal, label


def report_b_intervals(
    times: np.ndarray,
    signal: np.ndarray,
    intervals: Sequence[Interval],
    window: float,
    label: str,
) -> None:
    b_intervals = [it for it in intervals if normalize_phone(it.text) == "B"]
    if not b_intervals:
        print("No B intervals found in TextGrid.")
        return

    interval_offsets: List[float] = []
    window_offsets: List[float] = []

    if len(times) > 1:
        dt = times[1] - times[0]
    else:
        dt = 1.0
    fps = 1.0 / dt if dt > 0 else 1.0

    for idx, interval in enumerate(b_intervals, start=1):
        start = interval.start
        end = interval.end
        mid = 0.5 * (start + end)

        start_idx = max(0, int(np.floor(start * fps)))
        end_idx = max(start_idx + 1, min(len(times), int(np.ceil(end * fps))))

        start_idx = min(start_idx, len(signal) - 1)
        end_idx = min(end_idx, len(signal))

        start_val = signal[start_idx]
        mid_idx = min(len(signal) - 1, int(round(mid / dt)))
        mid_val = signal[mid_idx]
        end_val = signal[end_idx - 1]

        interval_slice = signal[start_idx:end_idx]
        min_idx_local = int(np.argmin(interval_slice))
        min_idx = start_idx + min_idx_local
        min_time = times[min_idx]
        min_val = signal[min_idx]

        window_start = max(0.0, mid - window)
        window_end = min(times[-1], mid + window)
        window_start_idx = max(0, int(np.floor(window_start / dt)))
        window_end_idx = min(len(signal), int(np.ceil(window_end / dt)))

        window_slice = signal[window_start_idx:window_end_idx]
        win_min_idx_local = int(np.argmin(window_slice))
        win_min_idx = window_start_idx + win_min_idx_local
        win_min_time = times[win_min_idx]
        win_min_val = signal[win_min_idx]

        print(
            f"B#{idx}: {start:.3f}s-{end:.3f}s (mid {mid:.3f}s) | "
            f"{label} start={start_val:.4f} mid={mid_val:.4f} end={end_val:.4f}"
        )
        print(
            f"  min in interval: {min_val:.4f} at {min_time:.3f}s "
            f"(offset {min_time - mid:+.3f}s)"
        )
        print(
            f"  min in window(+/-{window:.3f}s): {win_min_val:.4f} at {win_min_time:.3f}s "
            f"(offset {win_min_time - mid:+.3f}s)"
        )

        interval_offsets.append(min_time - mid)
        window_offsets.append(win_min_time - mid)

    def _summarize(name: str, offsets: List[float]) -> None:
        if not offsets:
            return
        arr = np.array(offsets, dtype=np.float32)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        median = float(np.median(arr))
        pos_ratio = float(np.mean(arr > 0)) * 100.0
        print(
            f"\n{name} offset summary (s): mean={mean:+.3f} std={std:.3f} "
            f"median={median:+.3f} positive={pos_ratio:.1f}%"
        )

    _summarize("Interval min", interval_offsets)
    _summarize(f"Window min (+/-{window:.3f}s)", window_offsets)


def main() -> None:
    args = parse_args()
    beat_root = Path(args.beat_root)
    textgrid_path = beat_root / f"{args.dataset_id}.TextGrid"
    if not textgrid_path.exists():
        raise FileNotFoundError(f"Missing TextGrid: {textgrid_path}")

    lip_shapes = [item.strip() for item in args.lip_shapes.split(",") if item.strip()]
    times, signal, label = compute_lip_gap_from_blendshapes(
        face_model_dir=Path(args.face_model_dir),
        beat_root=beat_root,
        dataset_id=args.dataset_id,
        analysis_fps=args.analysis_fps,
        lip_shapes=lip_shapes,
        signal_mode=args.signal_mode,
        apply_jaw_correction=not args.no_jaw_correction,
    )

    intervals = parse_textgrid_intervals(textgrid_path, args.phone_tier)
    report_b_intervals(times, signal, intervals, args.window, label)


if __name__ == "__main__":
    main()
