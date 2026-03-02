#!/usr/bin/env python3
"""
Plot lip aperture against TextGrid articulations with waveform context.

Inputs (default dataset: 1_wayne_0_75_75):
- .wav waveform
- .TextGrid phone/phoneme intervals
- .npy articulatory predictions (cols 0-7 tongue, cols 8-11 lips)
- .json BEAT blendshapes (mesh-based lip aperture from README landmark indices)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

TONGUE_COORD_COLS = 8  # 4 tongue points x (z, y)
LIP_COORD_COLS = 4     # upper-lip (2) + lower-lip (2)

# From README Multi-PIE 68 landmarks mapping:
# landmark 62 (upper inner lip mid) -> vertex 5533
# landmark 66 (lower inner lip mid) -> vertex 5517
UPPER_LIP_VERTEX_IDX = 5533
LOWER_LIP_VERTEX_IDX = 5517

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
TONGUE_ANIMATION_DIR = SCRIPT_DIR / "tongue_animation"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TONGUE_ANIMATION_DIR) not in sys.path:
    sys.path.insert(0, str(TONGUE_ANIMATION_DIR))

from face_model_io_trimesh import load_face_model_trimesh


@dataclass
class Interval:
    start: float
    end: float
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot lip aperture (articulatory vs blendshape) with TextGrid interval blocks."
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
        help="Root folder containing BEAT wav/json/TextGrid files",
    )
    parser.add_argument(
        "--tongue-npy-dir",
        default=str(SCRIPT_DIR / "outputs"),
        help="Directory containing WavLM tongue+lips .npy files",
    )
    parser.add_argument(
        "--face-model-dir",
        default=str(PROJECT_ROOT / "FaceXModel"),
        help="Face model directory",
    )
    parser.add_argument("--phone-tier", default="phones", help="TextGrid tier to read")
    parser.add_argument("--target-fps", type=float, default=50.0, help="Target FPS for all time series (resamples if needed)")
    parser.add_argument("--tongue-fps", type=float, default=50.0, help="Native FPS of .npy sequence (for resampling)")
    parser.add_argument("--beat-fps", type=float, default=60.0, help="Native FPS of BEAT JSON (for resampling)")
    parser.add_argument(
        "--smooth-frames",
        type=int,
        default=5,
        help="Moving-average window (frames), applied to both aperture curves",
    )
    parser.add_argument(
        "--window-start",
        type=float,
        default=None,
        help="Plot window start (seconds). Default: first non-empty interval start",
    )
    parser.add_argument(
        "--window-end",
        type=float,
        default=None,
        help="Plot window end (seconds). Default: last non-empty interval end",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output image path (PNG). Default: tongue_scripts/vis_output/<dataset>_lip_aperture_textgrid.png",
    )
    parser.add_argument(
        "--aperture-only",
        action="store_true",
        help="Plot only the two aperture curves (no waveform, no TextGrid labels, no interval separators)",
    )
    return parser.parse_args()


def map_beat_to_ict_names(beat_name: str) -> List[str]:
    truly_bilateral = ["browInnerUp", "cheekPuff"]
    if beat_name in truly_bilateral:
        return [f"{beat_name}_L", f"{beat_name}_R"]
    direct_mapping = ["jawLeft", "jawRight", "mouthLeft", "mouthRight"]
    if beat_name in direct_mapping:
        return [beat_name]
    if beat_name.endswith("Left"):
        return [f"{beat_name[:-4]}_L"]
    if beat_name.endswith("Right"):
        return [f"{beat_name[:-5]}_R"]
    return [beat_name]


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


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values
    window = max(1, int(window))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def zscore(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def resample_matrix(values: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    if np.isclose(source_fps, target_fps):
        return values
    if len(values) < 2:
        return values
    duration = len(values) / source_fps
    n_target = max(1, int(round(duration * target_fps)))
    x_source = np.linspace(0.0, duration, len(values), endpoint=False)
    x_target = np.linspace(0.0, duration, n_target, endpoint=False)
    out = np.empty((n_target, values.shape[1]), dtype=np.float32)
    for col in range(values.shape[1]):
        out[:, col] = np.interp(x_target, x_source, values[:, col])
    return out


def load_blendshape_lip_aperture(json_path: Path, face_model_dir: Path, beat_fps: float, target_fps: float) -> np.ndarray:
    """Load blendshapes and compute lip aperture from mesh vertices, then resample to target FPS."""
    with json_path.open("r", encoding="utf-8") as handle:
        anim_data = json.load(handle)

    beat_names = anim_data["names"]
    frames = anim_data["frames"]

    face_model = load_face_model_trimesh(str(face_model_dir))
    expr_names = face_model.expression_names
    expr_to_idx = {name: i for i, name in enumerate(expr_names)}

    source = np.zeros((len(frames), len(expr_names)), dtype=np.float32)
    for frame_idx, frame in enumerate(frames):
        weights = frame.get("weights", [])
        for beat_idx, weight in enumerate(weights):
            if beat_idx >= len(beat_names):
                break
            mapped = map_beat_to_ict_names(beat_names[beat_idx])
            for m in mapped:
                idx = expr_to_idx.get(m)
                if idx is not None:
                    source[frame_idx, idx] = float(weight)

    n_verts = len(face_model.neutral_verts)
    if UPPER_LIP_VERTEX_IDX >= n_verts or LOWER_LIP_VERTEX_IDX >= n_verts:
        raise RuntimeError(
            f"Lip vertex index out of range: upper={UPPER_LIP_VERTEX_IDX}, lower={LOWER_LIP_VERTEX_IDX}, n_verts={n_verts}"
        )

    lip_aperture = np.zeros(len(source), dtype=np.float32)
    for frame_idx in range(len(source)):
        row = source[frame_idx]
        weights = {name: float(val) for name, val in zip(expr_names, row) if val != 0.0}
        verts = face_model.deform(weights)
        upper = verts[UPPER_LIP_VERTEX_IDX]
        lower = verts[LOWER_LIP_VERTEX_IDX]
        lip_aperture[frame_idx] = float(np.linalg.norm(upper - lower))

    # Normalize closure baseline to zero (closed mouth => 0)
    lip_aperture = np.maximum(0.0, lip_aperture - float(np.min(lip_aperture)))

    # Resample BEAT from native beat_fps to target_fps
    lip_aperture = lip_aperture.reshape(-1, 1)
    lip_aperture = resample_matrix(lip_aperture, source_fps=beat_fps, target_fps=target_fps)
    return lip_aperture.squeeze()


def normalize_phone(label: str) -> str:
    return re.sub(r"[0-9]", "", label.upper())


def main() -> None:
    args = parse_args()

    beat_root = Path(args.beat_root)
    npy_dir = Path(args.tongue_npy_dir)
    face_model_dir = Path(args.face_model_dir)

    dataset_id = args.dataset_id
    wav_path = beat_root / f"{dataset_id}.wav"
    textgrid_path = beat_root / f"{dataset_id}.TextGrid"
    json_path = beat_root / f"{dataset_id}.json"
    npy_path = npy_dir / f"{dataset_id}.npy"

    for p in (wav_path, textgrid_path, json_path, npy_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = SCRIPT_DIR / "vis_output" / f"{dataset_id}_lip_aperture_textgrid.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Waveform
    sr, wav_data = wavfile.read(wav_path)
    if wav_data.ndim > 1:
        wav_data = np.mean(wav_data.astype(np.float32), axis=1)
    wav_data = wav_data.astype(np.float32)
    max_abs = float(np.max(np.abs(wav_data))) if len(wav_data) else 1.0
    if max_abs > 0:
        wav_data = wav_data / max_abs
    t_wav = np.arange(len(wav_data), dtype=np.float32) / float(sr)

    # Articulatory aperture from Euclidean distance between upper/lower lip points
    raw_motion = np.load(npy_path)
    if raw_motion.ndim != 2:
        raise ValueError(
            f"Expected .npy shape (T, D), got {raw_motion.shape}"
        )

    min_required_cols = TONGUE_COORD_COLS + LIP_COORD_COLS
    if raw_motion.shape[1] < min_required_cols:
        raise ValueError(
            f"Expected at least {min_required_cols} columns (8 tongue + 4 lips), got {raw_motion.shape[1]}"
        )

    # Exclude tongue columns [0:8], then use the next 4 columns [8:12] for lips.
    lip_motion = raw_motion[:, TONGUE_COORD_COLS:TONGUE_COORD_COLS + LIP_COORD_COLS].astype(np.float32)
    upper_point = lip_motion[:, 0:2]
    lower_point = lip_motion[:, 2:4]
    lip_aperture_art = np.linalg.norm(upper_point - lower_point, axis=1)
    
    # Resample articulatory data from native tongue_fps to target_fps
    lip_aperture_art = lip_aperture_art.reshape(-1, 1)  # (T,) -> (T, 1) for resample_matrix
    lip_aperture_art = resample_matrix(lip_aperture_art, source_fps=args.tongue_fps, target_fps=args.target_fps)
    lip_aperture_art = lip_aperture_art.squeeze()  # (T, 1) -> (T,)

    # Blendshape aperture from mesh lip vertices (README landmark-derived indices)
    lip_aperture_bs = load_blendshape_lip_aperture(json_path, face_model_dir, args.beat_fps, args.target_fps)

    # Optional smoothing
    lip_aperture_art = moving_average(lip_aperture_art, args.smooth_frames)
    lip_aperture_bs = moving_average(lip_aperture_bs, args.smooth_frames)

    # Z-score both aperture curves
    lip_aperture_art = zscore(lip_aperture_art)
    lip_aperture_bs = zscore(lip_aperture_bs)

    # Both are now at target_fps
    t_art = np.arange(len(lip_aperture_art), dtype=np.float32) / float(args.target_fps)
    t_bs = np.arange(len(lip_aperture_bs), dtype=np.float32) / float(args.target_fps)

    # TextGrid intervals (optional in aperture-only mode)
    intervals: List[Interval] = []
    if not args.aperture_only:
        intervals_all = parse_textgrid_intervals(textgrid_path, args.phone_tier)
        intervals = [iv for iv in intervals_all if iv.text.strip()]

    max_t = min(
        t_wav[-1] if len(t_wav) else 0.0,
        t_art[-1] if len(t_art) else 0.0,
        t_bs[-1] if len(t_bs) else 0.0,
    )

    if args.window_start is not None:
        window_start = max(0.0, float(args.window_start))
    elif intervals and not args.aperture_only:
        window_start = max(0.0, min(iv.start for iv in intervals))
    else:
        window_start = 0.0

    if args.window_end is not None:
        window_end = min(max_t, float(args.window_end))
    elif intervals and not args.aperture_only:
        window_end = min(max_t, max(iv.end for iv in intervals))
    else:
        window_end = max_t

    if window_end <= window_start:
        window_end = max_t

    wav_mask = (t_wav >= window_start) & (t_wav <= window_end)
    art_mask = (t_art >= window_start) & (t_art <= window_end)
    bs_mask = (t_bs >= window_start) & (t_bs <= window_end)

    if args.aperture_only:
        fig, ax_lip = plt.subplots(1, 1, figsize=(16, 4.5))
        ax_lip.plot(t_art[art_mask], lip_aperture_art[art_mask], label="Lip Aperture (Articulatory)", linewidth=1.8)
        ax_lip.plot(t_bs[bs_mask], lip_aperture_bs[bs_mask], label="Lip Aperture (Blendshapes)", linewidth=1.8)
        ax_lip.set_ylabel("Aperture (z-score)")
        ax_lip.set_xlabel("Time (s)")
        ax_lip.set_title("Lip aperture comparison")
        ax_lip.set_xlim(window_start, window_end)
        ax_lip.legend(loc="upper right")
        ax_lip.grid(alpha=0.2, linestyle=":")
    else:
        fig, (ax_wave, ax_lip) = plt.subplots(
            2,
            1,
            figsize=(16, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.2},
        )

        # Top: waveform
        ax_wave.plot(t_wav[wav_mask], wav_data[wav_mask], color="#1f77b4", linewidth=1.2)
        ax_wave.set_ylabel("Waveform")
        ax_wave.set_title(f"{dataset_id}: Waveform + TextGrid intervals")

        # Bottom: lip apertures
        ax_lip.plot(t_art[art_mask], lip_aperture_art[art_mask], label="Lip Aperture (Articulatory)", linewidth=1.8)
        ax_lip.plot(t_bs[bs_mask], lip_aperture_bs[bs_mask], label="Lip Aperture (Blendshapes)", linewidth=1.8)
        ax_lip.set_ylabel("Aperture (z-score)")
        ax_lip.set_xlabel("Time (s)")
        ax_lip.set_title("Lip aperture comparison")
        ax_lip.legend(loc="upper right")

        visible_intervals = [
            iv for iv in intervals if not (iv.end < window_start or iv.start > window_end)
        ]

        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for i, iv in enumerate(visible_intervals):
            s = max(window_start, iv.start)
            e = min(window_end, iv.end)
            if e <= s:
                continue
            color = palette[i % len(palette)]

            ax_wave.axvspan(s, e, color=color, alpha=0.06, linewidth=0)
            ax_lip.axvspan(s, e, color=color, alpha=0.06, linewidth=0)

            mid = 0.5 * (s + e)
            label = normalize_phone(iv.text)
            if label:
                ax_wave.text(mid, 0.99, label, transform=ax_wave.get_xaxis_transform(), ha="center", va="bottom", fontsize=11)
                ax_lip.text(mid, 0.72, label, transform=ax_lip.get_xaxis_transform(), ha="center", va="center", fontsize=11)

        boundaries = sorted({b for iv in visible_intervals for b in (iv.start, iv.end) if window_start <= b <= window_end})
        for b in boundaries:
            ax_wave.axvline(b, linestyle="--", linewidth=1.5, color="#1f77b4", alpha=0.9)
            ax_lip.axvline(b, linestyle="--", linewidth=1.5, color="#1f77b4", alpha=0.9)

        ax_wave.set_xlim(window_start, window_end)
        ax_wave.grid(alpha=0.2, linestyle=":")
        ax_lip.grid(alpha=0.2, linestyle=":")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
