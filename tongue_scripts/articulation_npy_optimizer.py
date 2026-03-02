#!/usr/bin/env python3
"""
Articulation-aware optimizer for tongue .npy motion files.

Current default rule focuses on /TH/ and /DH/, but the architecture supports
adding more phoneme-conditioned rules later.

Main idea:
- Load raw articulatory motion (.npy)
- Denormalize the 4 tongue anchors into 3D rig space
- Use TextGrid intervals to gate optimization by phoneme timing
- Apply rule-based target attraction (tip/blade) with smooth temporal blending
- Write optimized .npy back (first 8 columns updated)

Optional interactive mode lets you click:
1) a dental edge point on the face sagittal view
2) a desired tongue-tip contact target point
and optionally refine one target per interval.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
TONGUE_ANIMATION_DIR = SCRIPT_DIR / "tongue_animation"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TONGUE_ANIMATION_DIR) not in sys.path:
    sys.path.insert(0, str(TONGUE_ANIMATION_DIR))

from face_model_io_trimesh import load_face_model_trimesh
from generate_tongue_animation import (
    FaceKitTongueRig,
    TONGUE_SLICE,
    ANCHOR_INDICES,
    BONE_INDICES,
)


FPS_DEFAULT = 50.0
GUM_START_IDX = 14062
GUM_END_IDX = 17039
TIP_ANCHOR_IDX = 3
BLADE_ANCHOR_IDX = 2


@dataclass
class PhoneInterval:
    start: float
    end: float
    label: str


@dataclass
class ArticulationRule:
    name: str
    labels: List[str]
    ramp_seconds: float
    tip_strength: float
    blade_strength: float
    max_move: float
    temporal_smoothing: float


def normalize_phone(label: str) -> str:
    return re.sub(r"[0-9]", "", label.upper())


def parse_textgrid_intervals(textgrid_path: Path, tier_name: str) -> List[PhoneInterval]:
    intervals: List[PhoneInterval] = []
    in_tier = False
    current: Dict[str, str] = {}

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
            elif line.startswith("xmax ="):
                current["end"] = line.split("=", 1)[1].strip()
            elif line.startswith("text ="):
                txt = line.split("=", 1)[1].strip().strip('"')
                current["text"] = txt
                if {"start", "end", "text"} <= current.keys():
                    try:
                        s = float(current["start"])
                        e = float(current["end"])
                    except ValueError:
                        s, e = 0.0, 0.0
                    if txt.strip():
                        intervals.append(PhoneInterval(start=s, end=e, label=txt))
    return intervals


def raised_cosine_window(t: float, start: float, end: float, ramp: float) -> float:
    if t < start - ramp or t > end + ramp:
        return 0.0
    if start <= t <= end:
        return 1.0
    if t < start:
        x = (t - (start - ramp)) / max(ramp, 1e-6)
        return float(0.5 - 0.5 * np.cos(np.pi * x))
    x = (t - end) / max(ramp, 1e-6)
    return float(0.5 + 0.5 * np.cos(np.pi * x))


def raw_to_denorm_anchors(
    raw_ema_4x2: np.ndarray,
    std_4x2: np.ndarray,
    rig_anchors_4x3: np.ndarray,
    scalar: float,
) -> np.ndarray:
    denorm = np.zeros((len(raw_ema_4x2), 4, 3), dtype=np.float32)
    denorm[:, :, 0] = rig_anchors_4x3[:, 0][None, :]
    denorm[:, :, 1] = rig_anchors_4x3[:, 1][None, :] + raw_ema_4x2[:, :, 1] * std_4x2[:, 1][None, :] * scalar
    denorm[:, :, 2] = rig_anchors_4x3[:, 2][None, :] + raw_ema_4x2[:, :, 0] * std_4x2[:, 0][None, :] * scalar
    return denorm


def denorm_to_raw_anchors(
    denorm_4x3: np.ndarray,
    std_4x2: np.ndarray,
    rig_anchors_4x3: np.ndarray,
    scalar: float,
) -> np.ndarray:
    raw = np.zeros((len(denorm_4x3), 4, 2), dtype=np.float32)
    raw[:, :, 1] = (denorm_4x3[:, :, 1] - rig_anchors_4x3[:, 1][None, :]) / (
        np.maximum(np.abs(std_4x2[:, 1][None, :] * scalar), 1e-6)
    )
    raw[:, :, 0] = (denorm_4x3[:, :, 2] - rig_anchors_4x3[:, 2][None, :]) / (
        np.maximum(np.abs(std_4x2[:, 0][None, :] * scalar), 1e-6)
    )
    return raw


def estimate_default_teeth_yz(face_model) -> np.ndarray:
    verts = face_model.neutral_verts
    end_idx = min(GUM_END_IDX, len(verts))
    start_idx = min(GUM_START_IDX, end_idx)
    gum = verts[start_idx:end_idx]
    if len(gum) == 0:
        p = np.mean(verts, axis=0)
        return np.array([p[2], p[1]], dtype=np.float32)

    z_hi = np.percentile(gum[:, 2], 92)
    y_hi = np.percentile(gum[:, 1], 65)
    cand = gum[(gum[:, 2] >= z_hi) & (gum[:, 1] >= y_hi)]
    if len(cand) < 5:
        cand = gum[gum[:, 2] >= z_hi]
    if len(cand) == 0:
        cand = gum
    pt = np.mean(cand, axis=0)
    return np.array([pt[2], pt[1]], dtype=np.float32)


def parse_yz(s: Optional[str]) -> Optional[np.ndarray]:
    if not s:
        return None
    try:
        z_s, y_s = s.split(",")
        return np.array([float(z_s), float(y_s)], dtype=np.float32)
    except Exception:
        raise ValueError(f"Could not parse yz pair: {s!r}. Expected format: 'z,y'")


def _ginput_one(ax, title: str) -> np.ndarray:
    ax.set_title(title)
    plt.draw()
    pts = plt.ginput(1, timeout=-1)
    if not pts:
        raise RuntimeError("No point selected.")
    z, y = pts[0]
    ax.scatter([z], [y], c="yellow", s=70, marker="x")
    plt.draw()
    return np.array([z, y], dtype=np.float32)


def pick_global_targets_interactive(
    face_model,
    denorm_anchors: np.ndarray,
    active_frame_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    verts = face_model.neutral_verts
    midline = np.abs(verts[:, 0]) < 0.6
    yz_face = verts[midline][:, [2, 1]]

    tip_track = denorm_anchors[:, TIP_ANCHOR_IDX][:, [2, 1]]
    tip_th = tip_track[active_frame_idx] if len(active_frame_idx) else tip_track

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(yz_face[:, 0], yz_face[:, 1], s=1, c="#b0b0b0", alpha=0.35, label="face midline")
    if len(tip_th) > 0:
        ax.scatter(tip_th[:, 0], tip_th[:, 1], s=7, c="#d62728", alpha=0.35, label="tip in target windows")
    ax.set_xlabel("Z (Anterior →)")
    ax.set_ylabel("Y (Superior →)")
    ax.legend(loc="best")

    teeth_yz = _ginput_one(ax, "Click DENTAL EDGE target (teeth point), then Enter in plot")
    tip_yz = _ginput_one(ax, "Click desired TONGUE TIP contact target, then Enter in plot")

    plt.close(fig)
    return teeth_yz, tip_yz


def pick_interval_targets_interactive(
    denorm_anchors: np.ndarray,
    intervals: List[PhoneInterval],
    fps: float,
) -> Dict[int, np.ndarray]:
    targets: Dict[int, np.ndarray] = {}
    if not intervals:
        return targets

    fig, ax = plt.subplots(figsize=(9, 6))
    tip_track = denorm_anchors[:, TIP_ANCHOR_IDX][:, [2, 1]]

    for idx, ph in enumerate(intervals):
        s = max(0, int(np.floor(ph.start * fps)))
        e = min(len(tip_track), int(np.ceil(ph.end * fps)) + 1)
        seg = tip_track[s:e]
        if len(seg) == 0:
            continue

        ax.clear()
        ax.scatter(tip_track[:, 0], tip_track[:, 1], s=2, c="#cccccc", alpha=0.25)
        ax.scatter(seg[:, 0], seg[:, 1], s=10, c="#d62728", alpha=0.9)
        ax.set_xlabel("Z")
        ax.set_ylabel("Y")
        ax.set_title(f"Interval {idx+1}/{len(intervals)}: /{ph.label}/ {ph.start:.3f}-{ph.end:.3f}s | Click target")
        plt.draw()

        pt = plt.ginput(1, timeout=-1)
        if pt:
            z, y = pt[0]
            targets[idx] = np.array([z, y], dtype=np.float32)
            ax.scatter([z], [y], s=70, c="yellow", marker="x")
            plt.draw()

    plt.close(fig)
    return targets


def apply_rule_to_frame(
    anchors: np.ndarray,
    target_tip_yz: np.ndarray,
    weight: float,
    rule: ArticulationRule,
    prev_corrected: Optional[np.ndarray],
) -> np.ndarray:
    if weight <= 1e-6:
        return anchors.copy()

    corrected = anchors.copy()

    def _move_anchor(anchor_idx: int, strength: float, target_yz: np.ndarray):
        alpha = float(np.clip(strength * weight, 0.0, 1.0))
        if alpha <= 1e-6:
            return
        src = corrected[anchor_idx].copy()
        tgt = np.array([src[0], target_yz[1], target_yz[0]], dtype=np.float32)
        moved = (1.0 - alpha) * src + alpha * tgt
        delta = moved - src
        dist = float(np.linalg.norm(delta))
        if dist > rule.max_move:
            moved = src + delta * (rule.max_move / max(dist, 1e-6))
        corrected[anchor_idx] = moved

    _move_anchor(TIP_ANCHOR_IDX, rule.tip_strength, target_tip_yz)

    blade_tgt = 0.5 * corrected[BLADE_ANCHOR_IDX][[2, 1]] + 0.5 * target_tip_yz
    _move_anchor(BLADE_ANCHOR_IDX, rule.blade_strength, blade_tgt)

    if prev_corrected is not None:
        beta = float(np.clip(rule.temporal_smoothing, 0.0, 1.0))
        corrected = beta * prev_corrected + (1.0 - beta) * corrected

    return corrected


def build_active_intervals(all_intervals: List[PhoneInterval], labels: List[str]) -> List[PhoneInterval]:
    labels_norm = {normalize_phone(x) for x in labels}
    return [ph for ph in all_intervals if normalize_phone(ph.label) in labels_norm]


def best_interval_weight(t: float, intervals: List[PhoneInterval], ramp: float) -> Tuple[float, int]:
    best_w = 0.0
    best_i = -1
    for i, ph in enumerate(intervals):
        w = raised_cosine_window(t, ph.start, ph.end, ramp)
        if w > best_w:
            best_w = w
            best_i = i
    return best_w, best_i


def optimize_motion(
    denorm_anchors: np.ndarray,
    fps: float,
    intervals: List[PhoneInterval],
    rule: ArticulationRule,
    global_tip_target_yz: np.ndarray,
    interval_tip_targets: Optional[Dict[int, np.ndarray]] = None,
) -> np.ndarray:
    out = denorm_anchors.copy()
    prev_corr = None

    for i in range(len(out)):
        t = i / fps
        w, interval_idx = best_interval_weight(t, intervals, rule.ramp_seconds)
        if w <= 0.0:
            prev_corr = out[i].copy()
            continue

        target = global_tip_target_yz
        if interval_tip_targets and interval_idx in interval_tip_targets:
            target = interval_tip_targets[interval_idx]

        out[i] = apply_rule_to_frame(out[i], target, w, rule, prev_corr)
        prev_corr = out[i].copy()

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize tongue .npy by phoneme-conditioned articulation rules")
    parser.add_argument("--dataset-id", default="1_wayne_0_75_75")
    parser.add_argument("--npy-path", default=None, help="Input .npy path (default: tongue_scripts/outputs/<dataset>.npy)")
    parser.add_argument("--out-path", default=None, help="Output .npy path (default: tongue_scripts/outputs/<dataset>_optimized.npy)")
    parser.add_argument("--textgrid-path", default=None, help="TextGrid path (default: same folder layout as current scripts)")
    parser.add_argument("--phone-tier", default="phones")
    parser.add_argument("--face-model-dir", default=str(PROJECT_ROOT / "FaceXModel"))
    parser.add_argument("--std-path", default=str(SCRIPT_DIR / "normalising_vectors" / "JW13_4points_std.npy"))
    parser.add_argument("--scalar", type=float, default=0.20)
    parser.add_argument("--fps", type=float, default=FPS_DEFAULT)

    parser.add_argument("--labels", default="TH,DH", help="Comma-separated phone labels for this rule")
    parser.add_argument("--ramp-seconds", type=float, default=0.03)
    parser.add_argument("--tip-strength", type=float, default=0.95)
    parser.add_argument("--blade-strength", type=float, default=0.55)
    parser.add_argument("--max-move", type=float, default=1.10)
    parser.add_argument("--temporal-smoothing", type=float, default=0.30)

    parser.add_argument("--interactive-pick", action="store_true", default=True)
    parser.add_argument("--no-interactive-pick", dest="interactive_pick", action="store_false")
    parser.add_argument("--per-interval-picks", action="store_true", default=False)

    parser.add_argument("--teeth-yz", default=None, help="Manual teeth point as 'z,y'")
    parser.add_argument("--tip-target-yz", default=None, help="Manual tip target point as 'z,y'")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    npy_path = Path(args.npy_path) if args.npy_path else (SCRIPT_DIR / "outputs" / f"{args.dataset_id}.npy")
    out_path = Path(args.out_path) if args.out_path else (SCRIPT_DIR / "outputs" / f"{args.dataset_id}_optimized.npy")
    textgrid_path = Path(args.textgrid_path) if args.textgrid_path else (
        PROJECT_ROOT / "data" / "beat_cache" / "beat_english_v0.2.1" / "beat_english_v0.2.1" / "1" / f"{args.dataset_id}.TextGrid"
    )

    if not npy_path.exists():
        raise FileNotFoundError(f"Missing input npy: {npy_path}")
    if not textgrid_path.exists():
        raise FileNotFoundError(f"Missing TextGrid: {textgrid_path}")

    raw_motion = np.load(npy_path)
    if raw_motion.ndim != 2 or raw_motion.shape[1] < 8:
        raise ValueError(f"Expected (N, >=8) motion array, got shape={raw_motion.shape}")

    face_model = load_face_model_trimesh(str(Path(args.face_model_dir)))
    tongue_rig = FaceKitTongueRig(
        face_model.neutral_verts,
        face_model.faces,
        TONGUE_SLICE,
        ANCHOR_INDICES,
        BONE_INDICES,
        {
            "rotation_deg": 5,
            "thickness": 1.2,
            "shift_y": 0,
            "shift_z": 0,
            "std_scalar": args.scalar,
        },
    )

    std_raw = np.load(args.std_path)
    std_4x2 = std_raw.flatten()[:8].reshape(4, 2).astype(np.float32)

    raw_ema = raw_motion[:, :8].reshape(-1, 4, 2).astype(np.float32)
    denorm = raw_to_denorm_anchors(raw_ema, std_4x2, tongue_rig.anchors.astype(np.float32), args.scalar)

    all_intervals = parse_textgrid_intervals(textgrid_path, args.phone_tier)
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    active_intervals = build_active_intervals(all_intervals, labels)
    if not active_intervals:
        raise RuntimeError(f"No intervals found for labels={labels} in {textgrid_path}")

    rule = ArticulationRule(
        name="th_contact",
        labels=labels,
        ramp_seconds=args.ramp_seconds,
        tip_strength=args.tip_strength,
        blade_strength=args.blade_strength,
        max_move=args.max_move,
        temporal_smoothing=args.temporal_smoothing,
    )

    teeth_yz = parse_yz(args.teeth_yz)
    tip_target_yz = parse_yz(args.tip_target_yz)

    active_frame_idx = []
    for ph in active_intervals:
        s = max(0, int(np.floor(ph.start * args.fps)))
        e = min(len(denorm), int(np.ceil(ph.end * args.fps)) + 1)
        if e > s:
            active_frame_idx.extend(range(s, e))
    active_frame_idx = np.array(sorted(set(active_frame_idx)), dtype=int)

    if teeth_yz is None:
        teeth_yz = estimate_default_teeth_yz(face_model)

    if args.interactive_pick:
        clicked_teeth, clicked_tip = pick_global_targets_interactive(face_model, denorm, active_frame_idx)
        teeth_yz = clicked_teeth
        tip_target_yz = clicked_tip

    if tip_target_yz is None:
        tip_target_yz = teeth_yz.copy()

    interval_tip_targets = None
    if args.interactive_pick and args.per_interval_picks:
        interval_tip_targets = pick_interval_targets_interactive(denorm, active_intervals, args.fps)

    before_dist = np.linalg.norm(denorm[active_frame_idx, TIP_ANCHOR_IDX][:, [2, 1]] - teeth_yz[None, :], axis=1)

    denorm_opt = optimize_motion(
        denorm,
        args.fps,
        active_intervals,
        rule,
        global_tip_target_yz=tip_target_yz,
        interval_tip_targets=interval_tip_targets,
    )

    after_dist = np.linalg.norm(denorm_opt[active_frame_idx, TIP_ANCHOR_IDX][:, [2, 1]] - teeth_yz[None, :], axis=1)

    raw_opt = denorm_to_raw_anchors(denorm_opt, std_4x2, tongue_rig.anchors.astype(np.float32), args.scalar)

    out_motion = raw_motion.copy().astype(np.float32)
    out_motion[:, :8] = raw_opt.reshape(-1, 8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out_motion)

    sidecar = out_path.with_suffix(".json")
    report = {
        "input_npy": str(npy_path),
        "output_npy": str(out_path),
        "textgrid": str(textgrid_path),
        "rule": {
            "name": rule.name,
            "labels": rule.labels,
            "ramp_seconds": rule.ramp_seconds,
            "tip_strength": rule.tip_strength,
            "blade_strength": rule.blade_strength,
            "max_move": rule.max_move,
            "temporal_smoothing": rule.temporal_smoothing,
        },
        "targets": {
            "teeth_yz": teeth_yz.tolist(),
            "tip_target_yz": tip_target_yz.tolist(),
            "interval_targets": {str(k): v.tolist() for k, v in (interval_tip_targets or {}).items()},
        },
        "stats": {
            "active_intervals": len(active_intervals),
            "active_frames": int(len(active_frame_idx)),
            "tip_to_teeth_mean_before": float(np.mean(before_dist)) if len(before_dist) else None,
            "tip_to_teeth_mean_after": float(np.mean(after_dist)) if len(after_dist) else None,
        },
    }
    sidecar.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved optimized npy: {out_path}")
    print(f"Saved optimization report: {sidecar}")
    if len(before_dist):
        print(
            "Tip→teeth mean distance (target windows): "
            f"{np.mean(before_dist):.4f} -> {np.mean(after_dist):.4f}"
        )


if __name__ == "__main__":
    main()
