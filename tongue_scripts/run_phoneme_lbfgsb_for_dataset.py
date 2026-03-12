#!/usr/bin/env python3
"""
Run the phoneme L-BFGS-B optimizer on one dataset and emit render-ready motion.

Input motion shape:  (T, >=16) where first 8 columns are 4 anchors x (z, y).
Output motion shape: (T, >=16) same as input, but first 8 columns replaced
with optimized anchor trajectories re-encoded in the original normalized space.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
TONGUE_ANIMATION_DIR = SCRIPT_DIR / "tongue_animation"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TONGUE_ANIMATION_DIR) not in sys.path:
    sys.path.insert(0, str(TONGUE_ANIMATION_DIR))

from face_model_io_trimesh import load_face_model_trimesh  # type: ignore
from generate_tongue_animation import (  # type: ignore
    ANCHOR_INDICES,
    BONE_INDICES,
    TONGUE_SLICE,
    FaceKitTongueRig,
)
from tongue_scripts.phoneme_lbfgsb_optimizer import (
    FaceKitRigForwardAdapter,
    MouthBox,
    OptimizationConfig,
    OptimizationWeights,
    RegionRect3D,
    export_debug_report,
    optimize_utterance,
    parse_textgrid,
)


def parse_yz(value: Optional[str]) -> Optional[np.ndarray]:
    if not value:
        return None
    try:
        z_s, y_s = value.split(",")
        return np.array([float(z_s), float(y_s)], dtype=np.float64)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not parse yz pair {value!r}; expected 'z,y'.") from exc


def load_target_points_json(path: Path) -> dict[str, np.ndarray]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not parse target points JSON: {path}") from exc

    points = payload.get("points_yz")
    if not isinstance(points, dict):
        raise ValueError(f"Invalid target points JSON {path}: missing object field 'points_yz'")

    out: dict[str, np.ndarray] = {}
    for key in ("teeth_yz", "alveolar_yz", "interdental_yz"):
        value = points.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"Invalid value for '{key}' in {path}; expected [z, y]")
        out[key] = np.array([float(value[0]), float(value[1])], dtype=np.float64)
    return out


def shift_sequence(seq: np.ndarray, shift_frames: int, pad_mode: str) -> np.ndarray:
    if shift_frames == 0:
        return seq
    n = len(seq)
    if n == 0:
        return seq
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


def collect_span_indices(spans, target_class: str, total_frames: int) -> np.ndarray:
    idx: list[int] = []
    for sp in spans:
        if sp.phoneme_class != target_class:
            continue
        idx.extend(range(sp.start_frame, sp.end_frame + 1))
    if not idx:
        return np.zeros((0,), dtype=np.int32)
    return np.array(sorted(set(i for i in idx if 0 <= i < total_frames)), dtype=np.int32)


def estimate_oral_targets_yz(face_model) -> dict[str, np.ndarray]:
    """
    Estimate tip target points from neutral mesh geometry in sagittal view.

    Notes:
    - Uses gums/teeth vertices only (14062:16611), excluding tongue vertices.
    - `teeth_yz` is estimated from upper-anterior teeth cluster.
    - `interdental_yz` is midpoint between upper/lower anterior teeth clusters.
    - `alveolar_yz` is estimated from upper arch just posterior to upper teeth.
    """
    verts = np.asarray(face_model.neutral_verts, dtype=np.float64)
    gums = verts[14062:16611]
    if len(gums) == 0:
        p = np.mean(verts, axis=0)
        yz = np.array([p[2], p[1]], dtype=np.float64)
        return {
            "teeth_yz": yz.copy(),
            "alveolar_yz": np.array([yz[0] - 1.0, yz[1] + 0.8], dtype=np.float64),
            "interdental_yz": yz.copy(),
            "upper_front_yz": yz.copy(),
            "lower_front_yz": yz.copy(),
        }

    mid = gums[np.abs(gums[:, 0]) <= 0.6]
    if len(mid) < 16:
        mid = gums

    z_front = np.percentile(mid[:, 2], 90)
    front = mid[mid[:, 2] >= z_front]
    if len(front) < 12:
        front = mid

    y_split = np.median(front[:, 1])
    upper = front[front[:, 1] >= y_split]
    lower = front[front[:, 1] < y_split]
    if len(upper) == 0:
        idx = np.argsort(front[:, 1])
        upper = front[idx[-max(4, len(front) // 3) :]]
    if len(lower) == 0:
        idx = np.argsort(front[:, 1])
        lower = front[idx[: max(4, len(front) // 3)]]

    upper_front = np.mean(upper, axis=0)
    lower_front = np.mean(lower, axis=0)
    teeth_yz = np.array([upper_front[2], upper_front[1]], dtype=np.float64)
    interdental_yz = np.array(
        [(upper_front[2] + lower_front[2]) * 0.5, (upper_front[1] + lower_front[1]) * 0.5],
        dtype=np.float64,
    )

    upper_arch = mid[mid[:, 1] >= np.percentile(mid[:, 1], 65)]
    if len(upper_arch) < 12:
        upper_arch = mid
    z_hi = float(teeth_yz[0] - 0.4)
    z_lo = float(teeth_yz[0] - 1.6)
    alveolar_band = upper_arch[(upper_arch[:, 2] >= z_lo) & (upper_arch[:, 2] <= z_hi)]
    if len(alveolar_band) == 0:
        alveolar_band = upper_arch[upper_arch[:, 2] <= float(teeth_yz[0] - 0.2)]
    if len(alveolar_band) == 0:
        alveolar_yz = np.array([teeth_yz[0] - 1.0, teeth_yz[1] + 0.8], dtype=np.float64)
    else:
        alveolar_peak = alveolar_band[np.argmax(alveolar_band[:, 1])]
        alveolar_yz = np.array([alveolar_peak[2], alveolar_peak[1]], dtype=np.float64)

    return {
        "teeth_yz": teeth_yz,
        "alveolar_yz": alveolar_yz,
        "interdental_yz": interdental_yz,
        "upper_front_yz": np.array([upper_front[2], upper_front[1]], dtype=np.float64),
        "lower_front_yz": np.array([lower_front[2], lower_front[1]], dtype=np.float64),
    }


def apply_target_strategy(
    auto_targets: dict[str, np.ndarray],
    baseline_anchors: np.ndarray,
    spans,
    interdental_mode: str,
    alveolar_tip_alpha: float,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    interdental_mode_norm = str(interdental_mode).strip().lower()
    if interdental_mode_norm not in {"midpoint", "upper_edge"}:
        raise ValueError(f"Unsupported interdental mode: {interdental_mode}")
    alpha = float(np.clip(alveolar_tip_alpha, 0.0, 1.0))

    teeth_yz = auto_targets["teeth_yz"].copy()
    alveolar_yz = auto_targets["alveolar_yz"].copy()
    if interdental_mode_norm == "upper_edge":
        interdental_yz = auto_targets["upper_front_yz"].copy()
    else:
        interdental_yz = auto_targets["interdental_yz"].copy()

    tip_yz = np.asarray(baseline_anchors, dtype=np.float64)[:, 3, [2, 1]]
    alveolar_idx = collect_span_indices(spans, target_class="alveolar", total_frames=len(tip_yz))
    alveolar_tip_reference: Optional[np.ndarray] = None
    if len(alveolar_idx):
        alveolar_tip_reference = np.median(tip_yz[alveolar_idx], axis=0)
        alveolar_yz = (1.0 - alpha) * alveolar_yz + alpha * alveolar_tip_reference

    return teeth_yz, alveolar_yz, alveolar_tip_reference


def build_contact_region(
    yz: np.ndarray,
    extent_x: float,
    extent_z: float,
) -> RegionRect3D:
    # Region is a small horizontal patch centered at (z, y), spanning x and z.
    return RegionRect3D(
        point=np.array([0.0, float(yz[1]), float(yz[0])], dtype=np.float64),
        normal=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        tangent_u=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        tangent_v=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        extent_u=float(extent_x),
        extent_v=float(extent_z),
    )


def raw_to_denorm_anchors(
    raw_ema_4x2: np.ndarray,
    std_4x2: np.ndarray,
    rig_anchors_4x3: np.ndarray,
    scalar: float,
) -> np.ndarray:
    denorm = np.zeros((len(raw_ema_4x2), 4, 3), dtype=np.float64)
    denorm[:, :, 0] = rig_anchors_4x3[:, 0][None, :]
    denorm[:, :, 1] = (
        rig_anchors_4x3[:, 1][None, :]
        + raw_ema_4x2[:, :, 1] * std_4x2[:, 1][None, :] * scalar
    )
    denorm[:, :, 2] = (
        rig_anchors_4x3[:, 2][None, :]
        + raw_ema_4x2[:, :, 0] * std_4x2[:, 0][None, :] * scalar
    )
    return denorm


def denorm_to_raw_anchors(
    denorm_4x3: np.ndarray,
    std_4x2: np.ndarray,
    rig_anchors_4x3: np.ndarray,
    scalar: float,
) -> np.ndarray:
    raw = np.zeros((len(denorm_4x3), 4, 2), dtype=np.float64)
    denom_y = np.maximum(np.abs(std_4x2[:, 1][None, :] * scalar), 1e-6)
    denom_z = np.maximum(np.abs(std_4x2[:, 0][None, :] * scalar), 1e-6)
    raw[:, :, 1] = (denorm_4x3[:, :, 1] - rig_anchors_4x3[:, 1][None, :]) / denom_y
    raw[:, :, 0] = (denorm_4x3[:, :, 2] - rig_anchors_4x3[:, 2][None, :]) / denom_z
    return raw


def summarize_reports(reports: list) -> dict:
    target = [r for r in reports if r.span.phoneme_class in {"alveolar", "interdental"}]
    improved = [
        r
        for r in target
        if r.initial_tip_distance is not None
        and r.final_tip_distance is not None
        and r.final_tip_distance < r.initial_tip_distance
    ]
    worsened = [
        r
        for r in target
        if r.initial_tip_distance is not None
        and r.final_tip_distance is not None
        and r.final_tip_distance > r.initial_tip_distance
    ]
    loss_improved = [
        r
        for r in target
        if r.final_losses.get("total", 0.0) < r.initial_losses.get("total", 0.0)
    ]
    return {
        "target_spans": len(target),
        "tip_distance_improved": len(improved),
        "tip_distance_worsened": len(worsened),
        "total_loss_improved": len(loss_improved),
    }


def build_default_mouth_box(face_model, margin: float) -> MouthBox:
    verts = np.asarray(face_model.neutral_verts, dtype=np.float64)
    region = verts[14062:17039]
    if len(region) == 0:
        region = verts
    min_corner = np.min(region, axis=0) - float(margin)
    max_corner = np.max(region, axis=0) + float(margin)
    return MouthBox(min_corner=min_corner, max_corner=max_corner)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run phoneme L-BFGS-B optimization for one dataset and export render-ready .npy output."
    )
    parser.add_argument("--dataset-id", default="1_wayne_0_75_75")
    parser.add_argument("--speaker-id", default="1")
    parser.add_argument(
        "--beat-root",
        default=str(PROJECT_ROOT / "data" / "beat_cache" / "beat_english_v0.2.1" / "beat_english_v0.2.1"),
    )
    parser.add_argument("--motion-path", default=None, help="Input motion .npy path")
    parser.add_argument("--textgrid-path", default=None, help="Input TextGrid path")
    parser.add_argument("--phone-tier", default="phones")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output .npy path (default: tongue_scripts/outputs/<dataset>_lbfgsb.npy)",
    )
    parser.add_argument("--debug-json-path", default=None)
    parser.add_argument("--debug-csv-path", default=None)
    parser.add_argument("--debug-tip-path", default=None)
    parser.add_argument("--meta-json-path", default=None)

    parser.add_argument("--face-model-dir", default=str(PROJECT_ROOT / "FaceXModel"))
    parser.add_argument(
        "--std-path",
        default=str(SCRIPT_DIR / "normalising_vectors" / "JW13_4points_std.npy"),
    )
    parser.add_argument("--scalar", type=float, default=0.20)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument(
        "--pre-shift-seconds",
        type=float,
        default=0.0,
        help=(
            "Apply this sync shift to input motion BEFORE optimization. "
            "Positive value delays motion (same convention as phoneme_lag_probe recommendation)."
        ),
    )
    parser.add_argument("--pre-shift-pad-mode", choices=["edge", "zero"], default="edge")
    parser.add_argument("--save-pre-shifted-motion", default=None, help="Optional .npy path to save shifted pre-optimization motion.")

    parser.add_argument("--teeth-yz", default=None, help="Manual teeth edge as 'z,y'")
    parser.add_argument("--alveolar-yz", default=None, help="Manual alveolar target as 'z,y'")
    parser.add_argument("--interdental-yz", default=None, help="Manual interdental target as 'z,y'")
    parser.add_argument(
        "--interdental-mode",
        choices=["midpoint", "upper_edge"],
        default="upper_edge",
        help="Default interdental definition when not overridden by CLI/JSON.",
    )
    parser.add_argument(
        "--alveolar-tip-alpha",
        type=float,
        default=0.55,
        help=(
            "Blend ratio in [0,1] that moves alveolar target toward the median tip position "
            "in alveolar spans (0=mesh-only estimate, 1=tip-span median)."
        ),
    )
    parser.add_argument(
        "--target-points-json",
        default=None,
        help=(
            "Path to manual target JSON from tongue_target_point_editor.py. "
            "If omitted, will auto-try tongue_scripts/outputs/<dataset>_manual_tip_targets.json"
        ),
    )
    parser.add_argument("--region-extent-x", type=float, default=8.0)
    parser.add_argument("--region-extent-z", type=float, default=1.2)
    parser.add_argument("--use-mouth-box", action="store_true", default=True)
    parser.add_argument("--no-mouth-box", dest="use_mouth_box", action="store_false")
    parser.add_argument(
        "--mouth-box-margin",
        type=float,
        default=0.5,
        help="Expand auto mouth box by this margin in world units.",
    )

    parser.add_argument("--mode", choices=["delta", "absolute"], default="delta")
    parser.add_argument("--delta-bounds-mm", type=float, default=8.0)
    parser.add_argument("--tip-delta-bounds-mm", type=float, default=None)
    parser.add_argument("--tau-alveolar-mm", type=float, default=2.5)
    parser.add_argument("--tau-interdental-mm", type=float, default=2.0)
    parser.add_argument("--contact-window-fraction", type=float, default=0.4)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--seam-blend-frames", type=int, default=2)
    parser.add_argument("--mm-per-world-unit", type=float, default=1.0)

    parser.add_argument("--lambda-data", type=float, default=1.0)
    parser.add_argument("--lambda-contact-alveolar", type=float, default=3.0)
    parser.add_argument("--lambda-contact-interdental", type=float, default=3.0)
    parser.add_argument("--lambda-contact-other", type=float, default=0.0)
    parser.add_argument("--lambda-smooth", type=float, default=0.5)
    parser.add_argument("--lambda-prior", type=float, default=0.2)
    parser.add_argument("--lambda-compat", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    beat_root = Path(args.beat_root)
    speaker_dir = beat_root / str(args.speaker_id)

    motion_path = Path(args.motion_path) if args.motion_path else (SCRIPT_DIR / "outputs" / f"{args.dataset_id}.npy")
    textgrid_path = Path(args.textgrid_path) if args.textgrid_path else (speaker_dir / f"{args.dataset_id}.TextGrid")
    output_path = Path(args.output_path) if args.output_path else (SCRIPT_DIR / "outputs" / f"{args.dataset_id}_lbfgsb.npy")
    debug_json_path = (
        Path(args.debug_json_path)
        if args.debug_json_path
        else output_path.with_name(output_path.stem + "_debug.json")
    )
    debug_csv_path = (
        Path(args.debug_csv_path)
        if args.debug_csv_path
        else output_path.with_name(output_path.stem + "_summary.csv")
    )
    debug_tip_path = (
        Path(args.debug_tip_path)
        if args.debug_tip_path
        else output_path.with_name(output_path.stem + "_tip.csv")
    )
    meta_json_path = (
        Path(args.meta_json_path)
        if args.meta_json_path
        else output_path.with_name(output_path.stem + "_meta.json")
    )
    default_target_json = SCRIPT_DIR / "outputs" / f"{args.dataset_id}_manual_tip_targets.json"
    target_points_json_path = (
        Path(args.target_points_json)
        if args.target_points_json
        else (default_target_json if default_target_json.is_file() else None)
    )

    if not motion_path.is_file():
        raise SystemExit(f"Input motion not found: {motion_path}")
    if not textgrid_path.is_file():
        raise SystemExit(f"TextGrid not found: {textgrid_path}")
    if not Path(args.std_path).is_file():
        raise SystemExit(f"Std vector not found: {args.std_path}")
    if not Path(args.face_model_dir).is_dir():
        raise SystemExit(f"Face model directory not found: {args.face_model_dir}")

    print("=" * 80)
    print("PHONEME L-BFGS-B OPTIMIZER")
    print("=" * 80)
    print(f"dataset_id: {args.dataset_id}")
    print(f"motion:     {motion_path}")
    print(f"textgrid:   {textgrid_path}")
    print(f"output:     {output_path}")
    print("-" * 80)

    raw_motion = np.load(motion_path)
    if raw_motion.ndim != 2 or raw_motion.shape[1] < 8:
        raise SystemExit(f"Expected motion shape (T, >=8), got {raw_motion.shape}")
    shift_frames = int(round(float(args.pre_shift_seconds) * float(args.fps)))
    if shift_frames != 0:
        raw_motion = shift_sequence(raw_motion, shift_frames=shift_frames, pad_mode=args.pre_shift_pad_mode)
        print(f"pre-shift:      {float(args.pre_shift_seconds):+.3f}s ({shift_frames:+d} frames, mode={args.pre_shift_pad_mode})")
        if args.save_pre_shifted_motion:
            save_path = Path(args.save_pre_shifted_motion)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, raw_motion)
            print(f"saved shifted:  {save_path}")

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
            "std_scalar": float(args.scalar),
        },
    )

    std_raw = np.load(args.std_path).flatten()[:8].reshape(4, 2).astype(np.float64)
    raw_ema = raw_motion[:, :8].reshape(-1, 4, 2).astype(np.float64)
    baseline_anchors = raw_to_denorm_anchors(
        raw_ema_4x2=raw_ema,
        std_4x2=std_raw,
        rig_anchors_4x3=tongue_rig.anchors.astype(np.float64),
        scalar=float(args.scalar),
    )

    spans = parse_textgrid(
        textgrid_path=textgrid_path,
        fps=float(args.fps),
        tier_name=args.phone_tier,
        total_frames=len(baseline_anchors),
    )
    if not spans:
        raise SystemExit(f"No non-empty phoneme spans found in tier '{args.phone_tier}'")

    auto_targets = estimate_oral_targets_yz(face_model)
    strategy_teeth_yz, strategy_alveolar_yz, alveolar_tip_reference = apply_target_strategy(
        auto_targets=auto_targets,
        baseline_anchors=baseline_anchors,
        spans=spans,
        interdental_mode=args.interdental_mode,
        alveolar_tip_alpha=float(args.alveolar_tip_alpha),
    )
    strategy_interdental_yz = (
        auto_targets["upper_front_yz"].copy() if args.interdental_mode == "upper_edge" else auto_targets["interdental_yz"].copy()
    )
    json_targets: dict[str, np.ndarray] = {}
    if target_points_json_path is not None:
        if not target_points_json_path.is_file():
            raise SystemExit(f"Target points JSON not found: {target_points_json_path}")
        json_targets = load_target_points_json(target_points_json_path)

    manual_teeth_yz = parse_yz(args.teeth_yz)
    manual_alveolar_yz = parse_yz(args.alveolar_yz)
    manual_interdental_yz = parse_yz(args.interdental_yz)
    teeth_yz = (
        manual_teeth_yz
        if manual_teeth_yz is not None
        else (json_targets["teeth_yz"] if "teeth_yz" in json_targets else strategy_teeth_yz)
    )
    alveolar_yz = (
        manual_alveolar_yz
        if manual_alveolar_yz is not None
        else (json_targets["alveolar_yz"] if "alveolar_yz" in json_targets else strategy_alveolar_yz)
    )
    interdental_yz = (
        manual_interdental_yz
        if manual_interdental_yz is not None
        else (json_targets["interdental_yz"] if "interdental_yz" in json_targets else strategy_interdental_yz)
    )
    source_teeth = "cli" if manual_teeth_yz is not None else ("json" if "teeth_yz" in json_targets else "auto")
    source_alveolar = "cli" if manual_alveolar_yz is not None else ("json" if "alveolar_yz" in json_targets else "auto")
    source_interdental = (
        "cli" if manual_interdental_yz is not None else ("json" if "interdental_yz" in json_targets else "auto")
    )

    alveolar_region = build_contact_region(
        yz=alveolar_yz,
        extent_x=float(args.region_extent_x),
        extent_z=float(args.region_extent_z),
    )
    interdental_region = build_contact_region(
        yz=interdental_yz,
        extent_x=float(args.region_extent_x),
        extent_z=float(args.region_extent_z),
    )
    print(f"teeth_yz:       {teeth_yz.tolist()}")
    print(f"alveolar_yz:    {alveolar_yz.tolist()}")
    print(f"interdental_yz: {interdental_yz.tolist()}")
    print(f"target source:  teeth={source_teeth}, alveolar={source_alveolar}, interdental={source_interdental}")
    print(f"strategy:       interdental_mode={args.interdental_mode}, alveolar_tip_alpha={float(args.alveolar_tip_alpha):.3f}")
    if alveolar_tip_reference is not None:
        print(f"alv_tip_ref_yz: {alveolar_tip_reference.tolist()}")
    if target_points_json_path is not None:
        print(f"target json:    {target_points_json_path}")
    print(f"upper_front_yz: {auto_targets['upper_front_yz'].tolist()}")
    print(f"lower_front_yz: {auto_targets['lower_front_yz'].tolist()}")
    mouth_box = build_default_mouth_box(face_model, margin=float(args.mouth_box_margin)) if args.use_mouth_box else None

    config = OptimizationConfig(
        mode=args.mode,
        fps=float(args.fps),
        delta_bounds_mm=float(args.delta_bounds_mm),
        tip_delta_bounds_mm=args.tip_delta_bounds_mm,
        tau_alveolar_mm=float(args.tau_alveolar_mm),
        tau_interdental_mm=float(args.tau_interdental_mm),
        contact_window_fraction=float(args.contact_window_fraction),
        maxiter=int(args.maxiter),
        seam_blend_frames=int(args.seam_blend_frames),
        mm_per_world_unit=float(args.mm_per_world_unit),
    )
    weights = OptimizationWeights(
        lambda_data=float(args.lambda_data),
        lambda_contact_alveolar=float(args.lambda_contact_alveolar),
        lambda_contact_interdental=float(args.lambda_contact_interdental),
        lambda_contact_other=float(args.lambda_contact_other),
        lambda_smooth=float(args.lambda_smooth),
        lambda_prior=float(args.lambda_prior),
        lambda_compat=float(args.lambda_compat),
    )
    rig_forward = FaceKitRigForwardAdapter(tongue_rig=tongue_rig, face_model=None, include_face_mesh=False)

    optimized_anchors, reports = optimize_utterance(
        frames=np.arange(len(baseline_anchors), dtype=np.int32),
        wavlm_anchor_pred=baseline_anchors,
        jaw_controls=[0.0] * len(baseline_anchors),
        face_controls=[{} for _ in range(len(baseline_anchors))],
        phoneme_spans=spans,
        rig_forward=rig_forward,
        config=config,
        weights=weights,
        alveolar_region=alveolar_region,
        interdental_region=interdental_region,
        mouth_box=mouth_box,
    )

    raw_opt = denorm_to_raw_anchors(
        denorm_4x3=optimized_anchors,
        std_4x2=std_raw,
        rig_anchors_4x3=tongue_rig.anchors.astype(np.float64),
        scalar=float(args.scalar),
    )
    output_motion = raw_motion.copy()
    output_motion[:, :8] = raw_opt.reshape(len(raw_opt), 8).astype(raw_motion.dtype, copy=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, output_motion)
    export_debug_report(
        span_results=reports,
        json_output_path=debug_json_path,
        summary_csv_path=debug_csv_path,
        tip_output_path=debug_tip_path,
        baseline_anchor_traj=baseline_anchors,
        optimized_anchor_traj=optimized_anchors,
    )

    summary = summarize_reports(reports)
    meta_payload = {
        "dataset_id": args.dataset_id,
        "speaker_id": str(args.speaker_id),
        "input_motion": str(motion_path.resolve()),
        "output_motion": str(output_path.resolve()),
        "textgrid_path": str(textgrid_path.resolve()),
        "n_frames": int(len(baseline_anchors)),
        "n_spans": int(len(spans)),
        "regions": {
            "teeth_yz": teeth_yz.tolist(),
            "alveolar_yz": alveolar_yz.tolist(),
            "interdental_yz": interdental_yz.tolist(),
            "extent_x": float(args.region_extent_x),
            "extent_z": float(args.region_extent_z),
        },
        "target_estimation": {
            "upper_front_yz": auto_targets["upper_front_yz"].tolist(),
            "lower_front_yz": auto_targets["lower_front_yz"].tolist(),
            "teeth_source_vertices": [14062, 16611],
            "notes": "Auto targets estimated from sagittal gums/teeth region only (tongue excluded).",
        },
        "target_strategy": {
            "interdental_mode": str(args.interdental_mode),
            "alveolar_tip_alpha": float(args.alveolar_tip_alpha),
            "strategy_teeth_yz": strategy_teeth_yz.tolist(),
            "strategy_alveolar_yz": strategy_alveolar_yz.tolist(),
            "strategy_interdental_yz": strategy_interdental_yz.tolist(),
            "alveolar_tip_reference_yz": (
                alveolar_tip_reference.tolist() if alveolar_tip_reference is not None else None
            ),
        },
        "target_selection": {
            "teeth_source": source_teeth,
            "alveolar_source": source_alveolar,
            "interdental_source": source_interdental,
            "target_points_json": str(target_points_json_path.resolve()) if target_points_json_path is not None else None,
        },
        "pre_shift": {
            "seconds": float(args.pre_shift_seconds),
            "frames": int(shift_frames),
            "pad_mode": str(args.pre_shift_pad_mode),
            "saved_shifted_motion": str(Path(args.save_pre_shifted_motion).resolve()) if args.save_pre_shifted_motion else None,
        },
        "mouth_box": (
            {
                "min_corner": mouth_box.min_corner.tolist(),
                "max_corner": mouth_box.max_corner.tolist(),
                "margin": float(args.mouth_box_margin),
            }
            if mouth_box is not None
            else None
        ),
        "config": asdict(config),
        "weights": asdict(weights),
        "summary": summary,
        "artifacts": {
            "debug_json": str(debug_json_path.resolve()),
            "debug_csv": str(debug_csv_path.resolve()),
            "debug_tip": str(debug_tip_path.resolve()),
        },
    }
    meta_json_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    print("-" * 80)
    print("Optimization done.")
    print(f"saved motion:  {output_path}")
    print(f"saved debug:   {debug_json_path}")
    print(f"saved summary: {debug_csv_path}")
    print(f"target spans:  {summary['target_spans']}")
    print(f"tip improved:  {summary['tip_distance_improved']}")
    print(f"loss improved: {summary['total_loss_improved']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
