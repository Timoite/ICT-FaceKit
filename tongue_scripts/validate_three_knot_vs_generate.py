#!/usr/bin/env python3
"""
Validate the 3-knot-per-span parameterization against the current rendering path.

This script uses the same EMA loading + tongue rig deformation route used by
`generate_tongue_animation.py`, then compares:
1) baseline frame-wise anchors
2) span-wise 3-knot reconstructed anchors

It reports errors in both anchor space and deformed tongue mesh space.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
import importlib
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _append_project_root() -> None:
    import sys

    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


_append_project_root()

from tongue_scripts.phoneme_lbfgsb_optimizer import (  # noqa: E402
    build_knots_from_traj,
    interpolate_knots,
    parse_textgrid,
    span_positions_from_frames,
)
TONGUE_ANIM_DIR = PROJECT_ROOT / "tongue_scripts" / "tongue_animation"
if str(TONGUE_ANIM_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(TONGUE_ANIM_DIR))

face_model_io_trimesh = importlib.import_module("face_model_io_trimesh")
generate_tongue_animation = importlib.import_module("generate_tongue_animation")

load_face_model_trimesh = face_model_io_trimesh.load_face_model_trimesh
FaceKitTongueRig = generate_tongue_animation.FaceKitTongueRig
load_ema_motion = generate_tongue_animation.load_ema_motion
TONGUE_SLICE = generate_tongue_animation.TONGUE_SLICE
ANCHOR_INDICES = generate_tongue_animation.ANCHOR_INDICES
BONE_INDICES = generate_tongue_animation.BONE_INDICES
TONGUE_CONFIG = generate_tongue_animation.TONGUE_CONFIG


@dataclass(frozen=True)
class AnchorMetrics:
    rmse_world: float
    mae_world: float
    max_abs_world: float
    tip_rmse_world: float
    tip_max_abs_world: float
    rmse_per_anchor_world: list[float]


@dataclass(frozen=True)
class MeshMetrics:
    sampled_frames: int
    mean_vertex_rmse_world: float
    p95_vertex_l2_world: float
    max_vertex_l2_world: float
    mean_vertex_l2_world: float


@dataclass(frozen=True)
class ValidationReport:
    dataset_id: str
    n_frames: int
    n_spans: int
    n_spans_used: int
    span_stats: dict[str, Any]
    anchor_metrics_world: AnchorMetrics
    mesh_metrics_world: MeshMetrics
    mm_per_world_unit: float

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        mm = float(self.mm_per_world_unit)
        payload["anchor_metrics_mm"] = _convert_dict_values_to_mm(payload["anchor_metrics_world"], mm)
        payload["mesh_metrics_mm"] = _convert_dict_values_to_mm(payload["mesh_metrics_world"], mm)
        return payload


def _convert_dict_values_to_mm(data: dict[str, Any], mm_per_world_unit: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        out_key = key[:-6] + "_mm" if key.endswith("_world") else key
        if key == "sampled_frames":
            out[out_key] = value
            continue
        if isinstance(value, (float, int)):
            out[out_key] = float(value) * mm_per_world_unit
            continue
        if isinstance(value, list):
            out[out_key] = [float(v) * mm_per_world_unit for v in value]
            continue
        out[out_key] = value
    return out


def _build_reconstruction(
    baseline_anchor_traj: np.ndarray,
    spans: list[Any],
    frames: np.ndarray,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    recon = baseline_anchor_traj.copy()
    used = 0
    span_lengths: list[int] = []

    for span in spans:
        positions = span_positions_from_frames(span, frames)
        if len(positions) == 0:
            continue
        used += 1
        span_lengths.append(int(len(positions)))
        knots = build_knots_from_traj(baseline_anchor_traj, positions)
        recon[positions] = interpolate_knots(knots, len(positions))

    stats = {
        "min_span_frames": int(np.min(span_lengths)) if span_lengths else 0,
        "median_span_frames": float(np.median(span_lengths)) if span_lengths else 0.0,
        "mean_span_frames": float(np.mean(span_lengths)) if span_lengths else 0.0,
        "max_span_frames": int(np.max(span_lengths)) if span_lengths else 0,
    }
    return recon, used, stats


def _compute_anchor_metrics(baseline: np.ndarray, reconstructed: np.ndarray) -> AnchorMetrics:
    err = reconstructed - baseline
    abs_err = np.abs(err)
    sq_err = np.square(err)

    rmse = float(np.sqrt(np.mean(sq_err)))
    mae = float(np.mean(abs_err))
    max_abs = float(np.max(abs_err))
    tip_sq = sq_err[:, 3, :]
    tip_abs = abs_err[:, 3, :]
    tip_rmse = float(np.sqrt(np.mean(tip_sq)))
    tip_max = float(np.max(tip_abs))

    rmse_per_anchor = [
        float(np.sqrt(np.mean(sq_err[:, a, :])))
        for a in range(4)
    ]

    return AnchorMetrics(
        rmse_world=rmse,
        mae_world=mae,
        max_abs_world=max_abs,
        tip_rmse_world=tip_rmse,
        tip_max_abs_world=tip_max,
        rmse_per_anchor_world=rmse_per_anchor,
    )


def _sample_frame_indices(n_frames: int, max_sample_frames: int | None) -> np.ndarray:
    if n_frames <= 0:
        return np.zeros((0,), dtype=np.int32)
    if max_sample_frames is None or max_sample_frames <= 0 or max_sample_frames >= n_frames:
        return np.arange(n_frames, dtype=np.int32)
    idx = np.linspace(0, n_frames - 1, max_sample_frames, dtype=np.int32)
    return np.unique(idx)


def _compute_mesh_metrics(
    tongue_rig: FaceKitTongueRig,
    baseline: np.ndarray,
    reconstructed: np.ndarray,
    frame_indices: np.ndarray,
) -> MeshMetrics:
    frame_rmse: list[float] = []
    all_vertex_l2: list[np.ndarray] = []

    for i in frame_indices:
        base_mesh, _, _ = tongue_rig.deform(baseline[i])
        recon_mesh, _, _ = tongue_rig.deform(reconstructed[i])
        diff = recon_mesh - base_mesh
        diff_l2 = np.linalg.norm(diff, axis=1)
        frame_rmse.append(float(np.sqrt(np.mean(np.square(diff)))))
        all_vertex_l2.append(diff_l2)

    if not all_vertex_l2:
        return MeshMetrics(
            sampled_frames=0,
            mean_vertex_rmse_world=0.0,
            p95_vertex_l2_world=0.0,
            max_vertex_l2_world=0.0,
            mean_vertex_l2_world=0.0,
        )

    stacked = np.concatenate(all_vertex_l2, axis=0)
    return MeshMetrics(
        sampled_frames=int(len(frame_indices)),
        mean_vertex_rmse_world=float(np.mean(frame_rmse)),
        p95_vertex_l2_world=float(np.percentile(stacked, 95)),
        max_vertex_l2_world=float(np.max(stacked)),
        mean_vertex_l2_world=float(np.mean(stacked)),
    )


def _default_paths(dataset_id: str, speaker_id: str) -> tuple[Path, Path]:
    beat_root = PROJECT_ROOT / "data" / "beat_cache" / "beat_english_v0.2.1" / "beat_english_v0.2.1"
    speaker_dir = beat_root / speaker_id
    textgrid_path = speaker_dir / f"{dataset_id}.TextGrid"
    motion_path = PROJECT_ROOT / "tongue_scripts" / "outputs" / f"{dataset_id}.npy"
    return textgrid_path, motion_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare 3-knot phoneme span reconstruction against generate_tongue_animation deformation behavior."
    )
    parser.add_argument("--dataset-id", required=True, help="Dataset id stem, e.g. 1_wayne_0_75_75")
    parser.add_argument("--speaker-id", default="1", help="Speaker id used to resolve default TextGrid path")
    parser.add_argument("--textgrid-path", default=None, help="Optional explicit TextGrid path")
    parser.add_argument("--motion-path", default=None, help="Optional explicit WavLM output .npy path")
    parser.add_argument(
        "--std-path",
        default=str(PROJECT_ROOT / "tongue_scripts" / "normalising_vectors" / "JW13_4points_std.npy"),
    )
    parser.add_argument("--face-model-dir", default=str(PROJECT_ROOT / "FaceXModel"))
    parser.add_argument("--fps", type=float, default=50.0, help="FPS used for TextGrid-to-frame mapping")
    parser.add_argument("--std-scalar", type=float, default=float(TONGUE_CONFIG["std_scalar"]))
    parser.add_argument(
        "--mm-per-world-unit",
        type=float,
        default=1.0,
        help="Used only for mm conversion in the report (world_value * mm_per_world_unit).",
    )
    parser.add_argument(
        "--max-sample-frames",
        type=int,
        default=300,
        help="Max frames used for mesh-space comparison (uniform sampling).",
    )
    parser.add_argument("--report-json", default=None, help="Optional path to save JSON report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    default_textgrid, default_motion = _default_paths(args.dataset_id, args.speaker_id)
    textgrid_path = Path(args.textgrid_path) if args.textgrid_path else default_textgrid
    motion_path = Path(args.motion_path) if args.motion_path else default_motion
    std_path = Path(args.std_path)
    face_model_dir = Path(args.face_model_dir)

    if not textgrid_path.is_file():
        raise SystemExit(f"TextGrid not found: {textgrid_path}")
    if not motion_path.is_file():
        raise SystemExit(f"Motion file not found: {motion_path}")
    if not std_path.is_file():
        raise SystemExit(f"Std vector not found: {std_path}")
    if not face_model_dir.is_dir():
        raise SystemExit(f"Face model dir not found: {face_model_dir}")

    face_model = load_face_model_trimesh(face_model_dir)
    tongue_rig = FaceKitTongueRig(
        vertices=face_model.neutral_verts,
        faces=face_model.faces,
        tongue_slice=TONGUE_SLICE,
        anchor_indices_global=ANCHOR_INDICES,
        bone_ends_global=BONE_INDICES,
        config=dict(TONGUE_CONFIG),
    )
    baseline = load_ema_motion(
        motion_path,
        std_path,
        tongue_rig.anchors,
        scalar=float(args.std_scalar),
    ).astype(np.float64)

    spans = parse_textgrid(textgrid_path, fps=float(args.fps), total_frames=len(baseline))
    frames = np.arange(len(baseline), dtype=np.int32)
    reconstructed, used_spans, span_stats = _build_reconstruction(baseline, spans, frames)
    anchor_metrics = _compute_anchor_metrics(baseline, reconstructed)

    sampled_frames = _sample_frame_indices(len(baseline), args.max_sample_frames)
    mesh_metrics = _compute_mesh_metrics(tongue_rig, baseline, reconstructed, sampled_frames)

    report = ValidationReport(
        dataset_id=args.dataset_id,
        n_frames=int(len(baseline)),
        n_spans=int(len(spans)),
        n_spans_used=int(used_spans),
        span_stats=span_stats,
        anchor_metrics_world=anchor_metrics,
        mesh_metrics_world=mesh_metrics,
        mm_per_world_unit=float(args.mm_per_world_unit),
    )
    payload = report.as_dict()

    print("=" * 80)
    print("Three-Knot Validation (render path aligned with generate_tongue_animation.py)")
    print("=" * 80)
    print(f"dataset_id: {args.dataset_id}")
    print(f"textgrid:   {textgrid_path}")
    print(f"motion:     {motion_path}")
    print(f"frames:     {report.n_frames}")
    print(f"spans:      {report.n_spans} (used: {report.n_spans_used})")
    print("-" * 80)
    print("Anchor errors (world units):")
    print(f"  RMSE:      {anchor_metrics.rmse_world:.6f}")
    print(f"  MAE:       {anchor_metrics.mae_world:.6f}")
    print(f"  Max abs:   {anchor_metrics.max_abs_world:.6f}")
    print(f"  Tip RMSE:  {anchor_metrics.tip_rmse_world:.6f}")
    print(f"  Tip Max:   {anchor_metrics.tip_max_abs_world:.6f}")
    print(f"  Per-anchor RMSE [back,dorsum,blade,tip]: {anchor_metrics.rmse_per_anchor_world}")
    print("-" * 80)
    print("Deformed mesh errors (world units):")
    print(f"  sampled frames: {mesh_metrics.sampled_frames}")
    print(f"  mean vertex RMSE: {mesh_metrics.mean_vertex_rmse_world:.6f}")
    print(f"  mean vertex L2:   {mesh_metrics.mean_vertex_l2_world:.6f}")
    print(f"  p95 vertex L2:    {mesh_metrics.p95_vertex_l2_world:.6f}")
    print(f"  max vertex L2:    {mesh_metrics.max_vertex_l2_world:.6f}")
    print("-" * 80)
    print("Same metrics converted to mm:")
    mm = report.mm_per_world_unit
    print(f"  mm_per_world_unit: {mm}")
    print(f"  anchor RMSE (mm):  {anchor_metrics.rmse_world * mm:.6f}")
    print(f"  tip RMSE (mm):     {anchor_metrics.tip_rmse_world * mm:.6f}")
    print(f"  mesh p95 L2 (mm):  {mesh_metrics.p95_vertex_l2_world * mm:.6f}")
    print(f"  mesh max L2 (mm):  {mesh_metrics.max_vertex_l2_world * mm:.6f}")
    print("=" * 80)

    if args.report_json:
        out_path = Path(args.report_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
