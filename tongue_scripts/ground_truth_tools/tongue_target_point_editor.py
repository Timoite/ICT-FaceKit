#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "matplotlib>=3.8",
#     "trimesh",
# ]
# ///
"""
Interactive sagittal target-point editor for phoneme tip optimization.

This tool lets you manually place:
- teeth_yz
- alveolar_yz
- interdental_yz

on a Z-Y slice aligned with generate_tongue_animation.py.
The saved JSON can be used to drive optimizer target points directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, RadioButtons, Slider

SCRIPT_DIR = Path(__file__).parent.resolve()
TONGUE_SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TONGUE_SCRIPTS_DIR.parent
TONGUE_ANIMATION_DIR = TONGUE_SCRIPTS_DIR / "tongue_animation"
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
    process_beat_data,
)
from tongue_scripts.phoneme_lbfgsb_optimizer import parse_textgrid

MOUTH_REGION = slice(14062, 17039)


@dataclass
class TargetPoints:
    teeth_yz: np.ndarray
    alveolar_yz: np.ndarray
    interdental_yz: np.ndarray

    def copy(self) -> "TargetPoints":
        return TargetPoints(
            teeth_yz=self.teeth_yz.copy(),
            alveolar_yz=self.alveolar_yz.copy(),
            interdental_yz=self.interdental_yz.copy(),
        )


def parse_yz(text: Optional[str]) -> Optional[np.ndarray]:
    if not text:
        return None
    z_s, y_s = text.split(",")
    return np.array([float(z_s), float(y_s)], dtype=np.float64)


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


def shift_sequence(seq: np.ndarray, shift_frames: int, pad_mode: str = "edge") -> np.ndarray:
    if shift_frames == 0:
        return seq.copy()
    if len(seq) == 0:
        return seq.copy()

    if pad_mode == "edge":
        pad_start = seq[:1]
        pad_end = seq[-1:]
    elif pad_mode == "zero":
        pad_shape = list(seq.shape)
        pad_shape[0] = 1
        pad_start = np.zeros(pad_shape, dtype=seq.dtype)
        pad_end = np.zeros(pad_shape, dtype=seq.dtype)
    else:
        raise ValueError(f"Unsupported pad_mode: {pad_mode}")

    if shift_frames > 0:
        pad = np.repeat(pad_start, shift_frames, axis=0)
        shifted = np.concatenate([pad, seq[:-shift_frames]], axis=0)
    else:
        shift_frames = abs(shift_frames)
        pad = np.repeat(pad_end, shift_frames, axis=0)
        shifted = np.concatenate([seq[shift_frames:], pad], axis=0)
    return shifted


def estimate_oral_targets_yz(face_model) -> tuple[TargetPoints, np.ndarray, np.ndarray]:
    verts = np.asarray(face_model.neutral_verts, dtype=np.float64)
    gums = verts[14062:16611]
    if len(gums) == 0:
        p = np.mean(verts, axis=0)
        yz = np.array([p[2], p[1]], dtype=np.float64)
        return (
            TargetPoints(
                teeth_yz=yz.copy(),
                alveolar_yz=np.array([yz[0] - 1.0, yz[1] + 0.8], dtype=np.float64),
                interdental_yz=yz.copy(),
            ),
            yz.copy(),
            yz.copy(),
        )

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

    return (
        TargetPoints(
            teeth_yz=teeth_yz,
            alveolar_yz=alveolar_yz,
            interdental_yz=interdental_yz,
        ),
        np.array([upper_front[2], upper_front[1]], dtype=np.float64),
        np.array([lower_front[2], lower_front[1]], dtype=np.float64),
    )


def collect_span_indices(spans, target_class: str, total_frames: int) -> np.ndarray:
    idx: list[int] = []
    for sp in spans:
        if sp.phoneme_class != target_class:
            continue
        idx.extend(range(sp.start_frame, sp.end_frame + 1))
    if not idx:
        return np.zeros((0,), dtype=np.int32)
    return np.array(sorted(set(i for i in idx if 0 <= i < total_frames)), dtype=np.int32)


def apply_target_strategy(
    base_auto_targets: TargetPoints,
    upper_front_yz: np.ndarray,
    tip_yz: np.ndarray,
    alveolar_span_idx: np.ndarray,
    interdental_mode: str,
    alveolar_tip_alpha: float,
) -> tuple[TargetPoints, Optional[np.ndarray]]:
    mode = str(interdental_mode).strip().lower()
    if mode not in {"midpoint", "upper_edge"}:
        raise ValueError(f"Unsupported interdental_mode: {interdental_mode}")
    alpha = float(np.clip(alveolar_tip_alpha, 0.0, 1.0))

    out = base_auto_targets.copy()
    if mode == "upper_edge":
        out.interdental_yz = np.asarray(upper_front_yz, dtype=np.float64).copy()

    alv_ref = None
    if len(alveolar_span_idx):
        alv_ref = np.median(np.asarray(tip_yz, dtype=np.float64)[alveolar_span_idx], axis=0)
        out.alveolar_yz = (1.0 - alpha) * out.alveolar_yz + alpha * alv_ref
    return out, alv_ref


def compute_outline(mesh_pts: np.ndarray, num_bins: int = 80) -> tuple[np.ndarray, np.ndarray]:
    if len(mesh_pts) < 4:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty

    z = mesh_pts[:, 0]
    y = mesh_pts[:, 1]
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    if not np.isfinite(z_min) or not np.isfinite(z_max) or abs(z_max - z_min) < 1e-6:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty

    edges = np.linspace(z_min, z_max, num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    upper: list[list[float]] = []
    lower: list[list[float]] = []
    for idx in range(num_bins):
        if idx == num_bins - 1:
            mask = (z >= edges[idx]) & (z <= edges[idx + 1])
        else:
            mask = (z >= edges[idx]) & (z < edges[idx + 1])
        if not np.any(mask):
            continue
        y_bin = y[mask]
        upper.append([centers[idx], float(np.max(y_bin))])
        lower.append([centers[idx], float(np.min(y_bin))])

    return np.asarray(upper, dtype=np.float64), np.asarray(lower, dtype=np.float64)


def compute_zoom_bounds(
    mouth_pts_zy: np.ndarray,
    tongue_pts_zy: np.ndarray,
    tip_yz: np.ndarray,
    pad_z: float = 0.8,
    pad_y: float = 0.8,
) -> tuple[float, float, float, float]:
    all_pts = np.concatenate([mouth_pts_zy, tongue_pts_zy, tip_yz], axis=0)
    z_min, y_min = np.min(all_pts, axis=0)
    z_max, y_max = np.max(all_pts, axis=0)
    return z_min - pad_z, z_max + pad_z, y_min - pad_y, y_max + pad_y


class TongueTargetPointEditor:
    def __init__(
        self,
        dataset_id: str,
        speaker_id: str,
        beat_root: Path,
        face_model_dir: Path,
        motion_path: Path,
        compare_motion_path: Optional[Path],
        textgrid_path: Optional[Path],
        std_path: Path,
        scalar: float,
        fps: float,
        face_shift_seconds: float,
        region_extent_z: float,
        tau_alveolar_mm: float,
        tau_interdental_mm: float,
        output_path: Path,
        teeth_yz: Optional[np.ndarray],
        alveolar_yz: Optional[np.ndarray],
        interdental_yz: Optional[np.ndarray],
        interdental_mode: str,
        alveolar_tip_alpha: float,
    ) -> None:
        self.dataset_id = dataset_id
        self.speaker_id = speaker_id
        self.beat_root = beat_root
        self.face_model_dir = face_model_dir
        self.motion_path = motion_path
        self.compare_motion_path = compare_motion_path
        self.textgrid_path = textgrid_path
        self.std_path = std_path
        self.scalar = float(scalar)
        self.fps = float(fps)
        self.face_shift_seconds = float(face_shift_seconds)
        self.region_extent_z = float(region_extent_z)
        self.tau_alveolar_mm = float(tau_alveolar_mm)
        self.tau_interdental_mm = float(tau_interdental_mm)
        self.interdental_mode = str(interdental_mode)
        self.alveolar_tip_alpha = float(np.clip(alveolar_tip_alpha, 0.0, 1.0))
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        print("Loading face model ...")
        self.face_model = load_face_model_trimesh(str(self.face_model_dir))
        self.verts = np.asarray(self.face_model.neutral_verts, dtype=np.float64)
        self.json_path = self.beat_root / str(self.speaker_id) / f"{self.dataset_id}.json"
        self.face_seq: Optional[np.ndarray] = None
        if self.json_path.is_file():
            print("Loading BEAT face motion ...")
            self.face_seq = process_beat_data(str(self.json_path), self.face_model, target_fps=int(round(self.fps)))
            face_shift_frames = int(round(self.face_shift_seconds * self.fps))
            if face_shift_frames != 0:
                # Positive face_shift_seconds advances the face earlier relative to tongue/audio.
                self.face_seq = shift_sequence(self.face_seq, shift_frames=-face_shift_frames, pad_mode="edge")

        print("Preparing tongue rig ...")
        self.tongue_rig = FaceKitTongueRig(
            self.face_model.neutral_verts,
            self.face_model.faces,
            TONGUE_SLICE,
            ANCHOR_INDICES,
            BONE_INDICES,
            {
                "rotation_deg": 5,
                "thickness": 1.2,
                "shift_y": 0,
                "shift_z": 0,
                "std_scalar": self.scalar,
            },
        )

        print("Loading motion ...")
        raw = np.load(self.motion_path)
        if raw.ndim != 2 or raw.shape[1] < 8:
            raise ValueError(f"Expected motion shape (T, >=8), got {raw.shape}")
        raw_ema = raw[:, :8].reshape(-1, 4, 2).astype(np.float64)
        std = np.load(self.std_path).flatten()[:8].reshape(4, 2).astype(np.float64)
        self.anchor_seq = raw_to_denorm_anchors(raw_ema, std, self.tongue_rig.anchors.astype(np.float64), self.scalar)
        self.tip_yz = self.anchor_seq[:, 3][:, [2, 1]]
        self.current_frame = 0
        self.max_frame = len(self.tip_yz) - 1
        self.compare_anchor_seq: Optional[np.ndarray] = None
        self.compare_tip_yz: Optional[np.ndarray] = None
        if self.compare_motion_path is not None:
            raw_cmp = np.load(self.compare_motion_path)
            if raw_cmp.ndim != 2 or raw_cmp.shape[1] < 8:
                raise ValueError(f"Expected compare motion shape (T, >=8), got {raw_cmp.shape}")
            raw_cmp_ema = raw_cmp[:, :8].reshape(-1, 4, 2).astype(np.float64)
            self.compare_anchor_seq = raw_to_denorm_anchors(
                raw_cmp_ema, std, self.tongue_rig.anchors.astype(np.float64), self.scalar
            )
            self.compare_tip_yz = self.compare_anchor_seq[:, 3][:, [2, 1]]
            self.max_frame = min(self.max_frame, len(self.compare_anchor_seq) - 1)

        self.alveolar_span_idx = np.zeros((0,), dtype=np.int32)
        self.interdental_span_idx = np.zeros((0,), dtype=np.int32)
        self.phoneme_spans = []
        if self.textgrid_path is not None and self.textgrid_path.exists():
            self.phoneme_spans = parse_textgrid(
                self.textgrid_path, fps=self.fps, tier_name="phones", total_frames=len(self.tip_yz)
            )
            self.alveolar_span_idx = collect_span_indices(self.phoneme_spans, "alveolar", len(self.tip_yz))
            self.interdental_span_idx = collect_span_indices(self.phoneme_spans, "interdental", len(self.tip_yz))

        self.base_auto_targets, self.upper_front_yz, self.lower_front_yz = estimate_oral_targets_yz(self.face_model)
        self.auto_targets, self.alveolar_tip_reference_yz = apply_target_strategy(
            base_auto_targets=self.base_auto_targets,
            upper_front_yz=self.upper_front_yz,
            tip_yz=self.tip_yz,
            alveolar_span_idx=self.alveolar_span_idx,
            interdental_mode=self.interdental_mode,
            alveolar_tip_alpha=self.alveolar_tip_alpha,
        )
        self.targets = self.auto_targets.copy()
        if teeth_yz is not None:
            self.targets.teeth_yz = teeth_yz.copy()
        if alveolar_yz is not None:
            self.targets.alveolar_yz = alveolar_yz.copy()
        if interdental_yz is not None:
            self.targets.interdental_yz = interdental_yz.copy()

        if self.output_path.exists():
            self._load_from_json(self.output_path)

        self.target_names = ["teeth_yz", "alveolar_yz", "interdental_yz"]
        self.active_target = "alveolar_yz"
        self.dragging_target: Optional[str] = None
        self.drag_radius = 0.35
        self.zoomed_mouth_box = True
        self.zoom_padding = 0.8

        self.face_midline_mask = np.abs(self.verts[:, 0]) < 0.5
        self.face_midline_mask[TONGUE_SLICE] = False

        self.mouth_region = self.verts[MOUTH_REGION]
        self.mouth_midline_mask = np.abs(self.verts[:, 0]) < 0.5
        self.mouth_midline_mask[: MOUTH_REGION.start] = False
        self.mouth_midline_mask[TONGUE_SLICE] = False
        self.mouth_region_mid = self.mouth_region[np.abs(self.mouth_region[:, 0]) < 0.8][:, [2, 1]]

        self.tongue_rest = self.verts[TONGUE_SLICE]
        self.tongue_mid = self.tongue_rest[np.abs(self.tongue_rest[:, 0]) < 0.8][:, [2, 1]]

        self.zoom_xlim = None
        self.zoom_ylim = None
        z0, z1, y0, y1 = compute_zoom_bounds(self.mouth_region_mid, self.tongue_mid, self.tip_yz)
        self.zoom_xlim = (float(z0), float(z1))
        self.zoom_ylim = (float(y0), float(y1))
        self.full_xlim = (float(np.min(self.verts[:, 2])) - 0.5, float(np.max(self.verts[:, 2])) + 0.5)
        self.full_ylim = (float(np.min(self.verts[:, 1])) - 0.5, float(np.max(self.verts[:, 1])) + 0.5)

        self._setup_figure()
        self._refresh()

    def _target_array(self, name: str) -> np.ndarray:
        if name == "teeth_yz":
            return self.targets.teeth_yz
        if name == "alveolar_yz":
            return self.targets.alveolar_yz
        if name == "interdental_yz":
            return self.targets.interdental_yz
        raise KeyError(name)

    def _set_target(self, name: str, yz: np.ndarray) -> None:
        if name == "teeth_yz":
            self.targets.teeth_yz = yz.copy()
            return
        if name == "alveolar_yz":
            self.targets.alveolar_yz = yz.copy()
            return
        if name == "interdental_yz":
            self.targets.interdental_yz = yz.copy()
            return
        raise KeyError(name)

    def _load_from_json(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not load existing target file {path}: {exc}")
            return
        points = data.get("points_yz", {})
        for name in self.target_names_from_file():
            if name in points and isinstance(points[name], list) and len(points[name]) == 2:
                self._set_target(name, np.array([float(points[name][0]), float(points[name][1])], dtype=np.float64))
        print(f"Loaded existing targets from {path}")

    @staticmethod
    def target_names_from_file() -> list[str]:
        return ["teeth_yz", "alveolar_yz", "interdental_yz"]

    def _save_json(self) -> None:
        payload = {
            "dataset_id": self.dataset_id,
            "speaker_id": self.speaker_id,
            "motion_path": str(self.motion_path.resolve()),
            "textgrid_path": str(self.textgrid_path.resolve()) if self.textgrid_path is not None else None,
            "face_model_dir": str(self.face_model_dir.resolve()),
            "points_yz": {
                "teeth_yz": self.targets.teeth_yz.tolist(),
                "alveolar_yz": self.targets.alveolar_yz.tolist(),
                "interdental_yz": self.targets.interdental_yz.tolist(),
            },
            "points_xyz": {
                "teeth_xyz": [0.0, float(self.targets.teeth_yz[1]), float(self.targets.teeth_yz[0])],
                "alveolar_xyz": [0.0, float(self.targets.alveolar_yz[1]), float(self.targets.alveolar_yz[0])],
                "interdental_xyz": [0.0, float(self.targets.interdental_yz[1]), float(self.targets.interdental_yz[0])],
            },
            "auto_reference_yz": {
                "teeth_yz": self.base_auto_targets.teeth_yz.tolist(),
                "alveolar_yz": self.base_auto_targets.alveolar_yz.tolist(),
                "interdental_yz": self.base_auto_targets.interdental_yz.tolist(),
                "upper_front_yz": self.upper_front_yz.tolist(),
                "lower_front_yz": self.lower_front_yz.tolist(),
            },
            "strategy_reference_yz": {
                "interdental_mode": self.interdental_mode,
                "alveolar_tip_alpha": self.alveolar_tip_alpha,
                "teeth_yz": self.auto_targets.teeth_yz.tolist(),
                "alveolar_yz": self.auto_targets.alveolar_yz.tolist(),
                "interdental_yz": self.auto_targets.interdental_yz.tolist(),
                "alveolar_tip_reference_yz": (
                    self.alveolar_tip_reference_yz.tolist() if self.alveolar_tip_reference_yz is not None else None
                ),
            },
            "contact_params": {
                "region_extent_z": self.region_extent_z,
                "tau_alveolar_mm": self.tau_alveolar_mm,
                "tau_interdental_mm": self.tau_interdental_mm,
            },
            "notes": "Manually placed in tongue_target_point_editor.py (Z-Y sagittal).",
        }
        self.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved targets -> {self.output_path}")

    def _setup_figure(self) -> None:
        self.fig = plt.figure(figsize=(15, 9))
        self.fig.canvas.manager.set_window_title(f"Tongue Target Point Editor: {self.dataset_id}")

        self.ax = self.fig.add_axes([0.05, 0.16, 0.66, 0.80])
        self.ax_meta = self.fig.add_axes([0.73, 0.58, 0.25, 0.38])
        self.ax_meta.axis("off")
        self.ax_points = self.fig.add_axes([0.73, 0.16, 0.25, 0.34])
        self.ax_points.axis("off")

        self.ax_slider = self.fig.add_axes([0.05, 0.08, 0.52, 0.04])
        self.slider = Slider(self.ax_slider, "Frame", 0, self.max_frame, valinit=0, valstep=1)
        self.slider.on_changed(self._on_slider)

        self.ax_radio = self.fig.add_axes([0.58, 0.02, 0.13, 0.12])
        self.radio = RadioButtons(self.ax_radio, ("teeth_yz", "alveolar_yz", "interdental_yz"), active=1)
        self.radio.on_clicked(self._on_radio)

        self.ax_btn_save = self.fig.add_axes([0.73, 0.08, 0.07, 0.05])
        self.ax_btn_auto = self.fig.add_axes([0.81, 0.08, 0.07, 0.05])
        self.ax_btn_zoom = self.fig.add_axes([0.89, 0.08, 0.07, 0.05])
        self.ax_btn_quit = self.fig.add_axes([0.89, 0.02, 0.07, 0.05])
        self.btn_save = Button(self.ax_btn_save, "Save")
        self.btn_auto = Button(self.ax_btn_auto, "Reset")
        self.btn_zoom = Button(self.ax_btn_zoom, "Zoom")
        self.btn_quit = Button(self.ax_btn_quit, "Quit")
        self.btn_save.on_clicked(lambda _event: self._save_json())
        self.btn_auto.on_clicked(lambda _event: self._reset_auto())
        self.btn_zoom.on_clicked(lambda _event: self._toggle_zoom())
        self.btn_quit.on_clicked(lambda _event: self._save_and_quit())

        self.face_edge_artist = self.ax.scatter([], [], s=8, c="#7a7a7a", alpha=0.36, label="face cut-section")
        self.mouth_edge_artist = self.ax.scatter([], [], s=10, c="#bb7f7f", alpha=0.55, label="mouth region")
        (self.current_tongue_upper_line,) = self.ax.plot([], [], color="#d62728", linewidth=2.5, alpha=0.92, label="primary tongue edge")
        (self.current_tongue_lower_line,) = self.ax.plot([], [], color="#d62728", linewidth=2.5, alpha=0.92)
        (self.compare_tongue_upper_line,) = self.ax.plot([], [], color="#17becf", linewidth=2.3, alpha=0.92, label="compare tongue edge")
        (self.compare_tongue_lower_line,) = self.ax.plot([], [], color="#17becf", linewidth=2.3, alpha=0.92)

        self.alv_tip_ref_artist = self.ax.scatter([], [], s=95, c="#7f3c8d", marker="P", label="alveolar tip-reference")
        if self.alveolar_tip_reference_yz is not None:
            self.alv_tip_ref_artist.set_offsets(self.alveolar_tip_reference_yz[None, :])

        self.target_artists = {
            "teeth_yz": self.ax.scatter([], [], s=210, c="#ffffff", edgecolors="#1f77b4", marker="o", linewidths=2.0, zorder=10),
            "alveolar_yz": self.ax.scatter([], [], s=210, c="#fff3cd", edgecolors="#f39c12", marker="^", linewidths=2.0, zorder=10),
            "interdental_yz": self.ax.scatter([], [], s=210, c="#d4edda", edgecolors="#2ca25f", marker="s", linewidths=2.0, zorder=10),
        }
        self.auto_artists = {
            "teeth_yz": self.ax.scatter([], [], s=70, c="#1f77b4", marker="x", linewidths=2.0, zorder=9),
            "alveolar_yz": self.ax.scatter([], [], s=70, c="#f39c12", marker="x", linewidths=2.0, zorder=9),
            "interdental_yz": self.ax.scatter([], [], s=70, c="#2ca25f", marker="x", linewidths=2.0, zorder=9),
        }
        self.active_ring = self.ax.scatter([], [], s=360, facecolors="none", edgecolors="#111111", linewidths=1.5, zorder=11)
        self.current_tip_artist = self.ax.scatter([], [], s=120, c="#000000", marker="*", zorder=11, label="current frame tip")
        self.compare_tip_artist = self.ax.scatter([], [], s=100, c="#17becf", marker="o", edgecolors="black", linewidths=0.8, zorder=11)
        self.phoneme_text_artist = self.ax.text(
            0.01,
            0.99,
            "",
            transform=self.ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#666666"},
        )

        self.alv_rect = Rectangle((0, 0), 0, 0, linewidth=1.8, edgecolor="#f39c12", facecolor="none", linestyle="--", zorder=8)
        self.int_rect = Rectangle((0, 0), 0, 0, linewidth=1.8, edgecolor="#2ecc71", facecolor="none", linestyle="--", zorder=8)
        mouth_z = self.mouth_region_mid[:, 0]
        mouth_y = self.mouth_region_mid[:, 1]
        self.mouth_rect = Rectangle(
            (float(np.min(mouth_z)), float(np.min(mouth_y))),
            float(np.max(mouth_z) - np.min(mouth_z)),
            float(np.max(mouth_y) - np.min(mouth_y)),
            linewidth=1.6,
            edgecolor="#34495e",
            facecolor="none",
            linestyle=":",
            zorder=7,
        )
        self.ax.add_patch(self.alv_rect)
        self.ax.add_patch(self.int_rect)
        self.ax.add_patch(self.mouth_rect)

        self.ax.set_xlabel("Z (Anterior ->)")
        self.ax.set_ylabel("Y (Superior ->)")
        self.ax.set_title(
            f"Interactive Target Point Editor: {self.dataset_id}\n"
            "LMB: drag or place active target; keys 1/2/3 select target; S save; R reset auto; Z zoom; Q save+quit"
        )
        self.ax.grid(alpha=0.16)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_radio(self, label: str) -> None:
        self.active_target = str(label)
        self._refresh()

    def _on_slider(self, value: float) -> None:
        self.current_frame = int(value)
        self._refresh()

    def _closest_target(self, x: float, y: float) -> tuple[str, float]:
        click = np.array([x, y], dtype=np.float64)
        best_name = self.target_names[0]
        best_dist = 1e9
        for name in self.target_names:
            d = float(np.linalg.norm(self._target_array(name) - click))
            if d < best_dist:
                best_dist = d
                best_name = name
        return best_name, best_dist

    def _on_press(self, event) -> None:
        if event.button != 1 or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        nearest, dist = self._closest_target(x, y)
        if dist <= self.drag_radius:
            self.dragging_target = nearest
            self.active_target = nearest
            self._set_target(nearest, np.array([x, y], dtype=np.float64))
        else:
            self._set_target(self.active_target, np.array([x, y], dtype=np.float64))
        self._refresh()

    def _on_motion(self, event) -> None:
        if self.dragging_target is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self._set_target(self.dragging_target, np.array([float(event.xdata), float(event.ydata)], dtype=np.float64))
        self._refresh()

    def _on_release(self, _event) -> None:
        self.dragging_target = None

    def _on_key(self, event) -> None:
        if event.key == "1":
            self.active_target = "teeth_yz"
            self.radio.set_active(0)
        elif event.key == "2":
            self.active_target = "alveolar_yz"
            self.radio.set_active(1)
        elif event.key == "3":
            self.active_target = "interdental_yz"
            self.radio.set_active(2)
        elif event.key == "right":
            self.slider.set_val(min(self.max_frame, self.current_frame + 1))
        elif event.key == "left":
            self.slider.set_val(max(0, self.current_frame - 1))
        elif event.key in {"s", "e"}:
            self._save_json()
        elif event.key == "r":
            self._reset_auto()
        elif event.key == "z":
            self._toggle_zoom()
        elif event.key == "x":
            self.zoomed_mouth_box = False
            self._refresh()
        elif event.key == "q":
            self._save_and_quit()
        elif event.key == "i":
            print(
                "Current points: "
                f"teeth={self.targets.teeth_yz.tolist()} "
                f"alveolar={self.targets.alveolar_yz.tolist()} "
                f"interdental={self.targets.interdental_yz.tolist()}"
            )

    def _reset_auto(self) -> None:
        self.targets = self.auto_targets.copy()
        self._refresh()

    def _toggle_zoom(self) -> None:
        self.zoomed_mouth_box = not self.zoomed_mouth_box
        self._refresh()

    def _save_and_quit(self) -> None:
        self._save_json()
        plt.close(self.fig)

    def _current_phoneme_text(self) -> str:
        if not self.phoneme_spans:
            return "phoneme: n/a"
        frame = self.current_frame
        for span in self.phoneme_spans:
            if span.start_frame <= frame <= span.end_frame:
                return f"phoneme: {span.label} [{span.phoneme_class}]"
        return "phoneme: n/a"

    def _update_info_panel(self) -> None:
        self.ax_meta.clear()
        self.ax_meta.axis("off")
        status_lines = [
            f"Dataset: {self.dataset_id}",
            f"Speaker: {self.speaker_id}",
            f"Frame: {self.current_frame}/{self.max_frame}",
            f"Time: {self.current_frame / self.fps:.2f}s",
            self._current_phoneme_text(),
            "",
            f"Motion: {self.motion_path.name}",
            (
                f"Compare: {self.compare_motion_path.name}"
                if self.compare_motion_path is not None
                else "Compare: none"
            ),
            f"Face shift: {self.face_shift_seconds:+.3f}s",
            f"Active target: {self.active_target}",
            f"Zoom: {'mouth box' if self.zoomed_mouth_box else 'full view'}",
            "",
            "Defaults:",
            f"  mode={self.interdental_mode}, alpha={self.alveolar_tip_alpha:.2f}",
            f"  region_extent_z={self.region_extent_z:.2f}",
            f"  tau_alv={self.tau_alveolar_mm:.2f}",
            f"  tau_int={self.tau_interdental_mm:.2f}",
        ]
        self.ax_meta.text(
            0.02,
            0.98,
            "\n".join(status_lines),
            va="top",
            transform=self.ax_meta.transAxes,
            fontsize=9,
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "#f8f5e4", "alpha": 0.90},
        )

        self.ax_points.clear()
        self.ax_points.axis("off")
        point_lines = [
            "Targets (z, y):",
            f"  teeth_yz:       ({self.targets.teeth_yz[0]:.3f}, {self.targets.teeth_yz[1]:.3f})",
            f"  alveolar_yz:    ({self.targets.alveolar_yz[0]:.3f}, {self.targets.alveolar_yz[1]:.3f})",
            f"  interdental_yz: ({self.targets.interdental_yz[0]:.3f}, {self.targets.interdental_yz[1]:.3f})",
            "",
            "Auto reference:",
            f"  teeth_yz:       ({self.base_auto_targets.teeth_yz[0]:.3f}, {self.base_auto_targets.teeth_yz[1]:.3f})",
            f"  alveolar_yz:    ({self.base_auto_targets.alveolar_yz[0]:.3f}, {self.base_auto_targets.alveolar_yz[1]:.3f})",
            f"  interdental_yz: ({self.base_auto_targets.interdental_yz[0]:.3f}, {self.base_auto_targets.interdental_yz[1]:.3f})",
            "",
            f"Output JSON:",
            f"  {self.output_path}",
        ]
        self.ax_points.text(
            0.02,
            0.98,
            "\n".join(point_lines),
            va="top",
            transform=self.ax_points.transAxes,
            fontsize=9,
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "#eef6fb", "alpha": 0.92},
        )

    def _refresh(self) -> None:
        if self.face_seq is not None and self.current_frame < len(self.face_seq):
            weights = {n: v for n, v in zip(self.face_model.expression_names, self.face_seq[self.current_frame])}
            current_face_verts = self.face_model.deform(weights).copy()
        else:
            current_face_verts = self.verts
        self.face_edge_artist.set_offsets(current_face_verts[self.face_midline_mask][:, [2, 1]])
        self.mouth_edge_artist.set_offsets(current_face_verts[self.mouth_midline_mask][:, [2, 1]])

        current_tongue_verts, _, _ = self.tongue_rig.deform(self.anchor_seq[self.current_frame])
        current_tongue_mid = current_tongue_verts[np.abs(current_tongue_verts[:, 0]) < 0.8][:, [2, 1]]
        current_upper, current_lower = compute_outline(current_tongue_mid, num_bins=70)
        self.current_tongue_upper_line.set_data(current_upper[:, 0], current_upper[:, 1])
        self.current_tongue_lower_line.set_data(current_lower[:, 0], current_lower[:, 1])
        if self.compare_anchor_seq is not None and self.current_frame < len(self.compare_anchor_seq):
            compare_tongue_verts, _, _ = self.tongue_rig.deform(self.compare_anchor_seq[self.current_frame])
            compare_tongue_mid = compare_tongue_verts[np.abs(compare_tongue_verts[:, 0]) < 0.8][:, [2, 1]]
            compare_upper, compare_lower = compute_outline(compare_tongue_mid, num_bins=70)
            self.compare_tongue_upper_line.set_data(compare_upper[:, 0], compare_upper[:, 1])
            self.compare_tongue_lower_line.set_data(compare_lower[:, 0], compare_lower[:, 1])
            compare_tip = self.compare_tip_yz[self.current_frame]
            self.compare_tip_artist.set_offsets(compare_tip[None, :])
        else:
            self.compare_tongue_upper_line.set_data([], [])
            self.compare_tongue_lower_line.set_data([], [])
            self.compare_tip_artist.set_offsets(np.empty((0, 2)))

        for name in self.target_names:
            yz = self._target_array(name)
            self.target_artists[name].set_offsets(yz[None, :])
            self.auto_artists[name].set_offsets(self._target_array_from_auto(name)[None, :])

        active = self._target_array(self.active_target)
        self.active_ring.set_offsets(active[None, :])

        tip = self.tip_yz[self.current_frame]
        self.current_tip_artist.set_offsets(tip[None, :])
        self.phoneme_text_artist.set_text(
            f"{self.dataset_id} | frame {self.current_frame}/{self.max_frame} | "
            f"t={self.current_frame / self.fps:.2f}s\n{self._current_phoneme_text()}"
        )

        alv = self.targets.alveolar_yz
        ind = self.targets.interdental_yz
        self.alv_rect.set_x(float(alv[0] - self.region_extent_z))
        self.alv_rect.set_y(float(alv[1] - self.tau_alveolar_mm))
        self.alv_rect.set_width(float(2.0 * self.region_extent_z))
        self.alv_rect.set_height(float(2.0 * self.tau_alveolar_mm))

        self.int_rect.set_x(float(ind[0] - self.region_extent_z))
        self.int_rect.set_y(float(ind[1] - self.tau_interdental_mm))
        self.int_rect.set_width(float(2.0 * self.region_extent_z))
        self.int_rect.set_height(float(2.0 * self.tau_interdental_mm))

        if self.zoomed_mouth_box:
            self.ax.set_xlim(*self.zoom_xlim)
            self.ax.set_ylim(*self.zoom_ylim)
        else:
            self.ax.set_xlim(*self.full_xlim)
            self.ax.set_ylim(*self.full_ylim)

        self._update_info_panel()
        self.fig.canvas.draw_idle()

    def _target_array_from_auto(self, name: str) -> np.ndarray:
        if name == "teeth_yz":
            return self.auto_targets.teeth_yz
        if name == "alveolar_yz":
            return self.auto_targets.alveolar_yz
        if name == "interdental_yz":
            return self.auto_targets.interdental_yz
        raise KeyError(name)

    def run(self) -> None:
        plt.show()


def resolve_textgrid_path(beat_root: Path, speaker_id: str, dataset_id: str) -> Optional[Path]:
    direct = beat_root / f"{dataset_id}.TextGrid"
    if direct.is_file():
        return direct
    nested = beat_root / str(speaker_id) / f"{dataset_id}.TextGrid"
    if nested.is_file():
        return nested
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive sagittal target point editor for tip-contact optimization.")
    parser.add_argument("--dataset-id", default="1_wayne_0_75_75")
    parser.add_argument("--speaker-id", default="1")
    parser.add_argument(
        "--beat-root",
        default=str(PROJECT_ROOT / "data" / "beat_cache" / "beat_english_v0.2.1" / "beat_english_v0.2.1"),
    )
    parser.add_argument("--motion-path", default=None, help="Motion .npy path (defaults to tongue_scripts/outputs/<dataset>.npy)")
    parser.add_argument("--compare-motion-path", default=None, help="Optional second motion .npy to overlay frame-by-frame.")
    parser.add_argument("--textgrid-path", default=None, help="Optional explicit TextGrid path")
    parser.add_argument("--face-model-dir", default=str(PROJECT_ROOT / "FaceXModel"))
    parser.add_argument("--std-path", default=str(TONGUE_SCRIPTS_DIR / "normalising_vectors" / "JW13_4points_std.npy"))
    parser.add_argument("--scalar", type=float, default=0.20)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument(
        "--face-shift-seconds",
        type=float,
        default=0.0,
        help="Positive value advances BEAT face animation earlier relative to tongue/audio in the editor view.",
    )
    parser.add_argument("--interdental-mode", choices=["midpoint", "upper_edge"], default="upper_edge")
    parser.add_argument("--alveolar-tip-alpha", type=float, default=0.55)
    parser.add_argument("--region-extent-z", type=float, default=1.2)
    parser.add_argument("--tau-alveolar-mm", type=float, default=1.0)
    parser.add_argument("--tau-interdental-mm", type=float, default=1.0)
    parser.add_argument("--teeth-yz", default=None, help="Manual init teeth target as 'z,y'")
    parser.add_argument("--alveolar-yz", default=None, help="Manual init alveolar target as 'z,y'")
    parser.add_argument("--interdental-yz", default=None, help="Manual init interdental target as 'z,y'")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output JSON path (default: tongue_scripts/outputs/<dataset>_manual_tip_targets.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    beat_root = Path(args.beat_root)
    motion_path = Path(args.motion_path) if args.motion_path else (TONGUE_SCRIPTS_DIR / "outputs" / f"{args.dataset_id}.npy")
    textgrid_path = Path(args.textgrid_path) if args.textgrid_path else resolve_textgrid_path(beat_root, args.speaker_id, args.dataset_id)
    output_path = (
        Path(args.output_path)
        if args.output_path
        else (TONGUE_SCRIPTS_DIR / "outputs" / f"{args.dataset_id}_manual_tip_targets.json")
    )

    if not motion_path.is_file():
        raise SystemExit(f"motion not found: {motion_path}")
    if args.compare_motion_path and not Path(args.compare_motion_path).is_file():
        raise SystemExit(f"compare motion not found: {args.compare_motion_path}")
    if not Path(args.face_model_dir).is_dir():
        raise SystemExit(f"face model dir not found: {args.face_model_dir}")
    if not Path(args.std_path).is_file():
        raise SystemExit(f"std path not found: {args.std_path}")
    if textgrid_path is None:
        print("Warning: TextGrid not found. Alveolar/interdental span overlays will be disabled.")
    else:
        print(f"Using TextGrid: {textgrid_path}")

    editor = TongueTargetPointEditor(
        dataset_id=args.dataset_id,
        speaker_id=str(args.speaker_id),
        beat_root=beat_root,
        face_model_dir=Path(args.face_model_dir),
        motion_path=motion_path,
        compare_motion_path=(Path(args.compare_motion_path) if args.compare_motion_path else None),
        textgrid_path=textgrid_path,
        std_path=Path(args.std_path),
        scalar=float(args.scalar),
        fps=float(args.fps),
        face_shift_seconds=float(args.face_shift_seconds),
        region_extent_z=float(args.region_extent_z),
        tau_alveolar_mm=float(args.tau_alveolar_mm),
        tau_interdental_mm=float(args.tau_interdental_mm),
        output_path=output_path,
        teeth_yz=parse_yz(args.teeth_yz),
        alveolar_yz=parse_yz(args.alveolar_yz),
        interdental_yz=parse_yz(args.interdental_yz),
        interdental_mode=str(args.interdental_mode),
        alveolar_tip_alpha=float(args.alveolar_tip_alpha),
    )
    editor.run()


if __name__ == "__main__":
    main()
