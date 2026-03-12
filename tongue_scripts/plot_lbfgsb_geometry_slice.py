#!/usr/bin/env python3
"""
Plot a sagittal (Z-Y) slice for geometry review, aligned with
generate_tongue_animation.py matplotlib view.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

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
from tongue_scripts.phoneme_lbfgsb_optimizer import parse_textgrid


def parse_yz(value: str | None) -> np.ndarray | None:
    if not value:
        return None
    z_s, y_s = value.split(",")
    return np.array([float(z_s), float(y_s)], dtype=np.float64)


def estimate_oral_targets_yz(face_model) -> dict[str, np.ndarray]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot L-BFGS-B geometry slice for manual confirmation.")
    parser.add_argument("--dataset-id", default="1_wayne_0_75_75")
    parser.add_argument("--speaker-id", default="1")
    parser.add_argument(
        "--beat-root",
        default=str(PROJECT_ROOT / "data" / "beat_cache" / "beat_english_v0.2.1" / "beat_english_v0.2.1"),
    )
    parser.add_argument("--motion-path", default=None)
    parser.add_argument("--textgrid-path", default=None)
    parser.add_argument("--face-model-dir", default=str(PROJECT_ROOT / "FaceXModel"))
    parser.add_argument("--std-path", default=str(SCRIPT_DIR / "normalising_vectors" / "JW13_4points_std.npy"))
    parser.add_argument("--scalar", type=float, default=0.20)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--teeth-yz", default=None, help="Manual teeth target as 'z,y'")
    parser.add_argument("--alveolar-yz", default=None, help="Manual alveolar target as 'z,y'")
    parser.add_argument("--interdental-yz", default=None, help="Manual interdental target as 'z,y'")
    parser.add_argument("--region-extent-z", type=float, default=1.2)
    parser.add_argument("--tau-alveolar-mm", type=float, default=1.0)
    parser.add_argument("--tau-interdental-mm", type=float, default=1.0)
    parser.add_argument("--zoom-mouth-box", action="store_true", help="Zoom axes to the mouth-box candidate bounds.")
    parser.add_argument(
        "--zoom-padding",
        type=float,
        default=0.8,
        help="Padding applied around mouth-box bounds when --zoom-mouth-box is enabled.",
    )
    parser.add_argument(
        "--output-path",
        default=str(SCRIPT_DIR / "outputs" / "1_wayne_0_75_75_lbfgsb_geometry_slice.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    beat_root = Path(args.beat_root)
    speaker_dir = beat_root / str(args.speaker_id)
    motion_path = Path(args.motion_path) if args.motion_path else (SCRIPT_DIR / "outputs" / f"{args.dataset_id}.npy")
    textgrid_path = Path(args.textgrid_path) if args.textgrid_path else (speaker_dir / f"{args.dataset_id}.TextGrid")
    output_path = Path(args.output_path)

    if not motion_path.is_file():
        raise SystemExit(f"motion not found: {motion_path}")
    if not textgrid_path.is_file():
        raise SystemExit(f"textgrid not found: {textgrid_path}")
    if not Path(args.face_model_dir).is_dir():
        raise SystemExit(f"face model dir not found: {args.face_model_dir}")

    face_model = load_face_model_trimesh(str(Path(args.face_model_dir)))
    tongue_rig = FaceKitTongueRig(
        face_model.neutral_verts,
        face_model.faces,
        TONGUE_SLICE,
        ANCHOR_INDICES,
        BONE_INDICES,
        {"rotation_deg": 5, "thickness": 1.2, "shift_y": 0, "shift_z": 0, "std_scalar": float(args.scalar)},
    )

    raw_motion = np.load(motion_path)
    raw_ema = raw_motion[:, :8].reshape(-1, 4, 2).astype(np.float64)
    std_4x2 = np.load(args.std_path).flatten()[:8].reshape(4, 2).astype(np.float64)
    denorm = raw_to_denorm_anchors(raw_ema, std_4x2, tongue_rig.anchors.astype(np.float64), float(args.scalar))

    spans = parse_textgrid(textgrid_path, fps=float(args.fps), tier_name="phones", total_frames=len(denorm))
    alveolar_idx: list[int] = []
    interdental_idx: list[int] = []
    for sp in spans:
        if sp.phoneme_class == "alveolar":
            alveolar_idx.extend(range(sp.start_frame, sp.end_frame + 1))
        elif sp.phoneme_class == "interdental":
            interdental_idx.extend(range(sp.start_frame, sp.end_frame + 1))
    alveolar_idx = sorted(set(i for i in alveolar_idx if 0 <= i < len(denorm)))
    interdental_idx = sorted(set(i for i in interdental_idx if 0 <= i < len(denorm)))

    auto_targets = estimate_oral_targets_yz(face_model)
    manual_teeth_yz = parse_yz(args.teeth_yz)
    manual_alveolar_yz = parse_yz(args.alveolar_yz)
    manual_interdental_yz = parse_yz(args.interdental_yz)
    teeth_yz = manual_teeth_yz if manual_teeth_yz is not None else auto_targets["teeth_yz"]
    alveolar_yz = manual_alveolar_yz if manual_alveolar_yz is not None else auto_targets["alveolar_yz"]
    interdental_yz = manual_interdental_yz if manual_interdental_yz is not None else auto_targets["interdental_yz"]

    verts = np.asarray(face_model.neutral_verts, dtype=np.float64)
    midline = np.abs(verts[:, 0]) < 0.6
    yz_face = verts[midline][:, [2, 1]]
    tongue_rest = verts[TONGUE_SLICE]
    tongue_mid = tongue_rest[np.abs(tongue_rest[:, 0]) < 0.8][:, [2, 1]]
    gum_teeth = verts[14062:16611]
    gum_teeth_mid = gum_teeth[np.abs(gum_teeth[:, 0]) < 0.8][:, [2, 1]]

    region_verts = verts[14062:17039]
    mouth_min = np.min(region_verts, axis=0)
    mouth_max = np.max(region_verts, axis=0)

    tip_yz = denorm[:, 3][:, [2, 1]]
    tip_alv = tip_yz[alveolar_idx] if alveolar_idx else np.zeros((0, 2), dtype=np.float64)
    tip_int = tip_yz[interdental_idx] if interdental_idx else np.zeros((0, 2), dtype=np.float64)

    anchors_rest_yz = tongue_rig.anchors[:, [2, 1]]
    anchor_names = ["T4(back)", "T3(dorsum)", "T2(blade)", "T1(tip)"]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(yz_face[:, 0], yz_face[:, 1], s=2, c="#8d99ae", alpha=0.18, label="face shell (midline)")
    if len(gum_teeth_mid):
        ax.scatter(gum_teeth_mid[:, 0], gum_teeth_mid[:, 1], s=14, c="#f5b041", alpha=0.85, label="gums/teeth region")
    ax.scatter(tongue_mid[:, 0], tongue_mid[:, 1], s=10, c="#e74c3c", alpha=0.75, label="tongue body")
    ax.scatter(tip_yz[:, 0], tip_yz[:, 1], s=6, c="#e06666", alpha=0.18, label="tip trajectory (all)")
    if len(tip_alv):
        ax.scatter(tip_alv[:, 0], tip_alv[:, 1], s=8, c="#f39c12", alpha=0.65, label="tip in alveolar spans")
    if len(tip_int):
        ax.scatter(tip_int[:, 0], tip_int[:, 1], s=8, c="#2ecc71", alpha=0.65, label="tip in interdental spans")

    for i, name in enumerate(anchor_names):
        z, y = anchors_rest_yz[i]
        ax.scatter([z], [y], s=90, marker="x", c="#cc0000")
        ax.text(z + 0.12, y + 0.05, name, fontsize=9, color="#8b0000")

    point_suffix = "manual" if manual_teeth_yz is not None else "auto"
    ax.scatter([teeth_yz[0]], [teeth_yz[1]], s=190, c="#ffffff", edgecolors="#1f77b4", marker="o", linewidths=2.0, label=f"teeth_yz ({point_suffix})")
    ax.scatter([alveolar_yz[0]], [alveolar_yz[1]], s=190, c="#fff3cd", edgecolors="#f39c12", marker="^", linewidths=2.0, label="alveolar_yz")
    ax.scatter(
        [interdental_yz[0]],
        [interdental_yz[1]],
        s=190,
        c="#d4edda",
        edgecolors="#2ecc71",
        marker="s",
        linewidths=2.0,
        label="interdental_yz",
    )
    upper_front_yz = auto_targets["upper_front_yz"]
    lower_front_yz = auto_targets["lower_front_yz"]
    ax.scatter([upper_front_yz[0]], [upper_front_yz[1]], s=95, c="#2166ac", marker="D", label="upper front teeth centroid")
    ax.scatter([lower_front_yz[0]], [lower_front_yz[1]], s=95, c="#b2182b", marker="D", label="lower front teeth centroid")
    ax.plot(
        [upper_front_yz[0], lower_front_yz[0]],
        [upper_front_yz[1], lower_front_yz[1]],
        color="#2ecc71",
        linewidth=1.4,
        linestyle=":",
        alpha=0.9,
    )

    rz = float(args.region_extent_z)
    tau_a = float(args.tau_alveolar_mm)
    tau_i = float(args.tau_interdental_mm)
    rect_alv = Rectangle(
        (alveolar_yz[0] - rz, alveolar_yz[1] - tau_a),
        2 * rz,
        2 * tau_a,
        linewidth=1.8,
        edgecolor="#f39c12",
        facecolor="none",
        linestyle="--",
        label="alveolar contact band (z extent + tau)",
    )
    rect_int = Rectangle(
        (interdental_yz[0] - rz, interdental_yz[1] - tau_i),
        2 * rz,
        2 * tau_i,
        linewidth=1.8,
        edgecolor="#2ecc71",
        facecolor="none",
        linestyle="--",
        label="interdental contact band (z extent + tau)",
    )
    ax.add_patch(rect_alv)
    ax.add_patch(rect_int)

    mouth_rect = Rectangle(
        (mouth_min[2], mouth_min[1]),
        mouth_max[2] - mouth_min[2],
        mouth_max[1] - mouth_min[1],
        linewidth=1.5,
        edgecolor="#34495e",
        facecolor="none",
        linestyle=":",
        label="mouth box candidate (from 14062:17039)",
    )
    ax.add_patch(mouth_rect)

    ax.annotate(
        f"teeth_yz=({teeth_yz[0]:.2f}, {teeth_yz[1]:.2f})",
        xy=(teeth_yz[0], teeth_yz[1]),
        xytext=(teeth_yz[0] + 0.25, teeth_yz[1] + 0.35),
        fontsize=9,
        color="#1f77b4",
        arrowprops={"arrowstyle": "->", "color": "#1f77b4", "lw": 1.0},
    )
    ax.annotate(
        f"alveolar_yz=({alveolar_yz[0]:.2f}, {alveolar_yz[1]:.2f})",
        xy=(alveolar_yz[0], alveolar_yz[1]),
        xytext=(alveolar_yz[0] - 1.1, alveolar_yz[1] + 0.45),
        fontsize=9,
        color="#f39c12",
        arrowprops={"arrowstyle": "->", "color": "#f39c12", "lw": 1.0},
    )
    ax.annotate(
        f"interdental_yz=({interdental_yz[0]:.2f}, {interdental_yz[1]:.2f})",
        xy=(interdental_yz[0], interdental_yz[1]),
        xytext=(interdental_yz[0] + 0.3, interdental_yz[1] - 0.55),
        fontsize=9,
        color="#1f8f4d",
        arrowprops={"arrowstyle": "->", "color": "#1f8f4d", "lw": 1.0},
    )

    ax.set_xlabel("Z (Anterior ->)")
    ax.set_ylabel("Y (Superior ->)")
    ax.set_title(
        f"Geometry Slice Review: {args.dataset_id}\n"
        "View aligned with generate_tongue_animation matplotlib mode (Z-Y sagittal)"
    )
    ax.grid(alpha=0.15)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    if args.zoom_mouth_box:
        pad = float(args.zoom_padding)
        ax.set_xlim(float(mouth_min[2]) - pad, float(mouth_max[2]) + pad)
        ax.set_ylim(float(mouth_min[1]) - pad, float(mouth_max[1]) + pad)
        ax.set_title(
            f"Geometry Slice (Zoomed to Mouth Box): {args.dataset_id}\n"
            "View aligned with generate_tongue_animation matplotlib mode (Z-Y sagittal)"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"teeth_yz={teeth_yz.tolist()}")
    print(f"alveolar_yz={alveolar_yz.tolist()}")
    print(f"interdental_yz={interdental_yz.tolist()}")
    print(f"upper_front_yz={upper_front_yz.tolist()}")
    print(f"lower_front_yz={lower_front_yz.tolist()}")
    print(f"Saved geometry slice: {output_path}")


if __name__ == "__main__":
    main()
