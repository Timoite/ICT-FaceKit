#!/usr/bin/env python3
"""
Render a sagittal matplotlib video with two tongue motions overlaid in different colors.

The view is aligned with the repository's existing generate_tongue_animation.py
matplotlib mode, but adds:
 - dynamic facial cut-section from BEAT
 - dynamic mouth-region cut-section
 - baseline vs optimized tongue mesh/spline/anchors in the same frame
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
TONGUE_ANIM_DIR = SCRIPT_DIR / "tongue_animation"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TONGUE_ANIM_DIR) not in sys.path:
    sys.path.insert(0, str(TONGUE_ANIM_DIR))

from face_model_io_trimesh import load_face_model_trimesh  # type: ignore
import generate_tongue_animation as gta  # type: ignore

MOUTH_REGION = slice(14062, 17039)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render sagittal face-section video with overlaid baseline and optimized tongues."
    )
    parser.add_argument("--dataset-id", default="1_wayne_0_75_75")
    parser.add_argument("--speaker-id", default="1")
    parser.add_argument(
        "--beat-root",
        default=str(PROJECT_ROOT / "data" / "beat_cache" / "beat_english_v0.2.1" / "beat_english_v0.2.1"),
    )
    parser.add_argument(
        "--baseline-motion",
        default=str(SCRIPT_DIR / "outputs" / "1_wayne_0_75_75_pre_shift012.npy"),
    )
    parser.add_argument(
        "--optimized-motion",
        default=str(SCRIPT_DIR / "outputs" / "1_wayne_0_75_75_lbfgsb_manual_shift012_fixed.npy"),
    )
    parser.add_argument("--baseline-label", default="unoptimized")
    parser.add_argument("--optimized-label", default="optimized v2")
    parser.add_argument("--face-model-dir", default=str(PROJECT_ROOT / "FaceXModel"))
    parser.add_argument(
        "--std-path",
        default=str(SCRIPT_DIR / "normalising_vectors" / "JW13_4points_std.npy"),
    )
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--std-scalar", type=float, default=0.20)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--midline-width", type=float, default=0.5)
    parser.add_argument("--tongue-midline-width", type=float, default=1.0)
    parser.add_argument(
        "--output-video",
        default=str(SCRIPT_DIR / "outputs" / "1_wayne_0_75_75_tongue_overlay_matplotlib.mp4"),
    )
    parser.add_argument("--with-audio", action="store_true", default=True)
    parser.add_argument("--no-audio", dest="with_audio", action="store_false")
    return parser.parse_args()


def _mouth_bounds(face_model, tongue_rig, baseline_seq, optimized_seq) -> tuple[float, float, float, float]:
    mouth_mask = np.abs(face_model.neutral_verts[:, 0]) < 0.5
    mouth_mask[: MOUTH_REGION.start] = False
    mouth_mask[gta.TONGUE_SLICE] = False
    mouth_pts = face_model.neutral_verts[mouth_mask][:, [2, 1]]

    tongue_anchors = np.concatenate(
        [
            baseline_seq.reshape(-1, 3)[:, [2, 1]],
            optimized_seq.reshape(-1, 3)[:, [2, 1]],
            tongue_rig.vertices_rest[:, [2, 1]],
        ],
        axis=0,
    )
    all_pts = np.concatenate([mouth_pts, tongue_anchors], axis=0)
    z_min, y_min = all_pts.min(axis=0)
    z_max, y_max = all_pts.max(axis=0)
    pad_z = 0.8
    pad_y = 0.8
    return z_min - pad_z, z_max + pad_z, y_min - pad_y, y_max + pad_y


def _load_sequences(args: argparse.Namespace):
    beat_root = Path(args.beat_root)
    speaker_dir = beat_root / str(args.speaker_id)
    json_path = speaker_dir / f"{args.dataset_id}.json"
    wav_path = speaker_dir / f"{args.dataset_id}.wav"

    face_model = load_face_model_trimesh(str(Path(args.face_model_dir)))
    tongue_rig = gta.FaceKitTongueRig(
        face_model.neutral_verts,
        face_model.faces,
        gta.TONGUE_SLICE,
        gta.ANCHOR_INDICES,
        gta.BONE_INDICES,
        {
            "rotation_deg": 5,
            "thickness": 1.2,
            "shift_y": 0,
            "shift_z": 0,
            "std_scalar": float(args.std_scalar),
        },
    )

    face_seq = gta.process_beat_data(str(json_path), face_model, target_fps=int(args.fps))
    baseline_seq = gta.load_ema_motion(
        str(Path(args.baseline_motion)),
        str(Path(args.std_path)),
        tongue_rig.anchors,
        float(args.std_scalar),
    )
    optimized_seq = gta.load_ema_motion(
        str(Path(args.optimized_motion)),
        str(Path(args.std_path)),
        tongue_rig.anchors,
        float(args.std_scalar),
    )

    if args.max_seconds is None:
        frames = min(len(face_seq), len(baseline_seq), len(optimized_seq))
    else:
        frames = min(
            len(face_seq),
            len(baseline_seq),
            len(optimized_seq),
            max(1, int(round(float(args.max_seconds) * int(args.fps)))),
        )
    return face_model, tongue_rig, face_seq[:frames], baseline_seq[:frames], optimized_seq[:frames], wav_path


def _compute_outline(mesh_pts: np.ndarray, num_bins: int = 80) -> tuple[np.ndarray, np.ndarray]:
    if len(mesh_pts) < 4:
        empty = np.empty((0, 2), dtype=float)
        return empty, empty

    z = mesh_pts[:, 0]
    y = mesh_pts[:, 1]
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    if not np.isfinite(z_min) or not np.isfinite(z_max) or abs(z_max - z_min) < 1e-6:
        empty = np.empty((0, 2), dtype=float)
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
    return np.asarray(upper, dtype=float), np.asarray(lower, dtype=float)


def main() -> None:
    args = parse_args()
    output_video = Path(args.output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    face_model, tongue_rig, face_seq, baseline_seq, optimized_seq, wav_path = _load_sequences(args)
    frames = min(len(face_seq), len(baseline_seq), len(optimized_seq))
    if frames == 0:
        raise SystemExit("No frames available for overlay render.")

    face_midline = np.abs(face_model.neutral_verts[:, 0]) < float(args.midline_width)
    face_midline[gta.TONGUE_SLICE] = False

    mouth_midline = np.abs(face_model.neutral_verts[:, 0]) < float(args.midline_width)
    mouth_midline[: MOUTH_REGION.start] = False
    mouth_midline[gta.TONGUE_SLICE] = False

    x_min, x_max, y_min, y_max = _mouth_bounds(face_model, tongue_rig, baseline_seq, optimized_seq)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0.08, 0.10, 0.88, 0.84])
    ax.set_aspect("equal")
    ax.set_xlabel("Z (Anterior ->)")
    ax.set_ylabel("Y (Superior ->)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    face_scat = ax.scatter([], [], s=2, c="#8d8d8d", alpha=0.25, zorder=1, label="face cut-section")
    mouth_scat = ax.scatter([], [], s=4, c="#bb7f7f", alpha=0.45, zorder=2, label="mouth region")

    base_mesh = ax.scatter([], [], s=3, c="#f39c12", alpha=0.08, zorder=3)
    opt_mesh = ax.scatter([], [], s=3, c="#00bcd4", alpha=0.08, zorder=4)

    (base_upper,) = ax.plot([], [], color="#f39c12", lw=2.4, zorder=5, label=f"{args.baseline_label} edge")
    (base_lower,) = ax.plot([], [], color="#f39c12", lw=2.4, zorder=5)
    (opt_upper,) = ax.plot([], [], color="#00bcd4", lw=2.6, zorder=6, label=f"{args.optimized_label} edge")
    (opt_lower,) = ax.plot([], [], color="#00bcd4", lw=2.6, zorder=6)

    base_anchor = ax.scatter([], [], s=70, c="#b9770e", marker="x", linewidths=1.5, zorder=7)
    opt_anchor = ax.scatter([], [], s=70, c="#00838f", marker="+", linewidths=1.8, zorder=8)
    base_tip = ax.scatter([], [], s=85, c="#d35400", marker="o", edgecolors="black", linewidths=0.8, zorder=9)
    opt_tip = ax.scatter([], [], s=85, c="#26c6da", marker="o", edgecolors="black", linewidths=0.8, zorder=10)

    title = ax.text(
        0.01,
        0.99,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#444444"},
    )
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    def _tongue_profile(anc_3d):
        verts, _, _ = tongue_rig.deform(anc_3d)
        mid_mask = np.abs(verts[:, 0]) < float(args.tongue_midline_width)
        mesh_pts = verts[mid_mask][:, [2, 1]]
        upper_edge, lower_edge = _compute_outline(mesh_pts)
        anchors_zy = anc_3d[:, [2, 1]]
        tip_zy = anc_3d[-1:, [2, 1]]
        return mesh_pts, upper_edge, lower_edge, anchors_zy, tip_zy

    def update(frame_idx: int):
        weights = {n: v for n, v in zip(face_model.expression_names, face_seq[frame_idx])}
        face_verts = face_model.deform(weights).copy()

        face_scat.set_offsets(face_verts[face_midline][:, [2, 1]])
        mouth_scat.set_offsets(face_verts[mouth_midline][:, [2, 1]])

        base_mesh_pts, base_upper_pts, base_lower_pts, base_anchors_zy, base_tip_zy = _tongue_profile(
            baseline_seq[frame_idx]
        )
        opt_mesh_pts, opt_upper_pts, opt_lower_pts, opt_anchors_zy, opt_tip_zy = _tongue_profile(
            optimized_seq[frame_idx]
        )

        base_mesh.set_offsets(base_mesh_pts)
        opt_mesh.set_offsets(opt_mesh_pts)
        base_upper.set_data(base_upper_pts[:, 0], base_upper_pts[:, 1])
        base_lower.set_data(base_lower_pts[:, 0], base_lower_pts[:, 1])
        opt_upper.set_data(opt_upper_pts[:, 0], opt_upper_pts[:, 1])
        opt_lower.set_data(opt_lower_pts[:, 0], opt_lower_pts[:, 1])
        base_anchor.set_offsets(base_anchors_zy)
        opt_anchor.set_offsets(opt_anchors_zy)
        base_tip.set_offsets(base_tip_zy)
        opt_tip.set_offsets(opt_tip_zy)

        title.set_text(
            f"{args.dataset_id} | frame {frame_idx + 1}/{frames} | t={frame_idx / float(args.fps):.2f}s\n"
            f"orange={args.baseline_label}  cyan={args.optimized_label}"
        )
        return (
            face_scat,
            mouth_scat,
            base_mesh,
            opt_mesh,
            base_upper,
            base_lower,
            opt_upper,
            opt_lower,
            base_anchor,
            opt_anchor,
            base_tip,
            opt_tip,
            title,
        )

    print(f"Rendering overlay video to {output_video} ({frames} frames)...")
    anim = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / int(args.fps),
        blit=False,
    )
    anim.save(str(output_video), writer=FFMpegWriter(fps=int(args.fps)))
    plt.close(fig)

    if args.with_audio and wav_path.is_file():
        output_with_audio = output_video.with_name(output_video.stem + "_with_audio.mp4")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(output_video),
            "-i",
            str(wav_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_with_audio),
        ]
        print("$", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"Wrote {output_with_audio}")

    print(f"Wrote {output_video}")


if __name__ == "__main__":
    main()
