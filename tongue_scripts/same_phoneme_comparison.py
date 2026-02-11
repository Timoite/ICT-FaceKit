#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "matplotlib>=3.8",
#     "scipy>=1.11",
#     "trimesh",
#     "pyrender",
#     "opencv-python",
# ]
# ///
"""Unified pipeline for sagittal/oblique render, merge, and phoneme clips."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from face_model_io_trimesh import load_face_model_trimesh
    from test import process_beat_data, load_ema_motion, FaceKitTongueRig, TONGUE_CONFIG
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT))
    from tongue_scripts.face_model_io_trimesh import load_face_model_trimesh
    from tongue_scripts.test import (
        process_beat_data,
        load_ema_motion,
        FaceKitTongueRig,
        TONGUE_CONFIG,
    )

TONGUE_SLICE = slice(16611, 17039)
ANCHOR_INDICES = [16661, 16696, 16755, 16758]
BONE_INDICES = [16661, 16757]
FPS = 25
TONGUE_SOURCE_FPS = 50
TONGUE_SHIFT_SECONDS = 0.120


@dataclass
class PhoneInterval:
    idx: int
    start: float
    end: float
    label: str
    normalized: str


def normalize_phone(label: str) -> str:
    return re.sub(r"[0-9]", "", label.upper())


def _parse_textgrid_tier(textgrid_path: Path, tier_name: str):
    """Yield (idx, start, end, text) tuples for a TextGrid tier."""
    in_tier = False
    current: dict = {}
    idx = 0
    with textgrid_path.open("r", encoding="utf-8") as f:
        for raw in f:
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
                current["s"] = line.split("=", 1)[1].strip()
            elif line.startswith("xmax ="):
                current["e"] = line.split("=", 1)[1].strip()
            elif line.startswith("text ="):
                txt = line.split("=", 1)[1].strip()
                if txt.startswith('"') and txt.endswith('"'):
                    txt = txt[1:-1]
                current["t"] = txt
                if {"s", "e", "t"} <= current.keys():
                    try:
                        s, e = float(current["s"]), float(current["e"])
                    except ValueError:
                        s, e = 0.0, 0.0
                    yield idx, s, e, current["t"]
                    idx += 1


def parse_phones(path: Path) -> List[PhoneInterval]:
    out = []
    for idx, s, e, txt in _parse_textgrid_tier(path, "phones"):
        if txt.strip():
            out.append(
                PhoneInterval(idx=idx, start=s, end=e, label=txt, normalized=normalize_phone(txt))
            )
    return out


def _run_ffmpeg(cmd: List[str]):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def _atempo_chain(speed: float) -> str:
    """Build an atempo chain for speeds outside [0.5, 2.0]."""
    if speed <= 0:
        raise ValueError("speed must be > 0")
    factors = []
    remaining = speed
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    factors.append(remaining)
    return ",".join(f"atempo={f:.6f}" for f in factors)


def _get_video_duration(path: Path) -> float:
    """Return video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def _extend_sequence(seq: np.ndarray, frames: int) -> np.ndarray:
    if len(seq) >= frames:
        return seq
    pad = np.repeat(seq[-1:], frames - len(seq), axis=0)
    return np.concatenate([seq, pad], axis=0)


def _resample_ema_motion(ema_seq: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    if np.isclose(source_fps, target_fps):
        return ema_seq
    n_frames = len(ema_seq)
    duration = n_frames / source_fps
    n_target_frames = max(1, int(duration * target_fps))

    x_source = np.linspace(0.0, duration, n_frames)
    x_target = np.linspace(0.0, duration, n_target_frames)

    ema_flat = ema_seq.reshape(n_frames, -1)
    ema_resampled = np.empty((n_target_frames, ema_flat.shape[1]), dtype=ema_seq.dtype)
    for i in range(ema_flat.shape[1]):
        ema_resampled[:, i] = np.interp(x_target, x_source, ema_flat[:, i])

    return ema_resampled.reshape(n_target_frames, ema_seq.shape[1], ema_seq.shape[2])


def _shift_sequence(seq: np.ndarray, shift_frames: int) -> np.ndarray:
    if shift_frames == 0:
        return seq
    n = len(seq)
    if shift_frames > 0:
        pad = np.repeat(seq[:1], shift_frames, axis=0)
        shifted = np.concatenate([pad, seq], axis=0)[:n]
    else:
        shift_frames = abs(shift_frames)
        pad = np.repeat(seq[-1:], shift_frames, axis=0)
        shifted = np.concatenate([seq[shift_frames:], pad], axis=0)
    return shifted


def _load_face_and_sequences(
    dataset_id: str,
    beat_root: Path,
    npy_dir: Path,
    std_path: Path,
    face_model_dir: Path,
):
    json_path = beat_root / f"{dataset_id}.json"
    npy_path = npy_dir / f"{dataset_id}.npy"

    print("Loading face model ...")
    face_model = load_face_model_trimesh(str(face_model_dir))

    print("Loading BEAT blendshapes ...")
    try:
        face_seq = process_beat_data(str(json_path), face_model, target_fps=FPS)
    except Exception as exc:
        print(f"  Warning: BEAT JSON load failed ({exc}); using neutral face")
        face_seq = np.zeros((3000, len(face_model.expression_names)), dtype=np.float32)

    # --- FIX: LIP CLOSURE (Systematic Error Correction) ---
    # Shift 'jawOpen' so that its minimum value is 0.0 (fully closed).
    if "jawOpen" in face_model.expression_names:
        jaw_idx = face_model.expression_names.index("jawOpen")
        raw_vals = face_seq[:, jaw_idx]
        min_val = float(np.min(raw_vals))
        if min_val != 0.0:
            print(
                f"  [Correction] jawOpen min value: {min_val:.4f}. shifting by {-min_val:.4f} to 0.0"
            )
        face_seq[:, jaw_idx] = np.maximum(0.0, raw_vals - min_val)
    else:
        print("  [Warning] 'jawOpen' not found in expression names. Skipping lip correction.")

    print("Setting up tongue rig ...")
    tongue_rig = FaceKitTongueRig(
        face_model.neutral_verts,
        face_model.faces,
        TONGUE_SLICE,
        ANCHOR_INDICES,
        BONE_INDICES,
        TONGUE_CONFIG,
    )

    print("Loading EMA motion ...")
    ema_seq = load_ema_motion(
        str(npy_path),
        str(std_path),
        tongue_rig.anchors,
        TONGUE_CONFIG["std_scalar"],
    )
    if not np.isclose(TONGUE_SOURCE_FPS, FPS):
        ema_seq = _resample_ema_motion(ema_seq, TONGUE_SOURCE_FPS, FPS)
    if TONGUE_SHIFT_SECONDS != 0.0:
        shift_frames = int(round(TONGUE_SHIFT_SECONDS * FPS))
        ema_seq = _shift_sequence(ema_seq, shift_frames)
    return face_model, face_seq, tongue_rig, ema_seq


def render_sagittal(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import animation

    face_model, face_seq, tongue_rig, ema_seq = _load_face_and_sequences(
        args.dataset_id,
        Path(args.beat_root),
        Path(args.tongue_npy_dir),
        Path(args.std_path),
        Path(args.face_model_dir),
    )

    if args.max_seconds is not None:
        frames = max(1, int(args.max_seconds * FPS))
        face_seq = _extend_sequence(face_seq, frames)
        ema_seq = _extend_sequence(ema_seq, frames)
    else:
        frames = min(len(face_seq), len(ema_seq))

    face_midline = np.abs(face_model.neutral_verts[:, 0]) < 0.5
    face_midline[TONGUE_SLICE] = False

    gum_midline = np.abs(face_model.neutral_verts[:, 0]) < 0.5
    gum_midline[:14062] = False
    gum_midline[TONGUE_SLICE] = False

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0.08, 0.08, 0.9, 0.88])
    ax.set_aspect("equal")
    ax.set_xlabel("Z (Anterior ->)")
    ax.set_ylabel("Y (Superior ->)")
    ax.set_xlim(-2, 14)
    ax.set_ylim(-12, 2)

    face_scat = ax.scatter([], [], s=1, c="#aaaaaa", alpha=0.4, zorder=1)
    gum_scat = ax.scatter([], [], s=1, c="#cc9999", alpha=0.35, zorder=1)
    tongue_scat = ax.scatter([], [], s=3, c="#88bbff", alpha=0.5, zorder=2)
    (ema_line,) = ax.plot([], [], "b-", lw=1.5, alpha=0.5, zorder=3, label="EMA spline")
    ema_anch = ax.scatter([], [], s=80, c="blue", marker="x", linewidths=1.5, zorder=4)

    ax.legend(loc="upper left", fontsize=8)

    def init():
        face_scat.set_offsets(np.empty((0, 2)))
        gum_scat.set_offsets(np.empty((0, 2)))
        tongue_scat.set_offsets(np.empty((0, 2)))
        ema_line.set_data([], [])
        ema_anch.set_offsets(np.empty((0, 2)))
        return face_scat, gum_scat, tongue_scat, ema_line, ema_anch

    def update(frame_idx: int):
        weights = {n: v for n, v in zip(face_model.expression_names, face_seq[frame_idx])}
        verts = face_model.deform(weights).copy()

        face_pts = verts[face_midline][:, [2, 1]]
        gum_pts = verts[gum_midline][:, [2, 1]]
        face_scat.set_offsets(face_pts)
        gum_scat.set_offsets(gum_pts)

        ema_anc = ema_seq[frame_idx]
        t_verts, _, t_sp = tongue_rig.deform(ema_anc)
        t_mid = np.abs(t_verts[:, 0]) < 1.0
        tongue_scat.set_offsets(t_verts[t_mid][:, [2, 1]])

        u = np.linspace(0, 1, 100)
        sp_pts = t_sp(u)[:, [2, 1]]
        ema_line.set_data(sp_pts[:, 0], sp_pts[:, 1])
        ema_anch.set_offsets(ema_anc[:, [2, 1]])

        return face_scat, gum_scat, tongue_scat, ema_line, ema_anch

    output_path = Path(args.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Rendering {frames} frames to {output_path} ...")
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        interval=1000 / FPS,
        blit=False,
    )

    writer = animation.FFMpegWriter(fps=FPS, codec="libx264", bitrate=3000)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)

    if args.audio:
        audio_path = Path(args.audio)
        if audio_path.exists():
            muxed_path = output_path.with_name(output_path.stem + "_with_audio.mp4")
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(output_path),
                "-i",
                str(audio_path),
                "-c:v",
                "copy",
                "-c:a",
                "libmp3lame",
                "-q:a",
                "4",
                "-shortest",
                str(muxed_path),
            ]
            print("Muxing audio into sagittal video ...")
            subprocess.run(cmd, check=True)
            print(f"Wrote {muxed_path}")


def render_oblique(args: argparse.Namespace) -> None:
    import cv2
    import pyrender
    import trimesh

    face_model, face_seq, tongue_rig, ema_seq = _load_face_and_sequences(
        args.dataset_id,
        Path(args.beat_root),
        Path(args.tongue_npy_dir),
        Path(args.std_path),
        Path(args.face_model_dir),
    )

    if args.max_seconds is not None:
        frames = max(1, int(args.max_seconds * FPS))
        face_seq = _extend_sequence(face_seq, frames)
        ema_seq = _extend_sequence(ema_seq, frames)
    else:
        frames = min(len(face_seq), len(ema_seq))

    W, H = 1000, 700
    renderer = pyrender.OffscreenRenderer(W, H)
    output_path = Path(args.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    eye = np.array([0.0, -2.0, 35.0], dtype=np.float32)
    target = np.array([0.0, -2.0, 0.0], dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = np.column_stack((x, y, z))
    cam_pose[:3, 3] = eye

    mat_skin = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.5, 0.5, 0.5, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.8,
        alphaMode="OPAQUE",
    )
    mat_tongue = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 0.6, 0.6, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.2,
        alphaMode="OPAQUE",
    )
    mat_gums = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.7, 0.2, 0.2, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.2,
        alphaMode="OPAQUE",
    )

    scene_base = pyrender.Scene(bg_color=[0, 0, 0])
    spot_pose = cam_pose.copy()
    spot_pose[:3, 3] += [0, 10, -5]
    spot_light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=100,
        innerConeAngle=np.pi / 8,
        outerConeAngle=np.pi / 4,
    )
    fill_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=400)

    scene_base.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=cam_pose)
    scene_base.add(spot_light, pose=spot_pose)
    scene_base.add(fill_light, pose=cam_pose)

    is_tongue_vert = np.zeros(len(face_model.neutral_verts), dtype=bool)
    is_tongue_vert[tongue_rig.global_indices] = True

    is_gum_vert = np.zeros(len(face_model.neutral_verts), dtype=bool)
    is_gum_vert[14062:17039] = True
    is_gum_vert[is_tongue_vert] = False

    for i in range(frames):
        if i % 25 == 0:
            print(f"Rendering frame {i}/{frames} ...")

        weights = {name: val for name, val in zip(face_model.expression_names, face_seq[i])}
        verts = face_model.deform(weights).copy()

        t_verts, _, _ = tongue_rig.deform(ema_seq[i])
        verts[tongue_rig.global_indices] = t_verts

        current_faces = face_model.faces

        face_vert_is_tongue = is_tongue_vert[current_faces]
        is_tongue_face = face_vert_is_tongue.all(axis=1)

        face_vert_is_gum = is_gum_vert[current_faces]
        is_gum_face = face_vert_is_gum.all(axis=1)

        is_skin_face = ~(is_tongue_face | is_gum_face)

        faces_tongue = current_faces[is_tongue_face]
        faces_gum = current_faces[is_gum_face]
        faces_skin = current_faces[is_skin_face]

        nodes = []

        if len(faces_skin) > 0:
            tm_skin = trimesh.Trimesh(verts, faces_skin, process=False)
            mesh_skin = pyrender.Mesh.from_trimesh(tm_skin, material=mat_skin, smooth=True)
            if mesh_skin.primitives:
                for p in mesh_skin.primitives:
                    p.material.doubleSided = True
            nodes.append(scene_base.add(mesh_skin))

        if len(faces_tongue) > 0:
            tm_tongue = trimesh.Trimesh(verts, faces_tongue, process=False)
            mesh_tongue = pyrender.Mesh.from_trimesh(tm_tongue, material=mat_tongue, smooth=True)
            if mesh_tongue.primitives:
                for p in mesh_tongue.primitives:
                    p.material.doubleSided = True
            nodes.append(scene_base.add(mesh_tongue))

        if len(faces_gum) > 0:
            tm_gum = trimesh.Trimesh(verts, faces_gum, process=False)
            mesh_gum = pyrender.Mesh.from_trimesh(tm_gum, material=mat_gums, smooth=True)
            if mesh_gum.primitives:
                for p in mesh_gum.primitives:
                    p.material.doubleSided = True
            nodes.append(scene_base.add(mesh_gum))

        color, _ = renderer.render(scene_base)
        video.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

        for n in nodes:
            scene_base.remove_node(n)

    video.release()
    renderer.delete()


def combine_videos(args: argparse.Namespace) -> None:
    sagittal = Path(args.sagittal)
    oblique = Path(args.oblique)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(sagittal),
        "-i",
        str(oblique),
        "-filter_complex",
        "[0:v][1:v]hstack=inputs=2[v]",
        "-map",
        "[v]",
        "-map",
        "0:a",
        "-c:v",
        "libx264",
        "-c:a",
        "libmp3lame",
        "-q:a",
        "4",
        "-shortest",
        str(output),
    ]
    _run_ffmpeg(cmd)


def cut_phoneme_clips(args: argparse.Namespace) -> None:
    beat_root = Path(args.beat_root)
    textgrid_path = beat_root / f"{args.dataset_id}.TextGrid"
    audio_path = beat_root / f"{args.dataset_id}.wav"
    input_video = Path(args.input_video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not textgrid_path.exists():
        raise FileNotFoundError(textgrid_path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    if not input_video.exists():
        raise FileNotFoundError(input_video)

    video_duration = _get_video_duration(input_video)

    phones = parse_phones(textgrid_path)
    if not phones:
        raise RuntimeError("No phones found in TextGrid")

    manifest = {}

    for i, ph in enumerate(phones):
        left_idx = max(0, i - args.context)
        right_idx = min(len(phones) - 1, i + args.context)
        left = phones[left_idx] if left_idx < i else None
        right = phones[right_idx] if right_idx > i else None

        start = phones[left_idx].start
        end = phones[right_idx].end
        if start >= video_duration:
            continue
        end = min(end, video_duration)
        min_dur = 1.0 / FPS
        if end - start < min_dur:
            continue

        label = ph.normalized
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        out_name = f"idx{ph.idx}_{start:.2f}-{end:.2f}.mp4"
        out_path = label_dir / out_name

        setpts = 1.0 / args.speed
        atempo = _atempo_chain(args.speed)
        clip_start = start
        clip_duration = end - start
        marker_duration = max(1.0 / FPS, args.marker_seconds)
        target_start = (ph.start - clip_start) / args.speed
        target_end = (ph.end - clip_start) / args.speed
        marker_end = clip_duration / args.speed

        target_start = max(0.0, min(target_start, marker_end))
        target_end = max(0.0, min(target_end, marker_end))

        marker_filters = ""
        if args.mark_target:
            marker_filters = (
                ",drawbox=x=(iw/2-3):y=0:w=6:h=ih:color=red@0.95:"
                f"t=fill:enable='between(t,{target_start:.6f},{(target_start + marker_duration):.6f})'"
                ",drawbox=x=(iw/2-3):y=0:w=6:h=ih:color=red@0.95:"
                f"t=fill:enable='between(t,{target_end:.6f},{(target_end + marker_duration):.6f})'"
            )

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(input_video),
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(audio_path),
            "-filter_complex",
            f"[0:v]setpts={setpts}*PTS{marker_filters}[v];[1:a]{atempo}[a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-shortest",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "4",
            str(out_path),
        ]
        _run_ffmpeg(cmd)

        manifest.setdefault(label, []).append(
            {
                "phoneme": ph.label,
                "idx": ph.idx,
                "start": start,
                "end": end,
                "left": left.label if left else "",
                "right": right.label if right else "",
                "context": args.context,
                "clip": str(out_path),
            }
        )

    manifest_path = output_dir / f"{args.dataset_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest -> {manifest_path}")


def _add_render_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-id", default="1_wayne_0_75_75")
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
    )
    parser.add_argument("--tongue-npy-dir", default=str(SCRIPT_DIR / "outputs"))
    parser.add_argument(
        "--std-path",
        default=str(SCRIPT_DIR / "normalising_vectors" / "JW13_4points_std.npy"),
    )
    parser.add_argument("--face-model-dir", default=str(PROJECT_ROOT / "FaceXModel"))
    parser.add_argument("--max-seconds", type=float, default=None)


def _default_beat_root() -> Path:
    return (
        PROJECT_ROOT
        / "ADFA_EVALUATION"
        / "data"
        / "beat_cache_speaker1"
        / "beat_english_v0.2.1"
        / "beat_english_v0.2.1"
        / "1"
    )


def _run_default_pipeline() -> None:
    dataset_id = "1_wayne_0_75_75"
    beat_root = _default_beat_root()
    audio_path = beat_root / f"{dataset_id}.wav"
    audio_duration = _get_video_duration(audio_path)
    outputs_dir = SCRIPT_DIR / "outputs"

    sag_path = outputs_dir / f"{dataset_id}_sagittal.mp4"
    oblique_path = outputs_dir / f"{dataset_id}_oblique.mp4"
    sag_with_audio = outputs_dir / f"{dataset_id}_sagittal_with_audio.mp4"
    side_by_side = outputs_dir / f"{dataset_id}_side_by_side_with_audio.mp4"
    clips_dir = outputs_dir / "phoneme_comparision_videos"

    render_sagittal(
        argparse.Namespace(
            dataset_id=dataset_id,
            beat_root=str(beat_root),
            tongue_npy_dir=str(outputs_dir),
            std_path=str(SCRIPT_DIR / "normalising_vectors" / "JW13_4points_std.npy"),
            face_model_dir=str(PROJECT_ROOT / "FaceXModel"),
            output_video=str(sag_path),
            audio=str(audio_path),
            max_seconds=audio_duration,
        )
    )

    render_oblique(
        argparse.Namespace(
            dataset_id=dataset_id,
            beat_root=str(beat_root),
            tongue_npy_dir=str(outputs_dir),
            std_path=str(SCRIPT_DIR / "normalising_vectors" / "JW13_4points_std.npy"),
            face_model_dir=str(PROJECT_ROOT / "FaceXModel"),
            output_video=str(oblique_path),
            max_seconds=audio_duration,
        )
    )

    combine_videos(
        argparse.Namespace(
            sagittal=str(sag_with_audio),
            oblique=str(oblique_path),
            output=str(side_by_side),
        )
    )

    cut_phoneme_clips(
        argparse.Namespace(
            dataset_id=dataset_id,
            beat_root=str(beat_root),
            input_video=str(side_by_side),
            output_dir=str(clips_dir),
            speed=0.1,
            context=3,
            mark_target=True,
            marker_seconds=0.08,
        )
    )


def main() -> None:
    if len(sys.argv) == 1:
        _run_default_pipeline()
        return

    parser = argparse.ArgumentParser(
        description="Render sagittal/oblique videos, merge, and cut phoneme clips"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sag_parser = subparsers.add_parser("render-sagittal", help="Render sagittal video")
    _add_render_args(sag_parser)
    sag_parser.add_argument(
        "--output-video",
        default=str(SCRIPT_DIR / "outputs" / "1_wayne_0_75_75_sagittal.mp4"),
    )
    sag_parser.add_argument(
        "--audio",
        default=None,
        help="Optional WAV to mux into the rendered video",
    )
    sag_parser.set_defaults(func=render_sagittal)

    ob_parser = subparsers.add_parser("render-oblique", help="Render oblique video")
    _add_render_args(ob_parser)
    ob_parser.add_argument(
        "--output-video",
        default=str(SCRIPT_DIR / "outputs" / "1_wayne_0_75_75_oblique.mp4"),
    )
    ob_parser.set_defaults(func=render_oblique)

    comb_parser = subparsers.add_parser("combine", help="Combine sagittal + oblique")
    comb_parser.add_argument("--sagittal", required=True, help="Sagittal video (with audio)")
    comb_parser.add_argument("--oblique", required=True, help="Oblique video (no audio)")
    comb_parser.add_argument("--output", required=True, help="Output side-by-side video")
    comb_parser.set_defaults(func=combine_videos)

    cut_parser = subparsers.add_parser("cut-clips", help="Cut phoneme clips with context")
    cut_parser.add_argument("--dataset-id", default="1_wayne_0_75_75")
    cut_parser.add_argument(
        "--beat-root",
        default=str(_default_beat_root()),
    )
    cut_parser.add_argument(
        "--input-video",
        default=str(SCRIPT_DIR / "outputs" / "1_wayne_0_75_75_sagittal.mp4"),
    )
    cut_parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "outputs" / "phoneme_comparision_videos"),
    )
    cut_parser.add_argument("--speed", type=float, default=0.1, help="Playback speed")
    cut_parser.add_argument(
        "--context",
        type=int,
        default=3,
        help="Number of adjacent phonemes to include on each side",
    )
    cut_parser.add_argument(
        "--mark-target",
        action="store_true",
        default=True,
        help="Add red markers at target start/end in the clip",
    )
    cut_parser.add_argument(
        "--no-mark-target",
        action="store_false",
        dest="mark_target",
        help="Disable red markers at target start/end",
    )
    cut_parser.add_argument(
        "--marker-seconds",
        type=float,
        default=0.08,
        help="Duration of the red marker line in seconds",
    )
    cut_parser.set_defaults(func=cut_phoneme_clips)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
