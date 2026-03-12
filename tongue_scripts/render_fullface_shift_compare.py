#!/usr/bin/env python3
"""
Render FULL_FACE tongue videos with configurable global tongue shifts.

This wraps the rendering logic from generate_tongue_animation.py so we can
produce controlled sync comparisons for the same clip.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
TONGUE_ANIM_DIR = SCRIPT_DIR / "tongue_animation"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TONGUE_ANIM_DIR) not in sys.path:
    sys.path.insert(0, str(TONGUE_ANIM_DIR))

from face_model_io_trimesh import load_face_model_trimesh  # type: ignore
import generate_tongue_animation as gta  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render FULL_FACE videos for multiple tongue shifts.")
    parser.add_argument("--dataset-id", default="1_wayne_0_75_75")
    parser.add_argument("--speaker-id", default="1")
    parser.add_argument(
        "--beat-root",
        default=str(PROJECT_ROOT / "data" / "beat_cache" / "beat_english_v0.2.1" / "beat_english_v0.2.1"),
    )
    parser.add_argument(
        "--motion-path",
        default=str(SCRIPT_DIR / "outputs" / "1_wayne_0_75_75.npy"),
    )
    parser.add_argument("--face-model-dir", default=str(PROJECT_ROOT / "FaceXModel"))
    parser.add_argument(
        "--std-path",
        default=str(SCRIPT_DIR / "normalising_vectors" / "JW13_4points_std.npy"),
    )
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--std-scalar", type=float, default=0.20)
    parser.add_argument("--cutout-mode", action="store_true", default=False)
    parser.add_argument("--no-cutout-mode", dest="cutout_mode", action="store_false")
    parser.add_argument(
        "--shift-seconds",
        nargs="+",
        type=float,
        default=[0.0, 0.12],
        help="One or more global tongue delays in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "outputs" / "fullface_shift_compare"),
    )
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return parser.parse_args()


def apply_jawopen_offset_correction(face_seq: np.ndarray, face_model) -> np.ndarray:
    if "jawOpen" not in face_model.expression_names:
        return face_seq

    jaw_idx = face_model.expression_names.index("jawOpen")
    raw_vals = face_seq[:, jaw_idx]
    min_val = float(np.min(raw_vals))
    face_seq[:, jaw_idx] = np.maximum(0.0, raw_vals - min_val)
    return face_seq


def shift_sequence(seq: np.ndarray, shift_frames: int) -> np.ndarray:
    if shift_frames == 0:
        return seq
    if shift_frames > 0:
        pad = np.repeat(seq[:1], shift_frames, axis=0)
        return np.concatenate([pad, seq], axis=0)[: len(seq)]

    shift_frames = abs(shift_frames)
    pad = np.repeat(seq[-1:], shift_frames, axis=0)
    return np.concatenate([seq[shift_frames:], pad], axis=0)


def resample_ema(ema_seq: np.ndarray, source_fps: float, target_fps: float, target_len: int) -> np.ndarray:
    if len(ema_seq) == target_len and abs(source_fps - target_fps) < 1e-9:
        return ema_seq

    duration = len(ema_seq) / float(source_fps)
    x_source = np.linspace(0.0, duration, len(ema_seq))
    x_target = np.linspace(0.0, duration, target_len)
    ema_flat = ema_seq.reshape(len(ema_seq), -1)
    ema_resampled = np.zeros((target_len, ema_flat.shape[1]), dtype=np.float32)
    for idx in range(ema_flat.shape[1]):
        ema_resampled[:, idx] = interp1d(x_source, ema_flat[:, idx], kind="cubic")(x_target)
    return ema_resampled.reshape(target_len, 4, 3)


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    beat_root = Path(args.beat_root)
    speaker_dir = beat_root / str(args.speaker_id)
    motion_path = Path(args.motion_path)
    face_model_dir = Path(args.face_model_dir)
    std_path = Path(args.std_path)
    output_dir = Path(args.output_dir) / args.dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = speaker_dir / f"{args.dataset_id}.json"
    wav_path = speaker_dir / f"{args.dataset_id}.wav"

    if not motion_path.is_file():
        raise SystemExit(f"motion not found: {motion_path}")
    if not json_path.is_file():
        raise SystemExit(f"BEAT json not found: {json_path}")
    if not face_model_dir.is_dir():
        raise SystemExit(f"face model dir not found: {face_model_dir}")
    if not std_path.is_file():
        raise SystemExit(f"std path not found: {std_path}")

    print("Loading face model...")
    face_model = load_face_model_trimesh(str(face_model_dir))
    tongue_rig = gta.FaceKitTongueRig(
        face_model.neutral_verts,
        face_model.faces,
        gta.TONGUE_SLICE,
        gta.ANCHOR_INDICES,
        gta.BONE_INDICES,
        {
            "rotation_deg": gta.TONGUE_CONFIG["rotation_deg"],
            "thickness": gta.TONGUE_CONFIG["thickness"],
            "shift_y": gta.TONGUE_CONFIG["shift_y"],
            "shift_z": gta.TONGUE_CONFIG["shift_z"],
            "std_scalar": float(args.std_scalar),
        },
    )

    gta.FPS = int(args.fps)
    gta.CUTOUT_MODE = bool(args.cutout_mode)
    gta.MAX_SECONDS = args.max_seconds

    print("Loading animation controls and motion...")
    face_seq = gta.process_beat_data(str(json_path), face_model, target_fps=int(args.fps))
    face_seq = apply_jawopen_offset_correction(face_seq, face_model)

    ema_seq = gta.load_ema_motion(
        str(motion_path),
        str(std_path),
        tongue_rig.anchors,
        float(args.std_scalar),
    )
    ema_seq = resample_ema(ema_seq, source_fps=50.0, target_fps=float(args.fps), target_len=len(face_seq))

    for shift_seconds in args.shift_seconds:
        shift_tag = f"shift{int(round(shift_seconds * 1000.0)):03d}ms"
        video_no_audio = output_dir / f"{args.dataset_id}_FULL_FACE_{shift_tag}.mp4"
        video_with_audio = output_dir / f"{args.dataset_id}_FULL_FACE_{shift_tag}_with_audio.mp4"

        if args.skip_existing and video_with_audio.is_file():
            print(f"Skipping existing render: {video_with_audio}")
            continue

        print("=" * 80)
        print(f"Rendering FULL_FACE for shift={shift_seconds:.3f}s")
        shifted_ema = shift_sequence(ema_seq, int(round(shift_seconds * args.fps)))
        gta.TEMP_VIDEO = str(video_no_audio)
        gta.run_pyrender_video(face_model, tongue_rig, shifted_ema, face_seq)

        if wav_path.is_file():
            mux_audio(video_no_audio, wav_path, video_with_audio)
        else:
            video_no_audio.replace(video_with_audio)

        print(f"Saved: {video_with_audio}")


if __name__ == "__main__":
    main()
