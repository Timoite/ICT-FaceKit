#!/usr/bin/env python3
"""
Render FULL_FACE and MATPLOTLIB videos (as defined in generate_tongue_animation.py),
stack them side-by-side with ffmpeg, then run ADFA infer_pipeline VSR on the stacked video.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

SCRIPT_DIR = Path(__file__).resolve().parent
TONGUE_SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TONGUE_SCRIPTS_DIR.parent
TONGUE_ANIM_DIR = TONGUE_SCRIPTS_DIR / "tongue_animation"
ADFA_INFER_DIR = (
    PROJECT_ROOT
    / "ADFA_EVALUATION"
    / "Visual_Speech_Recognition_for_Multiple_Languages"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from tongue_scripts.tongue_animation.face_model_io_trimesh import load_face_model_trimesh  # type: ignore
from tongue_scripts.tongue_animation import generate_tongue_animation as gta  # type: ignore


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render FULL_FACE+MATPLOTLIB combo and run VSR.")
    parser.add_argument("--dataset-id", default="1_wayne_0_75_75")
    parser.add_argument("--speaker-id", default="1")
    parser.add_argument(
        "--beat-root",
        default=str(PROJECT_ROOT / "data" / "beat_cache" / "beat_english_v0.2.1" / "beat_english_v0.2.1"),
    )
    parser.add_argument(
        "--motion-path",
        default=str(TONGUE_SCRIPTS_DIR / "outputs" / "1_wayne_0_75_75_lbfgsb_manual.npy"),
    )
    parser.add_argument("--face-model-dir", default=str(PROJECT_ROOT / "FaceXModel"))
    parser.add_argument(
        "--std-path",
        default=str(TONGUE_SCRIPTS_DIR / "normalising_vectors" / "JW13_4points_std.npy"),
    )
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--max-seconds", type=float, default=None, help="Optional clip limit; default is full clip.")
    parser.add_argument("--std-scalar", type=float, default=0.20)
    parser.add_argument("--cutout-mode", action="store_true", default=False)
    parser.add_argument("--no-cutout-mode", dest="cutout_mode", action="store_false")
    parser.add_argument("--detector", default="mediapipe")
    parser.add_argument("--target-fps", type=int, default=25)
    parser.add_argument("--output-dir", default=str(TONGUE_SCRIPTS_DIR / "outputs" / "combo_vsr"))
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return parser.parse_args()


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
    textgrid_path = speaker_dir / f"{args.dataset_id}.TextGrid"

    if not motion_path.is_file():
        raise SystemExit(f"motion not found: {motion_path}")
    if not json_path.is_file():
        raise SystemExit(f"BEAT json not found: {json_path}")
    if not face_model_dir.is_dir():
        raise SystemExit(f"face model dir not found: {face_model_dir}")
    if not std_path.is_file():
        raise SystemExit(f"std path not found: {std_path}")
    if not textgrid_path.is_file():
        raise SystemExit(f"textgrid not found: {textgrid_path}")

    fullface_path = output_dir / f"{args.dataset_id}_FULL_FACE.mp4"
    matplotlib_path = output_dir / f"{args.dataset_id}_MATPLOTLIB.mp4"
    combo_no_audio = output_dir / f"{args.dataset_id}_FULL_FACE_plus_MATPLOTLIB.mp4"
    combo_audio = output_dir / f"{args.dataset_id}_FULL_FACE_plus_MATPLOTLIB_with_audio.mp4"
    vsr_out_dir = output_dir / "vsr_transcripts"
    vsr_out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading face model...")
    face_model = load_face_model_trimesh(str(face_model_dir))
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

    print("Loading animation controls and motion...")
    gta.FPS = int(args.fps)
    gta.CUTOUT_MODE = bool(args.cutout_mode)
    face_seq = gta.process_beat_data(str(json_path), face_model, target_fps=int(args.fps))
    ema_seq = gta.load_ema_motion(str(motion_path), str(std_path), tongue_rig.anchors, float(args.std_scalar))
    if args.max_seconds is None:
        gta.MAX_SECONDS = len(ema_seq) / float(args.fps)
    else:
        gta.MAX_SECONDS = float(args.max_seconds)

    if args.skip_existing and fullface_path.is_file():
        print(f"Skipping FULL_FACE render (exists): {fullface_path}")
    else:
        print("Rendering FULL_FACE...")
        gta.TEMP_VIDEO = str(fullface_path)
        gta.run_pyrender_video(face_model, tongue_rig, ema_seq, face_seq)

    if args.skip_existing and matplotlib_path.is_file():
        print(f"Skipping MATPLOTLIB render (exists): {matplotlib_path}")
    else:
        print("Rendering MATPLOTLIB...")
        gta.TEMP_VIDEO = str(matplotlib_path)
        gta.run_matplotlib_debug(tongue_rig, ema_seq)

    print("Stacking videos with ffmpeg...")
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(fullface_path),
            "-i",
            str(matplotlib_path),
            "-filter_complex",
            "[0:v]setpts=PTS-STARTPTS,scale=800:600[v0];"
            "[1:v]setpts=PTS-STARTPTS,scale=800:600[v1];"
            "[v0][v1]hstack=inputs=2[v]",
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(combo_no_audio),
        ]
    )
    final_combo = combo_no_audio
    if wav_path.is_file():
        print("Muxing audio...")
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(combo_no_audio),
                "-i",
                str(wav_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(combo_audio),
            ]
        )
        final_combo = combo_audio

    print("Running VSR infer_pipeline...")
    run(
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "infer_pipeline.py",
            "--video-path",
            str(final_combo),
            "--textgrid-path",
            str(textgrid_path),
            "--output-dir",
            str(vsr_out_dir),
            "--detector",
            str(args.detector),
            "--target-fps",
            str(int(args.target_fps)),
            "--print-silence-stats",
        ],
        cwd=ADFA_INFER_DIR,
    )

    transcript_path = vsr_out_dir / f"{final_combo.stem}.txt"
    print("=" * 80)
    print("DONE")
    print(f"FULL_FACE video: {fullface_path}")
    print(f"MATPLOTLIB video: {matplotlib_path}")
    print(f"COMBINED video: {final_combo}")
    print(f"VSR transcript: {transcript_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
