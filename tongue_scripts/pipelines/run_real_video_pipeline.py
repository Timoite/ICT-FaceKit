#!/usr/bin/env python3
"""Run the real-video SMIRK -> ARKit -> ICT FaceKit tongue pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

SCRIPT_DIR = Path(__file__).resolve().parent
TONGUE_SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TONGUE_SCRIPTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tongue_scripts.real_video.arkit_to_ict import (
    convert_csv_to_ict_outputs,
)
from tongue_scripts.real_video.extract_smirk_sequence import (
    SmirkExtractionConfig,
    extract_smirk_sequence,
)
from tongue_scripts.real_video.smirk_flame_to_arkit import fit_smirk_vertices_file


DEFAULT_WAVLM_CHECKPOINT = (
    TONGUE_SCRIPTS_DIR
    / "inversion_checkpoints"
    / "lora_multispeaker_consistency_alpha_0.25_threshold_0_vctk_vvn_4tonguepoints"
)
DEFAULT_STD_PATH = TONGUE_SCRIPTS_DIR / "normalising_vectors" / "JW13_4points_std.npy"
DEFAULT_FACE_MODEL_DIR = PROJECT_ROOT / "FaceXModel"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record-video pipeline: SMIRK geometry to ARKit/ICT coefficients plus WavLM tongue render."
    )
    parser.add_argument("--video", required=True, help="Input talking-head video.")
    parser.add_argument("--out-dir", required=True, help="Output directory for all derived artifacts.")
    parser.add_argument("--transcript", default=None, help="Transcript text or path to transcript text file.")
    parser.add_argument("--smirk-root", default=None, help="Local clone of https://github.com/georgeretsi/smirk.")
    parser.add_argument("--smirk-checkpoint", default=None, help="SMIRK checkpoint, e.g. pretrained_models/SMIRK_em1.pt.")
    parser.add_argument("--flame-model-path", default=None, help="FLAME2020 generic_model.pkl used by SMIRK.")
    parser.add_argument("--said-data-dir", default=None, help="SAiD data directory containing ARKit_blendshapes.txt and residuals.")
    parser.add_argument("--said-person-id", default=None, help="Optional VOCA person id inside SAiD blendshape_residuals.pickle.")
    parser.add_argument("--fps", type=float, default=25.0, help="Face/render frame rate. Keep 25 for VSR validation.")
    parser.add_argument("--device", default="auto", help="SMIRK torch device: auto, cpu, cuda, mps.")
    parser.add_argument("--face-model-dir", default=str(DEFAULT_FACE_MODEL_DIR))
    parser.add_argument("--std-path", default=str(DEFAULT_STD_PATH))
    parser.add_argument("--wavlm-checkpoint", default=str(DEFAULT_WAVLM_CHECKPOINT))
    parser.add_argument("--tongue-motion", default=None, help="Existing WavLM tongue .npy. If omitted, infer from audio.")
    parser.add_argument("--tongue-shift-seconds", type=float, default=0.120)
    parser.add_argument("--temporal-delta", type=float, default=0.1)
    parser.add_argument("--qp-chunk-size", type=int, default=120)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional SMIRK frame limit for quick smoke tests.")
    parser.add_argument("--no-crop", action="store_true", help="Disable SMIRK MediaPipe crop.")
    parser.add_argument("--skip-smirk", action="store_true", help="Reuse existing smirk_flame_vertices.npz.")
    parser.add_argument("--skip-fit", action="store_true", help="Reuse existing arkit_coeffs.csv.")
    parser.add_argument("--skip-wavlm", action="store_true", help="Do not infer WavLM tongue motion.")
    parser.add_argument("--skip-render", action="store_true", help="Stop after coefficient generation.")
    parser.add_argument("--skip-vsr", action="store_true", help="Do not run VSR even when transcript is provided.")
    return parser.parse_args()


def require_path(path: Path | None, label: str) -> Path:
    if path is None:
        raise SystemExit(f"--{label} is required for this step")
    if not path.exists():
        raise SystemExit(f"Missing {label.replace('-', ' ')}: {path}")
    return path


def extract_audio(video_path: Path, audio_path: Path) -> Path:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "quiet",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(audio_path),
    ]
    subprocess.run(cmd, check=True)
    if not audio_path.is_file():
        raise RuntimeError(f"ffmpeg did not create audio file: {audio_path}")
    return audio_path


def run_wavlm_inversion(wav_path: Path, out_path: Path, checkpoint_path: Path) -> Path:
    from tongue_scripts.inversion.invert import infer_ema

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing WavLM checkpoint: {checkpoint_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ema = infer_ema(wav_path, checkpoint_path)
    np.save(out_path, ema)
    print(f"Saved WavLM tongue motion: {out_path} shape={ema.shape}")
    return out_path


def resample_ema_to_face_frames(ema_seq: np.ndarray, face_frames: int, source_fps: float, target_fps: float) -> np.ndarray:
    if len(ema_seq) == face_frames:
        return ema_seq
    if len(ema_seq) < 2:
        return np.repeat(ema_seq[:1], face_frames, axis=0)

    x_source = np.arange(len(ema_seq), dtype=np.float32) / float(source_fps)
    x_target = np.arange(face_frames, dtype=np.float32) / float(target_fps)
    flat = ema_seq.reshape(len(ema_seq), -1)
    interp_kind = "cubic" if len(ema_seq) >= 4 else "linear"
    interp = interp1d(
        x_source,
        flat,
        axis=0,
        kind=interp_kind,
        bounds_error=False,
        fill_value=(flat[0], flat[-1]),
    )
    return interp(x_target).reshape(face_frames, *ema_seq.shape[1:]).astype(np.float32)


def render_outputs(
    face_model_dir: Path,
    ict_npz_path: Path,
    tongue_motion_path: Path,
    std_path: Path,
    audio_path: Path,
    out_dir: Path,
    fps: float,
    tongue_shift_seconds: float,
) -> tuple[Path, Path]:
    from tongue_scripts.rendering.render_dual_tongue_comparison import (
        ANCHOR_INDICES,
        BONE_INDICES,
        TONGUE_CONFIG,
        TONGUE_SLICE,
        apply_jawopen_offset_correction,
        merge_audio,
        render_video_with_dynamic_tongue,
        render_video_with_passive_tongue,
        shift_sequence,
    )
    from tongue_scripts.tongue_animation.face_model_io_trimesh import load_face_model_trimesh
    from tongue_scripts.tongue_animation.generate_tongue_animation import (
        FaceKitTongueRig,
        load_ema_motion,
    )

    face_model = load_face_model_trimesh(str(face_model_dir))
    ict_data = np.load(ict_npz_path, allow_pickle=True)
    face_seq = np.asarray(ict_data["coeffs"], dtype=np.float32)
    face_seq = apply_jawopen_offset_correction(face_seq, face_model)

    tongue_rig = FaceKitTongueRig(
        face_model.neutral_verts,
        face_model.faces,
        TONGUE_SLICE,
        ANCHOR_INDICES,
        BONE_INDICES,
        TONGUE_CONFIG,
    )
    ema_seq = load_ema_motion(
        str(tongue_motion_path),
        str(std_path),
        tongue_rig.anchors,
        TONGUE_CONFIG["std_scalar"],
    )
    ema_seq = resample_ema_to_face_frames(ema_seq, len(face_seq), source_fps=50.0, target_fps=fps)
    if tongue_shift_seconds:
        shift_frames = int(round(tongue_shift_seconds * fps))
        ema_seq = shift_sequence(ema_seq, shift_frames)

    dynamic_raw = out_dir / "real_video_with_tongue.mp4"
    passive_raw = out_dir / "real_video_passive_tongue.mp4"
    render_video_with_dynamic_tongue(
        face_model,
        face_seq,
        tongue_rig,
        ema_seq,
        str(dynamic_raw),
        fps=fps,
        max_seconds=None,
    )
    render_video_with_passive_tongue(
        face_model,
        face_seq,
        tongue_rig,
        str(passive_raw),
        fps=fps,
        max_seconds=None,
    )

    dynamic_audio = out_dir / "real_video_with_tongue_with_audio.mp4"
    passive_audio = out_dir / "real_video_passive_tongue_with_audio.mp4"
    merge_audio(str(dynamic_raw), str(audio_path), str(dynamic_audio))
    merge_audio(str(passive_raw), str(audio_path), str(passive_audio))
    return dynamic_audio, passive_audio


def _read_transcript(transcript_arg: str) -> str:
    try:
        path = Path(transcript_arg)
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        pass
    return transcript_arg.strip()


def run_vsr_report(video_path: Path, synthetic_videos: list[Path], transcript: str, out_dir: Path) -> Path:
    gt_path = out_dir / "ground_truth.txt"
    gt_path.write_text(transcript.strip() + "\n", encoding="utf-8")
    report_path = out_dir / "vsr_domain_gap_report.md"
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_vsr_ver.py"),
        "--videos",
        str(video_path),
        *[str(path) for path in synthetic_videos],
        "--ground-truth",
        str(gt_path),
        "--report-path",
        str(report_path),
        "--report-mode",
        "write",
        "--experiment-name",
        "real_video_smirk_domain_gap",
        "--dataset-id",
        video_path.stem,
        "--hypothesis",
        "compare original real video against SMIRK-derived ICT passive and active tongue renders",
    ]
    subprocess.run(cmd, check=True)
    return report_path


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not video_path.is_file():
        raise SystemExit(f"Missing input video: {video_path}")

    audio_path = extract_audio(video_path, out_dir / "audio.wav")
    print(f"Saved audio: {audio_path}")

    params_path = out_dir / "smirk_params.npz"
    vertices_path = out_dir / "smirk_flame_vertices.npz"
    if not args.skip_smirk:
        smirk_root = require_path(Path(args.smirk_root).expanduser().resolve() if args.smirk_root else None, "smirk-root")
        smirk_checkpoint = require_path(
            Path(args.smirk_checkpoint).expanduser().resolve() if args.smirk_checkpoint else None,
            "smirk-checkpoint",
        )
        flame_model_path = require_path(
            Path(args.flame_model_path).expanduser().resolve() if args.flame_model_path else None,
            "flame-model-path",
        )
        params_path, vertices_path = extract_smirk_sequence(
            SmirkExtractionConfig(
                video_path=video_path,
                out_dir=out_dir,
                smirk_root=smirk_root,
                checkpoint_path=smirk_checkpoint,
                flame_model_path=flame_model_path,
                fps=args.fps,
                device=args.device,
                crop=not args.no_crop,
                max_frames=args.max_frames,
            )
        )
    elif not vertices_path.is_file():
        raise SystemExit(f"--skip-smirk requested, but missing {vertices_path}")

    arkit_csv = out_dir / "arkit_coeffs.csv"
    diagnostics_json = out_dir / "arkit_fit_diagnostics.json"
    if not args.skip_fit:
        said_data_dir = require_path(
            Path(args.said_data_dir).expanduser().resolve() if args.said_data_dir else None,
            "said-data-dir",
        )
        fit_smirk_vertices_file(
            vertices_npz=vertices_path,
            said_data_dir=said_data_dir,
            coeffs_csv=arkit_csv,
            diagnostics_json=diagnostics_json,
            temporal_delta=args.temporal_delta,
            chunk_size=args.qp_chunk_size,
            said_person_id=args.said_person_id,
            fps=args.fps,
        )
    elif not arkit_csv.is_file():
        raise SystemExit(f"--skip-fit requested, but missing {arkit_csv}")

    ict_npz = out_dir / "ict_coeffs.npz"
    motion_json = out_dir / "arkit_face_motion.json"
    convert_csv_to_ict_outputs(
        arkit_csv=arkit_csv,
        face_model_dir=Path(args.face_model_dir).expanduser().resolve(),
        ict_npz=ict_npz,
        motion_json=motion_json,
        fps=args.fps,
        source_video=video_path,
    )
    print(f"Saved ICT coeffs: {ict_npz}")
    print(f"Saved transition JSON: {motion_json}")

    tongue_motion_path = Path(args.tongue_motion).expanduser().resolve() if args.tongue_motion else out_dir / "tongue_motion.npy"
    if args.tongue_motion:
        require_path(tongue_motion_path, "tongue-motion")
    elif not args.skip_wavlm:
        run_wavlm_inversion(audio_path, tongue_motion_path, Path(args.wavlm_checkpoint).expanduser().resolve())

    synthetic_videos: list[Path] = []
    if not args.skip_render:
        if not tongue_motion_path.is_file():
            raise SystemExit(
                "Rendering requires WavLM tongue motion. Provide --tongue-motion or omit --skip-wavlm."
            )
        dynamic_video, passive_video = render_outputs(
            face_model_dir=Path(args.face_model_dir).expanduser().resolve(),
            ict_npz_path=ict_npz,
            tongue_motion_path=tongue_motion_path,
            std_path=Path(args.std_path).expanduser().resolve(),
            audio_path=audio_path,
            out_dir=out_dir,
            fps=args.fps,
            tongue_shift_seconds=args.tongue_shift_seconds,
        )
        synthetic_videos = [dynamic_video, passive_video]
        print(f"Saved dynamic tongue video: {dynamic_video}")
        print(f"Saved passive tongue video: {passive_video}")

    if args.transcript and not args.skip_vsr:
        if not synthetic_videos:
            synthetic_videos = [
                out_dir / "real_video_with_tongue_with_audio.mp4",
                out_dir / "real_video_passive_tongue_with_audio.mp4",
            ]
        transcript = _read_transcript(args.transcript)
        report_path = run_vsr_report(video_path, synthetic_videos, transcript, out_dir)
        print(f"Saved VSR domain-gap report: {report_path}")


if __name__ == "__main__":
    main()
