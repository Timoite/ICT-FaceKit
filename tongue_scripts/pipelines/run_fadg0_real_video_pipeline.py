#!/usr/bin/env python3
"""Run FADG0 real-video SMIRK -> ARKit -> active/passive tongue renders."""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

SCRIPT_DIR = Path(__file__).resolve().parent
TONGUE_SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TONGUE_SCRIPTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_VIDEO_DIR = TONGUE_SCRIPTS_DIR / "real_video" / "fadg0" / "mp4"
DEFAULT_OUTPUT_ROOT = Path("/research/milsrg1/user_workspace/ht467/smirk_task/outputs/fadg0")
DEFAULT_SMIRK_ROOT = Path("/research/milsrg1/user_workspace/ht467/smirk_task/smirk")
DEFAULT_SAID_DATA_DIR = Path("/research/milsrg1/user_workspace/ht467/smirk_task/SAiD/data")
DEFAULT_SMIRK_PYTHON = Path("/research/milsrg1/user_workspace/ht467/venvs/smirk-facekit/bin/python")
DEFAULT_SMIRK_CHECKPOINT = DEFAULT_SMIRK_ROOT / "pretrained_models" / "SMIRK_em1.pt"
DEFAULT_FLAME_MODEL = DEFAULT_SMIRK_ROOT / "assets" / "FLAME2020" / "generic_model.pkl"
DEFAULT_WAVLM_CHECKPOINT = (
    TONGUE_SCRIPTS_DIR
    / "inversion_checkpoints"
    / "lora_multispeaker_consistency_alpha_0.25_threshold_0_vctk_vvn_4tonguepoints"
)
DEFAULT_FACE_MODEL_DIR = PROJECT_ROOT / "FaceXModel"
DEFAULT_MU_PATH = TONGUE_SCRIPTS_DIR / "normalising_vectors" / "JW13_4points_mu.npy"
DEFAULT_STD_PATH = TONGUE_SCRIPTS_DIR / "normalising_vectors" / "JW13_4points_std.npy"


def discover_videos(video: Path | None, video_dir: Path, smoke: bool) -> list[Path]:
    """Return one explicit video or sorted MP4s from a directory."""
    if video is not None:
        if not video.is_file():
            raise FileNotFoundError(f"Missing video: {video}")
        return [video]
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Missing video directory: {video_dir}")
    videos = sorted(path for path in video_dir.glob("*.mp4") if path.is_file())
    if not videos:
        raise FileNotFoundError(f"No .mp4 files found in {video_dir}")
    return videos[:1] if smoke else videos


def resample_sequence_to_frames(
    seq: np.ndarray,
    target_frames: int,
    source_fps: float,
    target_fps: float,
) -> np.ndarray:
    """Resample a time sequence to an exact target frame count."""
    seq = np.asarray(seq, dtype=np.float32)
    if len(seq) == target_frames:
        return seq
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    if len(seq) < 2:
        return np.repeat(seq[:1], target_frames, axis=0).astype(np.float32)

    x_source = np.arange(len(seq), dtype=np.float32) / float(source_fps)
    x_target = np.arange(target_frames, dtype=np.float32) / float(target_fps)
    flat = seq.reshape(len(seq), -1)
    kind = "cubic" if len(seq) >= 4 else "linear"
    interpolator = interp1d(
        x_source,
        flat,
        axis=0,
        kind=kind,
        bounds_error=False,
        fill_value=(flat[0], flat[-1]),
    )
    return interpolator(x_target).reshape(target_frames, *seq.shape[1:]).astype(np.float32)


def apply_render_shift(seq: np.ndarray, shift_seconds: float, fps: float) -> np.ndarray:
    """Apply renderer sign convention: positive shift delays the tongue sequence."""
    shift_frames = int(round(float(shift_seconds) * float(fps)))
    seq = np.asarray(seq)
    if shift_frames == 0:
        return seq
    if shift_frames > 0:
        pad = np.repeat(seq[:1], shift_frames, axis=0)
        return np.concatenate([pad, seq], axis=0)[: len(seq)]
    shift_frames = abs(shift_frames)
    pad = np.repeat(seq[-1:], shift_frames, axis=0)
    return np.concatenate([seq[shift_frames:], pad], axis=0)


def shift_tongue_motion_file(
    source_path: Path,
    output_path: Path,
    shift_seconds: float,
    tongue_fps: float,
) -> Path:
    """Save a shifted copy of the raw 50 fps tongue/lip motion .npy file."""
    motion = np.load(source_path)
    shifted = apply_render_shift(motion, shift_seconds=shift_seconds, fps=tongue_fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, shifted.astype(motion.dtype, copy=False))
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["process", "smirk-fit"], default="process")
    p.add_argument("--video", default=None, help="Single video to process. Defaults to first sorted FADG0 clip.")
    p.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--all", action="store_true", help="Process every MP4 in --video-dir.")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--tongue-fps", type=float, default=50.0)
    p.add_argument("--analysis-fps", type=float, default=50.0)
    p.add_argument("--max-lag-seconds", type=float, default=0.5)
    p.add_argument("--smooth-frames", type=int, default=5)
    p.add_argument("--scale-edge-trim-seconds", type=float, default=0.05)
    p.add_argument("--smirk-python", default=str(DEFAULT_SMIRK_PYTHON))
    p.add_argument("--smirk-root", default=str(DEFAULT_SMIRK_ROOT))
    p.add_argument("--smirk-checkpoint", default=str(DEFAULT_SMIRK_CHECKPOINT))
    p.add_argument("--flame-model-path", default=str(DEFAULT_FLAME_MODEL))
    p.add_argument("--said-data-dir", default=str(DEFAULT_SAID_DATA_DIR))
    p.add_argument("--said-person-id", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--wavlm-checkpoint", default=str(DEFAULT_WAVLM_CHECKPOINT))
    p.add_argument("--face-model-dir", default=str(DEFAULT_FACE_MODEL_DIR))
    p.add_argument("--mu-path", default=str(DEFAULT_MU_PATH))
    p.add_argument("--std-path", default=str(DEFAULT_STD_PATH))
    p.add_argument("--symlink-dir", default=str(PROJECT_ROOT / "tests"))
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def extract_audio(video_path: Path, audio_path: Path) -> Path:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
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
    return audio_path


def run_wavlm_inversion(wav_path: Path, out_path: Path, checkpoint_path: Path) -> Path:
    from tongue_scripts.inversion.invert import infer_ema

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ema = infer_ema(wav_path, checkpoint_path)
    np.save(out_path, ema)
    print(f"Saved WavLM tongue motion: {out_path} shape={ema.shape}")
    return out_path


def _install_chumpy_py311_shim() -> None:
    """Patch inspect.getargspec for chumpy under Python 3.11."""
    if hasattr(inspect, "getargspec"):
        return
    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = getargspec


def run_smirk_fit_stage(args: argparse.Namespace) -> None:
    _install_chumpy_py311_shim()
    from tongue_scripts.real_video.arkit_to_ict import convert_csv_to_ict_outputs
    from tongue_scripts.real_video.extract_smirk_sequence import (
        SmirkExtractionConfig,
        extract_smirk_sequence,
    )
    from tongue_scripts.real_video.smirk_flame_to_arkit import fit_smirk_vertices_file

    video_path = Path(args.video).expanduser().resolve()
    out_dir = Path(args.output_root).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _, vertices_path = extract_smirk_sequence(
        SmirkExtractionConfig(
            video_path=video_path,
            out_dir=out_dir,
            smirk_root=Path(args.smirk_root).expanduser().resolve(),
            checkpoint_path=Path(args.smirk_checkpoint).expanduser().resolve(),
            flame_model_path=Path(args.flame_model_path).expanduser().resolve(),
            fps=float(args.fps),
            device=args.device,
            crop=True,
        )
    )
    arkit_csv = out_dir / "arkit_coeffs.csv"
    diagnostics_json = out_dir / "arkit_fit_diagnostics.json"
    fit_smirk_vertices_file(
        vertices_npz=vertices_path,
        said_data_dir=Path(args.said_data_dir).expanduser().resolve(),
        coeffs_csv=arkit_csv,
        diagnostics_json=diagnostics_json,
        temporal_delta=0.1,
        chunk_size=120,
        said_person_id=args.said_person_id,
        fps=float(args.fps),
    )
    convert_csv_to_ict_outputs(
        arkit_csv=arkit_csv,
        face_model_dir=Path(args.face_model_dir).expanduser().resolve(),
        ict_npz=out_dir / "ict_coeffs.npz",
        motion_json=out_dir / "arkit_face_motion.json",
        fps=float(args.fps),
        source_video=video_path,
    )


def run_smirk_fit_subprocess(args: argparse.Namespace, video_path: Path, out_dir: Path) -> None:
    if args.skip_existing and (out_dir / "arkit_face_motion.json").is_file() and (out_dir / "ict_coeffs.npz").is_file():
        print(f"Reusing SMIRK/ARKit outputs in {out_dir}")
        return
    cmd = [
        str(Path(args.smirk_python).expanduser()),
        str(Path(__file__).resolve()),
        "--stage",
        "smirk-fit",
        "--video",
        str(video_path),
        "--output-root",
        str(out_dir),
        "--fps",
        str(args.fps),
        "--smirk-root",
        args.smirk_root,
        "--smirk-checkpoint",
        args.smirk_checkpoint,
        "--flame-model-path",
        args.flame_model_path,
        "--said-data-dir",
        args.said_data_dir,
        "--face-model-dir",
        args.face_model_dir,
        "--device",
        args.device,
    ]
    if args.said_person_id:
        cmd.extend(["--said-person-id", args.said_person_id])
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def _trimmed_minmax(values: np.ndarray, trim_frames: int) -> np.ndarray:
    if trim_frames > 0 and len(values) > 2 * trim_frames:
        ref = values[trim_frames:-trim_frames]
    else:
        ref = values
    if len(ref) == 0:
        return np.zeros_like(values, dtype=np.float32)
    v_min = float(np.min(ref))
    v_max = float(np.max(ref))
    if v_max - v_min <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(((values - v_min) / (v_max - v_min)).astype(np.float32), 0.0, 1.0)


def estimate_lip_aperture_shift(
    *,
    face_json: Path,
    tongue_motion: Path,
    face_model,
    mu_path: Path,
    std_path: Path,
    target_fps: float,
    face_fps: float,
    tongue_fps: float,
    max_lag_seconds: float,
    smooth_frames: int,
    scale_edge_trim_seconds: float,
    report_path: Path,
    plot_path: Path,
) -> dict:
    from tongue_scripts.analysis import lip_aperture_textgrid_plot as lap
    from tongue_scripts.analysis.estimate_lip_aperture_shifts import (
        load_articulatory_lip_aperture,
        load_blendshape_lip_aperture_with_model,
        safe_corr,
    )

    art = load_articulatory_lip_aperture(
        motion_path=tongue_motion,
        mu_path=mu_path,
        std_path=std_path,
        articulatory_scalar=0.20,
        tongue_fps=tongue_fps,
        target_fps=target_fps,
    )
    bs = load_blendshape_lip_aperture_with_model(
        face_json,
        face_model,
        beat_fps=face_fps,
        target_fps=target_fps,
    )
    art = lap.moving_average(art, smooth_frames)
    bs = lap.moving_average(bs, smooth_frames)
    trim_frames = max(0, int(round(scale_edge_trim_seconds * target_fps)))
    art = _trimmed_minmax(art, trim_frames)
    bs = _trimmed_minmax(bs, trim_frames)
    n = min(len(art), len(bs))
    art = art[:n]
    bs = bs[:n]
    max_lag_frames = int(round(max_lag_seconds * target_fps))
    correlations, best_lag_frames, best_corr = lap.compute_lag_correlation(
        art, bs, max_lag_frames
    )
    zero_corr = safe_corr(art, bs)
    render_shift_seconds = float(best_lag_frames) / float(target_fps)
    report = {
        "face_json": str(face_json),
        "tongue_motion": str(tongue_motion),
        "analysis_target_fps": float(target_fps),
        "face_fps": float(face_fps),
        "tongue_fps": float(tongue_fps),
        "render_fps": float(face_fps),
        "best_lag_frames": int(best_lag_frames),
        "best_lag_seconds": render_shift_seconds,
        "render_shift_seconds": render_shift_seconds,
        "render_shift_frames_25fps": int(round(render_shift_seconds * face_fps)),
        "tongue_shift_frames_50fps": int(round(render_shift_seconds * tongue_fps)),
        "best_correlation": float(best_corr),
        "zero_lag_correlation": float(zero_corr),
        "max_lag_seconds": float(max_lag_seconds),
        "smooth_frames": int(smooth_frames),
        "scale_edge_trim_seconds": float(scale_edge_trim_seconds),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        lags = np.arange(-max_lag_frames, max_lag_frames + 1)
        valid = np.isfinite(correlations)
        t = np.arange(n, dtype=np.float32) / float(target_fps)
        fig, axes = plt.subplots(2, 1, figsize=(11, 7))
        axes[0].plot(t, art, label="WavLM lip aperture")
        axes[0].plot(t, bs, label="SMIRK/FaceKit lip aperture")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(lags[valid] / float(target_fps), correlations[valid])
        axes[1].axvline(render_shift_seconds, color="black", linestyle="--")
        axes[1].axvline(0.0, color="gray", linestyle=":")
        axes[1].set_xlabel("Lag seconds; positive = WavLM leads face")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plot is diagnostic.
        report["plot_error"] = str(exc)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def render_outputs(args: argparse.Namespace, out_dir: Path) -> tuple[Path, Path]:
    from tongue_scripts.rendering.render_dual_tongue_comparison import (
        ANCHOR_INDICES,
        BONE_INDICES,
        TONGUE_CONFIG,
        TONGUE_SLICE,
        merge_audio,
        render_video_with_dynamic_tongue,
        render_video_with_passive_tongue,
    )
    from tongue_scripts.tongue_animation.face_model_io_trimesh import load_face_model_trimesh
    from tongue_scripts.tongue_animation.generate_tongue_animation import (
        FaceKitTongueRig,
        load_blendshape_json_sequence,
        load_ema_motion,
    )

    face_model = load_face_model_trimesh(args.face_model_dir)
    face_seq = load_blendshape_json_sequence(out_dir / "arkit_face_motion.json", face_model, target_fps=args.fps)
    tongue_rig = FaceKitTongueRig(
        face_model.neutral_verts,
        face_model.faces,
        TONGUE_SLICE,
        ANCHOR_INDICES,
        BONE_INDICES,
        TONGUE_CONFIG,
    )
    shift_report = estimate_lip_aperture_shift(
        face_json=out_dir / "arkit_face_motion.json",
        tongue_motion=out_dir / "tongue_motion.npy",
        face_model=face_model,
        mu_path=Path(args.mu_path),
        std_path=Path(args.std_path),
        target_fps=args.analysis_fps,
        face_fps=args.fps,
        tongue_fps=args.tongue_fps,
        max_lag_seconds=args.max_lag_seconds,
        smooth_frames=args.smooth_frames,
        scale_edge_trim_seconds=args.scale_edge_trim_seconds,
        report_path=out_dir / "lip_aperture_time_shift_analysis.json",
        plot_path=out_dir / "lip_aperture_time_shift_analysis.png",
    )
    ema_seq = load_ema_motion(
        str(shift_tongue_motion_file(
            out_dir / "tongue_motion.npy",
            out_dir / "tongue_motion_lipcorr_shifted.npy",
            shift_report["render_shift_seconds"],
            args.tongue_fps,
        )),
        str(args.std_path),
        tongue_rig.anchors,
        TONGUE_CONFIG["std_scalar"],
    )
    ema_seq = resample_sequence_to_frames(ema_seq, len(face_seq), args.tongue_fps, args.fps)

    active_raw = out_dir / f"{out_dir.name}_active_tongue.mp4"
    passive_raw = out_dir / f"{out_dir.name}_passive_tongue.mp4"
    active_audio = out_dir / f"{out_dir.name}_active_tongue_with_audio.mp4"
    passive_audio = out_dir / f"{out_dir.name}_passive_tongue_with_audio.mp4"
    render_video_with_dynamic_tongue(
        face_model, face_seq, tongue_rig, ema_seq, str(active_raw), fps=args.fps, max_seconds=None
    )
    merge_audio(str(active_raw), str(out_dir / "audio_16k.wav"), str(active_audio))
    render_video_with_passive_tongue(
        face_model, face_seq, tongue_rig, str(passive_raw), fps=args.fps, max_seconds=None
    )
    merge_audio(str(passive_raw), str(out_dir / "audio_16k.wav"), str(passive_audio))
    return active_audio, passive_audio


def symlink_outputs(args: argparse.Namespace, video_stem: str, out_dir: Path, active: Path, passive: Path) -> None:
    link_dir = Path(args.symlink_dir)
    link_dir.mkdir(parents=True, exist_ok=True)
    links = {
        f"fadg0_{video_stem}_active_tongue_with_audio.mp4": active,
        f"fadg0_{video_stem}_passive_tongue_with_audio.mp4": passive,
        f"fadg0_{video_stem}_arkit_face_motion.json": out_dir / "arkit_face_motion.json",
        f"fadg0_{video_stem}_lip_aperture_time_shift_analysis.json": out_dir / "lip_aperture_time_shift_analysis.json",
        f"fadg0_{video_stem}_lip_aperture_time_shift_analysis.png": out_dir / "lip_aperture_time_shift_analysis.png",
    }
    for name, target in links.items():
        link_path = link_dir / name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(target)


def process_video(args: argparse.Namespace, video_path: Path) -> dict:
    out_dir = Path(args.output_root).expanduser().resolve() / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_audio(video_path, out_dir / "audio_16k.wav")
    run_smirk_fit_subprocess(args, video_path.resolve(), out_dir)
    if not (args.skip_existing and (out_dir / "tongue_motion.npy").is_file()):
        run_wavlm_inversion(
            out_dir / "audio_16k.wav",
            out_dir / "tongue_motion.npy",
            Path(args.wavlm_checkpoint).expanduser().resolve(),
        )
    active, passive = render_outputs(args, out_dir)
    symlink_outputs(args, video_path.stem, out_dir, active, passive)
    return {"video": str(video_path), "out_dir": str(out_dir), "active": str(active), "passive": str(passive)}


def main() -> None:
    args = parse_args()
    if args.stage == "smirk-fit":
        run_smirk_fit_stage(args)
        return

    videos = discover_videos(
        video=Path(args.video).expanduser().resolve() if args.video else None,
        video_dir=Path(args.video_dir).expanduser().resolve(),
        smoke=not args.all,
    )
    print(f"Processing {len(videos)} video(s): {[path.name for path in videos]}")
    results = [process_video(args, video) for video in videos]
    summary_path = Path(args.output_root).expanduser().resolve() / "fadg0_pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
