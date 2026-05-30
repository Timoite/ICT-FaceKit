#!/usr/bin/env python3
"""Render VOCASets active-tongue videos with the selected best parameters.

The worker is safe to run from multiple tmux sessions. Jobs are shared through
atomic lock directories, large artifacts stay outside the repo, and final MP4s
are symlinked into tests/vocaset_outputs for inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")


@dataclass(frozen=True)
class ActiveBestParams:
    std_scalar: float = 0.27
    shift_z: float = 0.0
    rotation_deg: float = 5.0
    thickness: float = 1.2
    shift_y: float = 0.0


@dataclass(frozen=True)
class RenderJob:
    speaker: str
    sentence: str
    clip_id: str
    json_path: Path
    wav_path: Path
    transcript_path: Path


@dataclass(frozen=True)
class ActiveOutputPaths:
    clip_id: str
    out_dir: Path
    motion_path: Path
    raw_video: Path
    audio_video: Path
    link: Path


BEST_PARAMS = ActiveBestParams()
EXPERIMENT_NAME = "std0p27_z0p00_rot5"
DEFAULT_OUTPUT_ROOT = (
    Path("/research/milsrg1/user_workspace/ht467/smirk_task/outputs")
    / "vocasets_active_best"
    / EXPERIMENT_NAME
)
DEFAULT_ACTIVE_BEST_LINK_DIR = (
    PROJECT_ROOT
    / "tests"
    / "vocaset_outputs"
    / f"active_best_{EXPERIMENT_NAME}"
)


def _format_float(value: float) -> str:
    sign = "m" if value < 0 else ""
    return sign + f"{abs(value):.2f}".replace(".", "p")


def params_slug(params: ActiveBestParams = BEST_PARAMS) -> str:
    return (
        f"std{_format_float(params.std_scalar)}"
        f"_z{_format_float(params.shift_z)}"
        f"_rot{_format_float(params.rotation_deg)}"
    )


def sentence_index(sentence: str) -> int | None:
    match = re.fullmatch(r"sentence(\d+)", sentence)
    if not match:
        return None
    return int(match.group(1))


def has_sentence_transcript(transcript_path: Path, sentence: str) -> bool:
    index = sentence_index(sentence)
    if index is None or not transcript_path.is_file():
        return False
    lines = [
        line.strip()
        for line in transcript_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]
    return 1 <= index <= len(lines)


def collect_render_jobs(json_root: Path, wav_root: Path, transcript_root: Path) -> list[RenderJob]:
    jobs: list[RenderJob] = []
    for json_path in sorted(json_root.glob("*/*.json")):
        speaker = json_path.parent.name
        sentence = json_path.stem
        clip_id = f"{speaker}_{sentence}"
        wav_path = wav_root / f"{clip_id}.wav"
        transcript_path = transcript_root / f"{speaker}.txt"
        if not wav_path.is_file() or not has_sentence_transcript(transcript_path, sentence):
            continue
        jobs.append(
            RenderJob(
                speaker=speaker,
                sentence=sentence,
                clip_id=clip_id,
                json_path=json_path,
                wav_path=wav_path,
                transcript_path=transcript_path,
            )
        )
    return jobs


def active_output_paths(
    output_root: Path,
    link_dir: Path,
    speaker: str,
    sentence: str,
    params: ActiveBestParams = BEST_PARAMS,
) -> ActiveOutputPaths:
    clip_id = f"{speaker}_{sentence}"
    slug = params_slug(params)
    out_dir = output_root / speaker / sentence
    return ActiveOutputPaths(
        clip_id=clip_id,
        out_dir=out_dir,
        motion_path=out_dir / "tongue_motion" / "tongue_motion.npy",
        raw_video=out_dir / f"{clip_id}_{slug}_active_tongue.mp4",
        audio_video=out_dir / f"{clip_id}_{slug}_active_tongue_with_audio.mp4",
        link=link_dir / f"vocaset_{clip_id}_{slug}_active_tongue_with_audio.mp4",
    )


def split_jobs(jobs: Sequence, workers: int) -> list[list]:
    return [list(jobs[index::workers]) for index in range(workers)]


def claim_lock(lock_root: Path, clip_id: str) -> Path | None:
    lock_name = hashlib.sha1(clip_id.encode("utf-8")).hexdigest() + ".lock"
    lock_path = lock_root / lock_name
    try:
        lock_path.mkdir(parents=True)
    except FileExistsError:
        return None
    (lock_path / "clip_id.txt").write_text(clip_id + "\n", encoding="utf-8")
    return lock_path


def release_lock(lock_path: Path | None) -> None:
    if lock_path is None:
        return
    try:
        (lock_path / "clip_id.txt").unlink(missing_ok=True)
        lock_path.rmdir()
    except OSError:
        pass


def replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    tmp_link = link.with_name(f".{link.name}.{os.getpid()}.tmp")
    tmp_link.unlink(missing_ok=True)
    tmp_link.symlink_to(target)
    os.replace(tmp_link, link)


def render_one_job(
    job: RenderJob,
    paths: ActiveOutputPaths,
    *,
    force: bool,
    face_model,
    loaded_model_state: dict,
    fps: int,
    source_fps: float,
    params: ActiveBestParams,
) -> None:
    from tongue_scripts.inversion.invert import load_model, infer_ema
    from tongue_scripts.rendering.render_dual_tongue_comparison import (
        ANCHOR_INDICES,
        BONE_INDICES,
        STD_PATH,
        TONGUE_CONFIG,
        TONGUE_SLICE,
        apply_jawopen_offset_correction,
        merge_audio,
        render_video_with_dynamic_tongue,
    )
    from tongue_scripts.tongue_animation.generate_tongue_animation import (
        FaceKitTongueRig,
        load_blendshape_json_sequence,
        load_ema_motion,
    )
    from tongue_scripts.pipelines.grid_search_vocaset_active_tongue import (
        resample_ema_to_face_frames,
    )

    paths.out_dir.mkdir(parents=True, exist_ok=True)
    paths.motion_path.parent.mkdir(parents=True, exist_ok=True)

    if force or not paths.motion_path.is_file():
        if "model" not in loaded_model_state:
            loaded_model_state["model"] = load_model(Path(loaded_model_state["checkpoint"]))
        ema = infer_ema(
            job.wav_path,
            model=loaded_model_state["model"],
            max_seconds=loaded_model_state["max_seconds"],
        )
        import numpy as np

        np.save(str(paths.motion_path), ema)
        print(f"[MOTION] {paths.motion_path} shape={ema.shape}", flush=True)

    if not force and paths.audio_video.is_file():
        print(f"[SKIP] existing active video: {paths.audio_video}", flush=True)
        replace_symlink(paths.link, paths.audio_video)
        return

    config = dict(TONGUE_CONFIG)
    config.update(
        {
            "std_scalar": params.std_scalar,
            "shift_z": params.shift_z,
            "rotation_deg": params.rotation_deg,
            "thickness": params.thickness,
            "shift_y": params.shift_y,
        }
    )
    tongue_rig = FaceKitTongueRig(
        face_model.neutral_verts,
        face_model.faces,
        TONGUE_SLICE,
        ANCHOR_INDICES,
        BONE_INDICES,
        config,
    )
    face_seq = load_blendshape_json_sequence(
        job.json_path,
        face_model,
        source_fps=source_fps,
        target_fps=fps,
    )
    face_seq = apply_jawopen_offset_correction(face_seq, face_model)
    ema_seq = load_ema_motion(paths.motion_path, STD_PATH, tongue_rig.anchors, config["std_scalar"])
    ema_seq = resample_ema_to_face_frames(ema_seq, len(face_seq))
    print(f"[RENDER] {job.clip_id} -> {paths.audio_video}", flush=True)
    render_video_with_dynamic_tongue(
        face_model,
        face_seq,
        tongue_rig,
        ema_seq,
        str(paths.raw_video),
        fps=fps,
    )
    merge_audio(str(paths.raw_video), str(job.wav_path), str(paths.audio_video))
    replace_symlink(paths.link, paths.audio_video)
    print(f"[LINK] {paths.link}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-root",
        default=str(PROJECT_ROOT / "tests" / "vocasets" / "blendshape_json"),
    )
    parser.add_argument("--wav-root", default=str(PROJECT_ROOT / "tests" / "vocasets" / "wav"))
    parser.add_argument(
        "--transcript-root",
        default=str(PROJECT_ROOT / "tests" / "vocasets" / "sentencestext"),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--link-dir", default=str(DEFAULT_ACTIVE_BEST_LINK_DIR))
    parser.add_argument(
        "--checkpoint",
        default=str(
            PROJECT_ROOT
            / "tongue_scripts"
            / "inversion_checkpoints"
            / "lora_multispeaker_consistency_alpha_0.25_threshold_0_vctk_vvn_4tonguepoints"
        ),
    )
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--source-fps", type=float, default=60.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker-index", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--lock-namespace", default=EXPERIMENT_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_root = Path(args.json_root)
    wav_root = Path(args.wav_root)
    transcript_root = Path(args.transcript_root)
    output_root = Path(args.output_root)
    link_dir = Path(args.link_dir)
    lock_root = output_root / "_locks" / args.lock_namespace

    jobs = collect_render_jobs(json_root, wav_root, transcript_root)
    if args.workers is not None and args.worker_index is not None:
        jobs = split_jobs(jobs, args.workers)[args.worker_index]

    worker_name = os.environ.get("WORKER_NAME", f"worker_{args.worker_index}")
    print(f"[{worker_name}] active_best_jobs={len(jobs)}", flush=True)

    from tongue_scripts.tongue_animation.face_model_io_trimesh import load_face_model_trimesh

    face_model = load_face_model_trimesh(PROJECT_ROOT / "FaceXModel")
    loaded_model_state = {
        "checkpoint": args.checkpoint,
        "max_seconds": args.max_seconds,
    }

    for job in jobs:
        paths = active_output_paths(output_root, link_dir, job.speaker, job.sentence, BEST_PARAMS)
        if not args.force and paths.audio_video.is_file():
            replace_symlink(paths.link, paths.audio_video)
            print(f"[SKIP] existing active video: {paths.audio_video}", flush=True)
            continue

        lock_path = claim_lock(lock_root, job.clip_id)
        if lock_path is None:
            print(f"[SKIP] locked: {job.clip_id}", flush=True)
            continue
        try:
            render_one_job(
                job,
                paths,
                force=args.force,
                face_model=face_model,
                loaded_model_state=loaded_model_state,
                fps=args.fps,
                source_fps=args.source_fps,
                params=BEST_PARAMS,
            )
        finally:
            release_lock(lock_path)


if __name__ == "__main__":
    main()
