#!/usr/bin/env python3
"""Render VOCASets passive-tongue videos from multiple tmux workers.

Workers use per-clip lock directories so any number of processes can share the
same job list without rendering the same clip twice. Large MP4s are written
outside the repo and final videos are symlinked into tests/.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

from tongue_scripts.rendering.render_dual_tongue_comparison import (  # noqa: E402
    ANCHOR_INDICES,
    BONE_INDICES,
    TONGUE_CONFIG,
    TONGUE_SLICE,
    apply_jawopen_offset_correction,
    merge_audio,
    render_video_with_passive_tongue,
)
from tongue_scripts.tongue_animation.face_model_io_trimesh import (  # noqa: E402
    load_face_model_trimesh,
)
from tongue_scripts.tongue_animation.generate_tongue_animation import (  # noqa: E402
    FaceKitTongueRig,
    load_blendshape_json_sequence,
)

DEFAULT_PASSIVE_LINK_DIR = PROJECT_ROOT / "tests" / "vocaset_outputs" / "passive"


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
    tmp_link = link.with_name(f".{link.name}.{os.getpid()}.tmp")
    tmp_link.unlink(missing_ok=True)
    tmp_link.symlink_to(target)
    os.replace(tmp_link, link)


def main() -> None:
    worker_name = os.environ.get("WORKER_NAME", "worker")
    lock_namespace = os.environ.get("LOCK_NAMESPACE", "default")

    json_root = PROJECT_ROOT / "tests" / "vocasets" / "blendshape_json"
    wav_root = PROJECT_ROOT / "tests" / "vocasets" / "wav"
    out_root = Path(
        "/research/milsrg1/user_workspace/ht467/smirk_task/outputs/vocasets_passive"
    )
    lock_root = out_root / "_locks" / lock_namespace
    test_link_dir = DEFAULT_PASSIVE_LINK_DIR

    jobs = sorted(json_root.glob("*/*.json"))
    print(f"[{worker_name}] jobs_available={len(jobs)}", flush=True)

    face_model = load_face_model_trimesh(PROJECT_ROOT / "FaceXModel")
    tongue_rig = FaceKitTongueRig(
        face_model.neutral_verts,
        face_model.faces,
        TONGUE_SLICE,
        ANCHOR_INDICES,
        BONE_INDICES,
        TONGUE_CONFIG,
    )

    for json_path in jobs:
        speaker = json_path.parent.name
        sentence = json_path.stem
        clip_id = f"{speaker}_{sentence}"
        wav_path = wav_root / f"{clip_id}.wav"

        if not wav_path.is_file():
            print(f"[SKIP] missing wav: {wav_path}", flush=True)
            continue

        out_dir = out_root / speaker / sentence
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_video = out_dir / f"{clip_id}_passive_tongue.mp4"
        audio_video = out_dir / f"{clip_id}_passive_tongue_with_audio.mp4"

        if audio_video.is_file():
            print(f"[SKIP] existing: {audio_video}", flush=True)
        else:
            lock_path = claim_lock(lock_root, clip_id)
            if lock_path is None:
                print(f"[SKIP] locked: {clip_id}", flush=True)
                continue
            if audio_video.is_file():
                release_lock(lock_path)
                print(f"[SKIP] existing: {audio_video}", flush=True)
                continue

            print(f"[RENDER] {clip_id}", flush=True)
            face_seq = load_blendshape_json_sequence(
                json_path, face_model, source_fps=60, target_fps=25
            )
            face_seq = apply_jawopen_offset_correction(face_seq, face_model)
            try:
                render_video_with_passive_tongue(
                    face_model, face_seq, tongue_rig, str(raw_video), fps=25
                )
                merge_audio(str(raw_video), str(wav_path), str(audio_video))
            finally:
                release_lock(lock_path)

        link = test_link_dir / f"vocaset_{clip_id}_passive_tongue_with_audio.mp4"
        replace_symlink(link, audio_video)
        print(f"[LINK] {link}", flush=True)


if __name__ == "__main__":
    main()
