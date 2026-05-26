#!/usr/bin/env python3
"""Poll lip-aperture time-shift DB and render clips as soon as each shift is ready.

Rendered videos are kept outside the repository and exposed through symlinked
`tongue_scripts/outputs/<speaker>/<clip>/videos` directories.
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BEAT_ROOT = (
    PROJECT_ROOT / "data" / "beat_cache" / "beat_english_v0.2.1" / "beat_english_v0.2.1"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tongue_scripts" / "outputs"
DEFAULT_DB_PATH = (
    DEFAULT_OUTPUT_DIR / "time_shifts" / "lip_aperture_time_shifts.sqlite3"
)
DEFAULT_DONE_PATH = (
    DEFAULT_OUTPUT_DIR / "time_shifts" / "render_completed_clips_queue_done.txt"
)
DEFAULT_CLAIM_DIR = (
    DEFAULT_OUTPUT_DIR / "time_shifts" / "render_completed_clips_queue_claims"
)
DEFAULT_EXTERNAL_OUTPUT_DIR = (
    Path("/research/milsrg1/user_workspace/ht467/ICT-FaceKit-rendered-videos")
    / "tongue_scripts"
    / "outputs"
)
RENDER_DATASET_SCRIPT = (
    PROJECT_ROOT / "tongue_scripts" / "pipelines" / "run_render_dual_for_dataset.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render each clip as soon as it has an ok lip-aperture time-shift row."
    )
    parser.add_argument("--beat-root", default=str(DEFAULT_BEAT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--external-output-dir", default=str(DEFAULT_EXTERNAL_OUTPUT_DIR)
    )
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--done-path", default=str(DEFAULT_DONE_PATH))
    parser.add_argument("--claim-dir", default=str(DEFAULT_CLAIM_DIR))
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--tongue-shift-seconds", type=float, default=0.120)
    parser.add_argument(
        "--speaker-id",
        action="append",
        help="Only watch/render this speaker id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--exclude-speaker-id",
        action="append",
        default=[],
        help="Exclude a speaker id from this queue.",
    )
    parser.add_argument("--use-gpu", action="store_true")
    return parser.parse_args()


def speaker_sort_key(speaker_id: str) -> tuple[int, str]:
    return (
        (int(speaker_id), speaker_id) if speaker_id.isdigit() else (10**9, speaker_id)
    )


def clip_instance_id(dataset_id: str) -> str:
    for token in reversed(dataset_id.split("_")):
        if token.isdigit():
            return str(int(token))
    return dataset_id


def clip_key(speaker_id: str, dataset_id: str) -> str:
    return f"{speaker_id}\t{dataset_id}"


def read_done(done_path: Path) -> set[str]:
    if not done_path.exists():
        return set()
    return {line.strip() for line in done_path.read_text().splitlines() if line.strip()}


def mark_done(done_path: Path, speaker_id: str, dataset_id: str) -> None:
    done_path.parent.mkdir(parents=True, exist_ok=True)
    with done_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{clip_key(speaker_id, dataset_id)}\n")


def claim_name(speaker_id: str, dataset_id: str) -> str:
    safe_dataset_id = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in dataset_id
    )
    return f"{speaker_id}__{safe_dataset_id}.claim"


def try_claim(claim_dir: Path, speaker_id: str, dataset_id: str) -> Path | None:
    claim_dir.mkdir(parents=True, exist_ok=True)
    claim_path = claim_dir / claim_name(speaker_id, dataset_id)
    try:
        claim_path.mkdir()
    except FileExistsError:
        return None
    (claim_path / "created_by").write_text("claimed\n", encoding="utf-8")
    return claim_path


def release_claim(claim_path: Path | None) -> None:
    if claim_path is None:
        return
    marker = claim_path / "created_by"
    if marker.exists():
        marker.unlink()
    try:
        claim_path.rmdir()
    except OSError:
        pass


def estimator_running() -> bool:
    return (
        subprocess.run(
            ["pgrep", "-f", "estimate_lip_aperture_shifts.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def ready_shift_rows(
    db_path: Path,
    allowed: set[str] | None,
    excluded: set[str],
    done: set[str],
) -> list[sqlite3.Row]:
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT speaker_id, dataset_id, motion_path, created_at
            FROM lip_aperture_time_shifts
            WHERE status = 'ok'
            ORDER BY CAST(speaker_id AS INTEGER), dataset_id
            """
        ).fetchall()
    finally:
        conn.close()

    ready: list[sqlite3.Row] = []
    for row in rows:
        speaker_id = str(row["speaker_id"])
        dataset_id = str(row["dataset_id"])
        if allowed is not None and speaker_id not in allowed:
            continue
        if speaker_id in excluded:
            continue
        if clip_key(speaker_id, dataset_id) in done:
            continue
        ready.append(row)
    return ready


def count_ok_rows(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(db_path)
    try:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM lip_aperture_time_shifts WHERE status = 'ok'"
            ).fetchone()[0]
        )
    finally:
        conn.close()


def resolve_motion_path(
    output_dir: Path, speaker_id: str, dataset_id: str, db_motion_path: str | None
) -> Path:
    if db_motion_path:
        path = Path(db_motion_path)
        if path.exists():
            return path
    return (
        output_dir
        / speaker_id
        / clip_instance_id(dataset_id)
        / "tongue_motion"
        / f"{dataset_id}.npy"
    )


def ensure_video_symlink(
    *,
    output_dir: Path,
    external_output_dir: Path,
    speaker_id: str,
    dataset_id: str,
) -> Path:
    instance_id = clip_instance_id(dataset_id)
    repo_videos = output_dir / speaker_id / instance_id / "videos"
    external_videos = external_output_dir / speaker_id / instance_id / "videos"
    external_videos.mkdir(parents=True, exist_ok=True)
    repo_videos.parent.mkdir(parents=True, exist_ok=True)

    if repo_videos.is_symlink():
        return repo_videos

    if repo_videos.exists():
        for child in sorted(repo_videos.iterdir()):
            target = external_videos / child.name
            if target.exists():
                if (
                    child.is_file()
                    and target.is_file()
                    and child.stat().st_size == target.stat().st_size
                ):
                    child.unlink()
                else:
                    child.rename(external_videos / f"{child.name}.repo_move_duplicate")
            else:
                child.rename(target)
        repo_videos.rmdir()

    repo_videos.symlink_to(external_videos, target_is_directory=True)
    return repo_videos


def final_outputs_exist(video_dir: Path, dataset_id: str) -> bool:
    return (video_dir / f"{dataset_id}_with_tongue_with_audio.mp4").exists() and (
        video_dir / f"{dataset_id}_passive_tongue_with_audio.mp4"
    ).exists()


def render_clip(args: argparse.Namespace, row: sqlite3.Row) -> bool:
    speaker_id = str(row["speaker_id"])
    dataset_id = str(row["dataset_id"])
    output_dir = Path(args.output_dir)
    external_output_dir = Path(args.external_output_dir)
    video_dir = ensure_video_symlink(
        output_dir=output_dir,
        external_output_dir=external_output_dir,
        speaker_id=speaker_id,
        dataset_id=dataset_id,
    )

    if final_outputs_exist(video_dir, dataset_id):
        print(f"[SKIP existing] speaker={speaker_id} dataset={dataset_id}", flush=True)
        return True

    motion_path = resolve_motion_path(
        output_dir, speaker_id, dataset_id, row["motion_path"]
    )
    if not motion_path.exists():
        print(
            f"[SKIP missing motion] speaker={speaker_id} dataset={dataset_id} motion={motion_path}",
            flush=True,
        )
        return False

    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        str(RENDER_DATASET_SCRIPT),
        "--dataset-id",
        dataset_id,
        "--speaker-id",
        speaker_id,
        "--beat-root",
        str(Path(args.beat_root)),
        "--motion-path",
        str(motion_path),
        "--output-dir",
        str(output_dir),
        "--tongue-shift-seconds",
        str(args.tongue_shift_seconds),
        "--tongue-shift-db",
        str(Path(args.db_path)),
    ]
    if args.use_gpu:
        cmd.append("--use-gpu")

    print(f"[RENDER] speaker={speaker_id} dataset={dataset_id}", flush=True)
    print("[RUN] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    return True


def main() -> None:
    args = parse_args()
    done_path = Path(args.done_path)
    allowed = set(args.speaker_id) if args.speaker_id else None
    excluded = set(args.exclude_speaker_id or [])

    print(f"[QUEUE] Mode:                per-clip", flush=True)
    print(f"[QUEUE] Beat root:           {Path(args.beat_root)}", flush=True)
    print(f"[QUEUE] Repo output dir:     {Path(args.output_dir)}", flush=True)
    print(f"[QUEUE] External output dir: {Path(args.external_output_dir)}", flush=True)
    print(f"[QUEUE] DB path:             {Path(args.db_path)}", flush=True)
    print(f"[QUEUE] Done path:           {done_path}", flush=True)
    print(f"[QUEUE] Claim dir:           {Path(args.claim_dir)}", flush=True)
    if allowed:
        print(
            f"[QUEUE] Only speakers:       {sorted(allowed, key=speaker_sort_key)}",
            flush=True,
        )
    if excluded:
        print(
            f"[QUEUE] Excluded speakers:   {sorted(excluded, key=speaker_sort_key)}",
            flush=True,
        )

    while True:
        done = read_done(done_path)
        ready = ready_shift_rows(Path(args.db_path), allowed, excluded, done)

        if ready:
            print(f"[QUEUE] Ready clips: {len(ready)}", flush=True)
            for row in ready:
                speaker_id = str(row["speaker_id"])
                dataset_id = str(row["dataset_id"])
                claim_path = try_claim(Path(args.claim_dir), speaker_id, dataset_id)
                if claim_path is None:
                    continue
                try:
                    completed = render_clip(args, row)
                except subprocess.CalledProcessError as exc:
                    release_claim(claim_path)
                    print(
                        f"[ERROR] render failed speaker={speaker_id} dataset={dataset_id} exit={exc.returncode}; retrying later",
                        flush=True,
                    )
                    break
                if completed:
                    mark_done(done_path, speaker_id, dataset_id)
                else:
                    release_claim(claim_path)
            continue

        ok_total = count_ok_rows(Path(args.db_path))
        if not estimator_running():
            print(
                f"[QUEUE] No ready clips and estimator is not running. ok_rows={ok_total}. Exiting.",
                flush=True,
            )
            return

        print(
            f"[QUEUE] Waiting for new ok clips. ok_rows={ok_total} done={len(done)}",
            flush=True,
        )
        time.sleep(max(5, int(args.poll_seconds)))


if __name__ == "__main__":
    main()
