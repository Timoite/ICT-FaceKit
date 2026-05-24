#!/usr/bin/env python3
"""Estimate per-clip lip-aperture timing shifts and cache them for rendering.

This is the batch/database-oriented companion to lip_aperture_textgrid_plot.py.
It compares lip aperture from WavLM inversion output against lip aperture from
BEAT/ICT blendshapes and stores one reusable shift per dataset clip.

Convention:
- positive best_lag_seconds means articulatory/inversion lip aperture leads BEAT
- positive render_shift_seconds delays tongue motion in the renderer
- therefore render_shift_seconds == best_lag_seconds
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
TONGUE_SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TONGUE_SCRIPTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tongue_scripts.analysis import lip_aperture_textgrid_plot as lap
from tongue_scripts.pipelines.run_render_dual_for_dataset import clip_instance_id

DEFAULT_BEAT_ROOT = (
    PROJECT_ROOT / "data" / "beat_cache" / "beat_english_v0.2.1" / "beat_english_v0.2.1"
)
DEFAULT_ADFA_BEAT_ROOT = (
    PROJECT_ROOT
    / "ADFA_EVALUATION"
    / "data"
    / "beat_cache"
    / "beat_english_v0.2.1"
    / "beat_english_v0.2.1"
)
DEFAULT_OUTPUT_DIR = TONGUE_SCRIPTS_DIR / "outputs" / "time_shifts"
DEFAULT_DB_PATH = DEFAULT_OUTPUT_DIR / "lip_aperture_time_shifts.sqlite3"
DEFAULT_CSV_PATH = DEFAULT_OUTPUT_DIR / "lip_aperture_time_shifts.csv"


def default_beat_root() -> Path:
    """Prefer the live project cache, but fall back to ADFA's cache if it has the data."""
    if any(DEFAULT_BEAT_ROOT.glob("*/*.json")):
        return DEFAULT_BEAT_ROOT
    if any(DEFAULT_ADFA_BEAT_ROOT.glob("*/*.json")):
        return DEFAULT_ADFA_BEAT_ROOT
    return DEFAULT_BEAT_ROOT


SCHEMA = """
CREATE TABLE IF NOT EXISTS lip_aperture_time_shifts (
    speaker_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    instance_id TEXT NOT NULL,
    render_shift_seconds REAL,
    best_lag_seconds REAL,
    best_lag_frames INTEGER,
    best_correlation REAL,
    zero_lag_correlation REAL,
    window_start_seconds REAL,
    window_end_seconds REAL,
    target_fps REAL NOT NULL,
    tongue_fps REAL NOT NULL,
    beat_fps REAL NOT NULL,
    max_lag_seconds REAL NOT NULL,
    smooth_frames INTEGER NOT NULL,
    articulatory_scalar REAL NOT NULL,
    motion_path TEXT NOT NULL,
    beat_json_path TEXT NOT NULL,
    textgrid_path TEXT,
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (speaker_id, dataset_id)
);

CREATE INDEX IF NOT EXISTS idx_lip_aperture_time_shifts_dataset
ON lip_aperture_time_shifts(dataset_id);

CREATE INDEX IF NOT EXISTS idx_lip_aperture_time_shifts_status
ON lip_aperture_time_shifts(status);
"""

CSV_FIELDS = [
    "speaker_id",
    "dataset_id",
    "instance_id",
    "render_shift_seconds",
    "best_lag_seconds",
    "best_lag_frames",
    "best_correlation",
    "zero_lag_correlation",
    "window_start_seconds",
    "window_end_seconds",
    "target_fps",
    "tongue_fps",
    "beat_fps",
    "max_lag_seconds",
    "smooth_frames",
    "articulatory_scalar",
    "motion_path",
    "beat_json_path",
    "textgrid_path",
    "status",
    "error",
    "created_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate and cache per-sample lip-aperture time shifts for render reuse."
    )
    parser.add_argument(
        "--beat-root",
        default=str(default_beat_root()),
        help="Root containing speaker subfolders with BEAT .json/.TextGrid files.",
    )
    parser.add_argument(
        "--speaker-id",
        action="append",
        default=None,
        help="Speaker id to process. Can be passed multiple times. Default: all speaker dirs.",
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=None,
        help="Dataset id to process. Can be passed multiple times. Default: all .json files in selected speakers.",
    )
    parser.add_argument(
        "--motion-root",
        default=str(TONGUE_SCRIPTS_DIR / "outputs"),
        help=(
            "Root containing motion .npy files. The script searches common layouts: "
            "<root>/<dataset>.npy, <root>/<speaker>/<instance>/tongue_motion/<dataset>.npy, "
            "and recursive matches."
        ),
    )
    parser.add_argument(
        "--db-path", default=str(DEFAULT_DB_PATH), help="SQLite cache path."
    )
    parser.add_argument(
        "--csv-path", default=str(DEFAULT_CSV_PATH), help="Searchable CSV export path."
    )
    parser.add_argument("--face-model-dir", default=str(PROJECT_ROOT / "FaceXModel"))
    parser.add_argument(
        "--mu-path",
        default=str(TONGUE_SCRIPTS_DIR / "normalising_vectors" / "JW13_4points_mu.npy"),
    )
    parser.add_argument(
        "--std-path",
        default=str(
            TONGUE_SCRIPTS_DIR / "normalising_vectors" / "JW13_4points_std.npy"
        ),
    )
    parser.add_argument("--phone-tier", default="phones")
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument("--tongue-fps", type=float, default=50.0)
    parser.add_argument("--beat-fps", type=float, default=60.0)
    parser.add_argument("--articulatory-scalar", type=float, default=0.20)
    parser.add_argument("--smooth-frames", type=int, default=5)
    parser.add_argument("--max-lag-seconds", type=float, default=0.5)
    parser.add_argument("--scale-edge-trim-seconds", type=float, default=0.05)
    parser.add_argument(
        "--lag-window-seconds",
        type=float,
        default=None,
        help="Optional max analysis window length from first non-empty TextGrid interval.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute rows already present in DB."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional limit for quick/debug runs."
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Write skipped/error rows to the DB/CSV. By default only successful rows are stored.",
    )
    return parser.parse_args()


def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def discover_items(
    beat_root: Path, speaker_ids: list[str] | None, dataset_ids: set[str] | None
) -> list[tuple[str, str, Path]]:
    if speaker_ids:
        speaker_dirs = [beat_root / str(sid) for sid in speaker_ids]
    else:
        speaker_dirs = (
            sorted(path for path in beat_root.iterdir() if path.is_dir())
            if beat_root.exists()
            else []
        )

    items: list[tuple[str, str, Path]] = []
    for speaker_dir in speaker_dirs:
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name
        for json_path in sorted(speaker_dir.glob("*.json")):
            dataset_id = json_path.stem
            if dataset_ids is not None and dataset_id not in dataset_ids:
                continue
            items.append((speaker_id, dataset_id, json_path))
    return items


def find_motion_path(
    motion_root: Path, speaker_id: str, dataset_id: str
) -> Path | None:
    instance_id = clip_instance_id(dataset_id)
    candidates = [
        motion_root / f"{dataset_id}.npy",
        motion_root
        / str(speaker_id)
        / instance_id
        / "tongue_motion"
        / f"{dataset_id}.npy",
        motion_root / str(speaker_id) / f"{dataset_id}.npy",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    matches = (
        sorted(motion_root.glob(f"**/{dataset_id}.npy")) if motion_root.exists() else []
    )
    return matches[0] if matches else None


def _trimmed_minmax(values: np.ndarray, trim_frames: int) -> np.ndarray:
    if trim_frames > 0 and len(values) > 2 * trim_frames:
        ref = values[trim_frames:-trim_frames]
    else:
        ref = values
    if len(ref) == 0:
        return np.zeros_like(values, dtype=np.float32)
    v_min = float(np.min(ref))
    v_max = float(np.max(ref))
    v_range = v_max - v_min
    if v_range <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(((values - v_min) / v_range).astype(np.float32), 0.0, 1.0)


def load_articulatory_lip_aperture(
    motion_path: Path,
    mu_path: Path,
    std_path: Path,
    articulatory_scalar: float,
    tongue_fps: float,
    target_fps: float,
) -> np.ndarray:
    raw_motion = np.load(motion_path)
    if raw_motion.ndim != 2:
        raise ValueError(f"Expected motion shape (T, D), got {raw_motion.shape}")
    if raw_motion.shape[1] < lap.NORM_VECTOR_COLS:
        raise ValueError(
            f"Expected at least {lap.NORM_VECTOR_COLS} motion columns, got {raw_motion.shape[1]}"
        )

    mu = np.load(mu_path).astype(np.float32).reshape(-1)
    std = np.load(std_path).astype(np.float32).reshape(-1)
    if mu.size < lap.NORM_VECTOR_COLS or std.size < lap.NORM_VECTOR_COLS:
        raise ValueError(f"mu/std too small: mu={mu.size}, std={std.size}")

    denorm = (
        raw_motion[:, : lap.NORM_VECTOR_COLS].astype(np.float32)
        * std[: lap.NORM_VECTOR_COLS]
        * float(articulatory_scalar)
        + mu[: lap.NORM_VECTOR_COLS]
    )
    lip_motion = denorm[
        :, lap.TONGUE_COORD_COLS : lap.TONGUE_COORD_COLS + lap.LIP_COORD_COLS
    ]
    upper_point = lip_motion[:, 0:2]
    lower_point = lip_motion[:, 2:4]
    aperture = np.linalg.norm(upper_point - lower_point, axis=1).reshape(-1, 1)
    return lap.resample_matrix(
        aperture, source_fps=tongue_fps, target_fps=target_fps
    ).squeeze()


def analysis_window(
    textgrid_path: Path, phone_tier: str, max_t: float, lag_window_seconds: float | None
) -> tuple[float, float]:
    if not textgrid_path.is_file():
        return 0.0, max_t
    intervals = [
        iv
        for iv in lap.parse_textgrid_intervals(textgrid_path, phone_tier)
        if iv.text.strip()
    ]
    if not intervals:
        return 0.0, max_t
    window_start = max(0.0, min(iv.start for iv in intervals))
    window_end = min(max_t, max(iv.end for iv in intervals))
    if lag_window_seconds is not None:
        window_end = min(window_end, window_start + float(lag_window_seconds))
    if window_end <= window_start:
        return 0.0, max_t
    return window_start, window_end


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a = a[:n]
    b = b[:n]
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def estimate_one(
    *,
    speaker_id: str,
    dataset_id: str,
    json_path: Path,
    textgrid_path: Path,
    motion_path: Path,
    args: argparse.Namespace,
) -> dict:
    art = load_articulatory_lip_aperture(
        motion_path=motion_path,
        mu_path=Path(args.mu_path),
        std_path=Path(args.std_path),
        articulatory_scalar=float(args.articulatory_scalar),
        tongue_fps=float(args.tongue_fps),
        target_fps=float(args.target_fps),
    )
    bs = lap.load_blendshape_lip_aperture(
        json_path,
        Path(args.face_model_dir),
        beat_fps=float(args.beat_fps),
        target_fps=float(args.target_fps),
    )

    art = lap.moving_average(art, int(args.smooth_frames))
    bs = lap.moving_average(bs, int(args.smooth_frames))

    trim_frames = max(
        0, int(round(float(args.scale_edge_trim_seconds) * float(args.target_fps)))
    )
    art = _trimmed_minmax(art, trim_frames)
    bs = _trimmed_minmax(bs, trim_frames)

    t_art = np.arange(len(art), dtype=np.float32) / float(args.target_fps)
    t_bs = np.arange(len(bs), dtype=np.float32) / float(args.target_fps)
    max_t = min(
        float(t_art[-1]) if len(t_art) else 0.0, float(t_bs[-1]) if len(t_bs) else 0.0
    )
    window_start, window_end = analysis_window(
        textgrid_path, args.phone_tier, max_t, args.lag_window_seconds
    )

    art_mask = (t_art >= window_start) & (t_art <= window_end)
    bs_mask = (t_bs >= window_start) & (t_bs <= window_end)
    art_windowed = art[art_mask]
    bs_windowed = bs[bs_mask]

    max_lag_frames = int(round(float(args.max_lag_seconds) * float(args.target_fps)))
    _, best_lag_frames, best_corr = lap.compute_lag_correlation(
        art_windowed, bs_windowed, max_lag_frames
    )
    best_lag_seconds = float(best_lag_frames) / float(args.target_fps)
    zero_corr = safe_corr(art_windowed, bs_windowed)

    return {
        "speaker_id": str(speaker_id),
        "dataset_id": dataset_id,
        "instance_id": clip_instance_id(dataset_id),
        "render_shift_seconds": best_lag_seconds,
        "best_lag_seconds": best_lag_seconds,
        "best_lag_frames": int(best_lag_frames),
        "best_correlation": float(best_corr),
        "zero_lag_correlation": float(zero_corr),
        "window_start_seconds": float(window_start),
        "window_end_seconds": float(window_end),
        "target_fps": float(args.target_fps),
        "tongue_fps": float(args.tongue_fps),
        "beat_fps": float(args.beat_fps),
        "max_lag_seconds": float(args.max_lag_seconds),
        "smooth_frames": int(args.smooth_frames),
        "articulatory_scalar": float(args.articulatory_scalar),
        "motion_path": str(motion_path),
        "beat_json_path": str(json_path),
        "textgrid_path": str(textgrid_path) if textgrid_path.exists() else "",
        "status": "ok",
        "error": "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def upsert_row(conn: sqlite3.Connection, row: dict) -> None:
    cols = CSV_FIELDS
    placeholders = ", ".join(["?"] * len(cols))
    assignments = ", ".join(
        [
            f"{col}=excluded.{col}"
            for col in cols
            if col not in {"speaker_id", "dataset_id"}
        ]
    )
    conn.execute(
        f"""
        INSERT INTO lip_aperture_time_shifts ({", ".join(cols)})
        VALUES ({placeholders})
        ON CONFLICT(speaker_id, dataset_id) DO UPDATE SET {assignments}
        """,
        [row.get(col) for col in cols],
    )


def row_exists(conn: sqlite3.Connection, speaker_id: str, dataset_id: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM lip_aperture_time_shifts WHERE speaker_id = ? AND dataset_id = ? AND status = 'ok'",
        (str(speaker_id), dataset_id),
    )
    return cur.fetchone() is not None


def export_csv(conn: sqlite3.Connection, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = conn.execute(
        f"SELECT {', '.join(CSV_FIELDS)} FROM lip_aperture_time_shifts ORDER BY speaker_id, dataset_id"
    ).fetchall()
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in CSV_FIELDS})


def missing_row(
    speaker_id: str,
    dataset_id: str,
    json_path: Path,
    textgrid_path: Path,
    status: str,
    error: str,
    args: argparse.Namespace,
) -> dict:
    return {
        "speaker_id": str(speaker_id),
        "dataset_id": dataset_id,
        "instance_id": clip_instance_id(dataset_id),
        "render_shift_seconds": None,
        "best_lag_seconds": None,
        "best_lag_frames": None,
        "best_correlation": None,
        "zero_lag_correlation": None,
        "window_start_seconds": None,
        "window_end_seconds": None,
        "target_fps": float(args.target_fps),
        "tongue_fps": float(args.tongue_fps),
        "beat_fps": float(args.beat_fps),
        "max_lag_seconds": float(args.max_lag_seconds),
        "smooth_frames": int(args.smooth_frames),
        "articulatory_scalar": float(args.articulatory_scalar),
        "motion_path": "",
        "beat_json_path": str(json_path),
        "textgrid_path": str(textgrid_path) if textgrid_path.exists() else "",
        "status": status,
        "error": error,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    args = parse_args()
    beat_root = Path(args.beat_root)
    motion_root = Path(args.motion_root)
    db_path = Path(args.db_path)
    csv_path = Path(args.csv_path)

    conn = connect_db(db_path)
    dataset_filter = set(args.dataset_id) if args.dataset_id else None
    items = discover_items(beat_root, args.speaker_id, dataset_filter)
    if args.limit is not None:
        items = items[: int(args.limit)]

    if not items:
        raise SystemExit(f"No BEAT JSON files found under {beat_root}")

    print(f"Discovered {len(items)} candidate clips under {beat_root}")
    print(f"Writing SQLite cache: {db_path}")
    print(f"Writing CSV export:    {csv_path}")

    ok = skipped_existing = skipped_missing_motion = errors = 0
    for speaker_id, dataset_id, json_path in items:
        if not args.overwrite and row_exists(conn, speaker_id, dataset_id):
            skipped_existing += 1
            continue

        textgrid_path = json_path.with_suffix(".TextGrid")
        motion_path = find_motion_path(motion_root, speaker_id, dataset_id)
        if motion_path is None:
            skipped_missing_motion += 1
            print(f"[SKIP missing motion] speaker={speaker_id} dataset={dataset_id}")
            if args.include_missing:
                upsert_row(
                    conn,
                    missing_row(
                        speaker_id,
                        dataset_id,
                        json_path,
                        textgrid_path,
                        "missing_motion",
                        f"No {dataset_id}.npy found under {motion_root}",
                        args,
                    ),
                )
                conn.commit()
            continue

        try:
            row = estimate_one(
                speaker_id=speaker_id,
                dataset_id=dataset_id,
                json_path=json_path,
                textgrid_path=textgrid_path,
                motion_path=motion_path,
                args=args,
            )
            upsert_row(conn, row)
            conn.commit()
            ok += 1
            print(
                f"[OK] speaker={speaker_id} dataset={dataset_id} "
                f"shift={row['render_shift_seconds']:+.4f}s "
                f"frames={row['best_lag_frames']:+d} corr={row['best_correlation']:.3f}"
            )
        except (
            Exception
        ) as exc:  # keep batch runs moving; details go into searchable DB if requested
            errors += 1
            print(f"[ERROR] speaker={speaker_id} dataset={dataset_id}: {exc}")
            if args.include_missing:
                upsert_row(
                    conn,
                    missing_row(
                        speaker_id,
                        dataset_id,
                        json_path,
                        textgrid_path,
                        "error",
                        str(exc),
                        args,
                    ),
                )
                conn.commit()

    export_csv(conn, csv_path)
    conn.close()

    print("\n=== Time-shift estimation summary ===")
    print(f"OK: {ok}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Skipped missing motion: {skipped_missing_motion}")
    print(f"Errors: {errors}")
    print(f"SQLite: {db_path}")
    print(f"CSV:    {csv_path}")


if __name__ == "__main__":
    main()
