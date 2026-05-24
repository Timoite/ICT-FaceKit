#!/usr/bin/env python3
"""Small SQLite lookup helper for cached per-clip tongue render shifts."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def lookup_render_shift(
    db_path: str | Path, speaker_id: str, dataset_id: str
) -> float | None:
    """Return cached render_shift_seconds for one speaker/dataset, if present."""
    path = Path(db_path)
    if not path.is_file():
        return None

    conn = sqlite3.connect(path)
    try:
        cur = conn.execute(
            """
            SELECT render_shift_seconds
            FROM lip_aperture_time_shifts
            WHERE speaker_id = ?
              AND dataset_id = ?
              AND status = 'ok'
              AND render_shift_seconds IS NOT NULL
            """,
            (str(speaker_id), dataset_id),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return float(row[0])
    finally:
        conn.close()
