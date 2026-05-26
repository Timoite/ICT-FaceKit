#!/usr/bin/env python3
"""Run ADFA VSR for rendered active/passive tongue videos and store metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from jiwer import wer as jiwer_wer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ADFA_EVALUATION.export_textgrid_transcripts import intervals_to_transcript, parse_textgrid_words
from evaluation_script.ver import calculate_ver
from tongue_scripts.evaluation.evaluate_vsr_ver import normalize_for_wer


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tongue_scripts" / "outputs"
DEFAULT_ADFA_ROOT = PROJECT_ROOT / "ADFA_EVALUATION" / "Visual_Speech_Recognition_for_Multiple_Languages"
DEFAULT_BEAT_ROOT = PROJECT_ROOT / "data" / "beat_cache" / "beat_english_v0.2.1" / "beat_english_v0.2.1"
DEFAULT_DB = DEFAULT_OUTPUT_ROOT / "adfa_pilot" / "adfa_full_ctc06_results.sqlite3"
DEFAULT_CONFIG = DEFAULT_OUTPUT_ROOT / "adfa_pilot" / "configs" / "beam40_ctc06_penalty00.ini"
DEFAULT_ADFA_PYTHON = Path("/research/milsrg1/user_workspace/ht467/tools/uv/adfa-vsr/bin/python")


@dataclass(frozen=True)
class ClipPair:
    speaker_id: str
    dataset_id: str
    active_video: Path
    passive_video: Path
    textgrid_path: Path


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


class AdfaWorker:
    def __init__(self, python_bin: Path, adfa_root: Path, config_path: Path, detector: str) -> None:
        cmd = [
            str(python_bin),
            str(adfa_root / "adfa_jsonl_worker.py"),
            "--decode-config",
            str(config_path),
            "--detector",
            detector,
        ]
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(adfa_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._counter = 0
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            print(line.rstrip(), file=sys.stderr, flush=True)

    def infer(self, video_path: Path) -> str:
        if self.proc.poll() is not None:
            raise RuntimeError(f"ADFA worker exited with code {self.proc.returncode}")
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self._counter += 1
        request_id = str(self._counter)
        self.proc.stdin.write(json.dumps({"id": request_id, "video_path": str(video_path)}) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("ADFA worker closed stdout before responding")
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                print(line.rstrip(), file=sys.stderr, flush=True)
                continue
            if response.get("id") != request_id:
                continue
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error")))
            return str(response.get("hypothesis", "")).strip()

    def close(self) -> None:
        if self.proc.poll() is None:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=60)
    con.execute("pragma journal_mode=WAL")
    con.execute("pragma busy_timeout=60000")
    con.execute("pragma synchronous=NORMAL")
    return con


def init_db(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        create table if not exists condition_results (
            speaker_id text not null,
            dataset_id text not null,
            condition text not null,
            video_path text not null,
            textgrid_path text not null,
            config_path text not null,
            infer_mode text not null,
            detector text not null,
            hypothesis text,
            ground_truth text,
            wer_norm real,
            wer_raw real,
            ver real,
            composite_index real,
            viseme_accuracy real,
            word_accuracy real,
            hyp_words integer,
            ref_words integer,
            status text not null,
            error text,
            started_at text,
            completed_at text,
            primary key (dataset_id, condition, config_path, infer_mode)
        );

        create table if not exists pair_results (
            speaker_id text not null,
            dataset_id text not null,
            config_path text not null,
            infer_mode text not null,
            active_wer_norm real,
            passive_wer_norm real,
            delta_wer_passive_minus_active real,
            active_ver real,
            passive_ver real,
            delta_ver_passive_minus_active real,
            active_composite real,
            passive_composite real,
            delta_composite_passive_minus_active real,
            winner_by_composite text,
            completed_at text,
            primary key (dataset_id, config_path, infer_mode)
        );
        """
    )
    con.commit()


def result_exists(con: sqlite3.Connection, dataset_id: str, condition: str, config_path: Path) -> bool:
    row = con.execute(
        """
        select 1 from condition_results
        where dataset_id=? and condition=? and config_path=? and infer_mode='full' and status='ok'
        """,
        (dataset_id, condition, str(config_path)),
    ).fetchone()
    return row is not None


def upsert_condition(
    con: sqlite3.Connection,
    pair: ClipPair,
    condition: str,
    video_path: Path,
    config_path: Path,
    detector: str,
    ground_truth: str,
    hypothesis: str | None,
    status: str,
    error: str | None,
    started_at: str,
) -> None:
    completed_at = now_iso()
    values = {
        "speaker_id": pair.speaker_id,
        "dataset_id": pair.dataset_id,
        "condition": condition,
        "video_path": str(video_path),
        "textgrid_path": str(pair.textgrid_path),
        "config_path": str(config_path),
        "infer_mode": "full",
        "detector": detector,
        "hypothesis": hypothesis,
        "ground_truth": ground_truth,
        "status": status,
        "error": error,
        "started_at": started_at,
        "completed_at": completed_at,
        "wer_norm": None,
        "wer_raw": None,
        "ver": None,
        "composite_index": None,
        "viseme_accuracy": None,
        "word_accuracy": None,
        "hyp_words": None,
        "ref_words": len(normalize_for_wer(ground_truth).split()),
    }
    if status == "ok" and hypothesis is not None:
        hyp_lower = hypothesis.lower()
        ver, _, _ = calculate_ver(ground_truth, hyp_lower, vowel_mode="grouped")
        wer_raw = jiwer_wer(ground_truth, hyp_lower)
        wer_norm = jiwer_wer(normalize_for_wer(ground_truth), normalize_for_wer(hyp_lower))
        values.update(
            {
                "wer_norm": float(wer_norm),
                "wer_raw": float(wer_raw),
                "ver": float(ver),
                "composite_index": 0.5 * float(ver) + 0.5 * float(wer_norm),
                "viseme_accuracy": (1.0 - float(ver)) * 100.0,
                "word_accuracy": (1.0 - float(wer_norm)) * 100.0,
                "hyp_words": len(hypothesis.split()),
            }
        )

    columns = list(values)
    placeholders = ",".join("?" for _ in columns)
    assignments = ",".join(f"{col}=excluded.{col}" for col in columns if col not in {"dataset_id", "condition", "config_path", "infer_mode"})
    con.execute(
        f"""
        insert into condition_results ({','.join(columns)})
        values ({placeholders})
        on conflict(dataset_id, condition, config_path, infer_mode)
        do update set {assignments}
        """,
        [values[col] for col in columns],
    )
    con.commit()
    refresh_pair(con, pair.dataset_id, str(config_path))


def refresh_pair(con: sqlite3.Connection, dataset_id: str, config_path: str) -> None:
    rows = con.execute(
        """
        select speaker_id, condition, wer_norm, ver, composite_index
        from condition_results
        where dataset_id=? and config_path=? and infer_mode='full' and status='ok'
        """,
        (dataset_id, config_path),
    ).fetchall()
    by_condition = {row[1]: row for row in rows}
    if "active" not in by_condition or "passive" not in by_condition:
        return
    active = by_condition["active"]
    passive = by_condition["passive"]
    delta_wer = passive[2] - active[2]
    delta_ver = passive[3] - active[3]
    delta_comp = passive[4] - active[4]
    winner = "active" if active[4] < passive[4] else "passive"
    con.execute(
        """
        insert into pair_results (
            speaker_id, dataset_id, config_path, infer_mode,
            active_wer_norm, passive_wer_norm, delta_wer_passive_minus_active,
            active_ver, passive_ver, delta_ver_passive_minus_active,
            active_composite, passive_composite, delta_composite_passive_minus_active,
            winner_by_composite, completed_at
        )
        values (?, ?, ?, 'full', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(dataset_id, config_path, infer_mode) do update set
            active_wer_norm=excluded.active_wer_norm,
            passive_wer_norm=excluded.passive_wer_norm,
            delta_wer_passive_minus_active=excluded.delta_wer_passive_minus_active,
            active_ver=excluded.active_ver,
            passive_ver=excluded.passive_ver,
            delta_ver_passive_minus_active=excluded.delta_ver_passive_minus_active,
            active_composite=excluded.active_composite,
            passive_composite=excluded.passive_composite,
            delta_composite_passive_minus_active=excluded.delta_composite_passive_minus_active,
            winner_by_composite=excluded.winner_by_composite,
            completed_at=excluded.completed_at
        """,
        (
            active[0],
            dataset_id,
            config_path,
            active[2],
            passive[2],
            delta_wer,
            active[3],
            passive[3],
            delta_ver,
            active[4],
            passive[4],
            delta_comp,
            winner,
            now_iso(),
        ),
    )
    con.commit()


def discover_pairs(manifest_path: Path, output_root: Path, beat_root: Path, speaker_ids: set[str]) -> list[ClipPair]:
    pairs: list[ClipPair] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            speaker_id = row["speaker_id"]
            if speaker_id not in speaker_ids:
                continue
            dataset_id = row["dataset_id"]
            clip_id = dataset_id.rsplit("_", 1)[-1]
            video_dir = output_root / speaker_id / clip_id / "videos"
            active_video = first_existing(
                (
                    video_dir / f"{dataset_id}_with_tongue.mp4",
                    video_dir / f"{dataset_id}_with_tongue_with_audio.mp4",
                )
            )
            passive_video = first_existing(
                (
                    video_dir / f"{dataset_id}_passive_tongue.mp4",
                    video_dir / f"{dataset_id}_passive_tongue_with_audio.mp4",
                )
            )
            textgrid_path = beat_root / speaker_id / f"{dataset_id}.TextGrid"
            if active_video is not None and passive_video is not None and textgrid_path.is_file():
                pairs.append(
                    ClipPair(
                        speaker_id=speaker_id,
                        dataset_id=dataset_id,
                        active_video=active_video,
                        passive_video=passive_video,
                        textgrid_path=textgrid_path,
                    )
                )
    return sorted(pairs, key=lambda p: (int(p.speaker_id), p.dataset_id))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch ADFA full-video active/passive evaluation into SQLite.")
    parser.add_argument("--speaker-ids", nargs="+", required=True)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "manifest.csv")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--beat-root", type=Path, default=DEFAULT_BEAT_ROOT)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--adfa-python", type=Path, default=DEFAULT_ADFA_PYTHON)
    parser.add_argument("--adfa-root", type=Path, default=DEFAULT_ADFA_ROOT)
    parser.add_argument("--detector", default="mediapipe")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-clips", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    con = connect_db(args.db_path)
    init_db(con)
    pairs = discover_pairs(args.manifest_path, args.output_root, args.beat_root, set(args.speaker_ids))
    if args.max_clips is not None:
        pairs = pairs[: args.max_clips]
    print(f"[START] speakers={','.join(args.speaker_ids)} pairs={len(pairs)} db={args.db_path}", flush=True)

    worker = AdfaWorker(args.adfa_python, args.adfa_root, args.config_path, args.detector)
    try:
        for idx, pair in enumerate(pairs, start=1):
            ground_truth = intervals_to_transcript(parse_textgrid_words(pair.textgrid_path, "words"))
            for condition, video_path in (("active", pair.active_video), ("passive", pair.passive_video)):
                if not args.force and result_exists(con, pair.dataset_id, condition, args.config_path):
                    print(f"[{idx}/{len(pairs)}] SKIP {pair.dataset_id} {condition}", flush=True)
                    continue
                started_at = now_iso()
                try:
                    print(f"[{idx}/{len(pairs)}] RUN {pair.dataset_id} {condition}", flush=True)
                    hypothesis = worker.infer(video_path)
                    upsert_condition(
                        con,
                        pair,
                        condition,
                        video_path,
                        args.config_path,
                        args.detector,
                        ground_truth,
                        hypothesis,
                        "ok",
                        None,
                        started_at,
                    )
                    print(f"[{idx}/{len(pairs)}] OK {pair.dataset_id} {condition}", flush=True)
                except Exception as exc:
                    upsert_condition(
                        con,
                        pair,
                        condition,
                        video_path,
                        args.config_path,
                        args.detector,
                        ground_truth,
                        None,
                        "error",
                        repr(exc),
                        started_at,
                    )
                    print(f"[{idx}/{len(pairs)}] ERROR {pair.dataset_id} {condition}: {exc}", flush=True)
    finally:
        worker.close()
        con.close()
    print(f"[DONE] speakers={','.join(args.speaker_ids)}", flush=True)


if __name__ == "__main__":
    main()
