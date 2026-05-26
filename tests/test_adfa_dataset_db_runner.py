from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tongue_scripts.evaluation.run_adfa_dataset_to_db import (
    ClipPair,
    connect_db,
    discover_pairs,
    first_existing,
    init_db,
    upsert_condition,
)


def test_discover_pairs_requires_active_passive_and_textgrid(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    beat_root = tmp_path / "beat"
    manifest = output_root / "manifest.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "speaker_id,dataset_id,wav_path,npy_path,status,shape,has_json,has_textgrid,seconds,error\n"
        "1,1_wayne_0_55_55,wav,npy,ok,100x16,1,1,,\n"
        "2,2_scott_0_1_1,wav,npy,ok,100x16,1,1,,\n",
        encoding="utf-8",
    )
    video_dir = output_root / "1" / "55" / "videos"
    video_dir.mkdir(parents=True)
    (video_dir / "1_wayne_0_55_55_with_tongue_with_audio.mp4").write_bytes(b"active")
    (video_dir / "1_wayne_0_55_55_passive_tongue_with_audio.mp4").write_bytes(b"passive")
    textgrid_dir = beat_root / "1"
    textgrid_dir.mkdir(parents=True)
    (textgrid_dir / "1_wayne_0_55_55.TextGrid").write_text("tg", encoding="utf-8")

    pairs = discover_pairs(manifest, output_root, beat_root, {"1"})

    assert pairs == [
        ClipPair(
            speaker_id="1",
            dataset_id="1_wayne_0_55_55",
            active_video=video_dir / "1_wayne_0_55_55_with_tongue_with_audio.mp4",
            passive_video=video_dir / "1_wayne_0_55_55_passive_tongue_with_audio.mp4",
            textgrid_path=textgrid_dir / "1_wayne_0_55_55.TextGrid",
        )
    ]


def test_discover_pairs_prefers_silent_visual_render(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    beat_root = tmp_path / "beat"
    manifest = output_root / "manifest.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "speaker_id,dataset_id,wav_path,npy_path,status,shape,has_json,has_textgrid,seconds,error\n"
        "3,3_solomon_0_100_100,wav,npy,ok,100x16,1,1,,\n",
        encoding="utf-8",
    )
    video_dir = output_root / "3" / "100" / "videos"
    video_dir.mkdir(parents=True)
    active_silent = video_dir / "3_solomon_0_100_100_with_tongue.mp4"
    passive_silent = video_dir / "3_solomon_0_100_100_passive_tongue.mp4"
    active_muxed = video_dir / "3_solomon_0_100_100_with_tongue_with_audio.mp4"
    passive_muxed = video_dir / "3_solomon_0_100_100_passive_tongue_with_audio.mp4"
    for path in (active_silent, passive_silent, active_muxed, passive_muxed):
        path.write_bytes(b"mp4")
    textgrid_dir = beat_root / "3"
    textgrid_dir.mkdir(parents=True)
    textgrid_path = textgrid_dir / "3_solomon_0_100_100.TextGrid"
    textgrid_path.write_text("tg", encoding="utf-8")

    pairs = discover_pairs(manifest, output_root, beat_root, {"3"})

    assert pairs == [
        ClipPair(
            speaker_id="3",
            dataset_id="3_solomon_0_100_100",
            active_video=active_silent,
            passive_video=passive_silent,
            textgrid_path=textgrid_path,
        )
    ]


def test_first_existing_falls_back_to_muxed_video(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mp4"
    muxed = tmp_path / "muxed.mp4"
    muxed.write_bytes(b"mp4")

    assert first_existing((missing, muxed)) == muxed


def test_upsert_condition_refreshes_pair_result(tmp_path: Path) -> None:
    db_path = tmp_path / "results.sqlite3"
    con = connect_db(db_path)
    init_db(con)
    pair = ClipPair(
        speaker_id="1",
        dataset_id="1_wayne_0_1_1",
        active_video=tmp_path / "active.mp4",
        passive_video=tmp_path / "passive.mp4",
        textgrid_path=tmp_path / "clip.TextGrid",
    )
    config_path = tmp_path / "ctc06.ini"
    ground_truth = "hello world"

    upsert_condition(
        con,
        pair,
        "active",
        pair.active_video,
        config_path,
        "mediapipe",
        ground_truth,
        "hello world",
        "ok",
        None,
        "start",
    )
    upsert_condition(
        con,
        pair,
        "passive",
        pair.passive_video,
        config_path,
        "mediapipe",
        ground_truth,
        "hello there",
        "ok",
        None,
        "start",
    )

    row = con.execute(
        "select winner_by_composite, active_wer_norm, passive_wer_norm from pair_results"
    ).fetchone()

    assert row[0] == "active"
    assert row[1] == 0.0
    assert row[2] == 0.5
