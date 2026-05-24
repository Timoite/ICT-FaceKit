#!/usr/bin/env python3
"""Export lowercase transcripts from BEAT TextGrid word tiers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class WordInterval:
    start: float
    end: float
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one lowercase .txt transcript beside each TextGrid file."
    )
    parser.add_argument(
        "--root",
        default="ADFA_EVALUATION/data/beat_cache",
        help="Root directory to search recursively for .TextGrid files.",
    )
    parser.add_argument(
        "--tier-name",
        default="words",
        help="TextGrid interval tier to read.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .txt files.",
    )
    return parser.parse_args()


def parse_textgrid_words(textgrid_path: Path, tier_name: str) -> List[WordInterval]:
    intervals: List[WordInterval] = []
    in_tier = False
    current: dict[str, object] = {}

    with textgrid_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("item ["):
                in_tier = False
                continue
            if line.startswith('name = "'):
                tier = line.split("=", 1)[1].strip().strip('"')
                in_tier = tier == tier_name
                continue
            if not in_tier:
                continue
            if line.startswith("intervals ["):
                current = {}
                continue
            if line.startswith("xmin ="):
                current["start"] = float(line.split("=", 1)[1].strip())
                continue
            if line.startswith("xmax ="):
                current["end"] = float(line.split("=", 1)[1].strip())
                continue
            if line.startswith("text ="):
                text_value = line.split("=", 1)[1].strip()
                if text_value.startswith('"') and text_value.endswith('"'):
                    text_value = text_value[1:-1]
                current["text"] = text_value
                if {"start", "end", "text"} <= current.keys():
                    intervals.append(
                        WordInterval(
                            start=float(current["start"]),
                            end=float(current["end"]),
                            text=str(current["text"]),
                        )
                    )
    return intervals


def intervals_to_transcript(intervals: List[WordInterval]) -> str:
    words = []
    for interval in intervals:
        word = interval.text.strip().lower()
        if not word or word == "sp":
            continue
        words.append(word)
    return " ".join(words)


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    textgrids = sorted(root.rglob("*.TextGrid"))
    if not textgrids:
        raise SystemExit(f"No TextGrid files found under: {root}")

    written = 0
    skipped = 0
    for idx, textgrid_path in enumerate(textgrids, start=1):
        output_path = textgrid_path.with_suffix(".txt")
        if output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{idx}/{len(textgrids)}] skipped existing {output_path}")
            continue

        transcript = intervals_to_transcript(parse_textgrid_words(textgrid_path, args.tier_name))
        output_text = transcript + "\n" if transcript else ""
        output_path.write_text(output_text, encoding="utf-8")
        written += 1
        print(f"[{idx}/{len(textgrids)}] wrote {output_path}")

    print(f"Done. wrote={written} skipped={skipped} total={len(textgrids)}")


if __name__ == "__main__":
    main()
