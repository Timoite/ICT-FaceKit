#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

TEXTGRID_GLOB = "*.TextGrid"
TEXT_RE = re.compile(r'^\s*text\s*=\s*"(.*)"\s*$')
NAME_RE = re.compile(r'^\s*name\s*=\s*"(.*)"\s*$')
ITEM_RE = re.compile(r'^\s*item\s*\[(\d+)\]:\s*$')
INTERVALS_SIZE_RE = re.compile(r'^\s*intervals:\s*size\s*=\s*(\d+)\s*$')


def unescape_textgrid_string(value: str) -> str:
    return value.replace('""', '"').strip()


def extract_words_from_textgrid(path: Path, target_tier: str = "words") -> str:
    lines = path.read_text(encoding="utf-8").splitlines()

    in_target_item = False
    in_intervals = False
    words: list[str] = []

    for line in lines:
        if ITEM_RE.match(line):
            in_target_item = False
            in_intervals = False
            continue

        if not in_target_item:
            name_match = NAME_RE.match(line)
            if name_match and name_match.group(1) == target_tier:
                in_target_item = True
            continue

        if not in_intervals:
            if INTERVALS_SIZE_RE.match(line):
                in_intervals = True
            continue

        text_match = TEXT_RE.match(line)
        if not text_match:
            continue

        text = unescape_textgrid_string(text_match.group(1))
        if text:
            words.append(text)

    if not words:
        raise ValueError(f"No words found in tier '{target_tier}' for {path}")

    return " ".join(words)


def generate_transcripts(root: Path, overwrite: bool = False, tier: str = "words") -> tuple[int, int]:
    count = 0
    skipped = 0

    for textgrid_path in sorted(root.rglob(TEXTGRID_GLOB)):
        transcript_path = textgrid_path.with_suffix(".txt")
        if transcript_path.exists() and not overwrite:
            skipped += 1
            continue

        transcript = extract_words_from_textgrid(textgrid_path, target_tier=tier)
        transcript_path.write_text(transcript + "\n", encoding="utf-8")
        count += 1

    return count, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate .txt transcript files next to Praat TextGrid files by concatenating words from a target tier."
    )
    parser.add_argument("root", type=Path, help="Root directory to scan for .TextGrid files")
    parser.add_argument(
        "--tier",
        default="words",
        help="Name of the TextGrid interval tier to extract text from (default: words)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .txt files if present",
    )
    args = parser.parse_args()

    if not args.root.exists():
        raise SystemExit(f"Root path does not exist: {args.root}")

    created, skipped = generate_transcripts(args.root, overwrite=args.overwrite, tier=args.tier)
    print(f"Created {created} transcript file(s); skipped {skipped} existing file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
