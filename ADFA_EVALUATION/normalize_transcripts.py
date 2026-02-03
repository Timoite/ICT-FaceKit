#!/usr/bin/env python3
"""Normalize transcript files with the EnglishTextNormalizer.

Example:
    python normalize_transcripts.py --input-dir Visual_Speech_Recognition_for_Multiple_Languages/transcript-new \
        --output-dir Visual_Speech_Recognition_for_Multiple_Languages/transcript-normalized
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

try:
    from english import EnglishTextNormalizer
except ImportError as exc:  # pragma: no cover - aids debugging import issues
    raise SystemExit(
        "Unable to import EnglishTextNormalizer. Make sure english.py is on PYTHONPATH."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize transcripts and mirror the input directory structure."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "Visual_Speech_Recognition_for_Multiple_Languages/transcript-new"
        ),
        help="Directory containing raw transcripts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "Visual_Speech_Recognition_for_Multiple_Languages/transcript-normalized"
        ),
        help="Directory that will receive normalized transcripts.",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=[".txt"],
        help="File extensions to normalize (defaults to .txt).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover files and directories without writing output.",
    )
    return parser.parse_args()


def iter_files(root: Path, extensions: Sequence[str]) -> Iterable[Path]:
    normalized_ext = tuple(ext.lower() for ext in extensions) if extensions else None
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if normalized_ext and path.suffix.lower() not in normalized_ext:
            continue
        yield path


def normalize_text(text: str, normalizer: EnglishTextNormalizer) -> str:
    trailing_newline = text.endswith("\n")
    normalized_lines = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            normalized_lines.append("")
            continue
        normalized_lines.append(normalizer(stripped))
    normalized = "\n".join(normalized_lines)
    if trailing_newline:
        normalized += "\n"
    return normalized


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    if args.dry_run:
        print(f"Scanning {input_dir} -> {output_dir}")

    normalizer = EnglishTextNormalizer()
    files = list(iter_files(input_dir, args.extensions))
    total = len(files)

    for idx, src in enumerate(files, start=1):
        rel_path = src.relative_to(input_dir)
        dst = output_dir / rel_path
        if args.dry_run:
            print(f"[{idx}/{total}] {src} -> {dst}")
            continue

        ensure_parent(dst)
        text = src.read_text(encoding="utf-8")
        normalized = normalize_text(text, normalizer)
        dst.write_text(normalized, encoding="utf-8")
        print(f"[{idx}/{total}] normalized {rel_path}")


if __name__ == "__main__":
    main()
