#!/usr/bin/env python3
"""Extract ground truth transcript from BEAT TextGrid file."""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR.parent / "ADFA_EVALUATION"))

from compute_wer import parse_textgrid_words, intervals_to_tokens

TEXTGRID_PATH = (
    SCRIPT_DIR.parent / "ADFA_EVALUATION" /
    "Visual_Speech_Recognition_for_Multiple_Languages" /
    "data" / "beat_textgrids" / "1" / "1_wayne_0_75_75.TextGrid"
)
OUTPUT_FILE = SCRIPT_DIR / "ground_truth.txt"


def extract_ground_truth():
    if not TEXTGRID_PATH.exists():
        raise FileNotFoundError(f"TextGrid not found: {TEXTGRID_PATH}")

    print("="*60)
    print("GROUND TRUTH EXTRACTION")
    print("="*60)
    print(f"TextGrid: {TEXTGRID_PATH}")

    intervals = parse_textgrid_words(TEXTGRID_PATH, tier_name="words")
    tokens = intervals_to_tokens(intervals)

    print(f"  ✓ Found {len(intervals)} intervals")
    print(f"  ✓ Extracted {len(tokens)} words")

    transcript = " ".join(tokens)
    OUTPUT_FILE.write_text(transcript + "\n", encoding="utf-8")

    print(f"\nTranscript:\n{'-'*60}")
    print(transcript)
    print(f"{'-'*60}")
    print(f"\nSaved to: {OUTPUT_FILE}")

    if intervals:
        duration = intervals[-1].end
        print(f"  Duration: {duration:.2f}s")
        print(f"  Words/sec: {len(tokens)/duration:.2f}")

    return tokens


def main():
    try:
        tokens = extract_ground_truth()
        if len(tokens) > 0:
            print("\n✓ Ground truth extraction successful")
            return 0
        else:
            print("\n✗ No words extracted")
            return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
