#!/usr/bin/env python3
"""
Entry-point alias with a report-focused name.

Runs the same pipeline as evaluate_vsr_ver.py:
- VSR inference
- VER + WER computation
- Append-to-history composite report generation
"""

from evaluate_vsr_ver import main


if __name__ == "__main__":
    main()
