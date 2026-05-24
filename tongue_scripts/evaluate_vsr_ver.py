#!/usr/bin/env python3
"""Compatibility wrapper for the refactored tongue_scripts layout."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tongue_scripts.evaluation.evaluate_vsr_ver import *  # noqa: F401,F403

if __name__ == "__main__":
    from tongue_scripts.evaluation.evaluate_vsr_ver import main

    main()
