#!/usr/bin/env python3
"""Fine VOCASets std_scalar sweep: 0.20..0.40 by 0.01, th=1.4, rot=0."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLIP_ID = "FaceTalk_170725_00137_TA_sentence01"
EXPERIMENT_NAME = "std_scalar_0p20_0p40_step0p01_th1p40_rot0"
OUTPUT_ROOT = (
    Path("/research/milsrg1/user_workspace/ht467/smirk_task/outputs/vocasets_grid_search")
    / CLIP_ID
    / EXPERIMENT_NAME
)
LINK_ROOT = (
    PROJECT_ROOT
    / "tests"
    / "vocaset_outputs"
    / "grid_search"
    / EXPERIMENT_NAME
)


def build_fine_std_scalar_values() -> list[float]:
    return [round(value / 100.0, 2) for value in range(20, 41)]


def split_values(values: list[float], workers: int) -> list[list[float]]:
    return [values[index::workers] for index in range(workers)]


def build_fine_runner_argv(
    std_scalars: list[float] | None = None,
    render_only: bool = False,
    extra_args: list[str] | None = None,
) -> list[str]:
    values = build_fine_std_scalar_values() if std_scalars is None else std_scalars
    argv = [
        sys.executable,
        str(PROJECT_ROOT / "tongue_scripts" / "pipelines" / "grid_search_vocaset_active_tongue.py"),
        "--output-root",
        str(OUTPUT_ROOT),
        "--link-dir",
        str(LINK_ROOT / "videos"),
        "--std-scalars",
        *[str(value) for value in values],
        "--shift-z-values",
        "0.0",
        "--rotation-deg-values",
        "0.0",
        "--thickness-values",
        "1.4",
        "--shift-y-values",
        "0.0",
    ]
    if render_only:
        argv.append("--render-only")
    if extra_args:
        argv.extend(extra_args)
    return argv


def link_artifacts() -> None:
    LINK_ROOT.mkdir(parents=True, exist_ok=True)
    for name in (
        "grid_search_results.csv",
        "grid_search_summary.md",
        "grid_search_ver_plot.png",
        "adfa_grid_search_report.md",
    ):
        target = OUTPUT_ROOT / name
        if target.exists():
            link = LINK_ROOT / f"vocaset_{CLIP_ID}_{EXPERIMENT_NAME}_{name}"
            link.unlink(missing_ok=True)
            link.symlink_to(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--std-scalars", nargs="+", type=float, default=None)
    parser.add_argument(
        "--runner-extra-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to the generic VOCASets grid runner.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subprocess.run(
        build_fine_runner_argv(
            std_scalars=args.std_scalars,
            render_only=args.render_only,
            extra_args=args.runner_extra_arg,
        ),
        cwd=PROJECT_ROOT,
        check=True,
    )
    link_artifacts()


if __name__ == "__main__":
    main()
