#!/usr/bin/env python3
"""Create a research-style grid-search trajectory plot for VOCASets runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PARAMETER_COLUMNS = ["std_scalar", "shift_z", "rotation_deg", "thickness", "shift_y"]
STAGE_ORDER = {
    "coarse grid": 0,
    "std/shift refine": 1,
    "rotation refine": 2,
    "thickness refine": 3,
    "shift_y refine": 4,
}


def best_so_far(values: list[float]) -> list[float]:
    best: list[float] = []
    current = float("inf")
    for value in values:
        current = min(current, value)
        best.append(current)
    return best


def infer_search_stage(video_path: str) -> str:
    if "/refined_std_z/" in video_path:
        return "std/shift refine"
    if "/rotation_refine/" in video_path:
        return "rotation refine"
    if "/thickness_refine/" in video_path:
        return "thickness refine"
    if "/shift_y_refine/" in video_path:
        return "shift_y refine"
    return "coarse grid"


def load_rows(csv_path: Path) -> tuple[dict | None, list[dict]]:
    passive: dict | None = None
    active: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["ver_float"] = float(row["ver"])
            row["composite_float"] = float(row["composite_index"])
            if row.get("std_scalar"):
                row["stage"] = infer_search_stage(row["video"])
                for column in PARAMETER_COLUMNS:
                    row[f"{column}_float"] = float(row[column])
                active.append(row)
            else:
                passive = row

    active.sort(
        key=lambda row: (
            STAGE_ORDER.get(row["stage"], 99),
            row["std_scalar_float"],
            row["shift_z_float"],
            row["rotation_deg_float"],
            row["thickness_float"],
            row["shift_y_float"],
        )
    )
    return passive, active


def _plot_stage_spans(ax, rows: list[dict]) -> None:
    if not rows:
        return
    y_min, y_max = ax.get_ylim()
    start = 1
    current = rows[0]["stage"]
    colors = ["#f8fafc", "#eef6ff", "#f7f3ff", "#f1fff4", "#fff8eb"]
    stage_index = 0
    for idx, row in enumerate(rows, start=1):
        if row["stage"] != current:
            ax.axvspan(start - 0.5, idx - 0.5, color=colors[stage_index % len(colors)], zorder=0)
            ax.text((start + idx - 1) / 2, y_max, current, ha="center", va="bottom", fontsize=8)
            start = idx
            current = row["stage"]
            stage_index += 1
    ax.axvspan(start - 0.5, len(rows) + 0.5, color=colors[stage_index % len(colors)], zorder=0)
    ax.text((start + len(rows)) / 2, y_max, current, ha="center", va="bottom", fontsize=8)
    ax.set_ylim(y_min, y_max)


def plot_descent(csv_path: Path, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    passive, active = load_rows(csv_path)
    if not active:
        raise SystemExit(f"No active grid-search rows found in {csv_path}")

    trials = list(range(1, len(active) + 1))
    vers = [row["ver_float"] for row in active]
    best_curve = best_so_far(vers)
    best_idx = min(range(len(active)), key=lambda idx: active[idx]["ver_float"])
    best_row = active[best_idx]
    passive_ver = float(passive["ver"]) if passive else None

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 1.2, 1.2], hspace=0.42, wspace=0.22)
    ax_main = fig.add_subplot(gs[0, :])
    ax_std = fig.add_subplot(gs[1, 0])
    ax_shape = fig.add_subplot(gs[1, 1])
    ax_table = fig.add_subplot(gs[2, :])

    ax_main.plot(trials, vers, color="#94a3b8", marker="o", linewidth=1.0, label="trial VER")
    ax_main.plot(trials, best_curve, color="#0f766e", marker="o", linewidth=2.4, label="best so far")
    ax_main.scatter([best_idx + 1], [best_row["ver_float"]], s=130, color="#dc2626", zorder=4, label="best active")
    if passive_ver is not None:
        ax_main.axhline(passive_ver, color="black", linestyle="--", linewidth=1.4, label=f"passive baseline VER={passive_ver:.4f}")
    ax_main.set_title("VOCASets Active Tongue Parameter Search Trajectory", fontsize=15, pad=20)
    ax_main.set_xlabel("Evaluated active-tongue trial")
    ax_main.set_ylabel("VER (lower is better)")
    ax_main.grid(True, alpha=0.25)
    ax_main.legend(loc="upper right")
    _plot_stage_spans(ax_main, active)
    ax_main.annotate(
        f"best active\nVER={best_row['ver_float']:.4f}\nstd={best_row['std_scalar']}, z={best_row['shift_z']}, th={best_row['thickness']}",
        xy=(best_idx + 1, best_row["ver_float"]),
        xytext=(best_idx + 2, best_row["ver_float"] + 0.10),
        arrowprops={"arrowstyle": "->", "color": "#dc2626"},
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#fecaca"},
    )

    ax_std.plot(trials, [row["std_scalar_float"] for row in active], marker="o", label="std_scalar")
    ax_std.plot(trials, [row["shift_z_float"] for row in active], marker="o", label="shift_z")
    ax_std.set_title("Motion amplitude / depth parameters")
    ax_std.set_xlabel("Trial")
    ax_std.grid(True, alpha=0.25)
    ax_std.legend()

    ax_shape.plot(trials, [row["rotation_deg_float"] for row in active], marker="o", label="rotation_deg")
    ax_shape.plot(trials, [row["thickness_float"] for row in active], marker="o", label="thickness")
    ax_shape.plot(trials, [row["shift_y_float"] for row in active], marker="o", label="shift_y")
    ax_shape.set_title("Tongue geometry parameters")
    ax_shape.set_xlabel("Trial")
    ax_shape.grid(True, alpha=0.25)
    ax_shape.legend()

    top_rows = sorted(active, key=lambda row: row["composite_float"])[:5]
    cell_text = [
        [
            row["stage"],
            row["std_scalar"],
            row["shift_z"],
            row["rotation_deg"],
            row["thickness"],
            row["shift_y"],
            f"{row['ver_float']:.4f}",
            row["wer_norm"],
        ]
        for row in top_rows
    ]
    ax_table.axis("off")
    ax_table.set_title("Top active configurations", loc="left", pad=8)
    table = ax_table.table(
        cellText=cell_text,
        colLabels=["stage", "std", "z", "rot", "thick", "y", "VER", "WER"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default="tests/vocaset_outputs/grid_search/vocaset_FaceTalk_170725_00137_TA_sentence01_grid_search_results.csv",
    )
    parser.add_argument(
        "--out",
        default="tests/vocaset_outputs/grid_search/vocaset_FaceTalk_170725_00137_TA_sentence01_grid_search_descent.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_descent(Path(args.csv), Path(args.out))
    print(args.out)


if __name__ == "__main__":
    main()
