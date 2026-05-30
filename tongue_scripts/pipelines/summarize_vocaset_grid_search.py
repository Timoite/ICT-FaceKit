#!/usr/bin/env python3
"""Combine VOCASets grid-search CSVs and plot VER against varied parameters."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PARAMETER_COLUMNS = ["std_scalar", "shift_z", "rotation_deg", "thickness", "shift_y"]


def load_grid_rows(result_root: Path) -> list[dict]:
    rows_by_video: dict[str, dict] = {}
    for csv_path in sorted(result_root.rglob("grid_search_results.csv")):
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                video = row.get("video", "")
                if not video:
                    continue
                existing = rows_by_video.get(video)
                if existing is None or float(row["composite_index"]) < float(
                    existing["composite_index"]
                ):
                    rows_by_video[video] = row
    return sorted(rows_by_video.values(), key=lambda row: float(row["composite_index"]))


def active_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("std_scalar")]


def varied_parameter_names(rows: list[dict]) -> list[str]:
    names: list[str] = []
    for column in PARAMETER_COLUMNS:
        values = {row.get(column, "") for row in active_rows(rows)}
        if len(values) > 1:
            names.append(column)
    return names


def write_combined_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out = dict(row)
            out["rank"] = str(rank)
            writer.writerow(out)


def write_ver_plot(rows: list[dict], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    active = active_rows(rows)
    varied = varied_parameter_names(rows)
    if not active or not varied:
        return

    fig, axes = plt.subplots(
        len(varied),
        1,
        figsize=(9, max(4, 3.5 * len(varied))),
        squeeze=False,
    )
    passive_ver = [
        float(row["ver"]) for row in rows if not row.get("std_scalar") and row.get("ver")
    ]

    for ax, parameter in zip(axes[:, 0], varied):
        points_by_x: dict[float, list[float]] = {}
        for row in active:
            x_raw = row.get(parameter, "")
            if not x_raw:
                continue
            points_by_x.setdefault(float(x_raw), []).append(float(row["ver"]))

        xs = sorted(points_by_x)
        ys = [min(points_by_x[x]) for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=1.8)
        if passive_ver:
            ax.axhline(
                min(passive_ver),
                color="black",
                linestyle="--",
                linewidth=1.2,
                label=f"passive VER={min(passive_ver):.4f}",
            )
            ax.legend()
        ax.set_xlabel(parameter)
        ax.set_ylabel("Best VER")
        ax.set_title(f"Best VER vs {parameter}")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        default="/research/milsrg1/user_workspace/ht467/smirk_task/outputs/vocasets_grid_search/FaceTalk_170725_00137_TA_sentence01",
    )
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-plot", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root)
    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else result_root / "combined_grid_search_results.csv"
    )
    output_plot = (
        Path(args.output_plot)
        if args.output_plot
        else result_root / "combined_grid_search_ver_plot.png"
    )
    rows = load_grid_rows(result_root)
    write_combined_csv(rows, output_csv)
    write_ver_plot(rows, output_plot)
    print(f"rows={len(rows)}")
    print(f"csv={output_csv}")
    print(f"plot={output_plot}")


if __name__ == "__main__":
    main()
