#!/usr/bin/env python3
"""
Plot scalar sweep metrics and compare them against a passive-tongue baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot VER and WER from a scalar VSR sweep.")
    parser.add_argument("--summary-csv", required=True, help="CSV produced by run_scalar_vsr_sweep.py")
    parser.add_argument("--passive-json", required=True, help="Passive-tongue metrics JSON")
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Prefix for output figures. Will write *_ver.png and *_wer.png",
    )
    parser.add_argument("--title", default="Scalar Sweep Metrics")
    return parser.parse_args()


def load_summary_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in summary CSV: {path}")
    rows.sort(key=lambda row: float(row["scalar"]))
    return rows


def plot_metric(
    *,
    xs: list[float],
    ys: list[float],
    passive_value: float,
    ylabel: str,
    title: str,
    output_path: Path,
    best_x: float,
    best_y: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(xs, ys, color="#0f766e", marker="o", linewidth=2.2, label="Active tongue sweep")
    ax.axhline(
        passive_value,
        color="#b91c1c",
        linestyle="--",
        linewidth=2.0,
        label=f"Passive tongue = {passive_value:.4f}",
    )
    ax.scatter([best_x], [best_y], color="#f59e0b", s=90, zorder=5, label=f"Best active = {best_y:.4f}")
    ax.annotate(
        f"best @ {best_x:.3f}",
        xy=(best_x, best_y),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    ax.set_xlabel("std_scalar")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    passive_json = Path(args.passive_json)
    output_prefix = Path(args.output_prefix)

    rows = load_summary_rows(summary_csv)
    passive = json.loads(passive_json.read_text(encoding="utf-8"))

    xs = [float(row["scalar"]) for row in rows]
    ver = [float(row["ver"]) for row in rows]
    wer = [float(row["wer_norm"]) for row in rows]

    best_ver_idx = min(range(len(ver)), key=lambda i: ver[i])
    best_wer_idx = min(range(len(wer)), key=lambda i: wer[i])

    plot_metric(
        xs=xs,
        ys=ver,
        passive_value=float(passive["ver"]),
        ylabel="VER",
        title=f"{args.title} - VER",
        output_path=output_prefix.with_name(output_prefix.name + "_ver.png"),
        best_x=xs[best_ver_idx],
        best_y=ver[best_ver_idx],
    )
    plot_metric(
        xs=xs,
        ys=wer,
        passive_value=float(passive["wer_norm"]),
        ylabel="WER (normalized)",
        title=f"{args.title} - WER",
        output_path=output_prefix.with_name(output_prefix.name + "_wer.png"),
        best_x=xs[best_wer_idx],
        best_y=wer[best_wer_idx],
    )


if __name__ == "__main__":
    main()
