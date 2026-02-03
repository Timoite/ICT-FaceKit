#!/usr/bin/env python3
"""Visualize WER statistics to spot problematic speakers/files."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize results from wer_directory_report.csv")
    parser.add_argument("--report", default="wer_directory_report.csv", help="Path to wer_directory_report.csv")
    parser.add_argument("--output-dir", default="wer_plots", help="Directory to write generated figures")
    parser.add_argument("--top-speakers", type=int, default=8, help="How many worst speakers to highlight explicitly")
    parser.add_argument("--top-files", type=int, default=15, help="How many worst files to chart by WER")
    return parser.parse_args()


def load_report(report_path: Path) -> pd.DataFrame:
    df = pd.read_csv(report_path)
    df = df.loc[df["speaker_id"] != "-"].copy()
    df = df.loc[df["file"].str.upper() != "OVERALL"].copy()
    df["speaker_id"] = df["speaker_id"].astype(str)
    return cast(pd.DataFrame, df)


def plot_mean_wer_by_speaker(df: pd.DataFrame, output_path: Path) -> None:
    summary = (
        df.groupby("speaker_id")
        .agg(mean_wer=("wer", "mean"), median_wer=("wer", "median"), clips=("wer", "count"))
        .sort_values("mean_wer", ascending=False)
    )

    plt.figure(figsize=(12, 6))
    bars = plt.bar(summary.index, summary["mean_wer"], color="steelblue")
    plt.axhline(1.0, color="red", linestyle="--", label="WER = 100%")
    plt.ylabel("Mean WER")
    plt.xlabel("Speaker ID")
    plt.title("Mean WER per Speaker (higher is worse)")

    worst_idx = summary.index[:3]
    for idx in worst_idx:
        bar = bars[list(summary.index).index(idx)]
        bar.set_color("indianred")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_worst_speaker_boxplots(df: pd.DataFrame, output_path: Path, top_speakers: int) -> None:
    summary = (
        df.groupby("speaker_id")
        .agg(mean_wer=("wer", "mean"))
        .sort_values("mean_wer", ascending=False)
        .head(top_speakers)
    )
    worst_ids = summary.index.tolist()
    subset = df[df["speaker_id"].isin(worst_ids)]

    plt.figure(figsize=(12, 6))
    subset.boxplot(column="wer", by="speaker_id", grid=False)
    plt.suptitle("")
    plt.title(f"WER distribution for worst {len(worst_ids)} speakers")
    plt.xlabel("Speaker ID")
    plt.ylabel("WER")
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    
def plot_best_speaker_boxplots(df: pd.DataFrame, output_path: Path, top_speakers: int) -> None:
    summary = (
        df.groupby("speaker_id")
        .agg(mean_wer=("wer", "mean"))
        .sort_values("mean_wer", ascending=True)
        .head(top_speakers)
    )
    best_ids = summary.index.tolist()
    subset = df[df["speaker_id"].isin(best_ids)]

    plt.figure(figsize=(12, 6))
    subset.boxplot(column="wer", by="speaker_id", grid=False)
    plt.suptitle("")
    plt.title(f"WER distribution for best {len(best_ids)} speakers")
    plt.xlabel("Speaker ID")
    plt.ylabel("WER")
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_worst_files(df: pd.DataFrame, output_path: Path, top_files: int) -> None:
    worst = df.nlargest(top_files, "wer").copy()
    worst = worst.iloc[::-1]

    plt.figure(figsize=(10, max(6, top_files * 0.3)))
    plt.barh(worst["file"], worst["wer"], color="darkslateblue")
    plt.axvline(1.0, color="red", linestyle="--")
    plt.xlabel("WER")
    plt.ylabel("File")
    plt.title(f"Worst {top_files} Clips by WER")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_wer_distribution(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df["wer"], bins=30, color="teal", alpha=0.8)
    plt.axvline(df["wer"].median(), color="orange", linestyle="--", label="Median")
    plt.axvline(df["wer"].mean(), color="red", linestyle=":", label="Mean")
    plt.xlabel("WER")
    plt.ylabel("Count")
    plt.title("WER Distribution across all clips")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_all_speakers_boxplot(df: pd.DataFrame, output_path: Path) -> None:
    speaker_order = (
        df.groupby("speaker_id")
        .agg(mean_wer=("wer", "mean"))
        .sort_values("mean_wer", ascending=False)
        .index
        .tolist()
    )
    ordered_df = df.copy()
    ordered_df["speaker_id"] = pd.Categorical(ordered_df["speaker_id"], categories=speaker_order, ordered=True)

    plt.figure(figsize=(max(16, len(speaker_order) * 0.7), 6))
    ordered_df.boxplot(column="wer", by="speaker_id", grid=False)
    plt.suptitle("")
    plt.title("WER distribution for all speakers")
    plt.xlabel("Speaker ID")
    plt.ylabel("WER")
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    report_path = Path(args.report).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not report_path.is_file():
        raise SystemExit(f"Report not found: {report_path}")

    df = load_report(report_path)
    if df.empty:
        raise SystemExit("WER report has no speaker-level rows to visualize.")

    plot_mean_wer_by_speaker(df, output_dir / "mean_wer_by_speaker.png")
    plot_worst_speaker_boxplots(df, output_dir / "worst_speaker_boxplots.png", args.top_speakers)
    plot_best_speaker_boxplots(df, output_dir / "best_speaker_boxplots.png", args.top_speakers)
    plot_all_speakers_boxplot(df, output_dir / "all_speakers_boxplot.png")
    plot_worst_files(df, output_dir / "worst_files.png", args.top_files)
    plot_wer_distribution(df, output_dir / "wer_distribution.png")

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
