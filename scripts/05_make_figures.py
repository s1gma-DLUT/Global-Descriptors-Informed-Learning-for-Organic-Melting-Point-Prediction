#!/usr/bin/env python3
import argparse
import glob
import os
from datetime import datetime

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate prediction scatter figures from result CSV files.")
    parser.add_argument("--results_dir", default="./outputs", help="Directory containing prediction CSV files.")
    parser.add_argument("--output_dir", default="./reports/figures", help="Directory to save figures.")
    parser.add_argument("--target_col", default="MP", help="Experimental melting-point column.")
    parser.add_argument(
        "--pred_col",
        default="",
        help="Prediction column. If omitted, common prediction column names are tried.",
    )
    return parser.parse_args()


def find_prediction_column(df: pd.DataFrame, requested: str) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Prediction column not found: {requested}")
        return requested
    candidates = ["pred_3d_mean", "pred_mean", "prediction", "pred", "MP_pred"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"No prediction column found. Tried: {candidates}")


def plot_scatter(csv_path: str, output_dir: str, target_col: str, pred_col: str) -> str:
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column not found in {csv_path}: {target_col}")
    prediction_col = find_prediction_column(df, pred_col)

    x = pd.to_numeric(df[target_col], errors="coerce")
    y = pd.to_numeric(df[prediction_col], errors="coerce")
    valid = x.notna() & y.notna()
    if not valid.any():
        raise ValueError(f"No numeric target/prediction pairs in {csv_path}")

    x = x[valid]
    y = y[valid]
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())

    fig, ax = plt.subplots(figsize=(5.0, 4.5), dpi=200)
    ax.scatter(x, y, s=8, alpha=0.45, linewidths=0)
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    ax.set_xlabel("Experimental MP (degC)")
    ax.set_ylabel("Predicted MP (degC)")
    ax.set_title(os.path.splitext(os.path.basename(csv_path))[0])
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(csv_path))[0] + "_scatter.png")
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_figures(results_dir: str, output_dir: str, target_col: str, pred_col: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating figures...")
    print(f"Results directory: {results_dir}")

    made = []
    for csv_path in sorted(glob.glob(os.path.join(results_dir, "**", "*.csv"), recursive=True)):
        try:
            made.append(plot_scatter(csv_path, output_dir, target_col, pred_col))
        except ValueError as exc:
            print(f"Skipped {csv_path}: {exc}")

    if not made:
        raise FileNotFoundError("No scatter figures were generated from the available CSV files.")
    for path in made:
        print(f"Figure: {path}")


def main():
    args = parse_args()
    generate_figures(args.results_dir, args.output_dir, args.target_col, args.pred_col)


if __name__ == '__main__':
    main()
