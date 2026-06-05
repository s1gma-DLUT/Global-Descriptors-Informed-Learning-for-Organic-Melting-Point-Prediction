#!/usr/bin/env python3
import argparse
import glob
import os
from datetime import datetime

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate compact CSV tables from result files.")
    parser.add_argument("--results_dir", default="./outputs", help="Directory containing result CSV files.")
    parser.add_argument("--output_dir", default="./reports/tables", help="Directory to save summary tables.")
    return parser.parse_args()


def summarize_csv(path: str) -> dict:
    df = pd.read_csv(path)
    lower_cols = {col.lower(): col for col in df.columns}
    row = {"file": os.path.relpath(path), "rows": len(df)}

    for metric in ("mae", "rmse", "r2", "r^2"):
        if metric in lower_cols:
            row[metric.replace("^", "")] = pd.to_numeric(df[lower_cols[metric]], errors="coerce").mean()

    exp_col = next((lower_cols[name] for name in ("mp", "experimental", "experiment", "y_true") if name in lower_cols), None)
    pred_col = next(
        (col for col in df.columns if col.lower() in {"pred", "prediction", "pred_mean", "pred_3d_mean"}),
        None,
    )
    if exp_col and pred_col:
        exp = pd.to_numeric(df[exp_col], errors="coerce")
        pred = pd.to_numeric(df[pred_col], errors="coerce")
        valid = exp.notna() & pred.notna()
        if valid.any():
            err = pred[valid] - exp[valid]
            row["mae"] = err.abs().mean()
            row["rmse"] = (err.pow(2).mean()) ** 0.5

    return row


def generate_tables(results_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating tables...")
    print(f"Results directory: {results_dir}")

    csv_files = sorted(glob.glob(os.path.join(results_dir, "**", "*.csv"), recursive=True))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {results_dir}")

    summary = pd.DataFrame(summarize_csv(path) for path in csv_files)
    output_path = os.path.join(output_dir, "result_summary.csv")
    summary.to_csv(output_path, index=False)
    print(f"Summary table: {output_path}")


def main():
    args = parse_args()
    generate_tables(args.results_dir, args.output_dir)


if __name__ == '__main__':
    main()
