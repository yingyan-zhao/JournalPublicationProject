from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .metrics import author_productivity, concentration_summary


def run(input_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(input_csv)
    output_tables = output_dir / "tables"
    output_tables.mkdir(parents=True, exist_ok=True)

    productivity = author_productivity(df, credit="fractional")
    summary = concentration_summary(productivity)

    productivity.to_csv(output_tables / "author_productivity.csv", index=False)
    summary.to_csv(output_tables / "concentration_summary.csv", index=False)

    if "year" in df.columns:
        annual = []
        for year, year_df in df.groupby("year"):
            year_productivity = author_productivity(year_df, credit="fractional")
            row = concentration_summary(year_productivity).assign(year=year)
            annual.append(row)
        if annual:
            pd.concat(annual, ignore_index=True).sort_values("year").to_csv(
                output_tables / "annual_concentration_summary.csv",
                index=False,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze concentration among top-journal economics authors."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.input_csv, args.output_dir)


if __name__ == "__main__":
    main()

