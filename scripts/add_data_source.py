"""Add a `data_source` column to one or more parquet files for multi-teacher routing.

Usage:
    # Tag a single file:
    python scripts/add_data_source.py --input data/gsm8k/train.parquet --source gsm8k

    # Tag and merge multiple files into one:
    python scripts/add_data_source.py \
        --input data/gsm8k/train.parquet      --source gsm8k \
        --input data/humaneval/train.parquet   --source humaneval \
        --input data/alpaca/train.parquet      --source alpaca \
        --merge merged_train.parquet

    # Use a custom column name (default: data_source):
    python scripts/add_data_source.py --input data/gsm8k/train.parquet --source gsm8k --column task
"""

import argparse
from pathlib import Path

import pandas as pd


def add_data_source(input_path: str, source: str, column: str) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    if column in df.columns:
        print(f"  [warn] {input_path} already has column '{column}' — overwriting with '{source}'")
    df[column] = source
    print(f"  {input_path}: {len(df)} rows tagged as '{source}'")
    return df


def main():
    parser = argparse.ArgumentParser(description="Add data_source column to parquet files for multi-teacher routing.")
    parser.add_argument(
        "--input", "-i",
        action="append",
        required=True,
        help="Input parquet file path. Repeat for multiple files.",
    )
    parser.add_argument(
        "--source", "-s",
        action="append",
        required=True,
        help="data_source value for the corresponding --input. Must appear same number of times as --input.",
    )
    parser.add_argument(
        "--column", "-c",
        default="data_source",
        help="Column name to write (default: data_source). Must match distillation.teacher_key in your config.",
    )
    parser.add_argument(
        "--merge", "-m",
        default=None,
        help="If set, merge all tagged files into this single output parquet (shuffled).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffle when --merge is used (default: 42).",
    )
    parser.add_argument(
        "--suffix",
        default="_tagged",
        help="Suffix appended to each input filename for per-file output (default: _tagged). "
             "Ignored when --merge is used.",
    )
    args = parser.parse_args()

    if len(args.input) != len(args.source):
        parser.error(f"--input and --source must appear the same number of times "
                     f"(got {len(args.input)} inputs, {len(args.source)} sources)")

    dfs = []
    for input_path, source in zip(args.input, args.source):
        df = add_data_source(input_path, source, args.column)
        dfs.append((input_path, df))

    if args.merge:
        merged = pd.concat([df for _, df in dfs], ignore_index=True)
        merged = merged.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        merged.to_parquet(args.merge)
        print(f"\nMerged {len(merged)} rows -> {args.merge}")
    else:
        for input_path, df in dfs:
            p = Path(input_path)
            out = p.parent / f"{p.stem}{args.suffix}{p.suffix}"
            df.to_parquet(out)
            print(f"  -> {out}")


if __name__ == "__main__":
    main()
