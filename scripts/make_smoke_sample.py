#!/usr/bin/env python3
"""
Create a tiny *real-data* Parquet sample for CI smoke tests.

Examples:
  python scripts/make_smoke_sample.py --source /abs/path/to/dir_or_file \
      --rows 256 --out data/smoke_sample.parquet --seed 0
"""
import os, sys, glob, argparse, random
from typing import List
import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    print("Please `pip install pyarrow`", file=sys.stderr)
    raise

RAW_REQUIRED = [
    "vendor_id","pickup_datetime","dropoff_datetime",
    "passenger_count","trip_distance","pickup_longitude","pickup_latitude",
    "rate_code","dropoff_longitude","dropoff_latitude",
    "fare_amount","surcharge","mta_tax","tip_amount","tolls_amount","total_amount",
    "store_and_fwd_flag","payment_type"
]

def find_sources(src: str) -> List[str]:
    if os.path.isdir(src):
        files = sorted(glob.glob(os.path.join(src, "*.parquet")))
        if files:
            return files
        # fallback: CSVs (will read only the head)
        files = sorted(glob.glob(os.path.join(src, "*.csv")))
        if files:
            return files
        raise FileNotFoundError(f"No .parquet/.csv files in: {src}")
    if os.path.isfile(src):
        return [src]
    raise FileNotFoundError(f"Path not found: {src}")

def sample_parquet(files: List[str], rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tables, total = [], 0
    target = max(rows * 3, rows)  # read a bit more; downsample later
    for fp in files:
        pf = pq.ParquetFile(fp)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg)
            tables.append(t)
            total += t.num_rows
            if total >= target:
                break
        if total >= target:
            break
    if not tables:
        raise RuntimeError("No rows read from parquet")
    big = pa.concat_tables(tables, promote=True)
    df = big.to_pandas()
    if len(df) > rows:
        df = df.sample(n=rows, random_state=seed).reset_index(drop=True)
    return df

def sample_csv(files: List[str], rows: int, seed: int) -> pd.DataFrame:
    # Simplest approach: read head of first CSV, then downsample
    df = pd.read_csv(files[0])
    if len(df) > rows:
        df = df.sample(n=rows, random_state=seed).reset_index(drop=True)
    return df

def trim_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only columns needed by preprocessing; ignore extras to keep file small
    missing = [c for c in RAW_REQUIRED if c not in df.columns]
    if missing:
        print(f"[warn] Source missing columns {missing}. Keeping intersection only.", file=sys.stderr)
        keep = [c for c in RAW_REQUIRED if c in df.columns]
        return df[keep]
    return df[RAW_REQUIRED]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Path to a parquet/csv file or a directory containing them")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--out", default="data/smoke_sample.parquet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = find_sources(args.source)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if files[0].lower().endswith(".parquet"):
        df = sample_parquet(files, args.rows, args.seed)
    else:
        df = sample_csv(files, args.rows, args.seed)

    df = trim_columns(df)

    # Light dtype downcasting to keep file small
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        # safe downcast
        if df[col].between(0, 255).all(): df[col] = df[col].astype("uint8")
        elif df[col].between(0, 65535).all(): df[col] = df[col].astype("uint16")
        else: df[col] = df[col].astype("int32")

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, args.out, compression="snappy")
    size_kb = os.path.getsize(args.out) / 1024.0
    print(f"Wrote {args.out} ({size_kb:.1f} KB, {len(df)} rows, {df.shape[1]} cols)")

if __name__ == "__main__":
    main()

