#!/usr/bin/env python3

from pathlib import Path
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_PATH = Path(__file__).resolve()
SRC_SOURCE_PATH = BASE_PATH.parents[2] / "yellow_tripdata_2010-01.parquet"
OUT_PATH = BASE_PATH.parents[1] / "data" / "smoke_sample.parquet"
ROWS = 256      
SEED = 0       
RAW_REQUIRED = [
    "vendor_id","pickup_datetime","dropoff_datetime",
    "passenger_count","trip_distance","pickup_longitude","pickup_latitude",
    "rate_code","dropoff_longitude","dropoff_latitude",
    "fare_amount","surcharge","mta_tax","tip_amount","tolls_amount","total_amount",
    "store_and_fwd_flag","payment_type"
]

def _find_sources(src: Path) -> list[Path]:
    if src.is_dir():
        files = sorted(list(src.glob("*.parquet")) + list(src.glob("*.csv")))
        if not files:
            raise FileNotFoundError(f"No .parquet/.csv files in: {src}")
        return files
    if src.is_file():
        return [src]
    raise FileNotFoundError(f"Path not found: {src}")

def _sample_parquet(files: list[Path], rows: int, seed: int) -> pd.DataFrame:
    tables, total = [], 0
    target = max(rows * 3, rows)
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

def _sample_csv(files: list[Path], rows: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(files[0])
    if len(df) > rows:
        df = df.sample(n=rows, random_state=seed).reset_index(drop=True)
    return df

def _trim_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in RAW_REQUIRED if c in df.columns]
    if not keep:
        raise ValueError("None of the required columns were found in the source.")
    return df[keep]

def main():
    src = Path(SRC_SOURCE_PATH)
    files = _find_sources(src)
    out = OUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    if files[0].suffix.lower() == ".parquet":
        df = _sample_parquet(files, ROWS, SEED)
    else:
        df = _sample_csv(files, ROWS, SEED)

    df = _trim_columns(df)

    # downcast to keep file small
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        if df[col].between(0, 255).all(): df[col] = df[col].astype("uint8")
        elif df[col].between(0, 65535).all(): df[col] = df[col].astype("uint16")
        else: df[col] = df[col].astype("int32")

    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out, compression="snappy")
    size_kb = out.stat().st_size / 1024.0
    print(f"Wrote {out} ({size_kb:.1f} KB, {len(df)} rows, {df.shape[1]} cols)")

if __name__ == "__main__":
    main()
