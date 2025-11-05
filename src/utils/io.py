import os, glob
import pandas as pd

def read_parquet(path: str, nrows: int | None = None) -> pd.DataFrame:
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No .parquet files under directory: {path}")
        path = files[0]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet not found: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    return df.iloc[:nrows] if nrows else df

