import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from utils.io import read_parquet_first
from utils.preprocessing.s1 import preprocess_data

# Base = .../src
BASE_PATH = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BASE_PATH / "data" / "smoke_sample.parquet"

POST_REQUIRED = [
    "passenger_count","trip_distance","pickup_longitude","pickup_latitude","rate_code",
    "dropoff_longitude","dropoff_latitude","fare_amount","surcharge","mta_tax",
    "tip_amount","tolls_amount","total_amount","trip_duration",
    "pickup_year","pickup_month","pickup_day","pickup_weekday","pickup_hour","pickup_part_of_day"
]

def load_data(ctx):
    # Prefer explicit env/ctx; else use module-relative default
    p = ctx.get("data_path") or os.getenv("DATA_PATH", "")
    path = Path(p) if p else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"DATA_PATH not found: {path}. Either set DATA_PATH to your parquet "
            f"or create the sample at {DEFAULT_DATA_PATH} (run src/scripts/make_smoke_sample.py)."
        )

    nrows = int(ctx.get("rows", 256))
    df = read_parquet_first(str(path), nrows=nrows)
    df_proc = preprocess_data(df)

    missing = [c for c in POST_REQUIRED if c not in df_proc.columns]
    if missing:
        raise ValueError(f"Preprocess missing columns {missing}. Got {list(df_proc.columns)}")

    y = df_proc["trip_duration"].astype(np.float32)
    X = df_proc.drop(columns=["trip_duration"])
    return X, y

def build(model):
    # Keep it simple (no clone); scaler is harmless for trees and useful if you swap models later
    return SkPipeline(steps=[
        ("scaler", StandardScaler()),
        ("reg", model)
    ])

