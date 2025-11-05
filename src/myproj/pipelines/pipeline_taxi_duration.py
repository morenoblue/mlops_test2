import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from myproj.utils.io import read_parquet_first
from myproj.utils.preprocessing.s1 import preprocess_data

POST_REQUIRED = [
    "passenger_count","trip_distance","pickup_longitude","pickup_latitude","rate_code",
    "dropoff_longitude","dropoff_latitude","fare_amount","surcharge","mta_tax",
    "tip_amount","tolls_amount","total_amount","trip_duration",
    "pickup_year","pickup_month","pickup_day","pickup_weekday","pickup_hour","pickup_part_of_day"
]

def load_data(ctx):
    path = ctx.get("data_path") or os.getenv("DATA_PATH", "")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"DATA_PATH not found: {path}. Generate it with scripts/make_smoke_sample.py "
            "or set DATA_PATH to your parquet."
        )
    nrows = int(ctx.get("rows", 256))
    df = read_parquet_first(path, nrows=nrows)
    df_proc = preprocess_data(df)
    missing = [c for c in POST_REQUIRED if c not in df_proc.columns]
    if missing:
        raise ValueError(f"Preprocess missing columns {missing}. Got {list(df_proc.columns)}")
    y = df_proc["trip_duration"].astype(np.float32)
    X = df_proc.drop(columns=["trip_duration"])
    return X, y

def build(model):
    return Pipeline(steps=[
        ("scaler", StandardScaler()),  # harmless for trees; helpful if you swap models later
        ("reg", clone(model))
    ])

