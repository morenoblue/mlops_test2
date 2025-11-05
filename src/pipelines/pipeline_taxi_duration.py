import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline as Pipeline
from sklearn.preprocessing import StandardScaler
from utils.io import read_parquet
from utils.preprocessing.s1 import preprocess_data

BASE_PATH = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BASE_PATH / "data" / "smoke_sample.parquet"

POST_REQUIRED = [
    "passenger_count","trip_distance","pickup_longitude","pickup_latitude","rate_code",
    "dropoff_longitude","dropoff_latitude","fare_amount","surcharge","mta_tax",
    "tip_amount","tolls_amount","total_amount","trip_duration",
    "pickup_year","pickup_month","pickup_day","pickup_weekday","pickup_hour","pickup_part_of_day"
]

def load_data(ctx):
    nrows   = int(ctx.get("rows", 256))
    df      = read_parquet(str(DEFAULT_DATA_PATH), nrows=nrows)
    df_proc = preprocess_data(df)

    missing = [c for c in POST_REQUIRED if c not in df_proc.columns]
    if missing:
        raise ValueError(f"Preprocess missing columns {missing}. Got {list(df_proc.columns)}")

    y = df_proc["trip_duration"].astype(np.float32)
    X = df_proc.drop(columns=["trip_duration"])
    return X, y

def build(model):
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", model)
    ])

