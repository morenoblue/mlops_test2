import pandas as pd
import numpy as np

MINS = pd.Series({
    "passenger_count": 0, "trip_distance": 0.0,
    "pickup_longitude": -74.2, "pickup_latitude": 40.5,
    "rate_code": 0,
    "dropoff_longitude": -74.2, "dropoff_latitude": 40.5,
    "fare_amount": 0.0, "surcharge": 0.0, "mta_tax": 0.0,
    "tip_amount": 0.0, "tolls_amount": 0.0, "total_amount": 0.0,
    "trip_duration": 0.0,
    "pickup_year": 2000, "pickup_month": 1, "pickup_day": 1,
    "pickup_weekday": 0, "pickup_hour": 0, "pickup_part_of_day": 0
})
MAXS = pd.Series({
    "passenger_count": 5, "trip_distance": 50.0,
    "pickup_longitude": -73.7, "pickup_latitude": 41.0,
    "rate_code": 10,
    "dropoff_longitude": -73.7, "dropoff_latitude": 41.0,
    "fare_amount": 75.0, "surcharge": 2.0, "mta_tax": 0.5,
    "tip_amount": 20.0, "tolls_amount": 7.5, "total_amount": 100.0,
    "trip_duration": 7200.0,
    "pickup_year": 2020, "pickup_month": 12, "pickup_day": 31,
    "pickup_weekday": 6, "pickup_hour": 23, "pickup_part_of_day": 3
})

def _tod(hour: int) -> int:
    if 6 <= hour < 12:  return 0
    if 12 <= hour < 16: return 1
    if 16 <= hour < 22: return 2
    return 3

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    mins = MINS.reindex(df.columns)
    maxs = MAXS.reindex(df.columns)
    mask = (df >= mins) & (df <= maxs)
    df = df[mask.all(axis=1)].copy()
    df = (df - mins) / (maxs - mins)
    return df.astype(np.float32)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    casts = {
        "passenger_count": "uint8", "trip_distance": "float32",
        "pickup_longitude": "float32","pickup_latitude": "float32",
        "dropoff_longitude": "float32","dropoff_latitude": "float32",
        "fare_amount": "float32","surcharge": "float32","mta_tax": "float32",
        "tip_amount": "float32","tolls_amount": "float32","total_amount": "float32",
    }
    for c,t in casts.items():
        if c in df.columns: df[c] = df[c].astype(t)

    if "vendor_id" in df.columns:
        df = df.drop(columns=["vendor_id"])

    if "pickup_datetime" in df.columns:
        ts = pd.to_datetime(df["pickup_datetime"])
        df["pickup_year"]    = ts.dt.year.astype("uint16")
        df["pickup_month"]   = ts.dt.month.astype("uint8")
        df["pickup_day"]     = ts.dt.day.astype("uint8")
        df["pickup_weekday"] = ts.dt.weekday.astype("uint8")
        df["pickup_hour"]    = ts.dt.hour.astype("uint8")
        df["pickup_part_of_day"] = df["pickup_hour"].apply(_tod).astype("uint8")

    if "dropoff_datetime" in df.columns and "pickup_datetime" in df.columns:
        dur = (pd.to_datetime(df["dropoff_datetime"]) - pd.to_datetime(df["pickup_datetime"])) \
                .dt.total_seconds().astype("float32")
        df["trip_duration"] = dur
        df = df[df["trip_duration"] > 0]

    if "rate_code" in df.columns:
        df["rate_code"] = df["rate_code"].apply(lambda x: x if int(x) in range(11) else 0).astype("uint8")

    for col in ["store_and_fwd_flag","payment_type","pickup_datetime","dropoff_datetime"]:
        if col in df.columns: df = df.drop(columns=[col])

    return normalize(df)

