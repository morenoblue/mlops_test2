import os, sys, json, time, warnings, importlib
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def getenv(key, default, cast=str):
    v = os.getenv(key)
    if v is None or v == "":
        return default
    try:
        return cast(v)
    except Exception:
        print(f"[train] Bad {key}={v!r}; using {default}", flush=True)
        return default

def import_symbol(modpath: str, attr: str = None):
    try:
        mod = importlib.import_module(modpath)
    except Exception as e:
        print(f"[train] Import failed: {modpath}: {e}", file=sys.stderr)
        sys.exit(2)
    if attr:
        if not hasattr(mod, attr):
            print(f"[train] {modpath} missing required attribute {attr}", file=sys.stderr)
            sys.exit(2)
        return getattr(mod, attr)
    return mod

def seed_and_threads():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import random, numpy as np
        random.seed(0); np.random.seed(0)
    except Exception:
        pass

def build_ctx() -> Dict[str, Any]:
    return {
        "model_name": getenv("MODEL", "random_forest"),
        "pipeline_name": getenv("PIPELINE", "pipeline_taxi_duration"),
        "rows": getenv("ROWS", 256, int),
        "test_size": getenv("TEST_SIZE", 0.2, float),
        "seed": getenv("SEED", 123, int),
        # Optional override; pipeline has its own module-relative default
        "data_path": os.getenv("DATA_PATH", ""),
    }

def main():
    seed_and_threads()
    ctx = build_ctx()

    pl = import_symbol(f"pipelines.{ctx['pipeline_name']}")
    get_model = import_symbol(f"models.{ctx['model_name']}", "get")

    X, y = pl.load_data(ctx)
    estimator = pl.build(get_model())

    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=float(ctx["test_size"]), random_state=int(ctx["seed"])
        )
        estimator.fit(Xtr, ytr)
        y_pred = estimator.predict(Xte)
        mae = float(mean_absolute_error(yte, y_pred))
        mse = float(mean_squared_error(yte, y_pred))  # no 'squared' kwarg for compat
        rmse = float(mse ** 0.5)
        r2 = float(r2_score(yte, y_pred))
    dur = time.time() - start

    print(json.dumps({
        "model": ctx["model_name"],
        "pipeline": ctx["pipeline_name"],
        "duration_s": round(dur, 4),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "samples": int(len(X)),
        "features": int(X.shape[1]),
        "test_size": float(ctx["test_size"]),
    }))

if __name__ == "__main__":
    main()

