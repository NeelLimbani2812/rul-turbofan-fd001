"""Train the final RUL model (same recipe as the notebook) and save demo artifacts.

Run once from the repository root or from this folder:
    python app/train_model.py

It reads ../data/*.txt, trains XGBoost on the target capped at RUL_CAP,
prints the official-test metrics (compare them with the notebook), and writes
everything the Streamlit app needs into app/artifacts/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from rul_core import COLUMNS, RUL_CAP, ROLLING_WINDOW, add_rolling_features

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "random_state": 42,
    "n_jobs": -1,
}


def read(name, **kwargs):
    return pd.read_csv(DATA_DIR / name, sep=r"\s+", header=None, **kwargs)


def phm08_score(y_true, y_pred):
    errors = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(errors < 0, np.exp(-errors / 13) - 1, np.exp(errors / 10) - 1)))


def main():
    train_raw = read("train_FD001.txt", names=COLUMNS)
    test_raw = read("test_FD001.txt", names=COLUMNS)
    rul_df = pd.read_csv(DATA_DIR / "RUL_FD001.txt", header=None, names=["RUL"])

    max_cycle = train_raw.groupby("unit")["cycle"].max()
    train_raw["RUL"] = train_raw["unit"].map(max_cycle) - train_raw["cycle"]

    constant_columns = [
        col for col in train_raw.columns
        if col not in {"unit", "cycle", "RUL"}
        and train_raw[col].nunique(dropna=False) <= 1
    ]

    train_fe = add_rolling_features(train_raw)
    y = train_fe["RUL"].copy()
    X = train_fe.drop(columns=["RUL", "unit", *constant_columns])

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X, y.clip(upper=RUL_CAP))

    # Official test evaluation (last observed cycle per engine), same as the notebook.
    test_fe = add_rolling_features(test_raw)
    test_last = test_fe.groupby("unit", sort=True).tail(1).sort_values("unit")
    test_X = test_last.drop(columns=["unit", *constant_columns])
    assert list(test_X.columns) == list(X.columns), "Train/test feature mismatch"
    assert len(test_X) == len(rul_df) == 100
    pred = model.predict(test_X)
    true = rul_df["RUL"].to_numpy()
    err = pred - true
    metrics = {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "PHM08": phm08_score(true, pred),
        "Mean Error": float(np.mean(err)),
    }
    print("Official test metrics (cap %d):" % RUL_CAP)
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")
    print("Notebook reference: RMSE 17.92, MAE 13.20, PHM08 549.26, Mean Error -1.49")

    model.save_model(ARTIFACTS / "model.json")
    info = {
        "feature_columns": list(X.columns),
        "constant_columns": constant_columns,
        "rul_cap": RUL_CAP,
        "rolling_window": ROLLING_WINDOW,
        "official_test_metrics": {k: round(v, 2) for k, v in metrics.items()},
    }
    (ARTIFACTS / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    test_raw.to_csv(ARTIFACTS / "demo_test_FD001.csv", index=False)
    rul_df.to_csv(ARTIFACTS / "demo_rul_FD001.csv", index=False)
    print("Saved model and demo data to", ARTIFACTS)


if __name__ == "__main__":
    main()
