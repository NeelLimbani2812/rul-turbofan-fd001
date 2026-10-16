"""Shared feature-engineering code for the RUL demo.

The rolling-feature logic is identical to the notebook (notebooks/01_rul_fd001.ipynb)
so the demo app and the notebook always produce the same features.
"""
import pandas as pd

COLUMNS = ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
COLUMNS += [f"s_{i}" for i in range(1, 22)]

ROLLING_FEATURES = {
    "s_4": ["mean", "std"],
    "s_11": ["mean", "std"],
}
ROLLING_WINDOW = 5
RUL_CAP = 120


def add_rolling_features(frame, window=ROLLING_WINDOW):
    """Add per-engine rolling sensor statistics and remove incomplete windows."""
    result = frame.copy()
    for sensor, statistics in ROLLING_FEATURES.items():
        grouped = result.groupby("unit")[sensor]
        for statistic in statistics:
            name = f"{sensor}_roll_{statistic}_{window}"
            result[name] = (
                grouped.rolling(window)
                .agg(statistic)
                .reset_index(level=0, drop=True)
            )
    return result.dropna().copy()


def predict_engine(model, feature_columns, engine_raw):
    """Predict RUL for every cycle (from the first full rolling window) of ONE engine.

    engine_raw: raw rows of a single engine with the original dataset columns.
    Returns a DataFrame with columns: cycle, predicted_rul.
    """
    features = add_rolling_features(engine_raw)
    # Selecting by name (not position) fails loudly if a column is missing
    # and guarantees the same column order as during training.
    X = features[feature_columns]
    return pd.DataFrame({
        "cycle": features["cycle"].to_numpy(),
        "predicted_rul": model.predict(X),
    })
