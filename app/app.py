"""Streamlit demo: Remaining Useful Life (RUL) of NASA C-MAPSS FD001 turbofan engines."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

from rul_core import predict_engine

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

st.set_page_config(page_title="Turbofan RUL demo", layout="wide")


@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model(ARTIFACTS / "model.json")
    info = json.loads((ARTIFACTS / "model_info.json").read_text(encoding="utf-8"))
    return model, info


@st.cache_data
def load_data():
    test = pd.read_csv(ARTIFACTS / "demo_test_FD001.csv")
    final_rul = pd.read_csv(ARTIFACTS / "demo_rul_FD001.csv")["RUL"].to_numpy()
    units = sorted(test["unit"].unique())
    return test, dict(zip(units, final_rul))


model, info = load_model()
test, final_rul = load_data()

st.title("Turbofan Engine Remaining Useful Life (RUL)")
st.caption(
    "Demo of an XGBoost model trained on NASA C-MAPSS FD001. Pick a test engine and a "
    "cycle; the model only sees sensor data up to that cycle."
)

engine = st.sidebar.selectbox("Test engine", sorted(final_rul), index=0)
engine_raw = test[test["unit"] == engine].sort_values("cycle")
last_cycle = int(engine_raw["cycle"].max())

preds = predict_engine(model, info["feature_columns"], engine_raw)
# True RUL at an earlier cycle = official RUL at the last cycle + cycles still observed after it.
preds["true_rul"] = final_rul[engine] + (last_cycle - preds["cycle"])

first_cycle = int(preds["cycle"].min())
cycle = st.sidebar.slider("Observed up to cycle", first_cycle, last_cycle, last_cycle)
st.sidebar.caption(f"Last observed cycle for this engine: {last_cycle}")

now = preds[preds["cycle"] == cycle].iloc[0]
predicted, true, error = float(now["predicted_rul"]), float(now["true_rul"]), float(now["predicted_rul"] - now["true_rul"])

c1, c2, c3 = st.columns(3)
c1.metric("Predicted RUL (cycles)", f"{predicted:.0f}")
c2.metric("True RUL (cycles)", f"{true:.0f}")
c3.metric("Error (predicted - true)", f"{error:+.0f}")

# Illustrative thresholds only; not a real maintenance policy.
if predicted < 30:
    st.error("Predicted RUL below 30 cycles (illustrative threshold): inspection would be advised.")
elif predicted < 60:
    st.warning("Predicted RUL below 60 cycles (illustrative threshold): plan maintenance.")
else:
    st.success("Predicted RUL above 60 cycles (illustrative threshold).")

history = preds[preds["cycle"] <= cycle].set_index("cycle")
st.subheader("Predicted vs true RUL")
st.line_chart(history[["predicted_rul", "true_rul"]].rename(
    columns={"predicted_rul": "Predicted RUL", "true_rul": "True RUL"}))

st.subheader("Key sensors (up to the selected cycle)")
sensors = engine_raw[engine_raw["cycle"] <= cycle].set_index("cycle")
s1, s2 = st.columns(2)
s1.caption("s_11")
s1.line_chart(sensors[["s_11"]])
s2.caption("s_4")
s2.line_chart(sensors[["s_4"]])

with st.expander("About this model and its limits"):
    metrics = info["official_test_metrics"]
    st.markdown(
        f"""
- XGBoost regressor trained on FD001 training engines with the RUL target capped at **{info['rul_cap']} cycles**,
  using per-engine rolling mean/std (window {info['rolling_window']}) of sensors `s_4` and `s_11`.
- Official test metrics from the training script (last cycle of each test engine):
  RMSE {metrics['RMSE']}, MAE {metrics['MAE']}, PHM08 {metrics['PHM08']}, mean error {metrics['Mean Error']}.
- Because of the cap, the model cannot predict RUL far above {info['rul_cap']} cycles, so early-life predictions are
  systematically low.
- Only FD001 (one operating condition, one fault mode) was used. This is a portfolio demo, not a maintenance tool.
- The cap was chosen after seeing over-prediction on the official test set; see the project README for details.
- Data: NASA C-MAPSS FD001. A. Saxena and K. Goebel (2008), "Turbofan Engine Degradation Simulation Data Set", 
  NASA Prognostics Data Repository, NASA Ames Research Center.
"""
    )
