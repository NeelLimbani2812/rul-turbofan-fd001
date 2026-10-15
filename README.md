# Predictive Maintenance: RUL Estimation for Turbofan Engines

A machine learning project that estimates the **Remaining Useful Life (RUL)** of aircraft turbofan engines from sensor data in NASA's C-MAPSS dataset (FD001 subset).

## Project Overview

Remaining Useful Life (RUL) is the number of operating cycles an engine can still complete before failure. Predicting it supports predictive maintenance by indicating when an engine may need servicing.

This project builds per-engine rolling sensor features, compares three regression models with engine-grouped cross-validation, tests several RUL caps for the training target, and reports an uncapped and a capped XGBoost model on the official test set. Errors are analysed per engine and the final model is interpreted with SHAP.

## Dataset

**NASA C-MAPSS, FD001 subset** (from the NASA Prognostics Center of Excellence data repository). Three files are used:

- `train_FD001.txt`: 20,631 rows from 100 engines, each run until failure.
- `test_FD001.txt`: 13,096 rows from 100 engines, each stopped before failure.
- `RUL_FD001.txt`: the true RUL at the last observed cycle of each test engine.

Each row is one engine cycle with 3 operating settings and 21 sensor measurements.

## Approach

1. Load the data and run quality checks (no missing values; 7 constant columns removed: `setting_3`, `s_1`, `s_5`, `s_10`, `s_16`, `s_18`, `s_19`).
2. Build the training target: RUL = engine's last cycle minus current cycle.
3. Create trailing rolling mean and standard deviation (window 5, computed within each engine) for sensors `s_4` and `s_11`. Final feature matrix: 20,231 rows × 22 features (the first 4 cycles of each engine have no full window).
4. Compare Linear Regression, Random Forest and XGBoost with 5-fold `GroupKFold`, so no engine appears in both training and validation.
5. Choose an RUL cap for the training target using a training-only "random-cut" validation (see below).
6. Evaluate the uncapped and capped XGBoost models once on the official test set.
7. Analyse prediction errors per engine and explain the final model with SHAP.

XGBoost settings in the final notebook (fixed, no hyperparameter search): `n_estimators=100`, `max_depth=4`, `learning_rate=0.1`, `random_state=42`.

## Results

### Cross-validation (training engines, all cycles, uncapped target)

| Model | Mean CV RMSE |
|---|---:|
| Linear Regression | 39.98 |
| Random Forest | 37.66 |
| XGBoost | 37.40 |

### RUL cap selection (training engines only, random-cut validation)

For each of 20 repeats and each GroupKFold split, one random cycle with true RUL between 7 and 145 is sampled from every held-out engine, and models trained with different caps are scored on those rows.

| Cap | RMSE | MAE | Mean Error |
|---:|---:|---:|---:|
| 100 | 22.12 | 17.47 | -6.64 |
| 110 | 20.48 | 16.26 | -3.14 |
| **120** | **19.85** | 15.60 | -0.16 |
| 125 | 19.90 | 15.53 | +1.15 |
| 130 | 20.08 | 15.57 | +2.31 |

Cap 120 has the lowest RMSE in this validation. Cap 125 performs very similarly, while caps 110 and 130 have slightly higher RMSE. No statistical significance test was run, so these results do not establish that one cap is definitively better than another.

### Official test set (last observed cycle of each test engine)

| Model | RMSE | MAE | PHM08 Score | Mean Error |
|---|---:|---:|---:|---:|
| XGBoost, uncapped target | 26.08 | 18.63 | 7587.51 | +11.72 |
| XGBoost, RUL cap 120 | 17.92 | 13.20 | 549.26 | -1.49 |

The uncapped model over-predicts RUL on average (+11.72 cycles). Training on a target capped at 120 removes most of that bias and improves all four metrics. PHM08 is an asymmetric score where late predictions are penalised more than early ones; lower is better.

> Cross-validation RMSE and official-test metrics evaluate different rows (all cycles vs. one last cycle per engine) and must not be compared directly. 

### Model interpretation (SHAP)

The features with the largest mean absolute SHAP value on a 100-row training sample are `s_11_roll_mean_5` (11.79), `s_4_roll_mean_5` (10.11) and `cycle` (8.71), followed by `s_9` and `s_14`. SHAP describes how the fitted model uses its inputs; it does not prove causation, and correlated features share importance.

## Limitations

- **The cap is not a fully independent result.** The idea of capping came from seeing over-prediction in the uncapped model's official-test results, and the 7 to 145 validation range follows the official test label range. The final test score may therefore be somewhat optimistic compared with a completely untouched hold-out.
- Caps 120 and 125 perform very similarly in this validation, while caps 110 and 130 have slightly higher RMSE. No statistical significance test was run.
- A capped model cannot predict RUL far above the cap: 13 of the 100 test engines have true RUL above 120, and several of the largest under-predictions come from them.
- Only FD001 (one operating condition, one fault mode) is used; results may not carry over to FD002 to FD004.
- No hyperparameter search is included in the final notebook; an earlier small tuning attempt was not adopted.
- Predictions are estimates and are not a substitute for engineering inspection or safety procedures.

## Technologies

Python 3.11+ (tested on 3.13.9), NumPy 2.3.5, pandas 2.3.3, Matplotlib 3.10.6, scikit-learn 1.7.2, XGBoost 3.4.1, SHAP 0.52.0, Jupyter Notebook.

## Project Structure

```text
rul-turbofan-fd001/
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── notebooks/
│   └── 01_rul_fd001.ipynb
├── requirements.txt
└── README.md
```

## How to Run

1. Download `6. Turbofan Engine Degradation Simulation Data Set.zip` from the [NASA PCoE Data Set Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/). Create a `data/` folder in the project root (it is not tracked by Git) and extract `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` into it. The ZIP contains all four subsets (FD001–FD004).

2. Install the required libraries from the project root:

   ```bash
   pip install -r requirements.txt
   pip install notebook
   ```

3. Start Jupyter Notebook from the `notebooks/` folder:

   ```bash
   cd notebooks
   jupyter notebook
   ```

4. Open `01_rul_fd001.ipynb` and select **Kernel → Restart Kernel and Run All Cells…** to run all cells.

**Reproducibility note:** The scikit-learn version is pinned in `requirements.txt` because cross-validation results, including `GroupKFold` splits, can depend on the library version.

## Future Improvements

- Try additional sensors, rolling windows and slope features.
- Tune hyperparameters with engine-grouped validation.
- Compare other models and evaluate on FD002 to FD004.

## Acknowledgements

- Dataset: NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS).
- Original dataset reference: A. Saxena and K. Goebel, “Turbofan Engine Degradation Simulation Data Set,” NASA Ames Prognostics Data Repository, 2008.
