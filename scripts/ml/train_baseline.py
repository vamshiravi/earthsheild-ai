import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIG
# ============================================================

TEMPORAL = "data/processed/features/temporal_features.csv"
TARGET = "data/processed/features/flood_forecast_dataset.csv"

MODEL_PATH = "models/baseline_forecast.pkl"
PREDICTION_PATH = (
    "data/processed/features/"
    "baseline_forecast_predictions.csv"
)

DAYS = 30
HISTORY = 7

ALPHA = 1.0


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(TEMPORAL)
target_df = pd.read_csv(TARGET)

df["date"] = pd.to_datetime(df["date"])
target_df["date"] = pd.to_datetime(target_df["date"])

cells = len(df) // DAYS

print("Cells:", cells)
print("Days:", DAYS)


# ============================================================
# BUILD ENVIRONMENT TENSOR
# ============================================================

features = [
    "rainfall",
    "soil_moisture",
    "soil_moisture_available",
    "elevation"
]

X = df[features].to_numpy(
    dtype=np.float32
).reshape(
    DAYS,
    cells,
    len(features)
)

# Fill missing environmental values using
# training-safe simple zero after standardization.
for j in [0, 1, 3]:

    mean = np.nanmean(X[:, :, j])
    std = np.nanstd(X[:, :, j])

    X[:, :, j] = (
        X[:, :, j] - mean
    ) / (std + 1e-8)

    X[:, :, j] = np.nan_to_num(
        X[:, :, j]
    )


# ============================================================
# TARGET MATRIX
# ============================================================

dates = sorted(
    target_df["date"].unique()
)

date_to_idx = {
    d: i
    for i, d in enumerate(dates)
}

# cell order is the same as global grid
cell_to_idx = {
    cell_id: i
    for i, cell_id in enumerate(
        df["cell_id"].iloc[:cells]
    )
}

Y = np.full(
    (cells, len(dates)),
    np.nan,
    dtype=np.float32
)

for row in target_df.itertuples():

    i = cell_to_idx[row.cell_id]
    j = date_to_idx[row.date]

    Y[i, j] = row.flood_risk


# ============================================================
# CREATE SAMPLES
# ============================================================

def build_samples(day_indices):

    X_list = []
    y_list = []
    cell_list = []
    date_list = []

    for t in day_indices:

        start = t - HISTORY + 1

        sequence = X[
            start:t + 1
        ]

        # [7, cells, 4]
        # → [cells, 28]

        sequence = sequence.transpose(
            1, 0, 2
        ).reshape(
            cells,
            -1
        )

        valid = np.isfinite(
            Y[:, t]
        )

        X_list.append(
            sequence[valid]
        )

        y_list.append(
            Y[valid, t]
        )

        cell_list.extend(
            np.where(valid)[0]
        )

        date_list.extend(
            [dates[t]] * valid.sum()
        )

    return (
        np.vstack(X_list),
        np.concatenate(y_list),
        np.asarray(cell_list),
        np.asarray(date_list)
    )


# Same chronological split as EarthShield
train_days = range(
    HISTORY - 1,
    16
)

val_days = range(
    16,
    19
)

test_days = range(
    19,
    len(dates)
)


X_train, y_train, _, _ = build_samples(
    train_days
)

X_val, y_val, _, _ = build_samples(
    val_days
)

X_test, y_test, test_cells, test_dates = build_samples(
    test_days
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# TRAIN
# ============================================================

model = Ridge(
    alpha=ALPHA
)

print("Training Ridge baseline...")

model.fit(
    X_train,
    y_train
)


# ============================================================
# VALIDATION
# ============================================================

val_pred = np.clip(
    model.predict(X_val),
    0,
    1
)

val_mae = mean_absolute_error(
    y_val,
    val_pred
)

val_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        val_pred
    )
)

print(
    f"Validation MAE: {val_mae:.6f}"
)

print(
    f"Validation RMSE: {val_rmse:.6f}"
)


# ============================================================
# TEST
# ============================================================

test_pred = np.clip(
    model.predict(X_test),
    0,
    1
)

test_mae = mean_absolute_error(
    y_test,
    test_pred
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_pred
    )
)

correlation = np.corrcoef(
    test_pred,
    y_test
)[0, 1]


# ============================================================
# SAVE MODEL
# ============================================================

Path(
    MODEL_PATH
).parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    MODEL_PATH,
    "wb"
) as f:
    pickle.dump(
        model,
        f
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions = pd.DataFrame(
    {
        "cell_index": test_cells,
        "date": test_dates,
        "actual_flood_risk": y_test,
        "predicted_flood_risk": test_pred
    }
)

predictions.to_csv(
    PREDICTION_PATH,
    index=False
)


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 60)
print("BASELINE VALIDATION COMPLETE")
print("=" * 60)

print(
    f"Test MAE: {test_mae:.6f}"
)

print(
    f"Test RMSE: {test_rmse:.6f}"
)

print(
    f"Test correlation: {correlation:.4f}"
)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Predictions:",
    PREDICTION_PATH
)

print("=" * 60)