import pandas as pd
import numpy as np


PREDICTIONS = (
    "data/processed/features/"
    "forecast_predictions.csv"
)

SHIFT = (
    "data/processed/features/"
    "distribution_shift.csv"
)

OUTPUT = (
    "data/processed/features/"
    "shift_aware_predictions.csv"
)


# ============================================================
# LOAD
# ============================================================

pred = pd.read_csv(PREDICTIONS)
shift = pd.read_csv(SHIFT)

pred["date"] = pd.to_datetime(pred["date"])


# ============================================================
# MAP SHIFT DAY → DATE
# ============================================================

temporal_dates = pd.read_csv(
    "data/processed/features/temporal_features.csv",
    usecols=["date"]
)

temporal_dates["date"] = pd.to_datetime(
    temporal_dates["date"]
)

dates = (
    temporal_dates["date"]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

shift["date"] = shift["day_index"].apply(
    lambda x: dates.iloc[int(x)]
)


# ============================================================
# MERGE
# ============================================================

result = pred.merge(
    shift[
        [
            "date",
            "shift_score",
            "shift_ratio",
            "shift_status"
        ]
    ],
    on="date",
    how="left"
)


# ============================================================
# SHIFT-AWARE CONFIDENCE
# ============================================================

# Prototype confidence rule:
#
# LOW       → 0.90
# MODERATE  → 0.70
# HIGH      → 0.50

confidence_map = {
    "LOW": 0.90,
    "MODERATE": 0.70,
    "HIGH": 0.50
}

result["model_confidence"] = (
    result["shift_status"]
    .map(confidence_map)
    .fillna(0.50)
)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT,
    index=False
)

print("SHIFT-AWARE CONFIDENCE COMPLETE")
print("Rows:", len(result))
print("Saved:", OUTPUT)

print()
print(
    result[
        [
            "date",
            "shift_status",
            "shift_ratio",
            "model_confidence"
        ]
    ]
    .drop_duplicates()
    .to_string(index=False)
)