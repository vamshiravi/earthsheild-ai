import numpy as np
import pandas as pd


# ============================================================
# FILES
# ============================================================

PREDICTIONS = (
    "data/processed/features/"
    "forecast_predictions.csv"
)

SHIFT = (
    "data/processed/features/"
    "distribution_shift.csv"
)

TEMPORAL = (
    "data/processed/features/"
    "temporal_features.csv"
)

OUTPUT_DAILY = (
    "data/processed/features/"
    "robustness_daily.csv"
)

OUTPUT_SUMMARY = (
    "data/processed/features/"
    "robustness_summary.csv"
)


# ============================================================
# LOAD FORECAST RESULTS
# ============================================================

pred = pd.read_csv(
    PREDICTIONS
)

pred["date"] = pd.to_datetime(
    pred["date"]
)

print(
    "Forecast rows:",
    len(pred)
)


# ============================================================
# LOAD SHIFT RESULTS
# ============================================================

shift = pd.read_csv(
    SHIFT
)

print(
    "Shift rows:",
    len(shift)
)

print(
    "Shift columns:",
    list(shift.columns)
)


# ============================================================
# MAP EMBEDDING DAY INDEX → DATE
# ============================================================

dates = pd.read_csv(
    TEMPORAL,
    usecols=["date"]
)

dates["date"] = pd.to_datetime(
    dates["date"]
)

unique_dates = (
    dates["date"]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

shift["date"] = shift[
    "day_index"
].map(
    lambda x: unique_dates.iloc[int(x)]
)


# ============================================================
# KEEP ONLY FORECAST TEST DATES
# ============================================================

test_dates = set(
    pred["date"].unique()
)

shift = shift[
    shift["date"].isin(test_dates)
].copy()

print()
print(
    "Forecast test dates:",
    sorted(test_dates)
)


# ============================================================
# DAILY FORECAST METRICS
# ============================================================

daily_rows = []

for date, group in pred.groupby(
    "date"
):

    actual = group[
        "actual_flood_risk"
    ].to_numpy()

    predicted = group[
        "predicted_flood_risk"
    ].to_numpy()

    mae = np.mean(
        np.abs(
            predicted - actual
        )
    )

    rmse = np.sqrt(
        np.mean(
            (predicted - actual) ** 2
        )
    )

    correlation = np.corrcoef(
        predicted,
        actual
    )[0, 1]

    daily_rows.append(
        {
            "date": date,
            "mae": mae,
            "rmse": rmse,
            "correlation": correlation
        }
    )


daily = pd.DataFrame(
    daily_rows
)


# ============================================================
# MERGE SHIFT INFORMATION
# ============================================================

shift_info = shift[
    [
        "date",
        "shift_score",
        "baseline_mmd",
        "shift_ratio",
        "shift_status"
    ]
]

daily = daily.merge(
    shift_info,
    on="date",
    how="left"
)

daily = daily.sort_values(
    "date"
)


# ============================================================
# SAVE DAILY RESULTS
# ============================================================

daily.to_csv(
    OUTPUT_DAILY,
    index=False
)


# ============================================================
# GROUPED ROBUSTNESS METRICS
# ============================================================

summary_rows = []

for status, group in daily.groupby(
    "shift_status"
):

    dates_in_group = set(
        group["date"]
    )

    observations = pred[
        pred["date"].isin(
            dates_in_group
        )
    ]

    actual = observations[
        "actual_flood_risk"
    ].to_numpy()

    predicted = observations[
        "predicted_flood_risk"
    ].to_numpy()

    mae = np.mean(
        np.abs(
            predicted - actual
        )
    )

    rmse = np.sqrt(
        np.mean(
            (predicted - actual) ** 2
        )
    )

    correlation = np.corrcoef(
        predicted,
        actual
    )[0, 1]

    summary_rows.append(
        {
            "shift_status": status,
            "days": len(group),
            "samples": len(observations),
            "mae": mae,
            "rmse": rmse,
            "correlation": correlation
        }
    )


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# ============================================================
# ROBUSTNESS GAP
# ============================================================

low = summary[
    summary["shift_status"] == "LOW"
]

moderate = summary[
    summary["shift_status"] == "MODERATE"
]

print()
print("=" * 60)
print("ROBUSTNESS EVALUATION")
print("=" * 60)

print()
print("Daily results:")
print(
    daily.to_string(
        index=False
    )
)

print()
print("Grouped results:")
print(
    summary.to_string(
        index=False
    )
)

if not low.empty and not moderate.empty:

    low_mae = low.iloc[0]["mae"]
    moderate_mae = moderate.iloc[0]["mae"]

    robustness_gap = (
        moderate_mae -
        low_mae
    )

    degradation = (
        robustness_gap /
        (low_mae + 1e-8)
    ) * 100

    print()
    print(
        f"Low-shift MAE: "
        f"{low_mae:.6f}"
    )

    print(
        f"Moderate-shift MAE: "
        f"{moderate_mae:.6f}"
    )

    print(
        f"Robustness gap: "
        f"{robustness_gap:.6f}"
    )

    print(
        f"Relative degradation: "
        f"{degradation:.2f}%"
    )

else:

    print()
    print(
        "Not enough shift categories "
        "for a robustness-gap calculation."
    )


# ============================================================
# OUTPUT
# ============================================================

print()
print(
    "Daily output:",
    OUTPUT_DAILY
)

print(
    "Summary output:",
    OUTPUT_SUMMARY
)

print()
print("=" * 60)
print("ROBUSTNESS EVALUATION COMPLETE")
print("=" * 60)