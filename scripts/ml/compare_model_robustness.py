import numpy as np
import pandas as pd


# ============================================================
# FILES
# ============================================================

EARTHSHIELD = (
    "data/processed/features/"
    "forecast_predictions.csv"
)

BASELINE = (
    "data/processed/features/"
    "baseline_forecast_predictions.csv"
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
    "model_robustness_daily.csv"
)

OUTPUT_SUMMARY = (
    "data/processed/features/"
    "model_robustness_summary.csv"
)


# ============================================================
# LOAD
# ============================================================

earth = pd.read_csv(EARTHSHIELD)
baseline = pd.read_csv(BASELINE)
shift = pd.read_csv(SHIFT)

earth["date"] = pd.to_datetime(earth["date"])
baseline["date"] = pd.to_datetime(baseline["date"])


# ============================================================
# MAP SHIFT DAY INDEX → DATE
# ============================================================

dates = pd.read_csv(
    TEMPORAL,
    usecols=["date"]
)

dates["date"] = pd.to_datetime(dates["date"])

unique_dates = (
    dates["date"]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

shift["date"] = shift["day_index"].apply(
    lambda x: unique_dates.iloc[int(x)]
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def metrics(actual, predicted):

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

    return mae, rmse, correlation


# ============================================================
# DAILY RESULTS
# ============================================================

rows = []

for date in sorted(
    earth["date"].unique()
):

    earth_day = earth[
        earth["date"] == date
    ]

    baseline_day = baseline[
        baseline["date"] == date
    ]

    earth_mae, earth_rmse, earth_corr = metrics(
        earth_day["actual_flood_risk"].to_numpy(),
        earth_day["predicted_flood_risk"].to_numpy()
    )

    base_mae, base_rmse, base_corr = metrics(
        baseline_day["actual_flood_risk"].to_numpy(),
        baseline_day["predicted_flood_risk"].to_numpy()
    )

    shift_row = shift[
        shift["date"] == date
    ]

    if shift_row.empty:
        continue

    shift_row = shift_row.iloc[0]

    rows.append(
        {
            "date": date,

            "shift_score":
                shift_row["shift_score"],

            "shift_ratio":
                shift_row["shift_ratio"],

            "shift_status":
                shift_row["shift_status"],

            "earthshield_mae":
                earth_mae,

            "earthshield_rmse":
                earth_rmse,

            "earthshield_correlation":
                earth_corr,

            "baseline_mae":
                base_mae,

            "baseline_rmse":
                base_rmse,

            "baseline_correlation":
                base_corr
        }
    )


daily = pd.DataFrame(rows)

daily.to_csv(
    OUTPUT_DAILY,
    index=False
)


# ============================================================
# GROUP BY SHIFT STATUS
# ============================================================

summary_rows = []

for status, group in daily.groupby(
    "shift_status"
):

    dates_in_group = set(
        group["date"]
    )

    earth_group = earth[
        earth["date"].isin(
            dates_in_group
        )
    ]

    base_group = baseline[
        baseline["date"].isin(
            dates_in_group
        )
    ]

    earth_mae, earth_rmse, earth_corr = metrics(
        earth_group["actual_flood_risk"].to_numpy(),
        earth_group["predicted_flood_risk"].to_numpy()
    )

    base_mae, base_rmse, base_corr = metrics(
        base_group["actual_flood_risk"].to_numpy(),
        base_group["predicted_flood_risk"].to_numpy()
    )

    summary_rows.append(
        {
            "shift_status": status,
            "days": len(group),

            "earthshield_mae":
                earth_mae,

            "earthshield_rmse":
                earth_rmse,

            "earthshield_correlation":
                earth_corr,

            "baseline_mae":
                base_mae,

            "baseline_rmse":
                base_rmse,

            "baseline_correlation":
                base_corr
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
# ROBUSTNESS DEGRADATION
# ============================================================

print()
print("=" * 70)
print("MODEL ROBUSTNESS COMPARISON")
print("=" * 70)

print()
print(
    daily.to_string(index=False)
)

print()
print(
    summary.to_string(index=False)
)


# ------------------------------------------------------------
# Calculate degradation from LOW → MODERATE
# ------------------------------------------------------------

low = summary[
    summary["shift_status"] == "LOW"
]

moderate = summary[
    summary["shift_status"] == "MODERATE"
]

if not low.empty and not moderate.empty:

    # EarthShield
    earth_low = low.iloc[0]["earthshield_mae"]
    earth_mod = moderate.iloc[0]["earthshield_mae"]

    earth_gap = earth_mod - earth_low

    earth_degradation = (
        earth_gap /
        (earth_low + 1e-8)
    ) * 100

    # Baseline
    base_low = low.iloc[0]["baseline_mae"]
    base_mod = moderate.iloc[0]["baseline_mae"]

    base_gap = base_mod - base_low

    base_degradation = (
        base_gap /
        (base_low + 1e-8)
    ) * 100

    print()
    print("ROBUSTNESS DEGRADATION")
    print("-" * 70)

    print(
        f"EarthShield low-shift MAE: "
        f"{earth_low:.6f}"
    )

    print(
        f"EarthShield moderate-shift MAE: "
        f"{earth_mod:.6f}"
    )

    print(
        f"EarthShield degradation: "
        f"{earth_degradation:.2f}%"
    )

    print()

    print(
        f"Baseline low-shift MAE: "
        f"{base_low:.6f}"
    )

    print(
        f"Baseline moderate-shift MAE: "
        f"{base_mod:.6f}"
    )

    print(
        f"Baseline degradation: "
        f"{base_degradation:.2f}%"
    )

    print()

    print(
        f"Robustness advantage: "
        f"{base_degradation - earth_degradation:.2f} "
        f"percentage points"
    )


# ============================================================
# COMPLETE
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
print("=" * 70)
print("ROBUSTNESS COMPARISON COMPLETE")
print("=" * 70)