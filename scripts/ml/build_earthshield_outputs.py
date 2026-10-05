import pandas as pd


ANOMALY = (
    "data/processed/features/"
    "ssl_anomaly_scores.csv"
)

PREDICTIONS = (
    "data/processed/features/"
    "shift_aware_predictions.csv"
)

OUTPUT = (
    "data/processed/features/"
    "earthshield_outputs.csv"
)


# ============================================================
# LOAD
# ============================================================

anomaly = pd.read_csv(
    ANOMALY,
    usecols=[
        "cell_id",
        "latitude",
        "longitude",
        "date",
        "anomaly_score",
        "anomaly"
    ]
)

pred = pd.read_csv(
    PREDICTIONS
)

anomaly["date"] = pd.to_datetime(anomaly["date"])
pred["date"] = pd.to_datetime(pred["date"])


# ============================================================
# MERGE
# ============================================================

result = anomaly.merge(
    pred,
    on=["cell_id", "date"],
    how="inner"
)

print("Merged rows:", len(result))


# ============================================================
# CALIBRATE RELATIVE RISK THRESHOLDS
# ============================================================

low_threshold = result[
    "predicted_flood_risk"
].quantile(0.70)

high_threshold = result[
    "predicted_flood_risk"
].quantile(0.90)

print()
print(
    f"Low/Moderate threshold: "
    f"{low_threshold:.6f}"
)

print(
    f"Moderate/High threshold: "
    f"{high_threshold:.6f}"
)


# ============================================================
# RISK LEVEL
# ============================================================

def risk_level(value):

    if value >= high_threshold:
        return "HIGH"

    if value >= low_threshold:
        return "MODERATE"

    return "LOW"


result["risk_level"] = (
    result["predicted_flood_risk"]
    .apply(risk_level)
)


# ============================================================
# EARTHSHIELD STATUS
# ============================================================

def alert_level(row):

    if (
        row["risk_level"] == "HIGH"
        or (
            row["risk_level"] == "MODERATE"
            and row["anomaly"] == 1
        )
    ):
        return "HIGH ALERT"

    if (
        row["risk_level"] == "MODERATE"
        or row["anomaly"] == 1
    ):
        return "WATCH"

    return "NORMAL"


result["earthshield_status"] = (
    result.apply(
        alert_level,
        axis=1
    )
)


# ============================================================
# CONFIDENCE
# ============================================================

result["confidence_percent"] = (
    result["model_confidence"] * 100
).round(1)


# ============================================================
# FINAL COLUMNS
# ============================================================

result = result[
    [
        "cell_id",
        "latitude",
        "longitude",
        "date",

        "predicted_flood_risk",
        "risk_level",

        "anomaly_score",
        "anomaly",

        "shift_score",
        "shift_ratio",
        "shift_status",

        "model_confidence",
        "confidence_percent",

        "earthshield_status"
    ]
]


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 60)
print("CALIBRATED EARTHSHIELD OUTPUT COMPLETE")
print("=" * 60)

print()
print("Rows:", len(result))

print()
print("Risk levels:")
print(
    result["risk_level"]
    .value_counts()
)

print()
print("EarthShield status:")
print(
    result["earthshield_status"]
    .value_counts()
)

print()
print("Shift status:")
print(
    result["shift_status"]
    .value_counts()
)

print()
print("Saved:", OUTPUT)