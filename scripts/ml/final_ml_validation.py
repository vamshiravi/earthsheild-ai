import pandas as pd


FILE = (
    "data/processed/features/"
    "earthshield_outputs.csv"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(FILE)

df["date"] = pd.to_datetime(df["date"])

print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# BASIC CHECKS
# ============================================================

print()
print("Missing values:")
print(df.isna().sum())

print()
print("Duplicate cell-date pairs:")

duplicates = df.duplicated(
    subset=["cell_id", "date"]
).sum()

print(duplicates)


# ============================================================
# RANGE CHECKS
# ============================================================

print()
print("Flood risk range:")
print(
    df["predicted_flood_risk"].min(),
    df["predicted_flood_risk"].max()
)

print()
print("Confidence range:")
print(
    df["confidence_percent"].min(),
    df["confidence_percent"].max()
)


# ============================================================
# DATE CHECK
# ============================================================

print()
print(
    "Dates:",
    df["date"].min(),
    "to",
    df["date"].max()
)

print(
    "Unique dates:",
    df["date"].nunique()
)


# ============================================================
# CELL CHECK
# ============================================================

print()
print(
    "Unique cells:",
    df["cell_id"].nunique()
)


# ============================================================
# CATEGORICAL CHECKS
# ============================================================

print()
print("Risk levels:")
print(
    df["risk_level"].value_counts()
)

print()
print("Shift statuses:")
print(
    df["shift_status"].value_counts()
)

print()
print("EarthShield statuses:")
print(
    df["earthshield_status"].value_counts()
)


# ============================================================
# LOGICAL CONSISTENCY
# ============================================================

invalid_risk = (
    ((df["risk_level"] == "HIGH") &
     (df["predicted_flood_risk"] < 0))
    |
    ((df["risk_level"] == "MODERATE") &
     (df["predicted_flood_risk"] < 0))
)

invalid_confidence = ~df[
    "confidence_percent"
].between(0, 100)

print()
print(
    "Invalid risk records:",
    invalid_risk.sum()
)

print(
    "Invalid confidence records:",
    invalid_confidence.sum()
)


# ============================================================
# FINAL VERDICT
# ============================================================

checks = [
    df.isna().sum().sum() == 0,
    duplicates == 0,
    invalid_risk.sum() == 0,
    invalid_confidence.sum() == 0,
    df["cell_id"].nunique() > 0,
    df["date"].nunique() > 0
]

print()

if all(checks):

    print("=" * 60)
    print("FINAL ML VALIDATION: PASSED")
    print("=" * 60)

else:

    print("=" * 60)
    print("FINAL ML VALIDATION: CHECK FAILED")
    print("=" * 60)