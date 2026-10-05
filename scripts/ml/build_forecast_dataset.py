import pandas as pd


INPUT = "data/processed/features/temporal_features.csv"
OUTPUT = "data/processed/features/flood_forecast_dataset.csv"

HORIZON = 7


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["cell_id", "date"]
).reset_index(drop=True)


# ============================================================
# FUTURE 7-DAY RAINFALL
# t+1 ... t+7
# ============================================================

future_rain = (
    df.groupby("cell_id")["rainfall"]
    .transform(
        lambda x:
        x.shift(-1)
        .rolling(
            HORIZON,
            min_periods=HORIZON
        )
        .sum()
        .shift(-(HORIZON - 1))
    )
)

df["future_7d_rainfall"] = future_rain


# ============================================================
# CURRENT SOIL MOISTURE
# ============================================================

soil = df["soil_moisture"].fillna(
    df["soil_moisture"].median()
)


# ============================================================
# STANDARDIZE RISK COMPONENTS
# ============================================================

rain_mean = df["future_7d_rainfall"].mean()
rain_std = df["future_7d_rainfall"].std()

df["rainfall_risk"] = (
    df["future_7d_rainfall"] - rain_mean
) / (rain_std + 1e-8)


soil_mean = soil.mean()
soil_std = soil.std()

df["soil_risk"] = (
    soil - soil_mean
) / (soil_std + 1e-8)


# ============================================================
# FLOOD-RISK PROXY
# ============================================================

# Future rainfall = main driver
# Current soil moisture = supporting condition

df["flood_risk_raw"] = (
    0.70 * df["rainfall_risk"] +
    0.30 * df["soil_risk"]
)


# ============================================================
# SCALE TO 0–1
# ============================================================

minimum = df["flood_risk_raw"].min()
maximum = df["flood_risk_raw"].max()

df["flood_risk"] = (
    (df["flood_risk_raw"] - minimum) /
    (maximum - minimum + 1e-8)
)


# ============================================================
# REMOVE DAYS WITHOUT 7-DAY FUTURE
# ============================================================

df = df.dropna(
    subset=["future_7d_rainfall"]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print("CLEAN FORECAST DATASET COMPLETE")
print("Rows:", len(df))
print("Cells:", df["cell_id"].nunique())

print(
    "Date range:",
    df["date"].min(),
    "to",
    df["date"].max()
)

print()
print("Flood-risk proxy statistics:")
print(
    df["flood_risk"].describe()
)

print()
print("Saved:", OUTPUT)