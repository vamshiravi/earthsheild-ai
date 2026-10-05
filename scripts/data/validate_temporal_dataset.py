import pandas as pd

INPUT = "data/processed/features/temporal_features.csv"

df = pd.read_csv(INPUT, parse_dates=["date"])

print("Shape:", df.shape)
print("Dates:", df["date"].nunique())
print("Cells:", df["cell_id"].nunique())

print("\nDate range:")
print(df["date"].min(), "→", df["date"].max())

print("\nRows per date:")
print(df.groupby("date").size().describe())

print("\nMissing values:")
print(df.isna().sum())

print("\nFeatures:")
print(df.columns.tolist())

print("\nDuplicate cell-date pairs:")
print(
    df.duplicated(["cell_id", "date"]).sum()
)
