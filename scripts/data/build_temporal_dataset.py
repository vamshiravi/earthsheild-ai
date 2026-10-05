import json
import pandas as pd
from pathlib import Path

BASE = Path("data")

rainfall = pd.read_csv(
    BASE / "processed/environmental/temporal_rainfall.csv",
    parse_dates=["date"]
)

soil = pd.read_csv(
    BASE / "processed/environmental/temporal_soil_moisture.csv",
    parse_dates=["date"]
)

with open(BASE / "processed/environmental/global_elevation.json") as f:
    elevation = pd.DataFrame(json.load(f))

rainfall = rainfall.rename(columns={"id": "cell_id"})
soil = soil.rename(columns={"id": "cell_id"})
elevation = elevation.rename(columns={"id": "cell_id"})

elevation = elevation[["cell_id", "elevation"]]

df = rainfall.merge(
    soil[["cell_id", "date", "soil_moisture", "soil_moisture_available"]],
    on=["cell_id", "date"],
    how="left"
)

df = df.merge(
    elevation,
    on="cell_id",
    how="left"
)

df = df.sort_values(["date", "cell_id"])

output = BASE / "processed/features/temporal_features.csv"
output.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output, index=False)

print("COMPLETE")
print(f"Rows: {len(df)}")
print(f"Dates: {df['date'].nunique()}")
print(f"Cells/date: {df.groupby('date').size().min()}")
print(f"Columns: {len(df.columns)}")
print(f"SMAP valid: {df['soil_moisture_available'].sum()}")
print(f"Saved: {output}")