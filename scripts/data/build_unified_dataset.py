import json
import pandas as pd
from pathlib import Path

BASE = Path("data")

def load_json(path):
    with open(path) as f:
        return pd.DataFrame(json.load(f))

grid = load_json(BASE / "metadata/global_cells.json")
rainfall = load_json(BASE / "processed/environmental/global_rainfall.json")
soil = load_json(BASE / "processed/environmental/global_soil_moisture.json")
elevation = load_json(BASE / "processed/environmental/global_elevation.json")
sentinel = load_json(BASE / "metadata/global_cell_sentinel2.json")

# Normalize cell identifier
grid = grid.rename(columns={"id": "cell_id"})
elevation = elevation.rename(columns={"id": "cell_id"})

# Keep only required columns
rainfall = rainfall[["cell_id", "rainfall"]]
soil = soil[["cell_id", "soil_moisture"]]
elevation = elevation[["cell_id", "elevation"]]
sentinel = sentinel[["cell_id", "cloud_cover", "datetime"]]

# Merge everything onto the global grid
df = grid.merge(rainfall, on="cell_id", how="left")
df = df.merge(soil, on="cell_id", how="left")
df = df.merge(elevation, on="cell_id", how="left")
df = df.merge(sentinel, on="cell_id", how="left")

# Handle missing observations
df["soil_moisture"] = df["soil_moisture"].replace(-9999.0, pd.NA)

df["soil_moisture_available"] = (
    df["soil_moisture"].notna().astype(int)
)

df["sentinel2_available"] = (
    df["datetime"].notna().astype(int)
)

output = BASE / "processed/features/global_features.csv"
output.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output, index=False)

print("COMPLETE")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Soil moisture available: {df['soil_moisture_available'].sum()}")
print(f"Sentinel-2 available: {df['sentinel2_available'].sum()}")
print(f"Saved: {output}")