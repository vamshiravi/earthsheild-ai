import json
import re
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree

BASE = Path("data")
GRID = BASE / "metadata/global_cells.json"
INPUT = BASE / "raw/soil_moisture"
OUTPUT = BASE / "processed/environmental/temporal_soil_moisture.csv"

with open(GRID) as f:
    grid = pd.DataFrame(json.load(f))

grid_points = grid[["latitude", "longitude"]].to_numpy()

results = []

files = sorted(INPUT.glob("*.h5"))

print(f"SMAP files: {len(files)}")

for file in files:
    match = re.search(r"2025\d{4}", file.name)

    if not match:
        continue

    date = pd.to_datetime(match.group(), format="%Y%m%d")

    with h5py.File(file, "r") as h5:
        group = h5["Soil_Moisture_Retrieval_Data_PM"]

        lat = group["latitude_pm"][:]
        lon = group["longitude_pm"][:]
        sm = group["soil_moisture_dca_pm"][:]

    valid = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & np.isfinite(sm)
        & (sm > -9000)
    )

    source_points = np.column_stack([
        lat[valid],
        lon[valid]
    ])

    source_values = sm[valid]

    tree = cKDTree(source_points)

    distances, indices = tree.query(
        grid_points,
        distance_upper_bound=1.0
    )

    values = np.full(len(grid), np.nan)

    found = np.isfinite(distances)

    values[found] = source_values[indices[found]]

    day = grid.copy()

    day["date"] = date
    day["soil_moisture"] = values
    day["soil_moisture_available"] = np.isfinite(values).astype(int)

    results.append(day)

df = pd.concat(results, ignore_index=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("COMPLETE")
print(f"Rows: {len(df)}")
print(f"Dates: {df['date'].nunique()}")
print(f"Cells per date: {df.groupby('date').size().min()}")
print(f"Valid observations: {df['soil_moisture_available'].sum()}")
print(f"Missing observations: {df['soil_moisture'].isna().sum()}")
print(f"Saved: {OUTPUT}")