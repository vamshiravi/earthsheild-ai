import json
from pathlib import Path

import pandas as pd
import xarray as xr

BASE = Path("data")
GRID = BASE / "metadata/global_cells.json"
INPUT = BASE / "raw/rainfall"
OUTPUT = BASE / "processed/environmental/temporal_rainfall.csv"

with open(GRID) as f:
    cells = json.load(f)

grid = pd.DataFrame(cells)

results = []

files = sorted(INPUT.glob("*.nc4"))

print(f"Rainfall files: {len(files)}")

for file in files:
    date = file.name.split(".")[4][:8]

    ds = xr.open_dataset(file)
    rainfall = ds["precipitation"].isel(time=0)

    values = rainfall.sel(
        lat=xr.DataArray(grid["latitude"].values, dims="cell"),
        lon=xr.DataArray(grid["longitude"].values, dims="cell"),
        method="nearest",
    ).values

    day = grid[["id", "latitude", "longitude"]].copy()
    day["date"] = pd.to_datetime(date, format="%Y%m%d")
    day["rainfall"] = values

    results.append(day)

    ds.close()

df = pd.concat(results, ignore_index=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("COMPLETE")
print(f"Rows: {len(df)}")
print(f"Dates: {df['date'].nunique()}")
print(f"Cells per date: {df.groupby('date').size().min()}")
print(f"Saved: {OUTPUT}")