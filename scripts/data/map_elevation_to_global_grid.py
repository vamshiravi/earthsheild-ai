import json
import xarray as xr
from pathlib import Path

INPUT = "data/raw/elevation/copernicus_lowresnc_global.nc"
GRID = "data/metadata/global_cells.json"
OUTPUT = "data/processed/environmental/global_elevation.json"

ds = xr.open_dataset(INPUT)

with open(GRID) as file:
    cells = json.load(file)

results = []

for cell in cells:
    value = ds["elev"].sel(
        latitude=cell["latitude"],
        longitude=cell["longitude"],
        method="nearest"
    ).item()

    results.append({
        "id": cell["id"],
        "latitude": cell["latitude"],
        "longitude": cell["longitude"],
        "elevation": float(value)
    })

Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w") as file:
    json.dump(results, file)

print("COMPLETE")
print(f"Global cells: {len(cells)}")
print(f"Elevation values: {len(results)}")
print(f"Saved: {OUTPUT}")