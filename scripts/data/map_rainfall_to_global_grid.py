import json
import xarray as xr
from pathlib import Path

GRID = Path("data/metadata/global_cells.json")
INPUT = "data/raw/rainfall/3B-DAY.MS.MRG.3IMERG.20250930-S000000-E235959.V07B.nc4"
OUTPUT = Path("data/processed/environmental/global_rainfall.json")

with open(GRID) as f:
    cells = json.load(f)

ds = xr.open_dataset(INPUT)

results = []

for i, cell in enumerate(cells, 1):
    longitude = cell["longitude"] % 360

    value = ds["precipitation"].sel(
        lat=cell["latitude"],
        lon=longitude,
        method="nearest",
    ).item()

    results.append({
        "cell_id": cell["id"],
        "latitude": cell["latitude"],
        "longitude": cell["longitude"],
        "rainfall": float(value),
    })

    if i % 5000 == 0:
        print(f"Processed {i}/{len(cells)} cells")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w") as f:
    json.dump(results, f)

print(f"\nGlobal cells processed: {len(results)}")
print(f"Saved: {OUTPUT}")