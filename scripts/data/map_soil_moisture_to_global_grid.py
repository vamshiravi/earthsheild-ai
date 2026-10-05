import json
import h5py
import numpy as np
from pathlib import Path

GRID = Path("data/metadata/global_cells.json")
INPUT = Path(
    "data/raw/soil_moisture/"
    "SMAP_L3_SM_P_20250930_R19240_001.h5"
)
OUTPUT = Path(
    "data/processed/environmental/"
    "global_soil_moisture.json"
)

with open(GRID) as f:
    cells = json.load(f)

with h5py.File(INPUT, "r") as file:
    group = file["Soil_Moisture_Retrieval_Data_PM"]

    soil = group["soil_moisture_dca_pm"][:]
    lat = group["latitude_pm"][:]
    lon = group["longitude_pm"][:]

    valid = np.isfinite(soil) & np.isfinite(lat) & np.isfinite(lon)

    soil = soil[valid]
    lat = lat[valid]
    lon = lon[valid]

    results = []

    for i, cell in enumerate(cells, 1):
        distance = (
            (lat - cell["latitude"]) ** 2
            + (lon - cell["longitude"]) ** 2
        )

        index = np.argmin(distance)

        results.append({
            "cell_id": cell["id"],
            "latitude": cell["latitude"],
            "longitude": cell["longitude"],
            "soil_moisture": float(soil[index]),
        })

        if i % 5000 == 0:
            print(f"Processed {i}/{len(cells)} cells")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w") as f:
    json.dump(results, f)

print(f"\nGlobal cells processed: {len(results)}")
print(f"Saved: {OUTPUT}")