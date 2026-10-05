import json
import xarray as xr
from pathlib import Path

input_file = "data/raw/rainfall/3B-DAY.MS.MRG.3IMERG.20250930-S000000-E235959.V07B.nc4"
output_file = Path("data/processed/environmental/rainfall.json")

locations = [
    {"id": "region_001", "latitude": 13.1972, "longitude": 75.4156},
    {"id": "region_002", "latitude": 23.0225, "longitude": 72.5714},
    {"id": "region_003", "latitude": -1.2921, "longitude": 36.8219},
    {"id": "region_004", "latitude": 51.5074, "longitude": -0.1278},
    {"id": "region_005", "latitude": -23.5505, "longitude": -46.6333},
]

ds = xr.open_dataset(input_file)

results = []

for location in locations:
    longitude = location["longitude"] % 360

    value = ds["precipitation"].sel(
        lat=location["latitude"],
        lon=longitude,
        method="nearest",
    ).item()

    results.append({
        **location,
        "rainfall": float(value),
    })

output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print("Rainfall extracted:")
for result in results:
    print(result)

print(f"\nSaved: {output_file}")