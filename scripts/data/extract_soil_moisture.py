import json
import h5py
import numpy as np
from pathlib import Path

input_file = "data/raw/soil_moisture/SMAP_L3_SM_P_20250930_R19240_001.h5"
output_file = Path("data/processed/environmental/soil_moisture.json")

locations = [
    {"id": "region_001", "latitude": 13.1972, "longitude": 75.4156},
    {"id": "region_002", "latitude": 23.0225, "longitude": 72.5714},
    {"id": "region_003", "latitude": -1.2921, "longitude": 36.8219},
    {"id": "region_004", "latitude": 51.5074, "longitude": -0.1278},
    {"id": "region_005", "latitude": -23.5505, "longitude": -46.6333},
]

with h5py.File(input_file, "r") as file:
    group = file["Soil_Moisture_Retrieval_Data_PM"]

    soil_moisture = group["soil_moisture_dca_pm"][:]
    latitude = group["latitude_pm"][:]
    longitude = group["longitude_pm"][:]

    results = []

    for location in locations:
        distance = (
            (latitude - location["latitude"]) ** 2
            + (longitude - location["longitude"]) ** 2
        )

        index = np.unravel_index(np.nanargmin(distance), distance.shape)

        value = soil_moisture[index]

        results.append({
            **location,
            "soil_moisture": float(value),
        })

output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w") as file:
    json.dump(results, file, indent=2)

print("Soil moisture extracted:")

for result in results:
    print(result)

print(f"\nSaved: {output_file}")