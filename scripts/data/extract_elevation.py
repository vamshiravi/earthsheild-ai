import json
import requests
from pathlib import Path

locations = [
    {"id": "region_001", "latitude": 13.1972, "longitude": 75.4156},
    {"id": "region_002", "latitude": 23.0225, "longitude": 72.5714},
    {"id": "region_003", "latitude": -1.2921, "longitude": 36.8219},
    {"id": "region_004", "latitude": 51.5074, "longitude": -0.1278},
    {"id": "region_005", "latitude": -23.5505, "longitude": -46.6333},
]

url = "https://api.open-elevation.com/api/v1/lookup"

response = requests.post(
    url,
    json={"locations": [
        {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
        }
        for location in locations
    ]},
)

response.raise_for_status()

elevations = response.json()["results"]

results = []

for location, elevation in zip(locations, elevations):
    results.append({
        **location,
        "elevation": elevation["elevation"],
    })

output_file = Path("data/processed/environmental/elevation.json")
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w") as file:
    json.dump(results, file, indent=2)

print("Elevation extracted:")

for result in results:
    print(result)

print(f"\nSaved: {output_file}")
