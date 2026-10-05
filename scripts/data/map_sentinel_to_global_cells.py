import json
import requests
from pathlib import Path

GRID = Path("data/metadata/global_cells.json")
OUTPUT = Path("data/metadata/global_sentinel2_mapping.json")

SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"

with open(GRID) as f:
    cells = json.load(f)

results = []

# Process Earth in 10° × 10° spatial chunks
for lat in range(-90, 90, 10):
    for lon in range(-180, 180, 10):

        bbox = [
            lon,
            lat,
            min(lon + 10, 180),
            min(lat + 10, 90)
        ]

        payload = {
            "collections": ["sentinel-2-l2a"],
            "bbox": bbox,
            "datetime": "2025-09-30T00:00:00Z/2025-10-01T00:00:00Z",
            "query": {
                "eo:cloud_cover": {
                    "lt": 20
                }
            },
            "limit": 100,
            "fields": {
                "exclude": ["geometry"]
            }
        }

        try:
            response = requests.post(
                SEARCH_URL,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            features = response.json()["features"]

            for item in features:
                properties = item["properties"]

                results.append({
                    "sentinel2_id": item["id"],
                    "cloud_cover": properties.get("eo:cloud_cover"),
                    "datetime": properties.get("datetime"),
                    "bbox": item.get("bbox")
                })

            print(
                f"Chunk lat={lat}, lon={lon} | "
                f"scenes={len(features)} | "
                f"total={len(results)}"
            )

        except requests.RequestException as e:
            print(f"Chunk lat={lat}, lon={lon} failed: {e}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w") as f:
    json.dump(results, f)

print("\nSentinel-2 scenes collected:", len(results))
print("Saved:", OUTPUT)