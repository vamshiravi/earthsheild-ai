import json
import time
from pathlib import Path
import requests

SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"
OUTPUT = Path("data/metadata/sentinel2_global_20250930.json")

boxes = [
    [lon, lat, min(lon + 30, 180), min(lat + 30, 90)]
    for lat in range(-90, 90, 30)
    for lon in range(-180, 180, 30)
]

all_items = []

for i, bbox in enumerate(boxes, 1):

    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": "2025-09-30T00:00:00Z/2025-10-01T00:00:00Z",
        "query": {"eo:cloud_cover": {"lt": 20}},
        "limit": 100,
        "fields": {"exclude": ["geometry"]},
    }

    for attempt in range(3):
        try:
            response = requests.post(
                SEARCH_URL,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            break

        except requests.RequestException:
            if attempt == 2:
                print(f"Chunk {i} failed after 3 attempts")
                data = {"features": []}
            else:
                time.sleep(5)

    items = data["features"]
    all_items.extend(items)

    print(
        f"Chunk {i}/{len(boxes)} | "
        f"scenes: {len(items)} | "
        f"total: {len(all_items)}"
    )

    # Save after EVERY chunk
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w") as file:
        json.dump(all_items, file)

print(f"\nTotal scenes collected: {len(all_items)}")
print(f"Saved: {OUTPUT}")