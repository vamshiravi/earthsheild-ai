import json
from pathlib import Path

GRID = Path("data/metadata/global_cells.json")
SCENES = Path("data/metadata/global_sentinel2_mapping.json")
OUTPUT = Path("data/metadata/global_cell_sentinel2.json")

with open(GRID) as f:
    cells = json.load(f)

with open(SCENES) as f:
    scenes = json.load(f)

matches = []

for cell in cells:
    lat = cell["latitude"]
    lon = cell["longitude"]

    best = None

    for scene in scenes:
        min_lon, min_lat, max_lon, max_lat = scene["bbox"]

        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            if best is None or scene["cloud_cover"] < best["cloud_cover"]:
                best = scene

    if best:
        matches.append({
            "cell_id": cell["id"],
            "latitude": lat,
            "longitude": lon,
            "sentinel2_id": best["sentinel2_id"],
            "cloud_cover": best["cloud_cover"],
            "datetime": best["datetime"]
        })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w") as f:
    json.dump(matches, f, indent=2)

print(f"Global cells: {len(cells)}")
print(f"Sentinel-2 scenes: {len(scenes)}")
print(f"Cells matched: {len(matches)}")
print(f"Saved: {OUTPUT}")