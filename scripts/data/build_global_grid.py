import json
from pathlib import Path

cells = []

for lat in range(-89, 90):
    for lon in range(-180, 180):
        cells.append({
            "id": f"cell_{lat + 90:03d}_{lon + 180:03d}",
            "latitude": lat + 0.5,
            "longitude": lon + 0.5,
        })

output = Path("data/metadata/global_cells.json")
output.parent.mkdir(parents=True, exist_ok=True)

with open(output, "w") as file:
    json.dump(cells, file)

print(f"Global cells: {len(cells)}")
print(f"Saved: {output}")
