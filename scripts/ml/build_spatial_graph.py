import json
from pathlib import Path

import numpy as np

GRID = "data/metadata/global_cells.json"
OUTPUT = "data/processed/features/spatial_edges.npy"

with open(GRID) as f:
    cells = json.load(f)

positions = {
    (cell["latitude"], cell["longitude"]): i
    for i, cell in enumerate(cells)
}

edges = []

for i, cell in enumerate(cells):
    lat = cell["latitude"]
    lon = cell["longitude"]

    neighbors = [
        (lat + 1, lon),
        (lat - 1, lon),
        (lat, lon + 1),
        (lat, lon - 1),
    ]

    for neighbor in neighbors:
        if neighbor in positions:
            edges.append([i, positions[neighbor]])

edges = np.array(edges, dtype=np.int64)

Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
np.save(OUTPUT, edges)

print("GRAPH COMPLETE")
print(f"Nodes: {len(cells)}")
print(f"Edges: {len(edges)}")
print(f"Average neighbors: {len(edges) / len(cells):.2f}")
print(f"Saved: {OUTPUT}")