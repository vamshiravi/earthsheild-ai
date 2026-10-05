import json
from pathlib import Path

locations = json.loads(
    Path("data/metadata/sample_locations.json").read_text()
)

records = []

for location in locations:
    records.append({
        "region_id": location["id"],
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "date": "2025-12-31",
        "rainfall_mm": None,
        "source": "NASA GPM IMERG V07B"
    })

output = Path("data/processed/environmental/rainfall.json")
output.parent.mkdir(parents=True, exist_ok=True)

output.write_text(json.dumps(records, indent=2))

print(f"Created rainfall schema for {len(records)} regions")
print(f"Saved: {output}")