from pathlib import Path
from pystac_client import Client

catalog = Client.open("https://stac.dataspace.copernicus.eu/v1")

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[77.0, 12.0, 78.0, 13.0],
    datetime="2025-09-01/2025-09-30",
    query={"eo:cloud_cover": {"lte": 20}},
    max_items=1,
)

item = next(search.items(), None)

if item is None:
    print("No Sentinel-2 scene found.")
    raise SystemExit

print("Scene:", item.id)
print("Date:", item.datetime)
print("Cloud cover:", item.properties.get("eo:cloud_cover"))

output_dir = Path("data/raw/sentinel2")
output_dir.mkdir(parents=True, exist_ok=True)

bands = ["B02_10m", "B03_10m", "B04_10m", "B08_10m"]

for band in bands:
    asset = item.assets.get(band)

    if asset:
        print(f"\n{band}")
        print(asset.href)
    else:
        print(f"\n{band}: NOT FOUND")