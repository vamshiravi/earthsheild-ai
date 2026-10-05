from pathlib import Path

import numpy as np
import rasterio


data_dir = Path("data/raw/sentinel2")

red_file = next(data_dir.glob("*_B04_10m.jp2"))
nir_file = next(data_dir.glob("*_B08_10m.jp2"))

with rasterio.open(red_file) as red_src:
    red = red_src.read(1).astype("float32")
    profile = red_src.profile

with rasterio.open(nir_file) as nir_src:
    nir = nir_src.read(1).astype("float32")

denominator = nir + red

ndvi = np.divide(
    nir - red,
    denominator,
    out=np.zeros_like(nir),
    where=denominator != 0,
)

output_dir = Path("data/processed/features")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "ndvi.tif"
profile.update(
    driver="GTiff",
    dtype="float32",
    count=1,
)

with rasterio.open(output_file, "w", **profile) as dst:
    dst.write(ndvi, 1)

print(f"NDVI saved to: {output_file}")
print(f"Minimum: {ndvi.min():.3f}")
print(f"Maximum: {ndvi.max():.3f}")
print(f"Mean: {ndvi.mean():.3f}")