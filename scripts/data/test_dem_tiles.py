import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client(
    "s3",
    config=Config(signature_version=UNSIGNED)
)

tests = [
    (12.5, 77.5),   # Bengaluru
    (0.5, 6.5),     # known working tile
    (-0.5, 6.5),    # southern tile
]

for lat, lon in tests:

    lat_tile = int(lat // 1)
    lon_tile = int(lon // 1)

    lat_prefix = "N" if lat_tile >= 0 else "S"
    lon_prefix = "E" if lon_tile >= 0 else "W"

    tile = (
        f"Copernicus_DSM_COG_30_"
        f"{lat_prefix}{abs(lat_tile):02d}_00_"
        f"{lon_prefix}{abs(lon_tile):03d}_00_DEM"
    )

    key = f"{tile}/{tile}.tif"

    print(f"\nTesting: {lat}, {lon}")
    print(f"Tile: {tile}")

    try:
        result = s3.head_object(
            Bucket="copernicus-dem-90m",
            Key=key
        )

        print("FOUND")
        print(f"Size: {result['ContentLength']} bytes")

    except Exception as error:
        print("NOT FOUND")
        print(type(error).__name__)
        