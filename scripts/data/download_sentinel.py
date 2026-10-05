import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

access_key = os.getenv("CDSE_S3_ACCESS_KEY")
secret_key = os.getenv("CDSE_S3_SECRET_KEY")

if not access_key or not secret_key:
    raise RuntimeError("CDSE S3 credentials not found in .env")

s3 = boto3.client(
    "s3",
    endpoint_url="https://eodata.dataspace.copernicus.eu",
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
)

bucket = "eodata"

base = (
    "Sentinel-2/MSI/L2A/2025/12/31/"
    "S2C_MSIL2A_20251231T052231_N0511_R062_T43PGQ_20251231T083411.SAFE/"
    "GRANULE/L2A_T43PGQ_A006896_20251231T053319/"
    "IMG_DATA/R10m/"
)

bands = ["B02", "B03", "B04", "B08"]

for band in bands:
    filename = f"T43PGQ_20251231T052231_{band}_10m.jp2"
    key = base + filename
    output = Path("data/raw/sentinel2") / filename

    if output.exists():
        print(f"{band}: already downloaded")
        continue

    print(f"Downloading {band}...")
    s3.download_file(bucket, key, str(output))
    print(f"Saved: {output}")