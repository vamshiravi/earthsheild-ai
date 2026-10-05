import csv
import math
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(title="EarthShield AI")


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATA
# ============================================================

OUTPUT_FILE = Path(
    "data/processed/features/earthshield_outputs.csv"
)

records = []

if OUTPUT_FILE.exists():

    with open(
        OUTPUT_FILE,
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            records.append(row)

print(
    f"Loaded EarthShield records: {len(records)}"
)


# ============================================================
# MODELS
# ============================================================

class Region(BaseModel):
    latitude: float
    longitude: float


class RegionResponse(BaseModel):
    latitude: float
    longitude: float

    satellite_status: str
    environmental_status: str
    anomaly_status: str
    risk_status: str

    flood_risk: float
    risk_level: str

    anomaly_score: float
    anomaly: int

    shift_score: float
    shift_ratio: float
    shift_status: str

    confidence_percent: float
    earthshield_status: str

    date: str


# ============================================================
# HELPERS
# ============================================================

def find_nearest_record(
    latitude: float,
    longitude: float
):

    if not records:
        return None

    best = None
    best_distance = float("inf")

    for row in records:

        lat = float(row["latitude"])
        lon = float(row["longitude"])

        distance = (
            (lat - latitude) ** 2
            +
            (lon - longitude) ** 2
        )

        if distance < best_distance:

            best_distance = distance
            best = row

    return best


# ============================================================
# HEALTH
# ============================================================
@app.get("/")
def root():
    return {
        "name": "EarthShield AI",
        "status": "operational"
    }
    
@app.get("/api/health")
def health():

    return {
        "status": "operational"
    }


# ============================================================
# REGION
# ============================================================

@app.post(
    "/api/region",
    response_model=RegionResponse
)
def select_region(region: Region):

    row = find_nearest_record(
        region.latitude,
        region.longitude
    )

    if row is None:

        return RegionResponse(
            latitude=region.latitude,
            longitude=region.longitude,
            satellite_status="unavailable",
            environmental_status="unavailable",
            anomaly_status="unavailable",
            risk_status="unavailable",
            flood_risk=0.0,
            risk_level="UNKNOWN",
            anomaly_score=0.0,
            anomaly=0,
            shift_score=0.0,
            shift_ratio=0.0,
            shift_status="UNKNOWN",
            confidence_percent=0.0,
            earthshield_status="UNAVAILABLE",
            date=""
        )

    return RegionResponse(

        latitude=float(
            row["latitude"]
        ),

        longitude=float(
            row["longitude"]
        ),

        satellite_status="available",

        environmental_status="available",

        anomaly_status=(
            "anomalous"
            if row["anomaly"] == "1"
            else "normal"
        ),

        risk_status=row[
            "risk_level"
        ],

        flood_risk=float(
            row["predicted_flood_risk"]
        ),

        risk_level=row[
            "risk_level"
        ],

        anomaly_score=float(
            row["anomaly_score"]
        ),

        anomaly=int(
            row["anomaly"]
        ),

        shift_score=float(
            row["shift_score"]
        ),

        shift_ratio=float(
            row["shift_ratio"]
        ),

        shift_status=row[
            "shift_status"
        ],

        confidence_percent=float(
            row["confidence_percent"]
        ),

        earthshield_status=row[
            "earthshield_status"
        ],

        date=row["date"]
    )