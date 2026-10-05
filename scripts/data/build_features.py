import json
import pandas as pd

def load(path):
    with open(path) as f:
        return pd.DataFrame(json.load(f))

rainfall = load("data/processed/environmental/rainfall.json")
soil = load("data/processed/environmental/soil_moisture.json")
elevation = load("data/processed/environmental/elevation.json")

features = rainfall.merge(
    soil[["id", "soil_moisture"]],
    on="id"
).merge(
    elevation[["id", "elevation"]],
    on="id"
)

features = features.rename(columns={"id": "region_id"})

features.to_csv(
    "data/processed/features/multimodal_features.csv",
    index=False
)

print(features)
print("\nSaved: data/processed/features/multimodal_features.csv")