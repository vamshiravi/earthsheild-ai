import pandas as pd
from sklearn.ensemble import IsolationForest

INPUT = "data/processed/features/global_features.csv"

df = pd.read_csv(INPUT)

features = [
    "rainfall",
    "soil_moisture",
    "elevation",
]

X = df[features].copy()

# Use median values only for this baseline.
X = X.fillna(X.median())

model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42
)

model.fit(X)

df["anomaly_score"] = -model.score_samples(X)
df["anomaly"] = model.predict(X)

print("BASELINE COMPLETE")
print(f"Cells analyzed: {len(df)}")
print(f"Anomalies detected: {(df['anomaly'] == -1).sum()}")
print(
    f"Anomaly percentage: "
    f"{(df['anomaly'] == -1).mean() * 100:.2f}%"
)

output = "data/processed/features/baseline_anomaly.csv"
df.to_csv(output, index=False)

print(f"Saved: {output}")