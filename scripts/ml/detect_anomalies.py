import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


# ============================================================
# CONFIG
# ============================================================

EMBEDDINGS = (
    "data/processed/features/"
    "ssl_temporal_embeddings.npy"
)

DATA = (
    "data/processed/features/"
    "temporal_features.csv"
)

OUTPUT = (
    "data/processed/features/"
    "ssl_anomaly_scores.csv"
)

TRAIN_SAMPLES = 100000
TREES = 100


# ============================================================
# LOAD
# ============================================================

embeddings = np.load(
    EMBEDDINGS,
    mmap_mode="r"
)

df = pd.read_csv(DATA)

cells, days, dimensions = embeddings.shape

print("Embedding shape:", embeddings.shape)
print("Cells:", cells)
print("Days:", days)
print("Dimensions:", dimensions)


# ============================================================
# FLATTEN
# ============================================================

X = embeddings.reshape(
    cells * days,
    dimensions
)


# ============================================================
# TRAINING SAMPLE
# ============================================================

rng = np.random.default_rng(42)

sample_size = min(
    TRAIN_SAMPLES,
    len(X)
)

sample_indices = rng.choice(
    len(X),
    size=sample_size,
    replace=False
)

X_train = X[sample_indices]

print(
    "Isolation Forest training samples:",
    len(X_train)
)


# ============================================================
# MODEL
# ============================================================

detector = IsolationForest(
    n_estimators=TREES,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

print("Training Isolation Forest...")

detector.fit(X_train)

print("Training complete")


# ============================================================
# SCORE ALL EMBEDDINGS
# ============================================================

print("Calculating anomaly scores...")

scores = detector.decision_function(X)

# Larger = more normal
# Convert so larger = more anomalous

anomaly_score = -scores


# ============================================================
# PREDICTION
# ============================================================

prediction = detector.predict(X)

anomaly = (
    prediction == -1
).astype(int)


# ============================================================
# CREATE OUTPUT
# ============================================================

result = df[
    [
        "cell_id",
        "latitude",
        "longitude",
        "date"
    ]
].copy()

result["anomaly_score"] = anomaly_score
result["anomaly"] = anomaly


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT,
    index=False
)

print()
print("ANOMALY DETECTION COMPLETE")
print("Output:", OUTPUT)
print("Rows:", len(result))
print(
    "Anomalies:",
    result["anomaly"].sum()
)
print(
    "Anomaly rate:",
    f"{result['anomaly'].mean() * 100:.2f}%"
)