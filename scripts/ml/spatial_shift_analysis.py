import numpy as np
import pandas as pd
import json


# ============================================================
# FILES
# ============================================================

EMBEDDINGS = (
    "data/processed/features/"
    "causal_gnn_embeddings.npy"
)

PREDICTIONS = (
    "data/processed/features/"
    "forecast_predictions.csv"
)

GRID = "data/metadata/global_cells.json"

OUTPUT = (
    "data/processed/features/"
    "spatial_shift_analysis.csv"
)

SAMPLE_SIZE = 5000

rng = np.random.default_rng(42)


# ============================================================
# LOAD
# ============================================================

embeddings = np.load(
    EMBEDDINGS,
    mmap_mode="r"
)

pred = pd.read_csv(
    PREDICTIONS
)

with open(GRID) as f:
    grid = pd.DataFrame(json.load(f))


cells, days, dimensions = embeddings.shape

print("Embeddings:", embeddings.shape)
print("Prediction rows:", len(pred))
print("Grid cells:", len(grid))


# ============================================================
# ASSIGN LATITUDE DOMAINS
# ============================================================

def region(lat):

    if lat < -30:
        return "Southern_High"

    if lat < 0:
        return "Southern_Tropical"

    if lat < 30:
        return "Northern_Tropical"

    return "Northern_High"


grid["region"] = grid["latitude"].apply(
    region
)


# ============================================================
# CELL-LEVEL REPRESENTATION
# ============================================================

# Average the 30 daily GNN representations
# for each cell.

cell_embeddings = embeddings.mean(
    axis=1
)


# ============================================================
# MMD
# ============================================================

def linear_mmd(X, Y):

    n = min(
        len(X),
        len(Y),
        SAMPLE_SIZE
    )

    n -= n % 2

    if n < 4:
        return np.nan

    X = X[
        rng.choice(
            len(X),
            n,
            replace=False
        )
    ]

    Y = Y[
        rng.choice(
            len(Y),
            n,
            replace=False
        )
    ]

    X1 = X[::2]
    X2 = X[1::2]

    Y1 = Y[::2]
    Y2 = Y[1::2]

    # Median bandwidth
    combined = np.vstack(
        [X1[:500], Y1[:500]]
    )

    distances = np.sum(
        (
            combined[:, None, :]
            -
            combined[None, :, :]
        ) ** 2,
        axis=2
    )

    gamma = 1.0 / (
        2 * np.median(distances)
        + 1e-8
    )

    kxx = np.exp(
        -gamma *
        np.sum(
            (X1 - X2) ** 2,
            axis=1
        )
    )

    kyy = np.exp(
        -gamma *
        np.sum(
            (Y1 - Y2) ** 2,
            axis=1
        )
    )

    kxy = np.exp(
        -gamma *
        np.sum(
            (X1 - Y2) ** 2,
            axis=1
        )
    )

    return max(
        0.0,
        float(
            np.mean(
                kxx + kyy - 2 * kxy
            )
        )
    )


# ============================================================
# GLOBAL REFERENCE
# ============================================================

reference = cell_embeddings

print(
    "Reference cells:",
    len(reference)
)


# ============================================================
# REGION ANALYSIS
# ============================================================

rows = []

for name in sorted(
    grid["region"].unique()
):

    indices = grid.index[
        grid["region"] == name
    ].to_numpy()

    region_embeddings = cell_embeddings[
        indices
    ]

    # --------------------------------------------------------
    # Representation shift
    # --------------------------------------------------------

    shift_score = linear_mmd(
        reference,
        region_embeddings
    )

    # --------------------------------------------------------
    # Forecast performance
    # --------------------------------------------------------

    region_ids = set(
        grid.iloc[indices]["id"]
    )

    # Prediction file uses cell_id
    region_pred = pred[
        pred["cell_id"].isin(
            region_ids
        )
    ]

    actual = region_pred[
        "actual_flood_risk"
    ].to_numpy()

    predicted = region_pred[
        "predicted_flood_risk"
    ].to_numpy()

    mae = np.mean(
        np.abs(
            predicted - actual
        )
    )

    rmse = np.sqrt(
        np.mean(
            (predicted - actual) ** 2
        )
    )

    correlation = np.corrcoef(
        predicted,
        actual
    )[0, 1]

    rows.append(
        {
            "region": name,
            "cells": len(indices),
            "prediction_samples": len(region_pred),
            "mmd_shift": shift_score,
            "mae": mae,
            "rmse": rmse,
            "correlation": correlation
        }
    )

    print(
        f"{name:20s} | "
        f"Cells: {len(indices):6d} | "
        f"MMD: {shift_score:.6f} | "
        f"MAE: {mae:.6f}"
    )


# ============================================================
# SAVE
# ============================================================

result = pd.DataFrame(rows)

result = result.sort_values(
    "mmd_shift",
    ascending=False
)

result.to_csv(
    OUTPUT,
    index=False
)

print()
print("=" * 60)
print("SPATIAL SHIFT ANALYSIS COMPLETE")
print("=" * 60)

print(
    result.to_string(
        index=False
    )
)

print()
print(
    "Saved:",
    OUTPUT
)