import numpy as np
import pandas as pd


EMBEDDINGS = (
    "data/processed/features/"
    "causal_gnn_embeddings.npy"
)

OUTPUT = (
    "data/processed/features/"
    "distribution_shift.csv"
)

DAYS = 30

REFERENCE_START = 0
REFERENCE_END = 16

CURRENT_START = 16
CURRENT_END = 23

SAMPLE_SIZE = 10000


# ============================================================
# LOAD
# ============================================================

embeddings = np.load(
    EMBEDDINGS,
    mmap_mode="r"
)

cells, days, dimensions = embeddings.shape

print("Embeddings:", embeddings.shape)


# ============================================================
# MMD FUNCTION
# ============================================================

def rbf_mmd(X, Y):

    rng = np.random.default_rng(42)

    n = min(
        SAMPLE_SIZE,
        len(X)
    )

    m = min(
        SAMPLE_SIZE,
        len(Y)
    )

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
            m,
            replace=False
        )
    ]

    # --------------------------------------------------------
    # Median heuristic for kernel bandwidth
    # --------------------------------------------------------

    combined = np.vstack(
        [X[:2000], Y[:2000]]
    )

    distances = np.sum(
        (
            combined[:, None, :]
            -
            combined[None, :, :]
        ) ** 2,
        axis=2
    )

    median_distance = np.median(
        distances
    )

    gamma = 1.0 / (
        2.0 * median_distance
        + 1e-8
    )

    # --------------------------------------------------------
    # Kernel matrices
    # --------------------------------------------------------

    XX = np.exp(
        -gamma *
        np.sum(
            (X[:, None, :] - X[None, :, :]) ** 2,
            axis=2
        )
    )

    YY = np.exp(
        -gamma *
        np.sum(
            (Y[:, None, :] - Y[None, :, :]) ** 2,
            axis=2
        )
    )

    XY = np.exp(
        -gamma *
        np.sum(
            (X[:, None, :] - Y[None, :, :]) ** 2,
            axis=2
        )
    )

    mmd = (
        XX.mean()
        +
        YY.mean()
        -
        2 * XY.mean()
    )

    return float(mmd)


# ============================================================
# REFERENCE DISTRIBUTION
# ============================================================

reference = embeddings[
    :,
    REFERENCE_START:REFERENCE_END,
    :
].reshape(
    -1,
    dimensions
)

print(
    "Reference samples:",
    len(reference)
)


# ============================================================
# OVERALL CURRENT SHIFT
# ============================================================

current = embeddings[
    :,
    CURRENT_START:CURRENT_END,
    :
].reshape(
    -1,
    dimensions
)

overall_shift = rbf_mmd(
    reference,
    current
)

print(
    f"Overall MMD shift: {overall_shift:.6f}"
)


# ============================================================
# DAILY SHIFT
# ============================================================

rows = []

for day in range(
    CURRENT_START,
    CURRENT_END
):

    current_day = embeddings[
        :,
        day,
        :
    ]

    score = rbf_mmd(
        reference,
        current_day
    )

    rows.append(
        {
            "day_index": day,
            "shift_score": score
        }
    )

    print(
        f"Day {day + 1:02d} | "
        f"Shift score: {score:.6f}"
    )


# ============================================================
# NORMALIZE SHIFT SCORE
# ============================================================

result = pd.DataFrame(
    rows
)

minimum = result[
    "shift_score"
].min()

maximum = result[
    "shift_score"
].max()

result["normalized_shift"] = (
    (
        result["shift_score"]
        -
        minimum
    )
    /
    (
        maximum
        -
        minimum
        +
        1e-8
    )
)


# ============================================================
# INTERPRETATION
# ============================================================

def classify(score):

    if score < 0.33:
        return "LOW"

    if score < 0.66:
        return "MODERATE"

    return "HIGH"


result["shift_status"] = (
    result["normalized_shift"]
    .apply(classify)
)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT,
    index=False
)

print()
print("DISTRIBUTION SHIFT COMPLETE")
print(
    "Overall MMD:",
    f"{overall_shift:.6f}"
)
print(
    "Saved:",
    OUTPUT
)