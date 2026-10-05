import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# CONFIG
# ============================================================

EMBEDDINGS = (
    "data/processed/features/"
    "causal_gnn_embeddings.npy"
)

TARGET = (
    "data/processed/features/"
    "flood_forecast_dataset.csv"
)

GRID = "data/metadata/global_cells.json"

MODEL_PATH = "models/earthshield_forecast.pt"

PREDICTION_OUTPUT = (
    "data/processed/features/"
    "forecast_predictions.csv"
)

HISTORY = 7
BATCH_SIZE = 8192
EPOCHS = 5
LR = 0.001

INPUT_DIM = 24
HIDDEN_DIM = 32
HEADS = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(42)

print("Device:", DEVICE)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

embeddings = np.load(
    EMBEDDINGS,
    mmap_mode="r"
)

cells, days, dimensions = embeddings.shape

print("Embeddings:", embeddings.shape)

assert dimensions == INPUT_DIM


# ============================================================
# LOAD GLOBAL CELL ORDER
# ============================================================

with open(GRID) as f:
    grid = json.load(f)

cell_ids = [
    cell["id"]
    for cell in grid
]

assert len(cell_ids) == cells


# ============================================================
# LOAD TARGET DATA
# ============================================================

target_df = pd.read_csv(
    TARGET,
    usecols=[
        "cell_id",
        "date",
        "flood_risk"
    ]
)

target_df["date"] = pd.to_datetime(
    target_df["date"]
)

target_dates = np.array(
    sorted(
        target_df["date"].unique()
    )
)

num_target_days = len(target_dates)

print(
    "Forecast rows:",
    len(target_df)
)

print(
    "Target dates:",
    num_target_days
)


# ============================================================
# BUILD TARGET MATRIX
# ============================================================

# [cells, target_dates]
targets = np.full(
    (cells, num_target_days),
    np.nan,
    dtype=np.float32
)

cell_to_index = {
    cell_id: i
    for i, cell_id in enumerate(cell_ids)
}

date_to_index = {
    date: i
    for i, date in enumerate(target_dates)
}

cell_indices = target_df[
    "cell_id"
].map(
    cell_to_index
).to_numpy()

date_indices = target_df[
    "date"
].map(
    date_to_index
).to_numpy()

targets[
    cell_indices,
    date_indices
] = target_df[
    "flood_risk"
].to_numpy(
    dtype=np.float32
)


print(
    "Target matrix:",
    targets.shape
)

print(
    "Missing targets:",
    np.isnan(targets).sum()
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

# 23 target dates:
#
# train      → first 16
# validation → next 3
# test       → final 4

train_days = list(
    range(
        HISTORY - 1,
        16
    )
)

val_days = list(
    range(
        16,
        19
    )
)

test_days = list(
    range(
        19,
        num_target_days
    )
)


print()
print(
    "Train target dates:",
    [
        str(target_dates[i])[:10]
        for i in train_days
    ]
)

print(
    "Validation target dates:",
    [
        str(target_dates[i])[:10]
        for i in val_days
    ]
)

print(
    "Test target dates:",
    [
        str(target_dates[i])[:10]
        for i in test_days
    ]
)


# ============================================================
# MODEL
# ============================================================

class ForecastTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        self.input_projection = nn.Linear(
            INPUT_DIM,
            HIDDEN_DIM
        )

        self.position = nn.Parameter(
            torch.zeros(
                1,
                HISTORY,
                HIDDEN_DIM
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM,
            nhead=HEADS,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1
        )

        self.head = nn.Sequential(
            nn.Linear(
                HIDDEN_DIM,
                16
            ),
            nn.ReLU(),
            nn.Linear(
                16,
                1
            ),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = x + self.position

        x = self.transformer(x)

        # Most recent day
        x = x[:, -1, :]

        return self.head(
            x
        ).squeeze(-1)


model = ForecastTransformer().to(
    DEVICE
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

criterion = nn.HuberLoss()


# ============================================================
# EPOCH FUNCTION
# ============================================================

def run_epoch(
    day_indices,
    training=True
):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    start_time = time.perf_counter()

    rng = np.random.default_rng(
        42
    )

    for day_number, t in enumerate(
        day_indices,
        start=1
    ):

        # ----------------------------------------------------
        # Only use cells with a valid target
        # ----------------------------------------------------

        valid_cells = np.flatnonzero(
            np.isfinite(
                targets[:, t]
            )
        )

        if training:
            rng.shuffle(
                valid_cells
            )

        # ----------------------------------------------------
        # Batches
        # ----------------------------------------------------

        for start in range(
            0,
            len(valid_cells),
            BATCH_SIZE
        ):

            batch_cells = valid_cells[
                start:start + BATCH_SIZE
            ]

            # ------------------------------------------------
            # Previous 7 days
            # ------------------------------------------------

            start_day = t - HISTORY + 1

            batch_X = embeddings[
                batch_cells,
                start_day:t + 1,
                :
            ]

            batch_X = np.ascontiguousarray(
                batch_X,
                dtype=np.float32
            )

            xb = torch.from_numpy(
                batch_X
            ).to(DEVICE)

            yb = torch.from_numpy(
                targets[
                    batch_cells,
                    t
                ]
            ).to(DEVICE)

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            with torch.set_grad_enabled(
                training
            ):

                prediction = model(
                    xb
                )

                loss = criterion(
                    prediction,
                    yb
                )

                if training:

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        1.0
                    )

                    optimizer.step()

            batch_size = len(
                batch_cells
            )

            total_loss += (
                loss.item() *
                batch_size
            )

            total_samples += batch_size

    elapsed = (
        time.perf_counter()
        - start_time
    )

    return (
        total_loss / total_samples,
        elapsed
    )


# ============================================================
# TRAIN
# ============================================================

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    train_loss, train_time = run_epoch(
        train_days,
        training=True
    )

    val_loss, val_time = run_epoch(
        val_days,
        training=False
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"Time: {train_time:.1f}s"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        Path(
            MODEL_PATH
        ).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "best_val_loss":
                    best_val_loss,

                "history":
                    HISTORY
            },
            MODEL_PATH
        )

        print(
            "Best forecast model saved:",
            MODEL_PATH
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# TEST
# ============================================================

test_predictions = []
test_actual = []
test_cells = []
test_dates_output = []

with torch.inference_mode():

    for t in test_days:

        valid_cells = np.flatnonzero(
            np.isfinite(
                targets[:, t]
            )
        )

        for start in range(
            0,
            len(valid_cells),
            BATCH_SIZE
        ):

            batch_cells = valid_cells[
                start:start + BATCH_SIZE
            ]

            start_day = t - HISTORY + 1

            batch_X = np.ascontiguousarray(
                embeddings[
                    batch_cells,
                    start_day:t + 1,
                    :
                ],
                dtype=np.float32
            )

            xb = torch.from_numpy(
                batch_X
            ).to(DEVICE)

            prediction = model(
                xb
            ).cpu().numpy()

            actual = targets[
                batch_cells,
                t
            ]

            test_predictions.extend(
                prediction
            )

            test_actual.extend(
                actual
            )

            test_cells.extend(
                batch_cells
            )

            test_dates_output.extend(
                [target_dates[t]] *
                len(batch_cells)
            )


# ============================================================
# METRICS
# ============================================================

predictions = np.asarray(
    test_predictions
)

actual = np.asarray(
    test_actual
)

mae = np.mean(
    np.abs(
        predictions - actual
    )
)

rmse = np.sqrt(
    np.mean(
        (predictions - actual) ** 2
    )
)

correlation = np.corrcoef(
    predictions,
    actual
)[0, 1]


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

result = pd.DataFrame(
    {
        "cell_id": [
            cell_ids[i]
            for i in test_cells
        ],
        "date": test_dates_output,
        "actual_flood_risk": actual,
        "predicted_flood_risk": predictions
    }
)

result.to_csv(
    PREDICTION_OUTPUT,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("FORECAST TRAINING COMPLETE")
print(
    f"Best validation loss: "
    f"{best_val_loss:.6f}"
)
print(
    f"Test samples: "
    f"{len(actual)}"
)
print(
    f"Test MAE: "
    f"{mae:.6f}"
)
print(
    f"Test RMSE: "
    f"{rmse:.6f}"
)
print(
    f"Test correlation: "
    f"{correlation:.4f}"
)
print(
    "Model:",
    MODEL_PATH
)
print(
    "Predictions:",
    PREDICTION_OUTPUT
)
print("=" * 60)