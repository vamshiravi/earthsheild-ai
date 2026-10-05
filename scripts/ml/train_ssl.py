import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph


# ============================================================
# CONFIG
# ============================================================

DATA = "data/processed/features/temporal_features.csv"
EDGES = "data/processed/features/spatial_edges.npy"
MODEL_PATH = "models/earthshield_ssl.pt"

DAYS = 30
BATCH_SIZE = 1024
EPOCHS = 5

MASK_RATE = 0.20
LR = 0.001
HOPS = 2

GNN_HIDDEN = 24
TRANSFORMER_DIM = 48
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 1

torch.manual_seed(42)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {DEVICE}")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA)

cells = len(df) // DAYS

print(f"Cells: {cells}")
print(f"Days: {DAYS}")


# ============================================================
# ORIGINAL OBSERVATION MASKS
# ============================================================

rain_observed = torch.tensor(
    df["rainfall"].notna().to_numpy(),
    dtype=torch.bool
).reshape(DAYS, cells)

soil_observed = torch.tensor(
    df["soil_moisture"].notna().to_numpy(),
    dtype=torch.bool
).reshape(DAYS, cells)


# ============================================================
# NORMALIZATION
# ============================================================

stats = {}

for col in ["rainfall", "soil_moisture", "elevation"]:

    mean = float(df[col].mean())
    std = float(df[col].std())

    stats[col] = {
        "mean": mean,
        "std": std
    }

    df[col] = (
        (df[col] - mean) /
        (std + 1e-8)
    )

    df[col] = df[col].fillna(0)


# ============================================================
# DATA TENSOR
# [days, cells, features]
# ============================================================

X = torch.tensor(
    df[
        [
            "rainfall",
            "soil_moisture",
            "soil_moisture_available",
            "elevation"
        ]
    ].to_numpy(),
    dtype=torch.float32
).reshape(DAYS, cells, 4)

print("Data tensor:", tuple(X.shape))


# ============================================================
# GLOBAL GRAPH
# ============================================================

global_edges = torch.tensor(
    np.load(EDGES),
    dtype=torch.long
).T.contiguous()

print(
    "Global edges:",
    global_edges.shape[1]
)


# ============================================================
# PRECOMPUTE BATCH SUBGRAPHS
# ============================================================

batches = []

print("Building spatial subgraphs...")

graph_start = time.perf_counter()

for start in range(0, cells, BATCH_SIZE):

    end = min(
        start + BATCH_SIZE,
        cells
    )

    target_nodes = torch.arange(
        start,
        end
    )

    subset, local_edges, mapping, _ = k_hop_subgraph(
        target_nodes,
        num_hops=HOPS,
        edge_index=global_edges,
        relabel_nodes=True,
        num_nodes=cells
    )

    batches.append(
        {
            "subset": subset,
            "mapping": mapping,
            "edge_index": local_edges
        }
    )

    print(
        f"Batch {len(batches):02d} | "
        f"targets: {len(target_nodes)} | "
        f"graph nodes: {len(subset)} | "
        f"edges: {local_edges.shape[1]}"
    )

print(
    f"Subgraphs ready in "
    f"{time.perf_counter() - graph_start:.2f}s"
)

print(
    "Total batches:",
    len(batches)
)


# ============================================================
# MODEL
# ============================================================

class SpatialTemporalModel(nn.Module):

    def __init__(self):

        super().__init__()

        # 4 original features
        # + 3 SSL mask channels
        self.gnn1 = GCNConv(
            7,
            GNN_HIDDEN
        )

        self.gnn2 = GCNConv(
            GNN_HIDDEN,
            GNN_HIDDEN
        )

        self.projection = nn.Linear(
            GNN_HIDDEN,
            TRANSFORMER_DIM
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TRANSFORMER_DIM,
            nhead=TRANSFORMER_HEADS,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=TRANSFORMER_LAYERS
        )

        self.reconstruction = nn.Linear(
            TRANSFORMER_DIM,
            3
        )

    def forward(
        self,
        x,
        edge_index,
        num_nodes
    ):
        """
        x:
            [days, nodes, 7]
        """

        days = x.shape[0]

        # ----------------------------------------------------
        # Flatten time and nodes
        # ----------------------------------------------------

        x = x.reshape(
            days * num_nodes,
            7
        )

        # ----------------------------------------------------
        # Spatial GNN
        # Single vectorized pass
        # ----------------------------------------------------

        z = self.gnn1(
            x,
            edge_index
        )

        z = F.relu(z)

        z = self.gnn2(
            z,
            edge_index
        )

        # ----------------------------------------------------
        # Restore temporal structure
        # ----------------------------------------------------

        z = z.reshape(
            days,
            num_nodes,
            GNN_HIDDEN
        )

        # [days, nodes, features]
        # → [nodes, days, features]

        z = z.permute(
            1,
            0,
            2
        )

        # ----------------------------------------------------
        # Projection
        # ----------------------------------------------------

        z = self.projection(z)

        # ----------------------------------------------------
        # Temporal Transformer
        # ----------------------------------------------------

        z = self.transformer(z)

        # ----------------------------------------------------
        # Reconstruction
        # ----------------------------------------------------

        return self.reconstruction(z)


# ============================================================
# CREATE MODEL
# ============================================================

model = SpatialTemporalModel().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)


# ============================================================
# TRAIN
# ============================================================

best_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0.0
    batch_count = 0

    epoch_start = time.perf_counter()

    # Shuffle batch order
    order = torch.randperm(
        len(batches)
    )

    for step, batch_id in enumerate(
        order.tolist(),
        start=1
    ):

        batch = batches[batch_id]

        subset = batch["subset"]
        mapping = batch["mapping"]
        local_edges = batch["edge_index"]

        num_nodes = len(subset)

        # ----------------------------------------------------
        # Batch data
        # ----------------------------------------------------

        batch_X = X[:, subset, :].clone()

        rain_obs = rain_observed[
            :,
            subset
        ]

        soil_obs = soil_observed[
            :,
            subset
        ]

        # ----------------------------------------------------
        # Random mask
        # ----------------------------------------------------

        mask = (
            torch.rand(
                DAYS,
                num_nodes,
                3
            ) < MASK_RATE
        )

        mask[:, :, 0] &= rain_obs
        mask[:, :, 1] &= soil_obs

        if not mask.any():
            continue

        # ----------------------------------------------------
        # Mask environmental values
        # ----------------------------------------------------

        masked_X = batch_X.clone()

        continuous = masked_X[
            :,
            :,
            [0, 1, 3]
        ]

        continuous[mask] = 0.0

        masked_X[
            :,
            :,
            [0, 1, 3]
        ] = continuous

        # ----------------------------------------------------
        # Add mask channels
        # ----------------------------------------------------

        model_input = torch.cat(
            [
                masked_X,
                mask.float()
            ],
            dim=2
        )

        # ----------------------------------------------------
        # Build temporal-independent graph
        # ----------------------------------------------------

        edge_count = local_edges.shape[1]

        offsets = (
            torch.arange(DAYS)
            * num_nodes
        ).repeat_interleave(
            edge_count
        )

        edge_index = (
            local_edges.repeat(
                1,
                DAYS
            )
            + offsets.unsqueeze(0)
        )

        # ----------------------------------------------------
        # Move tensors
        # ----------------------------------------------------

        model_input = model_input.to(
            DEVICE
        )

        edge_index = edge_index.to(
            DEVICE
        )

        # Only target nodes participate in loss
        target = batch_X[
            :,
            mapping,
            :
        ][:, :, [0, 1, 3]]

        target = target.permute(
            1,
            0,
            2
        ).to(DEVICE)

        mask_target = mask[
            :,
            mapping,
            :
        ].permute(
            1,
            0,
            2
        ).to(DEVICE)

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            model_input,
            edge_index,
            num_nodes
        )

        output = output[
            mapping.to(DEVICE)
        ]

        # ----------------------------------------------------
        # Masked reconstruction loss
        # ----------------------------------------------------

        error = (
            output - target
        ) ** 2

        loss = error[
            mask_target
        ].mean()

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        elapsed = (
            time.perf_counter()
            - epoch_start
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Batch {step}/{len(batches)} | "
            f"Loss {loss.item():.5f} | "
            f"Elapsed {elapsed:.1f}s"
        )

    # ========================================================
    # EPOCH SUMMARY
    # ========================================================

    average_loss = (
        epoch_loss /
        max(batch_count, 1)
    )

    epoch_time = (
        time.perf_counter()
        - epoch_start
    )

    print()
    print(
        f"Epoch {epoch + 1}/{EPOCHS} COMPLETE | "
        f"Average Loss: {average_loss:.6f} | "
        f"Time: {epoch_time:.1f}s"
    )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if average_loss < best_loss:

        best_loss = average_loss

        Path(MODEL_PATH).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "normalization_stats":
                    stats,

                "cells":
                    cells,

                "days":
                    DAYS,

                "best_loss":
                    best_loss
            },
            MODEL_PATH
        )

        print(
            "Best model saved:",
            MODEL_PATH
        )

    print()


# ============================================================
# COMPLETE
# ============================================================

print("=" * 60)
print("SSL TRAINING COMPLETE")
print(f"Best loss: {best_loss:.6f}")
print(f"Model saved: {MODEL_PATH}")
print("=" * 60)