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

OUTPUT = (
    "data/processed/features/"
    "causal_gnn_embeddings.npy"
)

DAYS = 30
BATCH_SIZE = 2048
HOPS = 2
HIDDEN = 24

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA)

cells = len(df) // DAYS

print("Cells:", cells)
print("Days:", DAYS)


# ============================================================
# LOAD TRAINING CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

stats = checkpoint["normalization_stats"]

print(
    "Loaded SSL checkpoint | "
    "Best loss:",
    checkpoint["best_loss"]
)


# ============================================================
# NORMALIZE EXACTLY AS TRAINING
# ============================================================

for col in [
    "rainfall",
    "soil_moisture",
    "elevation"
]:

    mean = stats[col]["mean"]
    std = stats[col]["std"]

    df[col] = (
        (df[col] - mean) /
        (std + 1e-8)
    )

    df[col] = df[col].fillna(0)


# ============================================================
# BUILD INPUT
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
).reshape(
    DAYS,
    cells,
    4
)


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
# GNN
# ============================================================

class SpatialEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.gnn1 = GCNConv(
            7,
            HIDDEN
        )

        self.gnn2 = GCNConv(
            HIDDEN,
            HIDDEN
        )

    def forward(self, x, edge_index):

        z = self.gnn1(
            x,
            edge_index
        )

        z = F.relu(z)

        z = self.gnn2(
            z,
            edge_index
        )

        return z


# ============================================================
# LOAD TRAINED GNN WEIGHTS
# ============================================================

encoder = SpatialEncoder().to(DEVICE)

state = checkpoint["model_state_dict"]

encoder.gnn1.load_state_dict({
    "bias": state["gnn1.bias"],
    "lin.weight": state["gnn1.lin.weight"]
})

encoder.gnn2.load_state_dict({
    "bias": state["gnn2.bias"],
    "lin.weight": state["gnn2.lin.weight"]
})

encoder.eval()


# ============================================================
# OUTPUT
# ============================================================

Path(OUTPUT).parent.mkdir(
    parents=True,
    exist_ok=True
)

embeddings = np.lib.format.open_memmap(
    OUTPUT,
    mode="w+",
    dtype="float32",
    shape=(cells, DAYS, HIDDEN)
)


# ============================================================
# EXTRACT
# ============================================================

start_time = time.perf_counter()

for batch_number, start in enumerate(
    range(0, cells, BATCH_SIZE),
    start=1
):

    end = min(
        start + BATCH_SIZE,
        cells
    )

    target_nodes = torch.arange(
        start,
        end
    )

    # --------------------------------------------------------
    # Spatial neighborhood
    # --------------------------------------------------------

    subset, local_edges, mapping, _ = k_hop_subgraph(
        target_nodes,
        num_hops=HOPS,
        edge_index=global_edges,
        relabel_nodes=True,
        num_nodes=cells
    )

    num_nodes = len(subset)

    batch_X = X[
        :,
        subset,
        :
    ].clone()

    # --------------------------------------------------------
    # No SSL masking during extraction
    # --------------------------------------------------------

    mask = torch.zeros(
        DAYS,
        num_nodes,
        3
    )

    model_input = torch.cat(
        [
            batch_X,
            mask
        ],
        dim=2
    )

    # --------------------------------------------------------
    # Repeat graph across time
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Encode
    # --------------------------------------------------------

    with torch.inference_mode():

        z = encoder(
            model_input.reshape(
                DAYS * num_nodes,
                7
            ).to(DEVICE),
            edge_index.to(DEVICE)
        )

        z = z.reshape(
            DAYS,
            num_nodes,
            HIDDEN
        )

        # Keep target cells only
        z = z[
            :,
            mapping,
            :
        ]

        # [days, target cells, 24]
        # →
        # [target cells, days, 24]

        z = z.permute(
            1,
            0,
            2
        )

        embeddings[
            start:end
        ] = z.cpu().numpy()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Batch {batch_number} | "
        f"Cells {start}:{end} | "
        f"Elapsed {elapsed:.1f}s"
    )


# ============================================================
# FINISH
# ============================================================

del embeddings

print()
print("CAUSAL GNN EXTRACTION COMPLETE")
print("Output:", OUTPUT)
print(
    "Shape:",
    (cells, DAYS, HIDDEN)
)