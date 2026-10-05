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

TEMPORAL_OUTPUT = (
    "data/processed/features/ssl_temporal_embeddings.npy"
)

CELL_OUTPUT = (
    "data/processed/features/ssl_cell_embeddings.npy"
)

DAYS = 30
BATCH_SIZE = 1024
HOPS = 2

GNN_HIDDEN = 24
TRANSFORMER_DIM = 48
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 1

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
# NORMALIZE EXACTLY AS DURING TRAINING
# ============================================================

for col in [
    "rainfall",
    "soil_moisture",
    "elevation"
]:

    mean = df[col].mean()
    std = df[col].std()

    df[col] = (
        (df[col] - mean) /
        (std + 1e-8)
    )

    df[col] = df[col].fillna(0)


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


# ============================================================
# MODEL
# ============================================================

class SpatialTemporalModel(nn.Module):

    def __init__(self):

        super().__init__()

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

    def encode(
        self,
        x,
        edge_index,
        num_nodes
    ):

        x = x.reshape(
            DAYS * num_nodes,
            7
        )

        z = self.gnn1(
            x,
            edge_index
        )

        z = F.relu(z)

        z = self.gnn2(
            z,
            edge_index
        )

        z = z.reshape(
            DAYS,
            num_nodes,
            GNN_HIDDEN
        )

        z = z.permute(
            1,
            0,
            2
        )

        z = self.projection(z)

        z = self.transformer(z)

        return z


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model = SpatialTemporalModel().to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Loaded checkpoint with best loss:",
    checkpoint["best_loss"]
)


# ============================================================
# CREATE OUTPUT ARRAYS
# ============================================================

Path(TEMPORAL_OUTPUT).parent.mkdir(
    parents=True,
    exist_ok=True
)

temporal_embeddings = np.lib.format.open_memmap(
    TEMPORAL_OUTPUT,
    mode="w+",
    dtype="float32",
    shape=(cells, DAYS, TRANSFORMER_DIM)
)

cell_embeddings = np.lib.format.open_memmap(
    CELL_OUTPUT,
    mode="w+",
    dtype="float32",
    shape=(cells, TRANSFORMER_DIM)
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
    # Spatial halo
    # --------------------------------------------------------

    subset, local_edges, mapping, _ = k_hop_subgraph(
        target_nodes,
        num_hops=HOPS,
        edge_index=global_edges,
        relabel_nodes=True,
        num_nodes=cells
    )

    num_nodes = len(subset)

    batch_X = X[:, subset, :].clone()

    # No SSL masking during representation extraction
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

        embedding = model.encode(
            model_input.to(DEVICE),
            edge_index.to(DEVICE),
            num_nodes
        )

        # Keep only requested target cells
        embedding = embedding[
            mapping.to(DEVICE)
        ]

        embedding = embedding.cpu().numpy()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    temporal_embeddings[
        start:end
    ] = embedding

    cell_embeddings[
        start:end
    ] = embedding.mean(
        axis=1
    )

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
# FLUSH
# ============================================================

del temporal_embeddings
del cell_embeddings

print()
print("REPRESENTATION EXTRACTION COMPLETE")
print(
    "Temporal:",
    TEMPORAL_OUTPUT
)
print(
    "Cell:",
    CELL_OUTPUT
)

print("Temporal shape:", (cells, DAYS, TRANSFORMER_DIM))
print("Cell shape:", (cells, TRANSFORMER_DIM))
