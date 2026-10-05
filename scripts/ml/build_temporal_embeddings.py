import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SpatialGNN(torch.nn.Module):

    def __init__(self, input_dim=4, hidden_dim=32):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        return self.conv2(x, edge_index)


df = pd.read_csv(
    "data/processed/features/temporal_features.csv",
    parse_dates=["date"]
)

edges = np.load(
    "data/processed/features/spatial_edges.npy"
)

edge_index = torch.tensor(
    edges.T,
    dtype=torch.long
)

model = SpatialGNN()
model.eval()

features = [
    "rainfall",
    "soil_moisture",
    "soil_moisture_available",
    "elevation",
]

embeddings = []

with torch.no_grad():

    for date in sorted(df["date"].unique()):

        day = df[df["date"] == date].copy()

        X = day[features].copy()

        X["rainfall"] = X["rainfall"].fillna(
            X["rainfall"].median()
        )

        X["soil_moisture"] = X["soil_moisture"].fillna(
            X["soil_moisture"].median()
        )

        x = torch.tensor(
            X.values,
            dtype=torch.float32
        )

        z = model(x, edge_index)

        embeddings.append(z.numpy())

        print(
            f"{date.date()} → {z.shape}"
        )

embeddings = np.stack(embeddings)

output = "data/processed/features/temporal_gnn_embeddings.npy"

np.save(output, embeddings)

print("\nCOMPLETE")
print("Embedding tensor:", embeddings.shape)
print("Saved:", output)