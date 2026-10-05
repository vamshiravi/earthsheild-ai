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

        x = self.conv2(x, edge_index)

        return x


df = pd.read_csv(
    "data/processed/features/temporal_features.csv"
)

df = df[df["date"] == df["date"].min()].copy()

features = [
    "rainfall",
    "soil_moisture",
    "soil_moisture_available",
    "elevation",
]

X = df[features].copy()

X["rainfall"] = X["rainfall"].fillna(X["rainfall"].median())
X["soil_moisture"] = X["soil_moisture"].fillna(X["soil_moisture"].median())

x = torch.tensor(
    X.values,
    dtype=torch.float32
)

edges = np.load(
    "data/processed/features/spatial_edges.npy"
)

edge_index = torch.tensor(
    edges.T,
    dtype=torch.long
)

model = SpatialGNN()

with torch.no_grad():
    embeddings = model(x, edge_index)

print("GNN COMPLETE")
print("Input shape:", x.shape)
print("Edge index shape:", edge_index.shape)
print("Embedding shape:", embeddings.shape)
