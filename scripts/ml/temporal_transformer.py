import numpy as np
import torch
import torch.nn as nn


INPUT = "data/processed/features/temporal_gnn_embeddings.npy"


class TemporalTransformer(nn.Module):

    def __init__(
        self,
        input_dim=32,
        hidden_dim=64,
        heads=4,
        layers=2
    ):
        super().__init__()

        self.projection = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers
        )

    def forward(self, x):
        x = self.projection(x)
        return self.transformer(x)


embeddings = np.load(INPUT)

# Take only 256 cells for the test
x = torch.tensor(
    embeddings[:, :256, :],
    dtype=torch.float32
)

# [30, 256, 32] → [256, 30, 32]
x = x.permute(1, 0, 2)

model = TemporalTransformer()
model.eval()

with torch.no_grad():
    output = model(x)

print("TRANSFORMER TEST COMPLETE")
print("Input shape:", x.shape)
print("Output shape:", output.shape)