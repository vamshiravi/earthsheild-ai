import numpy as np
from pathlib import Path

INPUT = "data/processed/features/temporal_gnn_embeddings.npy"
OUTPUT = "data/processed/features/training_batches"

BATCH_SIZE = 256

embeddings = np.load(INPUT)

num_days, num_cells, features = embeddings.shape

Path(OUTPUT).mkdir(parents=True, exist_ok=True)

batch_id = 0

for start in range(0, num_cells, BATCH_SIZE):

    end = min(start + BATCH_SIZE, num_cells)

    batch = embeddings[:, start:end, :]

    np.save(
        f"{OUTPUT}/batch_{batch_id:03d}.npy",
        batch
    )

    batch_id += 1

print("BATCH PREPARATION COMPLETE")
print(f"Total cells: {num_cells}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Number of batches: {batch_id}")
print(f"Batch shape: {embeddings[:, :BATCH_SIZE, :].shape}")