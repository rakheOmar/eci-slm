#!/usr/bin/env python3
"""Simple dataloader for ECI-SLM training."""

import numpy as np
from pathlib import Path


class Dataloader:
    def __init__(self, data_path: str | Path, block_size: int, batch_size: int = 1):
        self.data_path = Path(data_path)
        self.block_size = block_size
        self.batch_size = batch_size

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.data = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        self.n = len(self.data)

    def __len__(self):
        return max(0, self.n - self.block_size) // self.batch_size

    def get_batch(self):
        ix = np.random.randint(0, self.n - self.block_size, size=(self.batch_size,))
        x = np.stack([self.data[i : i + self.block_size] for i in ix]).astype(np.int32)
        y = np.stack([self.data[i + 1 : i + 1 + self.block_size] for i in ix]).astype(
            np.int32
        )
        return x, y

    def get_batch_val(self, num_batches: int = 20):
        """Get random validation batches."""
        x_batch = []
        y_batch = []
        for _ in range(num_batches):
            x, y = self.get_batch()
            x_batch.append(x)
            y_batch.append(y)
        return np.concatenate(x_batch, axis=0), np.concatenate(y_batch, axis=0)


def create_dataloaders(
    data_dir: str | Path = ".",
    block_size: int = 512,
    batch_size: int = 1,
    train_file: str = "train.bin",
    val_file: str = "val.bin",
):
    data_dir = Path(data_dir)

    train_loader = None
    val_loader = None

    if (data_dir / train_file).exists():
        train_loader = Dataloader(data_dir / train_file, block_size, batch_size)
        print(
            f"Train: {len(train_loader):,} batches, {len(train_loader.data):,} tokens"
        )

    if (data_dir / val_file).exists():
        val_loader = Dataloader(data_dir / val_file, block_size, batch_size)
        print(f"Val: {len(val_loader):,} batches, {len(val_loader.data):,} tokens")

    return train_loader, val_loader


if __name__ == "__main__":
    import sys

    data_dir = (
        Path("notebooks") if Path("notebooks/train.bin").exists() else Path("data")
    )

    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])

    print(f"Loading data from: {data_dir}")
    train_loader, val_loader = create_dataloaders(
        data_dir, block_size=128, batch_size=8
    )

    if train_loader:
        print("\nTesting batch...")
        x, y = train_loader.get_batch()
        print(f"x shape: {x.shape}, dtype: {x.dtype}")
        print(f"y shape: {y.shape}, dtype: {y.dtype}")
