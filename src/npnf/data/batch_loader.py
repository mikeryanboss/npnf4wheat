"""Utilities for loading per-batch prediction results."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch


class BatchLoader:
    """Lazy loader that loads batches from disk on-demand."""

    def __init__(
        self,
        method_dir: Path,
        *,
        load_predictions: bool = True,
        load_batch_data: bool = True,
    ):
        """Initialize the batch loader.

        Args:
            method_dir: Path to the method directory containing batch subdirectories.
            load_predictions: Whether to load predictions.pt files.
            load_batch_data: Whether to load batch.pt files.
        """
        self.method_dir = Path(method_dir)
        self._total_batches = len(list((self.method_dir / "predictions").iterdir()))
        self._load_predictions = load_predictions
        self._load_batch_data = load_batch_data

    def __len__(self) -> int:
        """Return the total number of batches."""
        return self._total_batches

    def __iter__(self) -> Iterator[Any]:
        """Iterate over all batches."""
        for i in range(self._total_batches):
            yield self[i]

    def __getitem__(self, index: int) -> Any:
        """Get a batch by index.

        Args:
            index: Batch index (supports negative indexing).

        Returns:
            Tuple of (predictions, batch_data) if both are loaded,
            or a single dict if only one is loaded.
        """
        if index < 0:
            index = self._total_batches + index
        if index < 0 or index >= self._total_batches:
            raise IndexError(index)

        batch_dir = self.method_dir / "predictions" / f"{index:03d}"

        predictions: Any = None
        batch_data: Any = None

        if self._load_predictions:
            predictions = torch.load(batch_dir / "predictions.pt", weights_only=False)
        if self._load_batch_data:
            batch_data = torch.load(batch_dir / "batch.pt", weights_only=False)

        if self._load_predictions and self._load_batch_data:
            return predictions, batch_data
        if self._load_predictions:
            return predictions
        return batch_data

    def load_batch(self, batch_index: int) -> Any:
        """Load predictions and/or batch data for a single batch.

        Args:
            batch_index: Index of the batch to load.

        Returns:
            Tuple of (predictions, batch_data) if both are configured to load,
            or a single dict if only one is configured.
        """
        return self[batch_index]

    @property
    def config(self) -> dict | None:
        """Load config.json from method directory if present."""
        config_path = self.method_dir / "config.json"
        if config_path.exists():
            with config_path.open() as f:
                return json.load(f)
        return None
