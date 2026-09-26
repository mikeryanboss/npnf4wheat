from typing import Any

import torch
from loguru import logger
from torch.utils.data import Dataset


class WeightedYearsiteCombinedDataset(Dataset):
    """
    Combined dataset that samples from real and synthetic data with yearsite weighting.

    This dataset implements virtual epoch sampling:
    - Uses virtual_size to control epoch length
    - Maintains p_synth ratio through deterministic sampling decisions
    - Cycles through sub-datasets, restarting from index 0 when exhausted
    - DataLoader shuffling handles randomization for training

    Args:
        dataset_real: Real dataset (FIP1) - can be pre-balanced or regular
        dataset_synth: Synthetic dataset
        virtual_size: Number of samples per epoch (default: 10000)
        p_synth: Probability of sampling from synthetic dataset. If None (default),
                auto-calculates based on the ratio of synthetic units to real units.
        seed: Random seed for deterministic sampling decisions (default: 42)
    """

    def __init__(
        self,
        dataset_real,
        dataset_synth,
        virtual_size: int = 10000,
        p_synth: float | None = None,
        seed: int = 42,
    ):
        self.dataset_real = dataset_real
        self.dataset_synth = dataset_synth
        self.virtual_size = virtual_size
        self.seed = seed

        # Auto-calculate p_synth if not provided
        if p_synth is None:
            # Try to get yearsite count from real dataset if it's pre-balanced
            if hasattr(dataset_real, "get_yearsite_info"):
                yearsite_info = dataset_real.get_yearsite_info()
                num_real_units = yearsite_info["num_yearsites"]
                logger.info(
                    f"Using pre-balanced real dataset with {num_real_units} yearsites"
                )
            else:
                # For regular datasets, assume 1 unit (no balancing applied)
                num_real_units = 1
                logger.info(
                    "Real dataset is not pre-balanced. "
                    "Using 1 unit for p_synth calculation."
                )

            num_synth_units = self._count_synthetic_units(dataset_synth)
            total_units = num_real_units + num_synth_units

            if total_units == 0:
                self.p_synth = 0.0
            else:
                self.p_synth = num_synth_units / total_units

            logger.info(
                f"Auto-calculated p_synth = {self.p_synth:.3f} "
                f"({num_synth_units} synthetic units / {total_units} total units)"
            )
        else:
            self.p_synth = p_synth

        self.real_size = len(self.dataset_real)
        self.synth_size = len(self.dataset_synth)

        # Expected samples per epoch
        expected_synth_samples = int(self.virtual_size * self.p_synth)
        expected_real_samples = self.virtual_size - expected_synth_samples

        logger.info(
            f"Virtual epoch size: {self.virtual_size} "
            f"(~{expected_real_samples} real, ~{expected_synth_samples} synthetic)"
        )

        # Initialize sampling state
        self.real_index = 0
        self.synth_index = 0

    def __len__(self) -> int:
        return self.virtual_size

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index >= self.virtual_size:
            error_msg = (
                f"Index {index} out of range for dataset of size {self.virtual_size}"
            )
            raise IndexError(error_msg)

        # Deterministic choice based on index and seed
        choice_hash = (index * 31 + self.seed) % 1000
        use_synthetic = choice_hash < self.p_synth * 1000

        if use_synthetic:
            if self.synth_index >= self.synth_size:
                self.synth_index = 0

            sample = self.dataset_synth[self.synth_index]
            self.synth_index += 1
            source = "synthetic"
        else:
            if self.real_index >= self.real_size:
                self.real_index = 0

            sample = self.dataset_real[self.real_index]
            self.real_index += 1
            source = "real"

        sample["source_is_synth"] = torch.tensor(1.0 if source == "synthetic" else 0.0)

        return sample

    def _count_synthetic_units(self, dataset_synth) -> int:
        return int(dataset_synth.num_yearsites)

    def get_yearsite_info(self) -> dict[str, Any]:
        if hasattr(self.dataset_real, "get_yearsite_info"):
            return self.dataset_real.get_yearsite_info()
        return {"num_yearsites": 1, "message": "Real dataset is not pre-balanced"}
