from dataclasses import dataclass
from datetime import date
from typing import Any

import datasets
import numpy as np
import torch
from loguru import logger
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import Dataset

from .utils import YearsiteIndex


@dataclass(frozen=True)
class Fip1Facts:
    """Fixed facts of the FIP1 dataset.

    ``test_splits``: the four test splits, a subset of ``splits``.
    ``held_out_year``: the year that no calibration or training uses.
    """

    splits: tuple[str, ...] = (
        "train",
        "validation",
        "test_plot",
        "test_genotype",
        "test_environment",
        "test_genotype_environment",
    )
    test_splits: tuple[str, ...] = (
        "test_plot",
        "test_genotype",
        "test_environment",
        "test_genotype_environment",
    )
    years: tuple[int, ...] = (2016, 2017, 2018, 2019, 2021, 2022)
    held_out_year: int = 2019


class SingleItemAccessDataset(Dataset):
    """Wrapper that forces DataLoader to use __getitem__ instead of __getitems__.

    This avoids a bug in HuggingFace datasets' __getitems__() batch fetching when
    batches contain mixed None/non-None values in list-type columns (e.g., markers).
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def get_dataset_identity(self) -> dict[str, Any]:
        """Return a JSON-serializable identity, mirroring ``SyntheticDataset``.

        The prediction scripts store this in ``config.json``. HuggingFace's
        ``_fingerprint`` already hashes the split and every applied transform, so
        it plays the role the synthetic cache key plays.
        """
        return {
            "dataset_type": "Fip1HeightsDataset",
            "cache_key": str(self.dataset._fingerprint),  # noqa: SLF001
            "num_conditions": len(self),
        }

    def get_condition_metadata(self) -> dict[str, list[str]]:
        """Return plot, genotype and yearsite IDs in dataset index order.

        ``plot_uid`` is included because FIP1 replicate plots share a
        ``(genotype_id, yearsite_uid)`` pair, so only the plot is unique.
        """
        return {
            "plot_uid": [str(uid) for uid in self.dataset["plot_uid"]],
            "genotype_id": [str(gid) for gid in self.dataset["genotype_id"]],
            "yearsite_uid": [str(uid) for uid in self.dataset["yearsite_uid"]],
        }


def filter_height_increase_errors(
    values: torch.Tensor, days: torch.Tensor, eps_scale: float = 4.0
):
    """Advanced yet lightweight filter that removes artificial height recoveries
    after lodging.

    Parameters
    ----------
    values : torch.Tensor
        1-D tensor of height readings.
    days : torch.Tensor
        Matching tensor of day indices.
    eps_scale : float, optional
        Multiplier for the noise-based tolerance.  Default is 2.0 (≈95 % CI).

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with possibly reduced `height_values` and `height_days`.
    """
    if len(values) < 4:
        return {"height_values": values, "height_days": days}

    n = len(values)
    values = values.clone()  # avoid accidental in-place modification

    # 1. Peak (maximum height) detection
    peak_index = int(torch.argmax(values))
    peak_val = values[peak_index]

    # If the peak is at the end, nothing to filter
    if peak_index >= n - 2:
        return {"height_values": values, "height_days": days}

    # 2. Detect lodging as an 15 % sustained drop from the peak
    lodging_drop_fraction = 0.15
    post_peak = values[peak_index + 1 :]
    drop_mask = post_peak < peak_val * (1 - lodging_drop_fraction)
    if not drop_mask.any():
        return {"height_values": values, "height_days": days}

    lodging_local_index = int(torch.nonzero(drop_mask, as_tuple=False)[0])
    start_index = peak_index + 1 + lodging_local_index

    # 3. Fit a non-increasing envelope to the post-lodging segment
    x = torch.arange(start_index, n, dtype=torch.float32).numpy(force=True)
    y = values[start_index:].numpy(force=True)
    ir = IsotonicRegression(increasing=False)
    envelope_np = ir.fit_transform(x, y)
    envelope = torch.as_tensor(envelope_np, dtype=values.dtype)

    # 4. Robust per-curve noise estimate using MAD on pre-peak differences
    if peak_index >= 2:
        pre_diffs = values[1:peak_index] - values[: peak_index - 1]
        mad = torch.median(torch.abs(pre_diffs - pre_diffs.median()))
        sigma = 1.4826 * mad if mad > 0 else 0.02 * peak_val
    else:
        sigma = 0.02 * peak_val
    eps = eps_scale * sigma

    # 5. Build keep-mask
    keep = torch.ones(n, dtype=torch.bool)
    keep[start_index:] = values[start_index:] <= envelope + eps

    return {"height_values": values[keep], "height_days": days[keep]}


def standardize_values(
    heights: torch.Tensor,
    height_days: torch.Tensor,
    temperatures: torch.Tensor,
    height_values_std_mean: tuple[float, float],
    height_days_std_mean: tuple[float, float],
    temperature_values_std_mean: tuple[float, float],
) -> dict[str, torch.Tensor | list]:
    return {
        "height_values_standardized": (
            (heights - height_values_std_mean[1]) / height_values_std_mean[0]
        ).tolist(),
        "height_days_standardized": (
            (height_days - height_days_std_mean[1]) / height_days_std_mean[0]
        ).tolist(),
        "temperature_values_standardized": (
            (temperatures - temperature_values_std_mean[1])
            / temperature_values_std_mean[0]
        ),
    }


def standardize_dataset(
    dataset: datasets.Dataset,
    height_values_std_mean: tuple[float, float],
    temperature_values_std_mean: tuple[float, float],
    height_days_std_mean: tuple[float, float],
) -> datasets.Dataset:
    return dataset.map(
        standardize_values,
        input_columns=["height_values", "height_days", "temperature_values"],
        fn_kwargs={
            "height_values_std_mean": height_values_std_mean,
            "height_days_std_mean": height_days_std_mean,
            "temperature_values_std_mean": temperature_values_std_mean,
        },
    )


def process_height_and_temperature(
    harvest_year: int,
    height_values: list[float],
    height_dates: list[date | None],
    temperatures: list[float],
    zero_date: date,
) -> dict[str, torch.Tensor | list]:
    height_values = [np.clip(values, 0.0, 4.0) for values in height_values]
    height_days = [
        (date - zero_date.replace(year=harvest_year - 1)).days
        if date is not None
        else None
        for date in height_dates
    ]
    temperature_values = torch.as_tensor(temperatures).reshape(-1, 24)
    temperature_values = temperature_values[61 : 61 + 274]
    return {
        "height_values": height_values,
        "height_days": height_days,
        "temperature_values": temperature_values,
    }


def normalize_height_days(height_days: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"height_days_normalized": (height_days - 212.5) / 151.5}


def get_heights_dataset(
    split: str | list[str],
    zero_date: date = date(1900, 9, 1),
    height_values_std_mean=(0.3404, 0.4721),
    height_days_std_mean=(37.3556, 257.3499),
    temperature_values_std_mean=(7.6197, 8.6357),
    datasets_offline_path: str | None = None,
    standardize: bool = True,
    exclude_years: list[int] | None = None,
    **kwargs: Any,
) -> SingleItemAccessDataset:
    logger.info(f"Loading fip1 temperature & height dataset with split {split}!")

    if datasets_offline_path is None:
        if not isinstance(split, str):
            split = "+".join(split)
        dataset = datasets.load_dataset("mikeboss/FIP1", split=split)
    else:
        dataset = datasets.load_from_disk(datasets_offline_path)
        if not isinstance(dataset, datasets.DatasetDict):
            msg = f"Expected a DatasetDict at {datasets_offline_path}"
            raise TypeError(msg)
        if not isinstance(split, str):
            dataset = datasets.concatenate_datasets([dataset[s] for s in split])
        else:
            dataset = dataset[split]

    columns_to_select = [
        "harvest_year",
        "height_values",
        "height_dates",
        "temperature_air_200cm_values",
        "marker_biallelic_codes",
        "yearsite_uid",
        "plot_uid",
        "genotype_id",
    ]

    dataset = dataset.select_columns(columns_to_select)

    dataset = dataset.map(
        process_height_and_temperature,
        input_columns=[
            "harvest_year",
            "height_values",
            "height_dates",
            "temperature_air_200cm_values",
        ],
        remove_columns=["height_dates", "temperature_air_200cm_values"],
        fn_kwargs={"zero_date": zero_date},
    )

    if exclude_years:
        dataset = dataset.filter(
            lambda example: int(example["harvest_year"]) not in exclude_years
        )

    dataset = dataset.with_format("torch")

    dataset = dataset.map(
        filter_height_increase_errors, input_columns=["height_values", "height_days"]
    )
    dataset = dataset.map(normalize_height_days, input_columns=["height_days"])

    if standardize:
        return SingleItemAccessDataset(
            standardize_dataset(
                dataset,
                height_values_std_mean=height_values_std_mean,
                height_days_std_mean=height_days_std_mean,
                temperature_values_std_mean=temperature_values_std_mean,
            )
        )

    return SingleItemAccessDataset(dataset)


class BalancedFIP1Dataset(Dataset):
    """
    FIP1 dataset with yearsite-based weighted sampling.

    Addresses the imbalance where some years/sites have more samples than others
    by applying configurable weighting strategies during sampling.

    Args:
        split: Dataset split(s) to load (e.g., "train", ["train", "validation"])
        weight_strategy: Weighting strategy - "uniform", "inverse_frequency", or
            custom dict
        virtual_size: Number of samples per epoch. If None, uses natural dataset size.
        zero_date: Zero date for day calculations (passed to get_heights_dataset)
        height_values_std_mean: Standardization params for height values
        height_days_std_mean: Standardization params for height days
        temperature_values_std_mean: Standardization params for temperature
        datasets_offline_path: Path to offline datasets (if not using HuggingFace Hub)
        standardize: Whether to apply standardization
        **kwargs: Additional arguments passed to get_heights_dataset
    """

    def __init__(
        self,
        split: str | list[str],
        weight_strategy: str | dict[tuple[int, str], float] = "uniform",
        virtual_size: int | None = None,
        zero_date: date = date(1900, 9, 1),
        height_values_std_mean: tuple[float, float] = (0.3404, 0.4721),
        height_days_std_mean: tuple[float, float] = (37.3556, 257.3499),
        temperature_values_std_mean: tuple[float, float] = (7.6197, 8.6357),
        datasets_offline_path: str | None = None,
        standardize: bool = True,
        **kwargs: Any,
    ):
        # Load base dataset (now always includes site metadata)
        self.base_dataset = get_heights_dataset(
            split=split,
            zero_date=zero_date,
            height_values_std_mean=height_values_std_mean,
            height_days_std_mean=height_days_std_mean,
            temperature_values_std_mean=temperature_values_std_mean,
            datasets_offline_path=datasets_offline_path,
            standardize=standardize,
            **kwargs,
        )

        self.weight_strategy = weight_strategy
        self.virtual_size = (
            virtual_size if virtual_size is not None else len(self.base_dataset)
        )

        # Build yearsite index for weighted sampling
        self._build_yearsite_index()

        logger.info(
            f"BalancedFIP1Dataset initialized: {len(self.base_dataset)} base samples, "
            f"{self.virtual_size} virtual samples, "
            f"{len(self.yearsite_index.yearsites)} yearsites, "
            f"weight_strategy='{weight_strategy}'"
        )

    def _build_yearsite_index(self) -> None:
        """Build yearsite index for weighted sampling from base dataset."""
        if len(self.base_dataset) == 0:
            msg = "Base dataset is empty"
            raise ValueError(msg)

        # Validate required fields
        sample = self.base_dataset[0]
        if "harvest_year" not in sample or "yearsite_uid" not in sample:
            msg = "Base dataset must include 'harvest_year' and 'yearsite_uid' fields."
            raise ValueError(msg)

        # Extract years and sites from all samples
        years = []
        sites = []
        for i in range(len(self.base_dataset)):
            sample = self.base_dataset[i]
            years.append(int(sample["harvest_year"]))
            sites.append(str(sample["yearsite_uid"]))

        # Build yearsite index based on weight strategy
        if self.weight_strategy == "uniform":
            self.yearsite_index = YearsiteIndex.build_uniform_weights(
                years=years, sites=sites
            )
        elif self.weight_strategy == "inverse_frequency":
            self.yearsite_index = YearsiteIndex.build_inverse_frequency_weights(
                years=years, sites=sites
            )
        elif isinstance(self.weight_strategy, dict):
            self.yearsite_index = YearsiteIndex.build_custom_weights(
                years=years, sites=sites, custom_weights=self.weight_strategy
            )
        else:
            msg = (
                f"Unknown weight_strategy: {self.weight_strategy}. "
                f"Must be 'uniform', 'inverse_frequency', or a custom weights dict."
            )
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return virtual dataset size."""
        return self.virtual_size

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Get a sample with deterministic weighted sampling.

        Uses the index to deterministically sample a yearsite according to weights,
        then uniformly samples an index within that yearsite.

        Args:
            index: Sample index (0 to virtual_size-1)

        Returns:
            Sample dict from base dataset
        """
        if index >= self.virtual_size:
            msg = f"Index {index} out of range [0, {self.virtual_size})"
            raise IndexError(msg)

        # Deterministic per-sample choice based on index
        generator = torch.Generator()
        generator.manual_seed(index + 54321)  # Fixed seed offset for reproducibility

        # Sample yearsite according to weights
        yearsite = self.yearsite_index.sample_yearsite(generator)

        # Sample index within the yearsite
        real_index = self.yearsite_index.sample_index_from_yearsite(yearsite, generator)

        # Get sample from base dataset
        return self.base_dataset[real_index]

    def get_yearsite_info(self) -> dict[str, Any]:
        """
        Get information about the yearsite distribution and weights.

        Returns:
            Dict with yearsite statistics including counts, weights, and metadata
        """
        return self.yearsite_index.get_yearsite_info()
