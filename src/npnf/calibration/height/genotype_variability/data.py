"""FIP1 per-year aggregation for the genotype-variability analysis.

The FIP1 references for maximum height are the spatially corrected values in
the dataset: `height_final_blue` from a per-year SpATS fit with fixed genotype,
and `height_final_heritability`, the generalized heritability of a companion
fit with random genotype. The two fits share one spatial structure.
"""

from __future__ import annotations

from collections import defaultdict

import datasets
import numpy as np
import torch

from npnf.calibration.height.data import _detect_growth_end_day
from npnf.calibration.height.genotype_variability.traits import (
    effective_replicates,
    elongation_rate,
)


def load_fip1_metadata(dataset_path: str, splits: tuple[str, ...]) -> dict:
    """Load per-plot genotype labels, BLUEs and heritabilities from raw FIP1."""
    dataset_dict = datasets.load_from_disk(dataset_path)
    if not isinstance(dataset_dict, datasets.DatasetDict):
        msg = f"Expected a DatasetDict at {dataset_path}"
        raise TypeError(msg)
    dataset = datasets.concatenate_datasets([dataset_dict[split] for split in splits])
    columns = [
        "plot_uid",
        "harvest_year",
        "genotype_id",
        "height_final_blue",
        "height_final_heritability",
    ]
    return {row["plot_uid"]: row for row in dataset.select_columns(columns).to_list()}


def fip1_traits_by_year(heights_dataset: datasets.Dataset, metadata: dict) -> dict:
    """Group the FIP1 traits, BLUEs and heritability by year and genotype."""
    padded = torch.nn.utils.rnn.pad_sequence(
        list(heights_dataset["height_values"]),
        batch_first=True,
        padding_value=float("-inf"),
    )
    # Median of the three highest measurements, robust to one outlier.
    plot_max_heights = torch.topk(padded, 3, dim=1).values.median(dim=1).values.tolist()
    plot_elongation_rates = [
        elongation_rate(days.numpy(), heights.numpy())
        for days, heights in zip(
            heights_dataset["height_days"],
            heights_dataset["height_values"],
            strict=True,
        )
    ]

    # Same detector as the calibration, kept per plot to group by genotype.
    plot_growth_end_days = [
        _detect_growth_end_day(days.numpy(), heights.numpy())
        for days, heights in zip(
            heights_dataset["height_days"],
            heights_dataset["height_values"],
            strict=True,
        )
    ]

    per_year: dict[int, dict] = defaultdict(
        lambda: {
            "plot_max_height": [],
            "genotype_max_height": defaultdict(list),
            "genotype_elongation_rate": defaultdict(list),
            "genotype_growth_end_day": defaultdict(list),
            "blue": {},
            "heritability": float("nan"),
        }
    )
    for plot_uid, year, max_height, rate, growth_end in zip(
        heights_dataset["plot_uid"],
        heights_dataset["harvest_year"],
        plot_max_heights,
        plot_elongation_rates,
        plot_growth_end_days,
        strict=True,
    ):
        row = metadata[plot_uid]
        year = int(year)
        fip1_year = per_year[year]
        fip1_year["plot_max_height"].append(float(max_height))
        fip1_year["genotype_max_height"][row["genotype_id"]].append(float(max_height))
        fip1_year["genotype_elongation_rate"][row["genotype_id"]].append(rate)
        if growth_end is not None:
            fip1_year["genotype_growth_end_day"][row["genotype_id"]].append(
                float(growth_end)
            )
        if row["height_final_blue"] is not None:
            fip1_year["blue"][row["genotype_id"]] = float(row["height_final_blue"])
        if row["height_final_heritability"] is not None:
            # One value per year, broadcast to every plot row of that year.
            fip1_year["heritability"] = float(row["height_final_heritability"])

    for fip1_year in per_year.values():
        fip1_year["effective_replicates_max_height"] = effective_replicates(
            fip1_year["genotype_max_height"]
        )
        fip1_year["effective_replicates_elongation_rate"] = effective_replicates(
            fip1_year["genotype_elongation_rate"]
        )
        fip1_year["genotype_mean_elongation_rate"] = {
            genotype: float(np.mean(finite))
            for genotype, values in fip1_year["genotype_elongation_rate"].items()
            if (finite := [value for value in values if np.isfinite(value)])
        }
        fip1_year["effective_replicates_growth_end_day"] = effective_replicates(
            fip1_year["genotype_growth_end_day"]
        )
    return dict(per_year)
