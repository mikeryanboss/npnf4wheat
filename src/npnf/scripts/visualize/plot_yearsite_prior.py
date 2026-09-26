"""Plot prior max-height distribution for a specific yearsite."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from loguru import logger

from npnf.data.batch_loader import BatchLoader
from npnf.scripts.utils.batch_utils import compute_all_batch_max_heights
from npnf.scripts.utils.synthetic_batch_reconstruction import (
    SyntheticBatchReconstructionCache,
    reconstruct_synthetic_batch,
)


def _parse_yearsite(yearsite_uid: str) -> tuple[str, int]:
    """Parse yearsite_uid into (site, year). Format: '{site}_{year}'."""
    parts = yearsite_uid.rsplit("_", 1)
    return parts[0], int(parts[1])


def _build_filter_mask(
    metadata: dict,
    yearsite_uid: int | str | None = None,
    location: str | None = None,
    year: int | None = None,
) -> torch.Tensor | None:
    """Build a filter mask based on yearsite_uid, location, or year.

    Priority: yearsite_uid > location/year combination.
    Returns None if no filtering is needed.
    """
    if yearsite_uid is not None:
        return torch.as_tensor([ys == yearsite_uid for ys in metadata["yearsite_uid"]])

    # Parse site and year from yearsite_uid for location/year filtering
    parsed = [_parse_yearsite(ys) for ys in metadata["yearsite_uid"]]

    if location is not None and year is not None:
        return torch.as_tensor([site == location and yr == year for site, yr in parsed])
    if location is not None:
        return torch.as_tensor([site == location for site, _ in parsed])
    if year is not None:
        return torch.as_tensor([yr == year for _, yr in parsed])
    return None


def _load_all_batches(method_dir: Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Load and aggregate predictions and batch data across all batches.

    Args:
        method_dir: Path to method directory predictions/

    Returns:
        Tuple of (grid_max, batch_max_heights, metadata) where:
        - grid_max: Tensor of shape (total_samples, num_mc)
        - batch_max_heights: Tensor of shape (total_samples,)
        - metadata: Dict with 'yearsite_uid' list

    """
    loader = BatchLoader(method_dir)
    cache = SyntheticBatchReconstructionCache()

    all_grid_max = []
    all_max_heights = []
    all_yearsite_uids: list[str] = []

    for predictions, batch_dict in loader:
        batch_dict = reconstruct_synthetic_batch(method_dir, batch_dict, cache=cache)
        predictions_grid = predictions.select("grid").flatten()["grid"]
        predictions_grid = predictions_grid.mean(dim=1, keepdim=True)
        grid_max = predictions_grid[..., 0].max(dim=-1).values
        all_grid_max.append(grid_max)

        metadata = batch_dict["data"].flatten()
        batch_max_heights = compute_all_batch_max_heights(metadata)
        all_max_heights.append(batch_max_heights)
        all_yearsite_uids.extend(metadata["yearsite_uid"])

    return (
        torch.cat(all_grid_max, dim=0),
        torch.cat(all_max_heights, dim=0),
        {"yearsite_uid": all_yearsite_uids},
    )


def _plot_histogram(
    heights: torch.Tensor,
    bin_width: float,
    output_path: Path,
    title: str,
    batch_max_heights: torch.Tensor | None = None,
) -> None:
    heights_np = heights.numpy(force=True)

    if bin_width <= 0:
        msg = "bin_width must be positive."
        raise ValueError(msg)

    # Convert batch heights to numpy once if available
    batch_np = (
        batch_max_heights.numpy(force=True) if batch_max_heights is not None else None
    )

    # Calculate statistics
    mean_height = heights_np.mean()
    std_height = heights_np.std()

    # Compute bin range including batch heights if available
    h_min = float(heights_np.min())
    h_max = float(heights_np.max())

    if batch_np is not None:
        h_min = min(h_min, float(batch_np.min()))
        h_max = max(h_max, float(batch_np.max()))

    num_bins = max(1, int((h_max - h_min) / bin_width))
    bins = torch.linspace(h_min, h_min + num_bins * bin_width, num_bins + 1).numpy(
        force=True
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        heights_np,
        bins=bins,
        color="#386cb0",
        alpha=0.8,
        label="Predicted",
        edgecolor="black",
        linewidth=0.5,
    )

    # Overlay batch max heights if available
    if batch_np is not None:
        ax.hist(
            batch_np,
            bins=bins,
            color="#7fbc41",
            alpha=0.5,
            label="Ground truth (batch)",
            edgecolor="black",
            linewidth=0.5,
        )

    # Add batch statistics if available
    batch_stats = ""
    if batch_np is not None:
        batch_mean = batch_np.mean()
        batch_std = batch_np.std()
        num_batch = len(batch_np)
        batch_stats = f"\nBatch: {batch_mean:.3f} ± {batch_std:.3f} (n={num_batch})"

    ax.set_xlabel("Max height (m)")
    ax.set_ylabel("Sample count")
    ax.set_title(
        f"{title}\n"
        f"Predicted: {mean_height:.3f} ± {std_height:.3f} (n={len(heights_np)})"
        f"{batch_stats}"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved histogram to {output_path.as_posix()}")


def _plot_yearsite_prior_from_data(
    grid_max: torch.Tensor,
    batch_max_heights: torch.Tensor,
    metadata: dict,
    yearsite_uid: int | str | None = None,
    location: str | None = None,
    year: int | None = None,
    output_path: Path | None = None,
    bin_width: float = 0.05,
    results_path: Path | None = None,
) -> None:
    """Plot prior max-height distribution from preloaded data.

    This internal function accepts precomputed data to avoid redundant loading
    when generating multiple plots.

    Args:
        grid_max: Precomputed max heights from predictions (num_samples, num_mc)
        batch_max_heights: Precomputed max heights from batch (num_samples,)
        metadata: Flattened metadata dict containing yearsite_uid
        yearsite_uid: Yearsite to filter by (takes priority over location/year)
        location: Location (site) to filter by
        year: Year to filter by
        output_path: Output path for the plot
        bin_width: Width of histogram bins
        results_path: Results folder path (for default filename)

    """
    # Filter by mask
    mask = _build_filter_mask(metadata, yearsite_uid, location, year)
    if mask is not None:
        filtered_grid_max = grid_max[mask]
        filtered_batch_max = batch_max_heights[mask]
    else:
        filtered_grid_max = grid_max
        filtered_batch_max = batch_max_heights

    # Determine title and default filename based on filter type
    if yearsite_uid is not None:
        title = f"Prior max-height distribution for yearsite {yearsite_uid}"
        default_filename = f"prior_yearsite_{yearsite_uid}.png"
    elif location is not None and year is not None:
        title = f"Prior max-height distribution for {location} {year}"
        default_filename = f"prior_{location}_{year}.png"
    elif location is not None:
        title = f"Prior max-height distribution for location {location}"
        default_filename = f"prior_location_{location}.png"
    elif year is not None:
        title = f"Prior max-height distribution for year {year}"
        default_filename = f"prior_year_{year}.png"
    else:
        title = "Prior max-height distribution (all samples)"
        default_filename = "prior_all.png"

    if output_path is not None:
        output = Path(output_path)
    elif results_path is not None:
        output = results_path / default_filename
    else:
        output = Path(default_filename)

    _plot_histogram(
        heights=filtered_grid_max.flatten(),
        bin_width=bin_width,
        output_path=output,
        title=title,
        batch_max_heights=filtered_batch_max,
    )


def plot_yearsite_prior(
    results_folder: str,
    yearsite_uid: int | str | None = None,
    location: str | None = None,
    year: int | None = None,
    output_path: str | None = None,
    bin_width: float = 0.05,
) -> None:
    """Plot prior max-height distribution, optionally filtered.

    Args:
        results_folder: Path to method directory
            and predictions/ subdirectory
        yearsite_uid: Yearsite to filter by (takes priority over location/year)
        location: Location (site) to filter by
        year: Year to filter by
        output_path: Override output path for the plot
        bin_width: Width of histogram bins

    """
    results_path = Path(results_folder)

    grid_max, batch_max_heights, metadata = _load_all_batches(results_path)

    _plot_yearsite_prior_from_data(
        grid_max=grid_max,
        batch_max_heights=batch_max_heights,
        metadata=metadata,
        yearsite_uid=yearsite_uid,
        location=location,
        year=year,
        output_path=Path(output_path) if output_path else None,
        bin_width=bin_width,
        results_path=results_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-folder",
        type=str,
        required=True,
        help="Path to method directory predictions/",
    )
    parser.add_argument("--yearsite-uid", type=str, default=None)
    parser.add_argument("--location", type=str, default=None, help="Filter by location")
    parser.add_argument("--year", type=int, default=None, help="Filter by year")
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Override output path for the plot (defaults under results-folder).",
    )
    parser.add_argument("--bin-width", type=float, default=0.05)
    args = parser.parse_args()

    plot_yearsite_prior(
        results_folder=args.results_folder,
        yearsite_uid=args.yearsite_uid,
        location=args.location,
        year=args.year,
        output_path=args.output_path,
        bin_width=args.bin_width,
    )
