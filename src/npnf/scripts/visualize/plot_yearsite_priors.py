"""Generate prior max-height distribution plots for all yearsites in metadata."""

import argparse
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from npnf.scripts.visualize.plot_yearsite_prior import (
    _load_all_batches,
    _plot_yearsite_prior_from_data,
)


def plot_yearsite_priors(
    results_folder: str, output_dir: str | None = None, bin_width: float = 0.05
) -> None:
    """Generate prior plots for all yearsites found in metadata.

    Generates plots in hierarchical order:
    - plots/prior_all.png (all samples)
    - plots/site/prior_location_{site}.png (per location)
    - plots/year/prior_year_{year}.png (per year)
    - plots/yearsite/prior_yearsite_{yearsite_uid}.png (per yearsite)

    Args:
        results_folder: Path to method directory
            and predictions/ subdirectory
        output_dir: Override base output directory (defaults to results_folder).
            Plots are saved to a "plots" subfolder within this directory.
        bin_width: Width of histogram bins

    """
    results_path = Path(results_folder)

    logger.info("Loading predictions and batch data from all batches...")
    grid_max, batch_max_heights, metadata = _load_all_batches(results_path)

    unique_yearsites = sorted(set(metadata["yearsite_uid"]))

    # Parse sites and years from yearsite_uid (format: "{site}_{year}")
    sites_years = [ys.rsplit("_", 1) for ys in unique_yearsites]
    unique_sites = sorted({site for site, _ in sites_years})
    unique_years = sorted({int(year) for _, year in sites_years})

    logger.info(
        f"Found {len(unique_sites)} sites, {len(unique_years)} years, "
        f"{len(unique_yearsites)} yearsites"
    )

    # Determine output directory and create subfolders
    base_path = Path(output_dir) if output_dir else results_path
    output_path = base_path / "plots"
    site_path = output_path / "site"
    year_path = output_path / "year"
    yearsite_path = output_path / "yearsite"

    output_path.mkdir(parents=True, exist_ok=True)
    site_path.mkdir(parents=True, exist_ok=True)
    year_path.mkdir(parents=True, exist_ok=True)
    yearsite_path.mkdir(parents=True, exist_ok=True)

    errors = []

    # 1. Generate combined plot for all samples
    try:
        plot_file = output_path / "prior_all.png"
        _plot_yearsite_prior_from_data(
            grid_max=grid_max,
            batch_max_heights=batch_max_heights,
            metadata=metadata,
            output_path=plot_file,
            bin_width=bin_width,
        )
        logger.info("Generated combined plot for all samples")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to generate combined plot: {e}")
        errors.append(("all", str(e)))

    # 2. Generate per-location plots
    site_success = 0
    for site in tqdm(unique_sites, desc="Generating site plots"):
        try:
            plot_file = site_path / f"prior_location_{site}.png"
            _plot_yearsite_prior_from_data(
                grid_max=grid_max,
                batch_max_heights=batch_max_heights,
                metadata=metadata,
                location=site,
                output_path=plot_file,
                bin_width=bin_width,
            )
            site_success += 1
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to plot site {site}: {e}")
            errors.append((f"site:{site}", str(e)))

    logger.info(f"Generated {site_success}/{len(unique_sites)} site plots")

    # 3. Generate per-year plots
    year_success = 0
    for year in tqdm(unique_years, desc="Generating year plots"):
        try:
            plot_file = year_path / f"prior_year_{year}.png"
            _plot_yearsite_prior_from_data(
                grid_max=grid_max,
                batch_max_heights=batch_max_heights,
                metadata=metadata,
                year=year,
                output_path=plot_file,
                bin_width=bin_width,
            )
            year_success += 1
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to plot year {year}: {e}")
            errors.append((f"year:{year}", str(e)))

    logger.info(f"Generated {year_success}/{len(unique_years)} year plots")

    # 4. Generate per-yearsite plots
    yearsite_success = 0
    for yearsite_uid in tqdm(unique_yearsites, desc="Generating yearsite plots"):
        try:
            plot_file = yearsite_path / f"prior_yearsite_{yearsite_uid}.png"
            _plot_yearsite_prior_from_data(
                grid_max=grid_max,
                batch_max_heights=batch_max_heights,
                metadata=metadata,
                yearsite_uid=yearsite_uid,
                output_path=plot_file,
                bin_width=bin_width,
            )
            yearsite_success += 1
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to plot yearsite {yearsite_uid}: {e}")
            errors.append((f"yearsite:{yearsite_uid}", str(e)))

    logger.info(f"Generated {yearsite_success}/{len(unique_yearsites)} yearsite plots")

    # Summary
    if errors:
        logger.warning(f"Failed to generate {len(errors)} plots total")
        for name, error in errors:
            logger.warning(f"  - {name}: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate prior plots for all yearsites in metadata"
    )
    parser.add_argument(
        "--results-folder",
        type=str,
        required=True,
        help="Path to method directory with predictions/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override base output directory. Plots saved to 'plots' subfolder.",
    )
    parser.add_argument("--bin-width", type=float, default=0.05)
    args = parser.parse_args()

    plot_yearsite_priors(
        results_folder=args.results_folder,
        output_dir=args.output_dir,
        bin_width=args.bin_width,
    )
