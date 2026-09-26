"""Run full calibration pipeline (temperature + height) in one command.

Delegates to existing calibration scripts in dependency order:
1. Temperature calibration (extract + fit + optimize + merge)
2. Height calibration
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from npnf.scripts.calibration.height import run_pipeline as run_height_pipeline
from npnf.scripts.calibration.temperature import (
    run_pipeline as run_temperature_pipeline,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_full_pipeline(
    output_dir: Path,
    *,
    num_temperature_trials: int = 1000,
    num_height_trials: int = 3000,
    num_height_genotypes: int = 1000,
) -> None:
    """Run the full calibration pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("")
    logger.info("=" * 60)
    logger.info("FULL CALIBRATION PIPELINE")
    logger.info("=" * 60)
    logger.info("Started: %s", datetime.now(tz=UTC).isoformat())
    logger.info("Output directory: %s", output_dir)

    logger.info("")
    logger.info("[1/2] Temperature calibration")
    run_temperature_pipeline(output_dir=output_dir, num_trials=num_temperature_trials)
    logger.info("Temperature calibration complete")

    logger.info("")
    logger.info("[2/2] Height calibration")
    run_height_pipeline(
        output_dir=output_dir,
        num_trials=num_height_trials,
        num_genotypes=num_height_genotypes,
    )
    logger.info("Height calibration complete")

    logger.info("Completed: %s", datetime.now(tz=UTC).isoformat())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full calibration pipeline: temperature, height."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/calibration"),
        help="Directory for calibration artifacts (default: results/calibration)",
    )
    parser.add_argument(
        "--num-temperature-trials",
        type=int,
        default=1000,
        help="Optuna trials for temperature variance optimization (default: 1000)",
    )
    parser.add_argument(
        "--num-height-trials",
        type=int,
        default=3000,
        help="Optuna trials for height calibration (default: 3000)",
    )
    parser.add_argument(
        "--num-height-genotypes",
        type=int,
        default=1000,
        help="Genotypes sampled per height trial (default: 1000)",
    )
    args = parser.parse_args()

    run_full_pipeline(
        output_dir=args.output_dir,
        num_temperature_trials=args.num_temperature_trials,
        num_height_trials=args.num_height_trials,
        num_height_genotypes=args.num_height_genotypes,
    )


if __name__ == "__main__":
    sys.exit(main())
