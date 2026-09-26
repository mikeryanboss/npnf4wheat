"""Temperature calibration orchestrator.

Runs the calibration pipeline in order:
  [1/6] Load data and construct CalibrationData
  [2/6] Fit seasonal model + AR coefficient
  [3/6] Fit diurnal model + weather coupling
  [4/6] Extract variance parameters
  [5/6] Optimize variance parameters
  [6/6] Merge final parameters

Usage:
    python -m npnf.scripts.calibration.temperature [OPTIONS]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from npnf.calibration.temperature.data import (
    CalibrationData,
    FixedConstants,
    ObservedData,
    load_swiss_data,
)
from npnf.calibration.temperature.diurnal.visualize import plot_extracted_params
from npnf.calibration.temperature.pipeline import (
    build_final_params,
    extract_variance,
    fit_diurnal_model,
    fit_seasonal_model,
    optimize_variance_step,
)
from npnf.calibration.temperature.seasonal.visualize import plot_seasonal_fit
from npnf.calibration.temperature.variance.visualize import (
    plot_year_means_detrending,
    visualize_variance_diagnostics,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(
    output_dir: Path, num_trials: int = 1000, measurement_noise_std: float = 0.3
) -> None:
    """Run the full temperature calibration pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[1/6] Loading Swiss calibration data")
    swiss_data = load_swiss_data()
    observed = ObservedData.from_dataset(swiss_data)
    data = CalibrationData(
        observed=observed,
        constants=FixedConstants(measurement_noise_std=measurement_noise_std),
    )
    logger.info("Loaded %d site-year samples", observed.num_samples)

    logger.info("[2/6] Fit seasonal model + AR coefficient")
    data.seasonal = fit_seasonal_model(data)

    logger.info("[3/6] Fit diurnal model + weather coupling")
    data.diurnal = fit_diurnal_model(data)

    logger.info("[4/6] Extract variance parameters")
    data.optimization = extract_variance(data)

    logger.info("[5/6] Optimize variance parameters")
    data.variance, study = optimize_variance_step(
        data, output_dir, num_trials=num_trials
    )

    logger.info("[6/6] Merge final parameters")
    final_params = build_final_params(data, output_dir)

    # Visualization pass
    plot_extracted_params(observed, output_dir, timing=data.diurnal.timing)
    plot_seasonal_fit(observed, data.seasonal, output_dir)
    plot_year_means_detrending(observed, output_dir)
    visualize_variance_diagnostics(
        data, output_dir, study=study, swiss_data=swiss_data, final_params=final_params
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run temperature calibration pipeline")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/calibration"),
        help="Directory for calibration artifacts (default: results/calibration)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1000,
        help="Number of variance optimization trials (default: 1000)",
    )
    parser.add_argument(
        "--measurement-noise-std",
        type=float,
        default=0.3,
        help="Sensor measurement noise std (default: 0.3)",
    )
    args = parser.parse_args()

    run_pipeline(
        output_dir=args.output_dir,
        num_trials=args.num_trials,
        measurement_noise_std=args.measurement_noise_std,
    )


if __name__ == "__main__":
    sys.exit(main())
