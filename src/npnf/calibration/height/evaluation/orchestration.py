"""Calibration validation orchestration and metrics."""

from typing import Any

from npnf.calibration.height.evaluation.validation import (
    ValidationResult,
    validate_fip1_correlation,
    validate_height,
    validate_warm_short_correlation,
)


def validate_timing_start(predicted: float, target: float = 223) -> ValidationResult:
    """Validate timing start is within +/-10 days of target (Sep 1 based)."""
    threshold = 10
    passed = abs(predicted - target) <= threshold
    return ValidationResult(
        name="timing_start",
        passed=passed,
        actual=predicted,
        target=target,
        threshold=threshold,
        message=(
            f"Start {predicted:.0f}d "
            f"{'within' if passed else 'outside'} +/-{threshold}d of {target}d"
        ),
    )


def validate_timing_end(predicted: float, target: float = 284) -> ValidationResult:
    """Validate timing end is within +/-15 days of target (Sep 1 based)."""
    threshold = 15
    passed = abs(predicted - target) <= threshold
    return ValidationResult(
        name="timing_end",
        passed=passed,
        actual=predicted,
        target=target,
        threshold=threshold,
        message=(
            f"End {predicted:.0f}d "
            f"{'within' if passed else 'outside'} +/-{threshold}d of {target}d"
        ),
    )


def run_all_validations(metrics: dict[str, Any]) -> list[ValidationResult]:
    """Run all validations on computed metrics."""
    validators = {
        "height": validate_height,
        "timing_start": validate_timing_start,
        "timing_end": validate_timing_end,
        "warm_short_correlation": validate_warm_short_correlation,
        "fip1_year_correlation": validate_fip1_correlation,
    }
    return [fn(metrics[key]) for key, fn in validators.items() if key in metrics]
