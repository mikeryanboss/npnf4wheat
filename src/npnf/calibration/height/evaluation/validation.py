"""Height-domain validation functions."""

from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of a single validation test."""

    name: str
    passed: bool
    actual: float | str
    target: float | str
    threshold: float
    message: str


def validate_height(simulated_height: float, target: float = 0.876) -> ValidationResult:
    """Validate height is within ±10% of target.

    Args:
        simulated_height: Mean simulated height in meters.
        target: Target mean height (default: FIP1 mean 0.876m).

    Returns:
        ValidationResult with pass/fail status.
    """
    tolerance = 0.10
    lower = target * (1 - tolerance)
    upper = target * (1 + tolerance)
    passed = lower <= simulated_height <= upper
    return ValidationResult(
        name="height",
        passed=passed,
        actual=simulated_height,
        target=target,
        threshold=tolerance,
        message=(
            f"Height {simulated_height:.3f}m "
            f"{'within' if passed else 'outside'} [{lower:.3f}, {upper:.3f}]m"
        ),
    )


def validate_warm_short_correlation(correlation: float) -> ValidationResult:
    """Validate warm→short correlation is negative (r < -0.1).

    Args:
        correlation: Pearson correlation between mean temp and height.

    Returns:
        ValidationResult with pass/fail status.
    """
    threshold = -0.1
    passed = correlation < threshold
    return ValidationResult(
        name="warm_short_correlation",
        passed=passed,
        actual=correlation,
        target=threshold,
        threshold=threshold,
        message=f"Warm→short r={correlation:.3f} {'<' if passed else '>='} {threshold}",
    )


def validate_fip1_correlation(correlation: float) -> ValidationResult:
    """Validate FIP1 year correlation is positive (r > 0.5).

    Args:
        correlation: Pearson correlation between FIP1 and synthetic by year.

    Returns:
        ValidationResult with pass/fail status.
    """
    threshold = 0.5
    passed = correlation > threshold
    return ValidationResult(
        name="fip1_year_correlation",
        passed=passed,
        actual=correlation,
        target=threshold,
        threshold=threshold,
        message=f"FIP1 year r={correlation:.3f} {'>' if passed else '<='} {threshold}",
    )


def validate_year_pattern(year_heights: dict[int, float]) -> ValidationResult:
    """Validate 2016 is tallest, 2018 is shortest.

    Note: Removed from run_all_validations() by ETH-677. ETH-667 research
    proved the 2016-tallest requirement is unrealistic due to genotype
    confound (different panels per year, no overlap). Kept for reference.

    Args:
        year_heights: Dict mapping year to mean simulated height.

    Returns:
        ValidationResult with pass/fail status.
    """
    tallest_year = max(year_heights, key=lambda y: year_heights[y])
    shortest_year = min(year_heights, key=lambda y: year_heights[y])
    passed = tallest_year == 2016 and shortest_year == 2018
    actual_str = f"tallest={tallest_year}, shortest={shortest_year}"
    target_str = "tallest=2016, shortest=2018"
    return ValidationResult(
        name="year_pattern",
        passed=passed,
        actual=actual_str,
        target=target_str,
        threshold=0,
        message=(
            f"Year pattern: {actual_str} - {'CORRECT' if passed else 'INCORRECT'}"
        ),
    )
