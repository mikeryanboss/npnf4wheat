"""Height calibration date offset constants."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HeightDates:
    """Temporal indexing for the height calibration window.

    All day numbers are relative to Sep 1 (day 0).
    """

    # Nov 1 offset: temperature array index 0 = Nov 1 = day 61 from Sep 1
    day_temperature_start: int = 61

    # Temperature window: Nov 1 to Aug 1 (335 - 61)
    total_days: int = 274

    # Stem Elongation / Growth Season
    tau_start_day: int = 200  # Mid-March: photoperiod/vernalization gate
    plant_growth_stop_max: int = 322  # End of observable growth season

    @property
    def tau_start_idx(self) -> int:
        """Index into 274-day arrays where tau accumulation begins."""
        return self.tau_start_day - self.day_temperature_start  # 139

    @property
    def growing_period_end_idx(self) -> int:
        """Index into 274-day arrays where observable growth ends."""
        return self.plant_growth_stop_max - self.day_temperature_start  # 261

    @property
    def maximum_growth_period(self) -> int:
        """Number of days in the active growth window."""
        return self.plant_growth_stop_max - self.tau_start_day  # 122
