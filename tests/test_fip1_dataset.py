from datetime import date

import pytest
import torch

from npnf.data.datasets.fip1 import (
    filter_height_increase_errors,
    normalize_height_days,
    process_height_and_temperature,
)


def test_process_height_and_temperature_aligns_days_clips_heights_and_slices_season():
    harvest_year = 2018
    first_date = date(2018, 3, 1)
    last_date = date(2018, 6, 30)
    height_dates = [first_date, None, last_date]
    temperatures = torch.arange(365 * 24, dtype=torch.float32).tolist()

    out = process_height_and_temperature(
        harvest_year=harvest_year,
        height_values=[-0.5, 0.2, 5.0],
        height_dates=height_dates,
        temperatures=temperatures,
        zero_date=date(1900, 9, 1),
    )
    zero = date(2017, 9, 1)
    expected_days = [(first_date - zero).days, None, (last_date - zero).days]
    assert out["height_days"] == expected_days
    assert [float(value) for value in out["height_values"]] == pytest.approx(
        [0.0, 0.2, 4.0]
    )
    temperature_values = out["temperature_values"]
    assert isinstance(temperature_values, torch.Tensor)
    assert temperature_values.shape == (274, 24)
    # Season starts 61 days after the year's first hourly reading.
    assert temperature_values[0, 0].item() == 61 * 24
    assert temperature_values[-1, -1].item() == (61 + 274) * 24 - 1


def test_normalize_height_days_uses_synthetic_time_scaling():
    days = torch.tensor([212.5, 364.0])
    out = normalize_height_days(days)
    torch.testing.assert_close(
        out["height_days_normalized"], torch.tensor([0.0, 151.5 / 151.5])
    )


def test_normalized_days_match_filtered_days():
    values = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0, 0.6, 0.55, 1.0, 0.5, 0.5])
    days = torch.arange(200, 300, 10, dtype=torch.float32)
    filtered = filter_height_increase_errors(values, days)
    assert len(filtered["height_days"]) < len(days)
    normalized = normalize_height_days(filtered["height_days"])
    assert len(normalized["height_days_normalized"]) == len(filtered["height_values"])
