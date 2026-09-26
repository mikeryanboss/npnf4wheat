from collections.abc import Sequence

import datasets
import polars as pl


def load_weather_data(
    path: str,
    sites: Sequence[str] | None = None,
    exclude_sites: Sequence[str] | None = None,
    exclude_year: int | None = None,
    include_years: Sequence[int] | None = None,
    exclude_year_sites: dict[str, Sequence[int]] | None = None,
) -> datasets.Dataset:
    # excluded: Moudon (faulty temperature sensor, readings up to 48°C)
    # excluded: Assens (uncorrelated with regional climate, r=0.16 vs r>0.87)
    if exclude_sites is None:
        exclude_sites = ("Moudon", "Assens")
    # stuck/faulty sensors: multi-day zero diurnal range or sustained glitches
    if exclude_year_sites is None:
        exclude_year_sites = {
            "Delley": [2016],
            "Ellighausen": [2004, 2006],
            "Lindau": [2004, 2018],
            "SulzKunten": [2004, 2005, 2007, 2008],
        }

    df = pl.scan_csv(path, schema_overrides={"RH": pl.Float64}, try_parse_dates=True)

    if exclude_sites:
        df = df.filter(~pl.col("site").is_in(exclude_sites))

    if sites is not None:
        df = df.filter(pl.col("site").is_in(sites))

    # Filter by years
    if include_years is not None:
        df = df.filter(pl.col("harvest_year").is_in(include_years))
    if exclude_year is not None:
        df = df.filter(pl.col("harvest_year") != exclude_year)

    if exclude_year_sites is not None:
        exclude_conditions = []
        for site, years in exclude_year_sites.items():
            site_year_condition = (pl.col("site") == site) & (
                pl.col("harvest_year").is_in(years)
            )
            exclude_conditions.append(site_year_condition)

        if exclude_conditions:
            combined_exclude_condition = pl.any_horizontal(exclude_conditions)
            df = df.filter(~combined_exclude_condition)

    df = df.filter(
        ~(
            (pl.col("timestamp").dt.year() != pl.col("harvest_year"))
            & (
                pl.col("timestamp").dt.date()
                <= pl.col("timestamp").dt.replace(month=10, day=31)
            )
        )
        & ~(
            (pl.col("timestamp").dt.year() == pl.col("harvest_year"))
            & (
                pl.col("timestamp").dt.date()
                > pl.when(pl.col("timestamp").dt.is_leap_year())
                .then(pl.col("timestamp").dt.replace(month=7, day=31))
                .otherwise(pl.col("timestamp").dt.replace(month=8, day=1))
            )
        )
    )

    df = df.with_columns(
        pl.col("timestamp").dt.date().alias("temperature_dates"),
        pl.col("timestamp").dt.time().alias("temperature_times"),
    )

    df = (
        df.sort(["site", "harvest_year", "temperature_dates", "temperature_times"])
        .group_by(["site", "harvest_year"], maintain_order=True)
        .agg(
            pl.col("Temp").alias("temperature_values"),
            pl.col("temperature_dates"),
            pl.col("temperature_times"),
            pl.col("temperature_times").len().alias("num_times"),
        )
        .with_columns(
            pl.concat_str(
                [pl.col("site"), pl.col("harvest_year").cast(pl.Utf8)], separator="_"
            ).alias("yearsite_uid")
        )
    )

    df = df.filter(pl.col("num_times") == 274 * 24).drop(["num_times"])

    df = df.collect()

    return datasets.Dataset.from_polars(df).with_format("numpy")
