"""Present primary simulator calibration and real--real reference tables."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections.abc import Sequence
from pathlib import Path


def write_latex_table(
    path: Path,
    alignment: str,
    header: str,
    rows: Sequence[str],
    note: str | None = None,
) -> None:
    """Write formatted rows without changing table-specific values or precision."""
    lines = [
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        header,
        r"\hline",
        *rows,
        r"\hline",
        r"\end{tabular}",
    ]
    if note is not None:
        lines.extend([r"\par\smallskip", rf"\noindent {note}"])
    path.write_text("\n".join(lines) + "\n")


def write_comparison_tables(
    noise_dir: Path, reference_dir: Path, output_dir: Path
) -> None:
    """Write the two primary tables and their underlying analysis outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_simulator_comparison_table(noise_dir, output_dir)
    write_year_reference_table(reference_dir, output_dir)


def write_simulator_comparison_table(noise_dir: Path, output_dir: Path) -> None:
    """Present the combined complete-grid lodging-plus-noise calibration fit."""
    with (noise_dir / "combined_summary.csv").open() as f:
        rows = sorted(csv.DictReader(f), key=lambda row: int(row["year"]))
    with (noise_dir / "combined_settings.json").open() as f:
        settings = json.load(f)
    n_draws = int(rows[0]["n_draws"])
    scale = float(settings["weibull_scale"])
    sigma_cm = float(settings["sigma_m"]) * 100
    counts = ", ".join(f"{row['year']}: {row['n_real']}" for row in rows)
    selection = settings["population_selection"]
    exclusions = "; ".join(
        f"{year}: {item['n_retained']} of {item['n_available']} plots retained"
        for year, item in sorted(selection.items())
        if item["n_retained"] != item["n_available"]
    )
    table_rows = []
    for row in rows:
        year = int(row["year"])
        table_rows.append(
            f"{year} & ${float(row['mean']) * 1000:.3f} "
            f"\\pm {float(row['sd']) * 1000:.3f}$ "
            r"\\"
        )
    write_latex_table(
        output_dir / "combined_table.tex",
        "rr",
        r"Year & Simulator--FIP1 Sig-MMD$^2\times 10^3$ \\",
        table_rows,
        note=(
            r"Observed plots with complete grid measurements versus equally "
            f"many simulated trajectories (counts by year: {counts}). "
            + (f"Missing grid measurements: {exclusions}. " if exclusions else "")
            + r"Initial-offset years have observed height above the model at "
            r"the start of the window; this does not imply a constant offset "
            r"or an established cause. "
            r"Scores are mean $\pm$ sample SD across "
            f"{n_draws} independent paired lodging/noise draws "
            r"(denominator $n_{\mathrm{draws}}-1$). "
            f"Pooled Weibull scale: {scale:.4g}, fitted by matching the mean "
            r"Weibull lodging probability at retained plots' full-schedule "
            r"maximum observed heights to the observed detected lodging fraction, "
            r"using the fixed simulator height clamp and offset, without "
            r"detection correction. Gaussian noise "
            f"$\\sigma = {sigma_cm:.2f}$ cm (pooled RMS effective variation, "
            r"not pure sensor noise). Both pooled estimates use all comparison years. "
            f"All years use {rows[0]['n_shared_days']} shared anchors, "
            r"retaining measured heights within three days and sampling "
            r"simulated curves at the anchors. "
            r"SD describes simulation variability, not parameter uncertainty "
            r"or confidence intervals. Parameters use the same "
            r"observations: this is not independent validation or absolute "
            r"goodness of fit."
        ),
    )
    for filename in (
        "combined_runs.csv",
        "combined_summary.csv",
        "combined_settings.json",
    ):
        shutil.copyfile(noise_dir / filename, output_dir / filename)


def write_year_reference_table(reference_dir: Path, output_dir: Path) -> None:
    """Present the symmetric real--real year-reference matrix."""
    with (reference_dir / "reference_summary.csv").open() as f:
        rows = sorted(csv.DictReader(f), key=lambda row: int(row["year"]))
    n_splits = int(rows[0]["n_splits"])
    n_dates = int(rows[0]["n_shared_days"])
    n_min = min(int(row[key]) for row in rows for key in ("n_a_min", "n_b_min"))
    n_max = max(int(row[key]) for row in rows for key in ("n_a_max", "n_b_max"))
    with (reference_dir / "year_reference.csv").open() as f:
        year_rows = list(csv.DictReader(f))
    years = sorted({int(row[key]) for row in year_rows for key in ("year_a", "year_b")})
    diagonal = {
        int(row["year"]): (
            f"${float(row['real_mean']) * 1000:.3f} "
            f"\\pm {float(row['real_sd']) * 1000:.3f}$"
        )
        for row in rows
    }
    counts = ", ".join(f"{row['year']}: {row['n_real']}" for row in rows)
    indexed = {
        tuple(sorted((int(row["year_a"]), int(row["year_b"])))): float(
            row["sig_mmd_squared"]
        )
        for row in year_rows
    }
    table_rows = []
    for year_a in years:
        values = [
            diagonal[year_a]
            if year_a == year_b
            else f"{indexed[min(year_a, year_b), max(year_a, year_b)] * 1000:.3f}"
            for year_b in years
        ]
        table_rows.append(f"{year_a} & " + " & ".join(values) + r" \\")
    write_latex_table(
        output_dir / "year_reference_table.tex",
        "r" * (len(years) + 1),
        "Year & " + " & ".join(str(year) for year in years) + r" \\",
        table_rows,
        note=(
            r"Real--real Sig-MMD$^2 \times 1000$. Diagonal entries are "
            f"means $\\pm$ sample SD across {n_splits} within-year splits "
            r"of whole genotype groups, not comparisons of a population "
            r"with itself. "
            f"Split groups contain {n_min}--{n_max} plots. "
            r"Off-diagonal entries are single scores using all plots with "
            r"complete grid measurements for "
            f"the {len(year_rows)} unordered year pairs, displayed symmetrically. "
            f"Complete-grid plot counts by year: {counts}. "
            r"They have no SD because no repeated sampling is used. "
            f"All entries use the same {n_dates} anchor dates. "
            r"Diagonal SDs describe split variation, not confidence intervals "
            r"or total uncertainty; diagonal and off-diagonal sample sizes "
            r"differ. Year differences include genotype composition, weather "
            r"and measurement effects, not pure environmental effects. "
            r"These restricted-window references are not acceptability bounds."
        ),
    )
    for filename in (
        "reference_runs.csv",
        "reference_summary.csv",
        "year_reference.csv",
    ):
        shutil.copyfile(reference_dir / filename, output_dir / filename)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--noise-dir",
        type=Path,
        required=True,
        help="Saved combined lodging-plus-noise analysis.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Saved real--real reference comparisons.",
    )
    args = parser.parse_args()
    write_comparison_tables(args.noise_dir, args.reference_dir, args.output_dir)


if __name__ == "__main__":
    main()
