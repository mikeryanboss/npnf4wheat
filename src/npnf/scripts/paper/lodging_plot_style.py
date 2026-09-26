"""Semantic lodging-figure styling helpers for paper plots."""

from dataclasses import dataclass

import matplotlib.lines as mlines

from npnf.scripts.paper.style import COLORS

ROW_ORDER = ("Latent", "Latent + deterministic")
VARIANT_ORDER = (
    "Deterministic",
    "Gaussian",
    "NF-Prior",
    "NF-Posterior",
    "NF-Prior-Posterior",
)
VARIANT_COLORS = {
    "Deterministic": COLORS["blue"],
    "Gaussian": COLORS["cyan"],
    "NF-Prior": COLORS["orange"],
    "NF-Posterior": COLORS["green"],
    "NF-Prior-Posterior": COLORS["purple"],
}
SEMANTIC_GRID_SUPYLABEL_X = 0.055
SEMANTIC_GRID_BOTTOM = 0.135
SEMANTIC_GRID_XLABEL_Y = 0.095
SEMANTIC_GRID_LEGEND_Y = 0.035


@dataclass(frozen=True)
class BottomLegendLayout:
    """Shared bottom-legend placement for dual-panel lodging figures."""

    legend_bottom: float
    x_label_y: float
    legend_y: float
    ncol: int


@dataclass(frozen=True)
class ModelSemantic:
    """Semantic row and variant for a known paper model name."""

    row: str
    variant: str


MODEL_SEMANTICS = {
    "CNP": ModelSemantic(row="Latent", variant="Deterministic"),
    "LNP": ModelSemantic(row="Latent", variant="Gaussian"),
    "LNP-NF-Prior": ModelSemantic(row="Latent", variant="NF-Prior"),
    "LNP-NF-Posterior": ModelSemantic(row="Latent", variant="NF-Posterior"),
    "LNP-NF-Prior-Posterior": ModelSemantic(row="Latent", variant="NF-Prior-Posterior"),
    "ACNP": ModelSemantic(row="Latent + deterministic", variant="Deterministic"),
    "ANP": ModelSemantic(row="Latent + deterministic", variant="Gaussian"),
    "ANP-NF-Prior": ModelSemantic(row="Latent + deterministic", variant="NF-Prior"),
    "ANP-NF-Posterior": ModelSemantic(
        row="Latent + deterministic", variant="NF-Posterior"
    ),
    "ANP-NF-Prior-Posterior": ModelSemantic(
        row="Latent + deterministic", variant="NF-Prior-Posterior"
    ),
}


def semantic_model_groups(model_order: list[str]) -> dict[str, dict[str, str]] | None:
    """Return row -> variant -> model mapping for known paper model names.

    Returns None when any supplied model name is unknown so callers can fall
    back to generic one-color-per-model plotting.
    """
    if any(name not in MODEL_SEMANTICS for name in model_order):
        return None

    groups: dict[str, dict[str, str]] = {row: {} for row in ROW_ORDER}
    for name in model_order:
        semantic = MODEL_SEMANTICS[name]
        groups[semantic.row][semantic.variant] = name
    return groups


def bottom_legend_layout(
    *, handle_count: int, model_label_count: int
) -> BottomLegendLayout:
    """Return compact bottom-legend placement for generic lodging plots."""
    if model_label_count > 5:
        return BottomLegendLayout(
            legend_bottom=0.215,
            x_label_y=0.125,
            legend_y=0.095,
            ncol=(handle_count + 1) // 2,
        )
    return BottomLegendLayout(
        legend_bottom=0.165, x_label_y=0.095, legend_y=0.070, ncol=handle_count
    )


def variant_legend_handles(include_ground_truth: bool = True) -> list[mlines.Line2D]:
    """Build legend handles for lodging variant colors."""
    handles = [
        mlines.Line2D(
            [],
            [],
            color=VARIANT_COLORS[variant],
            linestyle="-",
            linewidth=2.5,
            label=variant,
        )
        for variant in VARIANT_ORDER
    ]
    if include_ground_truth:
        handles.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                linestyle="-",
                linewidth=2.5,
                label="Ground truth",
            )
        )
    return handles
