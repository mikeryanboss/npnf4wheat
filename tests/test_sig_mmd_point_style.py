from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from npnf.scripts.paper import sig_mmd_point_style as style


class RecordingAxes:
    def __init__(self) -> None:
        self.vlines_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.hlines_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.scatter_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def vlines(self, *args: object, **kwargs: object) -> None:
        self.vlines_calls.append((args, kwargs))

    def hlines(self, *args: object, **kwargs: object) -> None:
        self.hlines_calls.append((args, kwargs))

    def scatter(self, *args: object, **kwargs: object) -> None:
        self.scatter_calls.append((args, kwargs))


def test_draw_centered_summary_bars_draws_mean_median_iqr_only() -> None:
    ax = RecordingAxes()
    values = pd.Series([0.0, 1.0, 2.0, 100.0])
    plot_df = pd.DataFrame({"category": ["A"] * len(values), "score": values})

    style.draw_centered_summary_bars(
        ax,  # ty: ignore[invalid-argument-type]
        plot_df,
        x_col="category",
        y_col="score",
        x_order=["A"],
    )

    q25 = values.quantile(0.25)
    q75 = values.quantile(0.75)
    median = values.median()
    mean = values.mean()

    assert len(ax.vlines_calls) == 1
    vline_args, _ = ax.vlines_calls[0]
    assert vline_args[0] == 0
    assert vline_args[1] == pytest.approx(q25)
    assert vline_args[2] == pytest.approx(q75)

    assert len(ax.hlines_calls) == 2
    iqr_cap_args, _ = ax.hlines_calls[0]
    assert iqr_cap_args[0] == [pytest.approx(q25), pytest.approx(q75)]

    median_args, _ = ax.hlines_calls[1]
    assert median_args[0] == pytest.approx(median)

    assert len(ax.scatter_calls) == 1
    scatter_args, _ = ax.scatter_calls[0]
    assert scatter_args[0] == 0
    assert scatter_args[1] == pytest.approx(mean)


def test_swapped_fill_orders_and_styles_raw_conditioning_keys() -> None:
    _, _, condition_order = style.swapped_fill_orders(
        split_values=["seen"],
        method_values=["no context"],
        condition_values=["env_geno", "noenv_geno", "noenv_nogeno", "env_nogeno"],
    )

    assert condition_order == ["noenv_nogeno", "env_nogeno", "noenv_geno", "env_geno"]
    assert style.condition_fillstyles(condition_order) == {
        "noenv_nogeno": "none",
        "env_nogeno": "left",
        "noenv_geno": "right",
        "env_geno": "full",
    }


def test_swapped_fill_orders_and_styles_legacy_conditioning_labels() -> None:
    _, _, condition_order = style.swapped_fill_orders(
        split_values=["seen"],
        method_values=["no context"],
        condition_values=["E&G", "G", "P", "E"],
    )

    assert condition_order == ["P", "E", "G", "E&G"]
    assert style.condition_fillstyles(condition_order) == {
        "P": "none",
        "E": "left",
        "G": "right",
        "E&G": "full",
    }


def test_swapped_fill_handles_use_condition_display_labels() -> None:
    labels = [
        handle.get_label()
        for handle in style.swapped_fill_handles(
            method_order=["no context"],
            split_order=["seen"],
            condition_order=["noenv_nogeno", "env_nogeno", "noenv_geno", "env_geno"],
            condition_labels={
                "noenv_nogeno": "$\\emptyset$",
                "env_nogeno": "$e$",
                "noenv_geno": "$g$",
                "env_geno": "$g+e$",
            },
        )
    ]

    assert labels[2:6] == ["$\\emptyset$", "$e$", "$g$", "$g+e$"]


def test_swapped_fill_summary_handles_are_mean_median_iqr() -> None:
    labels = [
        handle.get_label()
        for handle in style.swapped_fill_handles(
            method_order=["no context"], split_order=["seen"], condition_order=["E"]
        )
    ]

    assert labels[-3:] == ["Mean", "Median", "IQR"]
    assert "5–95%" not in labels
    assert "25–75%" not in labels


def test_yerr_col_draws_error_bars_aligned_with_points():
    """Bars must be drawn inside the helper: x positions are RNG-jittered."""
    import matplotlib.pyplot as plt
    import pandas as pd

    from npnf.scripts.paper import sig_mmd_point_style as style

    frame = pd.DataFrame(
        {
            "model_name": ["LNP", "LNP"],
            "value": [1.0, 2.0],
            "split_label": ["seen", "seen"],
            "prediction_method_label": ["no context", "no context"],
            "conditioning_label": ["a", "a"],
            "sd": [0.1, 0.0],
        }
    )
    kwargs: dict[str, Any] = {
        "x_col": "model_name",
        "y_col": "value",
        "x_order": ["LNP"],
        "split_order": ["seen"],
        "method_order": ["no context"],
        "condition_order": ["a"],
    }

    _, ax_plain = plt.subplots()
    style.draw_swapped_fill_points(ax_plain, frame, **kwargs)
    baseline = len(ax_plain.containers)

    _, ax_bars = plt.subplots()
    style.draw_swapped_fill_points(ax_bars, frame, **kwargs, yerr_col="sd")
    # only the non-zero sd row gets a bar
    assert len(ax_bars.containers) == baseline + 1
    plt.close("all")


def test_error_bar_lower_end_is_clipped_at_zero():
    """A replicate sd can exceed the mean; a negative Sig-MMD is meaningless."""
    import matplotlib.pyplot as plt
    import pandas as pd

    from npnf.scripts.paper import sig_mmd_point_style as style

    frame = pd.DataFrame(
        {
            "model_name": ["LNP"],
            "value": [1.0],
            "split_label": ["seen"],
            "prediction_method_label": ["no context"],
            "conditioning_label": ["a"],
            "sd": [5.0],
        }
    )
    _, ax = plt.subplots()
    style.draw_swapped_fill_points(
        ax,
        frame,
        x_col="model_name",
        y_col="value",
        x_order=["LNP"],
        split_order=["seen"],
        method_order=["no context"],
        condition_order=["a"],
        yerr_col="sd",
    )
    segments = ax.containers[0][2][0].get_segments()[0]
    assert segments[:, 1].min() == pytest.approx(0.0)
    assert segments[:, 1].max() == pytest.approx(6.0)
    plt.close("all")


def _mean_markers(ax):
    """Summary mean markers, excluding the vline/hline collections."""
    from matplotlib.collections import PathCollection

    return [c for c in ax.collections if isinstance(c, PathCollection)]


def _summary_frame():
    import pandas as pd

    return pd.DataFrame(
        {
            "model_name": ["LNP"] * 4,
            "value": [1.0, 2.0, 10.0, 20.0],
            "split_label": ["seen", "seen", "unseen", "unseen"],
        }
    )


def test_summary_bars_pool_splits_by_default():
    import matplotlib.pyplot as plt

    from npnf.scripts.paper import sig_mmd_point_style as style

    _, ax = plt.subplots()
    style.draw_centered_summary_bars(
        ax, _summary_frame(), x_col="model_name", y_col="value", x_order=["LNP"]
    )
    # one mean marker for the pooled bar (the rest are vline/hline collections)
    assert len(_mean_markers(ax)) == 1
    plt.close("all")


def test_summary_bars_split_by_test_set_when_requested():
    """Pooling an easy and a hard split makes the IQR describe split difficulty."""
    import matplotlib.pyplot as plt

    from npnf.scripts.paper import sig_mmd_point_style as style

    _, ax = plt.subplots()
    style.draw_centered_summary_bars(
        ax,
        _summary_frame(),
        x_col="model_name",
        y_col="value",
        x_order=["LNP"],
        split_col="split_label",
        split_order=["seen", "unseen"],
    )
    markers = _mean_markers(ax)
    assert len(markers) == 2
    means = sorted(float(c.get_offsets()[0][1]) for c in markers)
    assert means == pytest.approx([1.5, 15.0])
    # the two bars are drawn at different x so they do not overlap
    xs = sorted(float(c.get_offsets()[0][0]) for c in markers)
    assert xs[0] < xs[1]
    plt.close("all")


def test_summary_split_offsets_are_symmetric():
    from npnf.scripts.paper import sig_mmd_point_style as style

    assert style.summary_split_offsets(1) == [0.0]
    offsets = style.summary_split_offsets(2)
    assert offsets[0] == pytest.approx(-offsets[1])


def test_seed_spread_summary_shows_cell_sd_and_typical_seed_sd():
    """The replicate glyph separates spread over cells from spread over seeds."""
    import matplotlib.pyplot as plt
    import pandas as pd

    from npnf.scripts.paper import sig_mmd_point_style as style

    plot_df = pd.DataFrame(
        {
            "model_name": ["LNP"] * 4,
            "value": [1.0, 2.0, 10.0, 20.0],
            "seed_sd": [0.1, 0.1, 0.5, 0.7],
        }
    )
    _, ax = plt.subplots()
    style.draw_centered_summary_bars(
        ax,
        plot_df,
        x_col="model_name",
        y_col="value",
        x_order=["LNP"],
        seed_sd_col="seed_sd",
    )
    mean = plot_df["value"].mean()
    cell_sd = plot_df["value"].std(ddof=1)
    seed_sd = ((plot_df["seed_sd"] ** 2).mean()) ** 0.5
    from matplotlib.collections import LineCollection

    thin, thick = (
        c
        for c in ax.collections
        if isinstance(c, LineCollection) and len(c.get_segments()) == 1
    )
    assert thin.get_segments()[0][:, 1] == pytest.approx(
        [mean - cell_sd, mean + cell_sd]
    )
    assert thick.get_segments()[0][:, 1] == pytest.approx(
        [mean - seed_sd, mean + seed_sd]
    )
    assert [float(c.get_offsets()[0][1]) for c in _mean_markers(ax)] == pytest.approx(
        [mean]
    )
    labels = [
        h.get_label()
        for h in style.swapped_fill_handles(
            ["no context"], ["seen"], ["noenv_nogeno"], seed_spread=True
        )
    ]
    assert labels[-3:] == ["Mean", "± sd over seeds", "± sd over settings"]
    plt.close("all")


def test_geometric_seed_spread_uses_factors_around_the_geometric_mean():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.collections import LineCollection

    from npnf.scripts.paper import sig_mmd_point_style as style

    plot_df = pd.DataFrame(
        {
            "model_name": ["LNP"] * 3,
            "value": [0.5, 1.0, 2.0],
            "seed_sd": [np.log(2.0)] * 3,
        }
    )
    _, ax = plt.subplots()
    style.draw_centered_summary_bars(
        ax,
        plot_df,
        x_col="model_name",
        y_col="value",
        x_order=["LNP"],
        seed_sd_col="seed_sd",
        geometric=True,
    )
    thin, thick = (
        c
        for c in ax.collections
        if isinstance(c, LineCollection) and len(c.get_segments()) == 1
    )
    # log values -ln2, 0, ln2: geometric mean 1, geometric sd over cells 2
    assert thin.get_segments()[0][:, 1] == pytest.approx([0.5, 2.0])
    assert thick.get_segments()[0][:, 1] == pytest.approx([0.5, 2.0])
    assert [float(c.get_offsets()[0][1]) for c in _mean_markers(ax)] == pytest.approx(
        [1.0]
    )
    labels = [
        h.get_label()
        for h in style.swapped_fill_handles(
            ["no context"], ["seen"], ["noenv_nogeno"], seed_spread=True, geometric=True
        )
    ]
    assert labels[-3:] == [
        "Geometric mean",
        "Geometric sd over seeds",
        "Geometric sd over settings",
    ]
    plt.close("all")


def test_geometric_point_bars_are_factors():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.collections import LineCollection

    from npnf.scripts.paper import sig_mmd_point_style as style

    plot_df = pd.DataFrame(
        {
            "model_name": ["LNP"],
            "value": [1.0],
            "sd": [np.log(4.0)],
            "split_label": ["seen"],
            "prediction_method_label": ["no context"],
            "conditioning_label": ["none"],
        }
    )
    _, ax = plt.subplots()
    style.draw_swapped_fill_points(
        ax,
        plot_df,
        x_col="model_name",
        y_col="value",
        x_order=["LNP"],
        split_order=["seen"],
        method_order=["no context"],
        condition_order=["none"],
        yerr_col="sd",
        geometric=True,
    )
    (bar,) = [
        c
        for c in ax.collections
        if isinstance(c, LineCollection) and len(c.get_segments()) == 1
    ]
    assert bar.get_segments()[0][:, 1] == pytest.approx([0.25, 4.0])
    plt.close("all")
