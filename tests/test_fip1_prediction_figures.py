import numpy as np

from npnf.scripts.paper.fip1_prediction_figures import default_plot


def test_default_plot_is_nearest_the_90th_height_percentile():
    observed = {
        f"plot-{index}": {"heights": np.array([0.0, height])}
        for index, height in enumerate(np.linspace(0.5, 1.5, 11))
    }
    # The 90th percentile of 0.5, 0.6, ..., 1.5 is 1.4.
    assert default_plot(observed) == "plot-9"
