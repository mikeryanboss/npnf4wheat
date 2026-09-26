import numpy as np
import pandas as pd
import pytest
import torch

from npnf.scripts.paper.fip1_final_height import (
    final_height,
    genotype_year_means,
    run_identity,
    score,
)


def test_final_height_ignores_one_outlier_and_a_later_drop():
    curve = torch.tensor([0.1, 0.5, 0.9, 1.0, 1.0, 1.0, 1.6, 0.4, 0.4])
    assert final_height(curve).item() == pytest.approx(1.0)


def test_final_height_is_per_draw():
    curves = torch.tensor([[0.2, 0.8, 0.8, 0.8], [0.1, 0.5, 0.6, 0.7]])
    assert final_height(curves).tolist() == pytest.approx([0.8, 0.6])


def test_genotype_year_means_average_plots_and_keep_the_blue():
    plots = pd.DataFrame(
        {
            "genotype_id": [1, 1, 1, 2],
            "harvest_year": [2016, 2016, 2017, 2016],
            "predicted": [0.8, 1.0, 1.2, 0.7],
            "height_final_blue": [0.95, 0.95, 1.1, 0.75],
        }
    )
    means = genotype_year_means(plots).set_index(["genotype_id", "harvest_year"])
    assert means.loc[(1, 2016), "predicted"] == pytest.approx(0.9)
    assert means.loc[(1, 2016), "observed"] == pytest.approx(0.95)
    assert len(means) == 3


def test_score_averages_within_year_scores():
    units = pd.DataFrame(
        {
            "harvest_year": [2016] * 3 + [2017] * 3,
            "predicted": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "observed": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        }
    )
    r, rmse = score(units, correlation=True)
    assert r == pytest.approx(0.0)  # +1 in 2016, -1 in 2017
    assert rmse == pytest.approx((0.0 + np.sqrt(8 / 3)) / 2)
    r, _ = score(units, correlation=False)
    assert np.isnan(r)


def test_run_identity():
    assert run_identity("ANP-NF-Prior-FIP-2930-fip1_pretrained_1m_seed13") == (
        "ANP-NF-Prior",
        "pretrained",
        13,
    )
    assert run_identity("LodgingMixtureSpline-FIP-2930") == (
        "LodgingMixtureSpline-FIP-2930",
        "fit",
        0,
    )
