# ruff: noqa: SLF001

import types
from typing import Any

import pytest
import torch

import npnf.data.datasets.synthetic as synthetic_mod
from npnf.calibration.height.constants import HeightDates
from npnf.data.datasets.synthetic import SyntheticDataset
from npnf.data.synthetic.height.genotype import GenotypeParams
from tests.test_synthetic_dataset_cache_key import _make_proxy


def _make_generation_inputs() -> tuple[Any, Any, Any]:
    proxy = _make_proxy(genotype_indices=[0, 1], yearsite_indices=[0])
    proxy.num_days = 274
    proxy.dates = HeightDates()
    num_params = len(GenotypeParams.field_names())
    params = torch.rand(2, num_params)
    params[:, -1] = 600.0
    genotype_pool = types.SimpleNamespace(params=params, markers=torch.zeros(2, 31))
    yearsite_pool = types.SimpleNamespace(temperatures=torch.full((1, 274, 24), 12.0))
    return proxy, genotype_pool, yearsite_pool


@pytest.fixture
def restore_matmul_precision():
    precision = torch.backends.cuda.matmul.fp32_precision
    yield
    torch.backends.cuda.matmul.fp32_precision = precision


def test_growth_runs_with_tf32_and_restores_precision(
    monkeypatch, restore_matmul_precision
) -> None:
    precisions: list[str] = []

    def record_growth(**kwargs) -> torch.Tensor:
        precisions.append(torch.backends.cuda.matmul.fp32_precision)
        return torch.zeros(kwargs["yearsite_indices"].shape[0], kwargs["num_days"])

    monkeypatch.setattr(synthetic_mod, "_compute_growth_chunked", record_growth)
    torch.backends.cuda.matmul.fp32_precision = "ieee"

    SyntheticDataset._compute_intermediate(*_make_generation_inputs())

    assert precisions == ["tf32"]
    assert torch.backends.cuda.matmul.fp32_precision == "ieee"


def test_precision_is_restored_when_growth_fails(
    monkeypatch, restore_matmul_precision
) -> None:
    def fail_growth(**kwargs) -> torch.Tensor:
        raise RuntimeError

    monkeypatch.setattr(synthetic_mod, "_compute_growth_chunked", fail_growth)
    torch.backends.cuda.matmul.fp32_precision = "ieee"

    with pytest.raises(RuntimeError):
        SyntheticDataset._compute_intermediate(*_make_generation_inputs())

    assert torch.backends.cuda.matmul.fp32_precision == "ieee"
