from pathlib import Path

import numpy as np
import torch

from npnf.scripts.paper import lodging_probability as lp


def test_load_ground_truth_projects_synthetic_oracle_metadata_to_draw_zero(
    monkeypatch,
) -> None:
    batch = {
        "data": {
            "has_lodged": torch.tensor(
                [[True, False, False], [False, True, True]], dtype=torch.bool
            ),
            "height": {
                "Y_original": torch.tensor(
                    [
                        [
                            [[1.0], [4.0], [2.0]],
                            [[9.0], [9.0], [9.0]],
                            [[8.0], [8.0], [8.0]],
                        ],
                        [
                            [[0.5], [0.2], [0.8]],
                            [[7.0], [7.0], [7.0]],
                            [[6.0], [6.0], [6.0]],
                        ],
                    ],
                    dtype=torch.float32,
                )
            },
        }
    }

    class FakeLoader:
        def __init__(self, method_dir: Path, *, load_predictions: bool = False):
            assert method_dir == Path("unused")
            assert load_predictions is False

        def __iter__(self):
            yield batch

    monkeypatch.setattr(lp, "BatchLoader", FakeLoader)

    ground_truth = lp._load_ground_truth(Path("unused"))  # noqa: SLF001

    assert ground_truth.labels.tolist() == [True, False]
    np.testing.assert_allclose(ground_truth.max_heights, np.array([4.0, 0.8]))
