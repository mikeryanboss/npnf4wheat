from pathlib import Path

import numpy as np
import torch

from npnf.scripts.paper import lodging_absolute_drop as lad


def test_calculate_binned_drop_means_masks_low_lodged_counts() -> None:
    means, total_counts, lodged_counts = lad.calculate_binned_drop_means(
        values=np.array([0.7, 0.72, 0.85, 0.86]),
        drop_abs=np.array([0.1, 0.3, 0.5, 0.7]),
        lodged_mask=np.array([True, False, True, True]),
        bins=np.array([0.6, 0.8, 1.0]),
        min_lodged_count=2,
    )

    np.testing.assert_array_equal(total_counts, np.array([2, 2]))
    np.testing.assert_array_equal(lodged_counts, np.array([1, 2]))
    assert np.isnan(means[0])
    np.testing.assert_allclose(means[1], 0.6)


def test_load_ground_truth_drops_uses_flag_and_draw_zero(monkeypatch) -> None:
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

    monkeypatch.setattr(lad, "BatchLoader", FakeLoader)

    ground_truth = lad._load_ground_truth_drops(Path("unused"))  # noqa: SLF001

    assert ground_truth.labels.tolist() == [True, False]
    np.testing.assert_allclose(ground_truth.max_heights, np.array([4.0, 0.8]))
    np.testing.assert_allclose(ground_truth.drop_abs, np.array([2.0, 0.0]))
