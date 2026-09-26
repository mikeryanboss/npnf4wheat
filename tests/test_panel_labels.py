from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

from npnf.utils import add_panel_labels


def test_add_panel_labels_writes_letters_in_row_order() -> None:
    fig, axes = plt.subplots(nrows=2, ncols=4, squeeze=False)

    add_panel_labels(axes.flat)

    labels = [[text.get_text() for text in panel.texts] for panel in axes.flat]
    assert labels == [["A"], ["B"], ["C"], ["D"], ["E"], ["F"], ["G"], ["H"]]
    plt.close(fig)
