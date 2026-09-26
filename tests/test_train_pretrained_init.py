from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from npnf.scripts.utils.training import load_pretrained_weights


class _Tiny(nn.Module):
    def __init__(self, marker_dim: int) -> None:
        super().__init__()
        self.shared = nn.Linear(4, 4, bias=False)
        self.markers = nn.Linear(marker_dim, 4, bias=False)


def _checkpoint(tmp_path: Path, model: nn.Module) -> Path:
    folder = tmp_path / "checkpoints" / "checkpoint-10"
    folder.mkdir(parents=True)
    save_file(model.state_dict(), str(folder / "model.safetensors"))
    return folder


def test_load_pretrained_weights_copies_matching_and_skips_mismatched(tmp_path):
    torch.manual_seed(0)
    source = _Tiny(marker_dim=3)
    source_checkpoint = _checkpoint(tmp_path, source)

    target = _Tiny(marker_dim=5)
    marker_before = target.markers.weight.detach().clone()

    skipped = load_pretrained_weights(target, str(source_checkpoint))

    assert skipped == ["markers.weight"]
    assert torch.equal(target.shared.weight, source.shared.weight)
    assert torch.equal(target.markers.weight, marker_before)


def test_load_pretrained_weights_resolves_latest_checkpoint(tmp_path):
    source = _Tiny(marker_dim=3)
    _checkpoint(tmp_path, source)
    target = _Tiny(marker_dim=3)

    skipped = load_pretrained_weights(target, str(tmp_path / "checkpoints"))

    assert skipped == []
    assert torch.equal(target.markers.weight, source.markers.weight)
