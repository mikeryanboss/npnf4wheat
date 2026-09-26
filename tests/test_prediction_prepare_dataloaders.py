from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from npnf.scripts.utils import prediction


class NoDeepcopyDataloader:
    def __deepcopy__(self, memo):
        msg = "dataloader was deep-copied"
        raise AssertionError(msg)


@dataclass
class SingleSplitDataloaders:
    test_plot: NoDeepcopyDataloader


@dataclass
class MultiSplitDataloaders:
    test_plot: NoDeepcopyDataloader
    test_unseen: NoDeepcopyDataloader


class FakeModel:
    def __init__(self):
        self.loaded_state_dict = None
        self.eval_called = False

    def load_state_dict(self, state_dict):
        self.loaded_state_dict = state_dict

    def eval(self):
        self.eval_called = True


class FakeAccelerator:
    def __init__(self):
        self.prepare_calls = []

    def prepare(self, *objects):
        self.prepare_calls.append(objects)
        if len(objects) == 1:
            return objects[0]
        return objects


def test_shallow_dataloaders_dict_preserves_multi_split_names_without_deepcopy():
    test_plot = NoDeepcopyDataloader()
    test_unseen = NoDeepcopyDataloader()

    result = prediction.dataloaders_to_shallow_dict(
        MultiSplitDataloaders(test_plot=test_plot, test_unseen=test_unseen)
    )

    assert result == {"test_plot": test_plot, "test_unseen": test_unseen}


def test_prepare_model_and_dataloaders_does_not_deepcopy_dataloaders(
    monkeypatch, tmp_path
):
    checkpoint_dir = tmp_path / "checkpoint-1"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").touch()
    dataloader = NoDeepcopyDataloader()
    model = FakeModel()
    accelerator = FakeAccelerator()

    monkeypatch.setattr(prediction, "resolve_checkpoint", Path)

    def fake_load_file(path):
        assert path == str(checkpoint_dir / "model.safetensors")
        return {"weight": torch.tensor(1.0)}

    monkeypatch.setattr("safetensors.torch.load_file", fake_load_file)

    prepared_model, dataloaders, resolved = prediction.prepare_model_and_dataloaders(
        model=model,
        dataloaders=SingleSplitDataloaders(test_plot=dataloader),
        checkpoint_folder=str(checkpoint_dir),
        accelerator=accelerator,  # ty: ignore[invalid-argument-type]
    )

    assert prepared_model is model
    assert dataloaders == {"test_plot": dataloader}
    assert resolved == str(checkpoint_dir)
    assert torch.equal(model.loaded_state_dict["weight"], torch.tensor(1.0))
    assert model.eval_called
    assert accelerator.prepare_calls == [(model,), (dataloader,)]
