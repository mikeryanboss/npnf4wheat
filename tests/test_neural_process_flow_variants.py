import hashlib
import os
from collections import OrderedDict
from pathlib import Path

import pytest
import tensordict
import torch

from npnf.models.configs.models import (
    SYNTH_MARKERS_HIDDEN_CHANNELS,
    SYNTH_MARKERS_INPUT_DIM,
)
from npnf.models.neural_process import (
    ANP,
    ANP_NF_Posterior,
    ANP_NF_Prior,
    ANP_NF_Prior_Posterior,
    LNP_NF_Posterior,
    LNP_NF_Prior,
    LNP_NF_Prior_Posterior,
)
from npnf.models.utils import consolidate_tensordict_jagged_dim

MODEL_KWARGS = {
    "hidden_dim": 16,
    "latent_dim": 4,
    "num_transformer_layers": 1,
    "num_transformer_heads": 1,
    "markers_input_dim": 2,
    "markers_hidden_channels": [4],
}

ANP_FLOW_STATE_KEY_BASELINES = {
    ANP_NF_Posterior: (
        69,
        "e3a152ce719283c0061300dac2418220fa1cf2535c86929ef5e01b203dfe2dc1",
    ),
    ANP_NF_Prior: (
        73,
        "ef08fc5a8865c29176805baa784412f46f9f239eb2e3e73a6a31b0b87a2d7250",
    ),
    ANP_NF_Prior_Posterior: (
        109,
        "c4a54d2ce8911ce3e0dec2ffd276fc27f8cba5b4e69bd01fa6e33459fd89a034",
    ),
}

ANP_FLOW_CLASSES = tuple(ANP_FLOW_STATE_KEY_BASELINES)
LNP_FLOW_CLASSES = (LNP_NF_Posterior, LNP_NF_Prior, LNP_NF_Prior_Posterior)


def _make_points(lengths: list[int]) -> tensordict.TensorDict:
    samples = []
    for length in lengths:
        x = torch.linspace(0.0, 1.0, length).unsqueeze(-1)
        samples.append(
            tensordict.TensorDict(
                {"X": x, "X_normalized": x.clone(), "Y": torch.sin(x)}, device="cpu"
            )
        )
    return consolidate_tensordict_jagged_dim(
        tensordict.lazy_stack(samples).densify(layout=torch.jagged)
    )


def _batch() -> tuple[
    tensordict.TensorDict, tensordict.TensorDict, tensordict.TensorDict
]:
    return _make_points([3, 4]), _make_points([5, 5]), _make_points([2, 3])


def _state_key_baseline(model_cls: type[torch.nn.Module]) -> tuple[int, str]:
    keys = list(model_cls(**MODEL_KWARGS).state_dict())
    digest = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    return len(keys), digest


@pytest.mark.parametrize("model_cls", ANP_FLOW_CLASSES)
def test_anp_flow_state_dict_keys_match_pre_lnp_nf_baseline(model_cls):
    assert _state_key_baseline(model_cls) == ANP_FLOW_STATE_KEY_BASELINES[model_cls]


@pytest.mark.parametrize("model_cls", ANP_FLOW_CLASSES + LNP_FLOW_CLASSES)
def test_flow_train_and_eval_smoke(model_cls):
    context_priors, context_posteriors, targets = _batch()
    model = model_cls(**MODEL_KWARGS)

    model.train()
    train_output = model(
        context_priors=context_priors.clone(),
        context_posteriors=context_posteriors.clone(),
        targets=targets.clone(),
        get_loss=True,
        get_samples=True,
        sample_from_posterior=True,
        num_samples=2,
    )
    assert {"loss", "kl", "likelihood"} <= set(train_output["loss_dict"])
    assert torch.isfinite(train_output["loss_dict"]["loss"])
    assert train_output["predictions"]["target"].shape[:2] == torch.Size([2, 2])

    model.eval()
    with torch.no_grad():
        eval_output = model(
            context_priors=context_priors.clone(),
            context_posteriors=None,
            targets=targets.clone(),
            get_loss=False,
            get_samples=True,
            sample_from_posterior=False,
            num_samples=2,
        )
    assert eval_output["predictions"]["target"].shape[:2] == torch.Size([2, 2])
    assert eval_output["predictions"]["target_scale"].shape[:2] == torch.Size([2, 2])


@pytest.mark.parametrize("model_cls", LNP_FLOW_CLASSES)
def test_lnp_flow_prior_latents_do_not_include_target_tokens(model_cls):
    context_priors, _, targets = _batch()
    model = model_cls(**MODEL_KWARGS)

    latent_output = model(
        context_priors=context_priors.clone(),
        context_posteriors=None,
        targets=targets.clone(),
        get_loss=False,
        get_samples=False,
    )
    prior_targets = latent_output["prior"]["latents"]["target"]
    assert [len(target) for target in prior_targets] == [0, 0]

    model.eval()
    with torch.no_grad():
        prediction_output = model(
            context_priors=context_priors.clone(),
            context_posteriors=None,
            targets=targets.clone(),
            get_loss=False,
            get_samples=True,
            sample_from_posterior=False,
            num_samples=2,
        )
    assert [
        len(batch_predictions[0])
        for batch_predictions in prediction_output["predictions"]["target"]
    ] == [2, 3]


@pytest.mark.parametrize("model_cls", [ANP_NF_Posterior, ANP_NF_Prior_Posterior])
def test_anp_posterior_flows_load_legacy_zuko_base_keys(model_cls):
    state_dict = model_cls(**MODEL_KWARGS).state_dict()
    legacy_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key == "posterior_flow.base.loc":
            legacy_state_dict["posterior_flow.base._0"] = value
        elif key == "posterior_flow.base.scale":
            legacy_state_dict["posterior_flow.base._1"] = value
        else:
            legacy_state_dict[key] = value

    model_cls(**MODEL_KWARGS).load_state_dict(legacy_state_dict)


@pytest.mark.parametrize(
    ("relative_checkpoint", "model_cls"),
    [
        ("ANP-512k/checkpoints/checkpoint-3000000/model.safetensors", ANP),
        (
            "ANP-NF-Prior-512k/checkpoints/checkpoint-3000000/model.safetensors",
            ANP_NF_Prior,
        ),
        (
            "ANP-NF-Posterior-512k/checkpoints/checkpoint-3000000/model.safetensors",
            ANP_NF_Posterior,
        ),
        (
            (
                "ANP-NF-Prior-Posterior-512k/checkpoints/checkpoint-3000000/"
                "model.safetensors"
            ),
            ANP_NF_Prior_Posterior,
        ),
    ],
)
def test_existing_project_checkpoints_load_when_available(
    relative_checkpoint, model_cls
):
    project_dir = os.getenv("NPNF_PROJECT_DIR")
    if project_dir is None:
        pytest.skip("NPNF_PROJECT_DIR is not set")

    checkpoint_path = Path(project_dir) / relative_checkpoint
    if not checkpoint_path.exists():
        pytest.skip(f"checkpoint not available: {checkpoint_path}")

    from safetensors.torch import load_file

    state_dict = load_file(str(checkpoint_path), device="cpu")
    model = model_cls(
        markers_input_dim=SYNTH_MARKERS_INPUT_DIM,
        markers_hidden_channels=SYNTH_MARKERS_HIDDEN_CHANNELS,
    )
    model.load_state_dict(state_dict)


def test_lnp_flow_configs_are_registered():
    from hydra_zen import store

    expected_names = {
        "LNP-NF-Prior",
        "LNP-NF-Posterior",
        "LNP-NF-Prior-Posterior",
        "LNP-NF-Prior-FIP",
        "LNP-NF-Posterior-FIP",
        "LNP-NF-Prior-Posterior-FIP",
    }
    for name in expected_names:
        assert store.get_entry("model", name) is not None
