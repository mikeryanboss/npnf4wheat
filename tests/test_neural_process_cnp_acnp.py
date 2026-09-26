import tensordict
import torch

from npnf.models.neural_process import ACNP, CNP
from npnf.models.utils import consolidate_tensordict_jagged_dim
from tests.utils import tensor_at


def _make_model(model_cls: type[CNP]) -> CNP:
    return model_cls(
        hidden_dim=16,
        latent_dim=4,
        num_transformer_layers=1,
        num_transformer_heads=1,
        markers_input_dim=2,
        markers_hidden_channels=[4],
    )


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


def _batch() -> tuple[tensordict.TensorDict, tensordict.TensorDict]:
    return _make_points([3, 4]), _make_points([2, 3])


def _prediction_lengths(model_cls: type[CNP]) -> list[int]:
    context_priors, targets = _batch()
    model = _make_model(model_cls)

    output = model(
        context_priors=context_priors.clone(),
        targets=targets.clone(),
        get_loss=False,
        get_samples=True,
    )

    target_predictions = output["predictions"]["target"]
    return [len(batch_predictions[0]) for batch_predictions in target_predictions]


def test_cnp_prior_latents_do_not_include_target_tokens():
    context_priors, targets = _batch()
    model = _make_model(CNP)

    output = model(
        context_priors=context_priors.clone(),
        targets=targets.clone(),
        get_loss=False,
        get_samples=False,
    )

    assert [len(target) for target in output["prior"]["latents"]["target"]] == [0, 0]


def test_acnp_prior_latents_include_target_tokens():
    context_priors, targets = _batch()
    model = _make_model(ACNP)

    output = model(
        context_priors=context_priors.clone(),
        targets=targets.clone(),
        get_loss=False,
        get_samples=False,
    )

    assert [len(target) for target in output["prior"]["latents"]["target"]] == [2, 3]


def test_cnp_and_acnp_predict_for_all_target_points():
    assert _prediction_lengths(CNP) == [2, 3]
    assert _prediction_lengths(ACNP) == [2, 3]


def test_cnp_predicts_with_dense_equal_length_targets():
    context_priors = _make_points([3, 3])
    targets = _make_points([2, 2])
    assert not tensor_at(targets, "X").is_nested

    model = _make_model(CNP)

    output = model(
        context_priors=context_priors.clone(),
        targets=targets.clone(),
        get_loss=False,
        get_samples=True,
    )

    target_predictions = output["predictions"]["target"]
    assert [len(batch_predictions[0]) for batch_predictions in target_predictions] == [
        2,
        2,
    ]


def test_cnp_combine_paths_skips_nested_target_holes():
    model = _make_model(CNP)
    values = torch.arange(6 * model.hidden_dim, dtype=torch.float32).reshape(
        6, model.hidden_dim
    )
    targets = torch.nested.nested_tensor_from_jagged(
        values=values,
        offsets=torch.tensor([0, 2, 3, 6]),
        lengths=torch.tensor([1, 1, 2]),
    )
    z = torch.zeros(3, 1, model.hidden_dim)

    combined = model.combine_paths(z=z, targets=targets)

    combined_rows = list(combined)
    assert torch.equal(combined_rows[0], values[0:1])
    assert torch.equal(combined_rows[1], values[2:3])
    assert torch.equal(combined_rows[2], values[3:5])


def test_acnp_state_dict_keys_match_cnp_for_legacy_checkpoint_loading():
    cnp = _make_model(CNP)
    acnp = _make_model(ACNP)

    assert set(cnp.state_dict()) == set(acnp.state_dict())
    acnp.load_state_dict(cnp.state_dict(), strict=True)


def test_cnp_and_acnp_configs_are_registered():
    from hydra_zen import store

    import npnf.models.configs.models  # noqa: F401

    for name in ("CNP", "ACNP", "CNP-FIP", "ACNP-FIP"):
        assert store.get_entry("model", name) is not None
