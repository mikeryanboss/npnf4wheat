from npnf.scripts.paper.lodging_plot_style import MODEL_SEMANTICS, semantic_model_groups


def test_semantic_model_groups_maps_known_paper_models() -> None:
    groups = semantic_model_groups(list(MODEL_SEMANTICS))

    assert groups is not None
    assert groups["Latent"]["Deterministic"] == "CNP"
    assert groups["Latent + deterministic"]["Gaussian"] == "ANP"
    grouped_models = [
        model for variants in groups.values() for model in variants.values()
    ]
    assert sorted(grouped_models) == sorted(MODEL_SEMANTICS)


def test_semantic_model_groups_returns_none_for_unknown_model() -> None:
    assert semantic_model_groups(["CNP", "Model 1"]) is None
