"""Tests for resolve_model_from_checkpoint using training config vs model store."""

import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

from npnf.models.neural_process import ACNP
from npnf.scripts.utils.prediction import resolve_model_from_checkpoint

CNP_TARGET = "npnf.models.neural_process.models.neural_process.CNP"
ACNP_TARGET = "npnf.models.neural_process.models.neural_process.ACNP"

# Minimal valid model parameters for a CNP instantiation
MINIMAL_MODEL_KWARGS = {
    "hidden_dim": 16,
    "latent_dim": 4,
    "num_transformer_layers": 1,
    "num_transformer_heads": 1,
    "markers_input_dim": 2,
    "markers_hidden_channels": [4],
}


def _make_fake_checkpoint_dir(tmp_path: Path, run_name: str) -> Path:
    """Create a fake checkpoint directory structure.

    Returns the path to the checkpoint directory.
    """
    project_dir = tmp_path / run_name
    checkpoint_dir = project_dir / "checkpoints" / "checkpoint-1000"
    checkpoint_dir.mkdir(parents=True)
    return checkpoint_dir


def _write_train_config(
    project_dir: Path, model_kwargs: dict, model_target: str = CNP_TARGET
) -> None:
    """Write a train_config.yaml with a model section."""
    cfg = {"model": {"_target_": model_target, **model_kwargs}}
    path = project_dir / "train_config.yaml"
    OmegaConf.save(OmegaConf.create(cfg), path)


def test_uses_train_config_when_present():
    """Model instantiated from train_config.yaml with non-default latent_dim."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        run_name = "CNP-test"
        checkpoint_dir = _make_fake_checkpoint_dir(tmp_path, run_name)
        project_dir = checkpoint_dir.parent.parent

        # Train config says latent_dim=8 (non-default)
        model_kwargs = {**MINIMAL_MODEL_KWARGS, "latent_dim": 8}
        _write_train_config(project_dir, model_kwargs)

        model = resolve_model_from_checkpoint(str(checkpoint_dir))

        assert model.latent_dim == 8, (
            f"Expected latent_dim=8 from train_config.yaml, got {model.latent_dim}"
        )


def test_uses_train_config_with_extra_layers():
    """Model instantiated from train_config.yaml with extra encoder/decoder layers."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        run_name = "CNP-test"
        checkpoint_dir = _make_fake_checkpoint_dir(tmp_path, run_name)
        project_dir = checkpoint_dir.parent.parent

        model_kwargs = {
            **MINIMAL_MODEL_KWARGS,
            "latent_dim": 16,
            "num_latent_encoder_layers": 6,
            "num_latent_decoder_layers": 6,
            "num_decoder_layers": 4,
        }
        _write_train_config(project_dir, model_kwargs)

        model = resolve_model_from_checkpoint(str(checkpoint_dir))

        assert model.latent_dim == 16
        # latent_encoder is MLP: check number of Linear layers (one per MLP layer)
        encoder_linear_layers = sum(
            1
            for m in model.latent_encoder.modules()  # ty: ignore[unresolved-attribute]
            if isinstance(m, torch.nn.Linear)
        )
        assert encoder_linear_layers == 6, (
            f"Expected 6 linear layers in latent_encoder, got {encoder_linear_layers}"
        )


def test_falls_back_to_model_store_when_no_train_config():
    """When train_config.yaml is missing, falls back to model store defaults."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        run_name = "CNP-test"
        checkpoint_dir = _make_fake_checkpoint_dir(tmp_path, run_name)

        # No train_config.yaml written

        model = resolve_model_from_checkpoint(str(checkpoint_dir))

        # Default latent_dim from CNP constructor is 32
        assert model.latent_dim == 32, (
            f"Expected default latent_dim=32 from model store, got {model.latent_dim}"
        )


def test_uses_migrated_acnp_train_config_when_present():
    """Migrated ACNP train_config.yaml instantiates ACNP."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        run_name = "ACNP-test"
        checkpoint_dir = _make_fake_checkpoint_dir(tmp_path, run_name)
        project_dir = checkpoint_dir.parent.parent

        model_kwargs = {**MINIMAL_MODEL_KWARGS, "latent_dim": 8}
        _write_train_config(project_dir, model_kwargs, model_target=ACNP_TARGET)

        model = resolve_model_from_checkpoint(str(checkpoint_dir))

        assert isinstance(model, ACNP)
        assert model.latent_dim == 8


def test_acnp_falls_back_to_model_store_when_no_train_config():
    """ACNP run names resolve by model store longest-prefix fallback."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        run_name = "ACNP-test"
        checkpoint_dir = _make_fake_checkpoint_dir(tmp_path, run_name)

        model = resolve_model_from_checkpoint(str(checkpoint_dir))

        assert isinstance(model, ACNP)
        assert model.latent_dim == 32


def test_train_config_without_model_section_falls_back():
    """When train_config.yaml exists but has no model section, falls back."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        run_name = "CNP-test"
        checkpoint_dir = _make_fake_checkpoint_dir(tmp_path, run_name)
        project_dir = checkpoint_dir.parent.parent

        # Write config without model section
        cfg = {"some_other_key": "value"}
        OmegaConf.save(OmegaConf.create(cfg), project_dir / "train_config.yaml")

        model = resolve_model_from_checkpoint(str(checkpoint_dir))

        # Falls back to model store defaults
        assert model.latent_dim == 32
