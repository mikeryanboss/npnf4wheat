"""Training precision must not depend on `accelerate launch` or the environment."""

from omegaconf import OmegaConf

from npnf.configs import neural_process


def _mixed_precision(config) -> str:
    return OmegaConf.structured(config).accelerator.mixed_precision


def test_training_uses_bf16_and_prediction_uses_fp32():
    assert _mixed_precision(neural_process.TrainConfig) == "bf16"
    for config in (
        neural_process.ValidateConfig,
        neural_process.TestVaryingConfig,
        neural_process.MaxHeightPredictionConfig,
        neural_process.NoContextPriorMetadataSamplingConfig,
        neural_process.RandomContextPredictionConfig,
    ):
        assert _mixed_precision(config) == "no"
