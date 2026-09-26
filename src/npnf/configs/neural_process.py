import os

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from hydra_zen import make_config
from omegaconf import MISSING
from typing_extensions import TypedDict

from npnf.configs.utils import builds, populate_builds
from npnf.data.configs.dataloaders import (
    dataloaders as _,  # noqa: F401 - registers Hydra stores
)
from npnf.models.configs.models import ANP_Config  # noqa: F401

ProjectConfigurationConfig = populate_builds(
    ProjectConfiguration, project_dir="${oc.env:NPNF_PROJECT_DIR}/${run_name}"
)

# Precision is explicit, so it does not depend on `accelerate launch` or
# ACCELERATE_MIXED_PRECISION (#463): prediction and evaluation run in fp32 ...
AcceleratorConfig = builds(
    Accelerator,
    log_with="wandb",
    project_config=ProjectConfigurationConfig,
    mixed_precision="no",
)
# ... and training always runs in bf16 mixed precision.
TrainAcceleratorConfig = builds(
    Accelerator, mixed_precision="bf16", builds_bases=(AcceleratorConfig,)
)


class PredictionSaveDefaults(TypedDict, closed=True):
    save_targets_std: bool
    save_grid_std: bool
    save_dtype: str


class OneStepPredictionSaveDefaults(TypedDict, closed=True):
    save_targets_std: bool
    save_grid_std: bool
    save_dtype: str
    save_batch_dtype: str
    batch_schema: str
    grid: str


PREDICTION_SAVE_DEFAULTS: PredictionSaveDefaults = {
    "save_targets_std": False,
    "save_grid_std": False,
    "save_dtype": "float16",
}
ONE_STEP_PREDICTION_SAVE_DEFAULTS: OneStepPredictionSaveDefaults = {
    **PREDICTION_SAVE_DEFAULTS,
    "save_batch_dtype": "float16",
    "batch_schema": "compact",
    # "synthetic" is the 32-point canonical grid; FIP1 needs "fip1" (daily), whose
    # days cover the real observation dates.
    "grid": "synthetic",
}


TrainConfig = make_config(
    run_name="",
    seed=42,
    dataloaders=MISSING,
    model=MISSING,
    accelerator=TrainAcceleratorConfig,
    num_training_samples=3_000_000,
    per_device_batch_size=64,
    optimizer_type="adamw",
    learning_rate=1e-3,
    weight_decay=0.01,
    use_lr_schedule=True,  # Enable cosine LR schedule with warmup (adamw only)
    num_warmup_samples=3072,  # Number of samples for LR warmup
    use_temperature_probability_train=0.25,
    use_marker_probability_train=0.25,
    use_full_context_probability_train=0.9,
    small_context_prior_length_upper_bound_train=3,
    use_temperature_validation=False,
    use_marker_validation=False,
    max_context_prior_length=50,
    set_mode="nested-noprior",
    max_distinct=128,
    min_target=10,
    context_prior_anneal_fraction=0.25,
    context_prior_length_initial=0,
    context_prior_length_final=50,
    context_prior_anneal_shape="cosine",
    num_samples=16,
    kl_hold_fraction=0.25,
    kl_ramp_fraction=0.75,
    kl_weight_initial=0.0,
    kl_weight_peak=0.5,
    kl_anneal_shape="cosine",
    debug=False,
    checkpoint_frequency=100_000,
    checkpoint_folder=None,
    resume_training=True,
    validation_frequency=100_000,
    datasets_offline_path=os.getenv("NPNF_DATASET_PATH"),
    hydra_defaults=[
        "_self_",
        {"dataloaders": "synth_train_512k_dataloaders"},
        {"model": "ANP"},
        {"override hydra/launcher": "slurm"},
    ],
)

ValidateConfig = make_config(
    run_name="validate",
    seed=42,
    dataloader=MISSING,
    model=MISSING,
    accelerator=AcceleratorConfig,
    per_device_batch_size=16,
    checkpoint_folder="/data-kp/mike/projects/neural_process/train/checkpoints",
    datasets_offline_path=os.getenv("NPNF_DATASET_PATH"),
    hydra_defaults=[
        "_self_",
        {"dataloader": "fip1_val_dataloader"},
        {"model": "ANP-FIP"},
        {"override hydra/launcher": "slurm"},
    ],
)

TestVaryingConfig = make_config(
    run_name="test/varying",
    seed=42,
    dataloaders=MISSING,
    num_samples=64,
    use_temperature=True,
    use_marker=True,
    predict_uncertainty=True,
    predict_sequential=False,
    predict_random=False,
    num_random_trials=3,
    max_num_context_prior=49,
    num_batches=None,
    accelerator=AcceleratorConfig,
    per_device_batch_size=384,
    save_dtype="float16",
    checkpoint_folder="/data-kp/mike/projects/neural_process/train/checkpoints",
    datasets_offline_path=os.getenv("NPNF_DATASET_PATH"),
    hydra_defaults=[
        "_self_",
        {"dataloaders": "synth_test_plot_dataloaders"},
        {"override hydra/launcher": "slurm_prediction"},
    ],
)

MaxHeightPredictionConfig = make_config(
    run_name="test/max_height",
    seed=42,
    dataloaders=MISSING,
    accelerator=AcceleratorConfig,
    per_device_batch_size=1280,
    checkpoint_folder="/data-kp/mike/projects/neural_process/train/checkpoints",
    use_temperature=True,
    use_marker=True,
    num_batches=None,
    **ONE_STEP_PREDICTION_SAVE_DEFAULTS,
    datasets_offline_path=os.getenv("NPNF_DATASET_PATH"),
    hydra_defaults=[
        "_self_",
        {"dataloaders": "synth_test_plot_dataloaders"},
        {"override hydra/launcher": "slurm_prediction"},
    ],
)

NoContextPriorMetadataSamplingConfig = make_config(
    run_name="test/no_context_prior_metadata",
    seed=42,
    dataloaders=MISSING,
    accelerator=AcceleratorConfig,
    checkpoint_folder="/data-kp/mike/projects/neural_process/train/checkpoints",
    use_temperature=True,
    use_marker=True,
    per_device_batch_size=1280,
    num_batches=None,
    **ONE_STEP_PREDICTION_SAVE_DEFAULTS,
    datasets_offline_path=os.getenv("NPNF_DATASET_PATH"),
    hydra_defaults=[
        "_self_",
        {"dataloaders": "synth_test_plot_dataloaders"},
        {"override hydra/launcher": "slurm_prediction"},
    ],
)

RandomContextPredictionConfig = make_config(
    run_name="test/random_context",
    seed=42,
    dataloaders=MISSING,
    accelerator=AcceleratorConfig,
    checkpoint_folder="/data-kp/mike/projects/neural_process/train/checkpoints",
    use_temperature=True,
    use_marker=True,
    num_samples=64,
    context_seed=42,
    min_context=1,
    max_context=10,
    per_device_batch_size=1280,
    num_batches=None,
    **ONE_STEP_PREDICTION_SAVE_DEFAULTS,
    datasets_offline_path=os.getenv("NPNF_DATASET_PATH"),
    hydra_defaults=[
        "_self_",
        {"dataloaders": "synth_test_plot_dataloaders"},
        {"override hydra/launcher": "slurm_prediction"},
    ],
)
