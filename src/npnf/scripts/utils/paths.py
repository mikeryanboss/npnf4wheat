"""Shared utilities for building standardized results paths.

Path structure: results/{dataloader_name}/{model}/{checkpoint}/{method}/[{cond}/]

Components:
- dataloader_name: Dataloader config name (e.g., "fip1_test_dataloaders") or special
  identifier ("prior", "all")
- model: Model run name extracted from checkpoint path
- checkpoint: Checkpoint identifier (e.g., checkpoint-15000000)
- method: Prediction strategy (sequential, uncertainty, random, no_context, etc.)
- conditioning: Optional flags for env (temperature) and geno (marker) conditioning
"""

import json
import os
import re
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

# Known dataset sizes for parsing run names (order matters: check larger first)
DATASET_SIZE_PATTERNS = ["-n512k", "-n64k", "-n8k", "-n1k"]


def get_dataloader_name_from_hydra() -> str:
    return HydraConfig.get().runtime.choices["dataloaders"]


def get_model_name_from_hydra() -> str:
    return HydraConfig.get().runtime.choices["model"]


def get_conditioning_suffix(use_temperature: bool, use_marker: bool) -> str:
    """Return conditioning suffix based on enabled flags.

    Args:
        use_temperature: Whether temperature/environment conditioning is enabled
        use_marker: Whether genotype/marker conditioning is enabled

    Returns:
        Conditioning suffix string (always explicit, e.g., 'env_geno', 'noenv_nogeno')
    """
    env_part = "env" if use_temperature else "noenv"
    geno_part = "geno" if use_marker else "nogeno"
    return f"{env_part}_{geno_part}"


def extract_model_info(checkpoint_path: str) -> tuple[str, str]:
    """Extract model name and checkpoint name from checkpoint path.

    Handles paths like:
    - /path/to/outputs/ModelName/checkpoints/checkpoint-15000000
    - /path/to/outputs/ModelName/checkpoints/checkpoint-15000000/

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        Tuple of (model_name, checkpoint_name)
    """
    path = Path(checkpoint_path).resolve()
    parts = path.parts

    # Handle trailing slash
    if parts[-1] == "":
        parts = parts[:-1]

    # Path structure: .../ModelName/checkpoints/checkpoint-X
    checkpoint_name = parts[-1]  # checkpoint-X
    model_name = parts[-3]  # ModelName (skip "checkpoints" directory)

    return model_name, checkpoint_name


def checkpoint_step(checkpoint_name: str) -> int | None:
    """Training step of a ``checkpoint-N`` directory name, else None."""
    match = re.fullmatch(r"checkpoint-(\d+)", checkpoint_name)
    return int(match.group(1)) if match else None


def normalize_checkpoint(checkpoint: str) -> str:
    """``latest``, or ``checkpoint-N`` from ``N`` or ``checkpoint-N`` with N > 0."""
    checkpoint = checkpoint.strip()
    if checkpoint == "latest":
        return checkpoint
    suffix = checkpoint.removeprefix("checkpoint-")
    if not suffix.isdigit() or int(suffix) <= 0:
        msg = "--checkpoint must be latest, a positive integer, or checkpoint-N."
        raise ValueError(msg)
    return f"checkpoint-{suffix}"


def parse_run_name(run_name: str) -> tuple[str, str | None]:
    """Extract base model name and suffix from run name (strips dataset size).

    Run names follow format: {model}-{size}[-{suffix}]
    where size is one of: n1k, n8k, n64k, n512k

    Args:
        run_name: Full run name (e.g., "ANP-n64k-joint_targets")

    Returns:
        Tuple of (base_model, suffix) where suffix is None if not present

    Examples:
        - "ANP-n64k" -> ("ANP", None)
        - "ANP-n64k-joint_targets" -> ("ANP", "joint_targets")
    """
    for size_pattern in DATASET_SIZE_PATTERNS:
        if size_pattern in run_name:
            parts = run_name.split(size_pattern)
            base_model = parts[0]
            suffix = parts[1].lstrip("-") if len(parts) > 1 and parts[1] else None
            return base_model, suffix
    return run_name, None


def build_results_path(
    dataloader_name: str,
    model: str,
    checkpoint: str,
    method: str,
    split_name: str | None = None,
    use_temperature: bool = False,
    use_marker: bool = False,
    base_dir: str | None = None,
    dataset_size: int | None = None,
    run_suffix: str | None = None,
) -> Path:
    """Build standardized results path.

    Path structure:
        {base_dir}/{dataloader_name}/{model}/[{dataset_size}/][{run_suffix}/]{checkpoint}/[{split}/]{conditioning}/{method}/

    Args:
        dataloader_name: Dataloader config name (e.g., "fip1_test_dataloaders") or
            special identifier ("prior", "all")
        model: Model run name
        checkpoint: Checkpoint identifier
        method: Prediction method (sequential, uncertainty, random, no_context, etc.)
        split_name: Optional split name within the dataloader config (e.g., "test_all")
        use_temperature: Whether temperature conditioning is enabled
        use_marker: Whether marker conditioning is enabled
        base_dir: Base directory for results; defaults to $NPNF_RESULTS_DIR
        dataset_size: Optional training dataset size for scaling analysis
        run_suffix: Optional run suffix (e.g., "joint_targets") for training variants

    Returns:
        Path object for the results directory
    """
    path = Path(base_dir or os.environ["NPNF_RESULTS_DIR"]) / dataloader_name / model
    if dataset_size is not None:
        path = path / str(dataset_size)
    if run_suffix is not None:
        path = path / run_suffix
    path = path / checkpoint
    if split_name:
        path = path / split_name

    conditioning = get_conditioning_suffix(use_temperature, use_marker)
    return path / conditioning / method


def build_results_path_from_checkpoint(
    checkpoint_path: str,
    dataloader_name: str,
    method: str,
    split_name: str | None = None,
    use_temperature: bool = False,
    use_marker: bool = False,
    base_dir: str | None = None,
    dataset_size: int | None = None,
    run_suffix: str | None = None,
) -> Path:
    """Build results path extracting model info from checkpoint path.

    Convenience function that combines extract_model_info and build_results_path.

    Args:
        checkpoint_path: Path to the checkpoint directory
        dataloader_name: Dataloader config name (e.g., "fip1_test_dataloaders") or
            special identifier ("prior", "all")
        method: Prediction method
        split_name: Optional split name within the dataloader config (e.g., "test_all")
        use_temperature: Whether temperature conditioning is enabled
        use_marker: Whether marker conditioning is enabled
        base_dir: Base directory for results; defaults to $NPNF_RESULTS_DIR
        dataset_size: Optional training dataset size for scaling analysis
        run_suffix: Optional run suffix (e.g., "joint_targets") for training variants

    Returns:
        Path object for the results directory
    """
    model, checkpoint = extract_model_info(checkpoint_path)
    return build_results_path(
        dataloader_name=dataloader_name,
        model=model,
        checkpoint=checkpoint,
        method=method,
        split_name=split_name,
        use_temperature=use_temperature,
        use_marker=use_marker,
        base_dir=base_dir,
        dataset_size=dataset_size,
        run_suffix=run_suffix,
    )


def get_dataset_size_from_checkpoint(checkpoint_path: str) -> int | None:
    """Extract training dataset size from saved config.

    Looks for train_config.yaml at the project_dir level (parent of checkpoints folder)
    and computes the training dataset size from genotype/yearsite indices.

    Args:
        checkpoint_path: Path to checkpoint directory
            (e.g., /data-kp/.../run_name/checkpoints/checkpoint-1000000)

    Returns:
        Training dataset size (G x Y) or None if config not found
    """
    project_dir = Path(checkpoint_path).parent.parent
    config_path = project_dir / "train_config.yaml"

    if not config_path.exists():
        return None

    cfg = OmegaConf.load(config_path)
    dataset_cfg = cfg.dataloaders.train.dataset
    if "genotype_indices" not in dataset_cfg or "yearsite_indices" not in dataset_cfg:
        # FIP1 runs train on a fixed real dataset with no factorial index ranges;
        # there is no scaling size to record, so the path component is omitted.
        return None
    g_indices = dataset_cfg.genotype_indices
    y_indices = dataset_cfg.yearsite_indices
    return len(g_indices) * len(y_indices)


SIZE_LABELS = {
    1024: "1k",  # 64x16
    8192: "8k",  # 256x32
    65536: "64k",  # 1024x64
    524288: "512k",  # 4096x128
}


def format_dataset_size(num_samples: int) -> str:
    """Map dataset size to clean tier label (1k, 10k, 100k, 1m)."""
    return SIZE_LABELS.get(num_samples, str(num_samples))


def save_config(path: Path, **config) -> None:
    """Save configuration as JSON file.

    Args:
        path: Directory path where config.json will be saved
        **config: Configuration key-value pairs to save
    """
    config_path = path / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        json.dump(config, f, indent=2, default=str)


def load_config(path: Path) -> dict:
    """Load configuration from JSON file.

    Args:
        path: Directory path containing config.json

    Returns:
        Configuration dictionary
    """
    config_path = path / "config.json"
    with config_path.open() as f:
        return json.load(f)
