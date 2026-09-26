"""Hydra configuration utilities for NPNF scripts."""

from collections.abc import Callable
from typing import Any

from accelerate.utils import set_seed
from hydra_zen import store, zen


def run_hydra_main(
    config_class: type,
    main_func: Callable,
    config_name: str = "default",
    config_path: str = "../../configs",
) -> Any:
    """Standard Hydra entry point with seed pre-call.

    Args:
        config_class: The Hydra config dataclass to register.
        main_func: The main function to run with Hydra.
        config_name: Name to register the config under.
        config_path: Relative path to config directory.

    Returns:
        The result of hydra_main execution.
    """
    store(config_class, name=config_name)
    store.add_to_hydra_store()

    pre_seed = zen(set_seed)
    task_function = zen(main_func, pre_call=pre_seed)

    return task_function.hydra_main(
        config_name=config_name, version_base=None, config_path=config_path
    )
