"""Schedulers for training hyperparameters."""

import math


class KLWeightScheduler:
    """Scheduler for KL divergence loss weight with optional hold + ramp."""

    def __init__(
        self,
        hold_steps: int = 0,
        ramp_steps: int = 0,
        shape: str = "cosine",
        initial_weight: float = 0.0,
        peak_weight: float = 1.0,
    ) -> None:
        if hold_steps < 0:
            msg = f"hold_steps must be non-negative, got {hold_steps}"
            raise ValueError(msg)
        if ramp_steps < 0:
            msg = f"ramp_steps must be non-negative, got {ramp_steps}"
            raise ValueError(msg)

        if shape not in ("linear", "cosine", "constant"):
            msg = f"shape must be 'linear', 'cosine', or 'constant', got '{shape}'"
            raise ValueError(msg)

        self.hold_steps = hold_steps
        self.ramp_steps = ramp_steps
        self.shape = shape
        self.initial_weight = initial_weight
        self.peak_weight = peak_weight
        self._step_count = 0

    def step(self) -> None:
        """Advance the scheduler by one step."""
        self._step_count += 1

    def get_weight(self) -> float:
        """Get current KL weight based on hold + annealing schedule."""
        if self._step_count < self.hold_steps:
            return 0.0
        if self.ramp_steps == 0:
            return self.peak_weight

        ramp_step = self._step_count - self.hold_steps
        if ramp_step >= self.ramp_steps:
            return self.peak_weight

        progress = ramp_step / self.ramp_steps
        if self.shape == "linear":
            weight = (
                self.initial_weight
                + (self.peak_weight - self.initial_weight) * progress
            )
        elif self.shape == "cosine":
            cos_progress = (1 - math.cos(math.pi * progress)) / 2
            weight = (
                self.initial_weight
                + (self.peak_weight - self.initial_weight) * cos_progress
            )
        elif self.shape == "constant":
            weight = self.peak_weight
        else:
            msg = f"Unknown shape: {self.shape}"
            raise ValueError(msg)

        return weight

    def state_dict(self) -> dict:
        """Return state dictionary for checkpointing.

        Returns:
            Dictionary containing scheduler state
        """
        return {
            "_step_count": self._step_count,
            "hold_steps": self.hold_steps,
            "ramp_steps": self.ramp_steps,
            "shape": self.shape,
            "initial_weight": self.initial_weight,
            "peak_weight": self.peak_weight,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint.

        Args:
            state_dict: Dictionary containing scheduler state
        """
        self._step_count = state_dict["_step_count"]
        self.hold_steps = state_dict["hold_steps"]
        self.ramp_steps = state_dict["ramp_steps"]
        self.shape = state_dict["shape"]
        self.initial_weight = state_dict["initial_weight"]
        self.peak_weight = state_dict["peak_weight"]

    def __repr__(self) -> str:
        return (
            f"KLWeightScheduler(hold_steps={self.hold_steps}, "
            f"ramp_steps={self.ramp_steps}, "
            f"shape='{self.shape}', "
            f"initial_weight={self.initial_weight}, "
            f"peak_weight={self.peak_weight}, "
            f"current_step={self._step_count})"
        )


class ContextLengthScheduler:
    """Scheduler that anneals the sampled context-set size during training."""

    def __init__(
        self,
        total_steps: int,
        shape: str = "linear",
        initial_length: float = 4.0,
        final_length: float = 50.0,
        min_length: float = 1.0,
    ) -> None:
        if total_steps <= 0:
            msg = f"total_steps must be positive, got {total_steps}"
            raise ValueError(msg)
        if shape not in ("linear", "cosine", "constant"):
            msg = f"shape must be 'linear', 'cosine', or 'constant', got '{shape}'"
            raise ValueError(msg)
        if min_length <= 0:
            msg = f"min_length must be positive, got {min_length}"
            raise ValueError(msg)

        self.total_steps = total_steps
        self.shape = shape
        self.initial_length = initial_length
        self.final_length = final_length
        self.min_length = min_length
        self._step_count = 0

    def step(self) -> None:
        self._step_count += 1

    def get_length(self) -> int:
        if self._step_count >= self.total_steps:
            length = self.final_length
        else:
            progress = self._step_count / self.total_steps
            if self.shape == "linear":
                length = self.initial_length + (
                    (self.final_length - self.initial_length) * progress
                )
            elif self.shape == "cosine":
                cos_progress = (1 - math.cos(math.pi * progress)) / 2
                length = self.initial_length + (
                    (self.final_length - self.initial_length) * cos_progress
                )
            elif self.shape == "constant":
                length = self.final_length
            else:
                msg = f"Unknown shape: {self.shape}"
                raise ValueError(msg)

        length = max(self.min_length, length)
        return max(1, round(length))

    def state_dict(self) -> dict:
        return {
            "_step_count": self._step_count,
            "total_steps": self.total_steps,
            "shape": self.shape,
            "initial_length": self.initial_length,
            "final_length": self.final_length,
            "min_length": self.min_length,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._step_count = state_dict["_step_count"]
        self.total_steps = state_dict["total_steps"]
        self.shape = state_dict["shape"]
        self.initial_length = state_dict["initial_length"]
        self.final_length = state_dict["final_length"]
        self.min_length = state_dict.get("min_length", 1.0)

    def __repr__(self) -> str:
        return (
            "ContextLengthScheduler("
            f"total_steps={self.total_steps}, "
            f"shape='{self.shape}', "
            f"initial_length={self.initial_length}, "
            f"final_length={self.final_length}, "
            f"min_length={self.min_length}, "
            f"current_step={self._step_count})"
        )


class WeightDecayScheduler:
    """Scheduler for weight decay with cosine annealing.

    Anneals weight decay from an initial value to a final value over a specified
    number of training steps using a cosine schedule.

    Args:
        optimizer: PyTorch optimizer to adjust weight decay for
        total_steps: Number of steps for the complete schedule
        initial_weight_decay: Starting weight decay value
        final_weight_decay: Final weight decay value after annealing completes
        shape: Annealing schedule shape ('linear', 'cosine', 'constant')

    Example:
        >>> optimizer = torch.optim.AdamW(
        ...     model.parameters(), lr=1e-3, weight_decay=0.1
        ... )
        >>> scheduler = WeightDecayScheduler(
        ...     optimizer,
        ...     total_steps=1000,
        ...     initial_weight_decay=0.1,
        ...     final_weight_decay=0.01,
        ... )
        >>> for step in range(1000):
        ...     # training step
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        initial_weight_decay: float,
        final_weight_decay: float,
        shape: str = "cosine",
    ) -> None:
        if total_steps <= 0:
            msg = f"total_steps must be positive, got {total_steps}"
            raise ValueError(msg)

        if shape not in ("linear", "cosine", "constant"):
            msg = f"shape must be 'linear', 'cosine', or 'constant', got '{shape}'"
            raise ValueError(msg)

        self.optimizer = optimizer
        self.total_steps = total_steps
        self.initial_weight_decay = initial_weight_decay
        self.final_weight_decay = final_weight_decay
        self.shape = shape
        self._step_count = 0

        # Store base weight decay for each parameter group
        self.base_weight_decays = [
            group.get("weight_decay", 0.0) for group in self.optimizer.param_groups
        ]

    def step(self) -> None:
        """Advance the scheduler by one step and update optimizer weight decay."""
        self._step_count += 1
        weight_decay = self.get_weight_decay()

        for param_group in self.optimizer.param_groups:
            param_group["weight_decay"] = weight_decay

    def get_weight_decay(self) -> float:
        """Get current weight decay based on the annealing schedule.

        Returns:
            Current weight decay (between initial and final weight_decay)
        """
        if self._step_count >= self.total_steps:
            return self.final_weight_decay

        progress = self._step_count / self.total_steps

        if self.shape == "linear":
            weight_decay = (
                self.initial_weight_decay
                + (self.final_weight_decay - self.initial_weight_decay) * progress
            )
        elif self.shape == "cosine":
            cos_progress = (1 - math.cos(math.pi * progress)) / 2
            weight_decay = (
                self.initial_weight_decay
                + (self.final_weight_decay - self.initial_weight_decay) * cos_progress
            )
        elif self.shape == "constant":
            weight_decay = self.final_weight_decay
        else:
            msg = f"Unknown shape: {self.shape}"
            raise ValueError(msg)

        return weight_decay

    @property
    def current_weight_decay(self) -> float:
        """Get the current weight decay value."""
        return self.get_weight_decay()

    def state_dict(self) -> dict:
        """Return state dictionary for checkpointing.

        Returns:
            Dictionary containing scheduler state
        """
        return {
            "_step_count": self._step_count,
            "total_steps": self.total_steps,
            "shape": self.shape,
            "initial_weight_decay": self.initial_weight_decay,
            "final_weight_decay": self.final_weight_decay,
            "base_weight_decays": self.base_weight_decays,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint.

        Args:
            state_dict: Dictionary containing scheduler state
        """
        self._step_count = state_dict["_step_count"]
        self.total_steps = state_dict["total_steps"]
        self.shape = state_dict["shape"]
        self.initial_weight_decay = state_dict["initial_weight_decay"]
        self.final_weight_decay = state_dict["final_weight_decay"]
        self.base_weight_decays = state_dict["base_weight_decays"]

    def __repr__(self) -> str:
        return (
            f"WeightDecayScheduler(total_steps={self.total_steps}, "
            f"shape='{self.shape}', "
            f"initial_weight_decay={self.initial_weight_decay}, "
            f"final_weight_decay={self.final_weight_decay}, "
            f"current_step={self._step_count})"
        )
