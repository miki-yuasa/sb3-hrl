"""Built-in random option to encourage meta-controller exploration."""

from __future__ import annotations

from typing import Any, Callable, Optional

from gymnasium import spaces

from .base import BaseOption


class RandomOption(BaseOption):
    """Option that samples primitive actions uniformly from env action space.

    Parameters
    ----------
    action_space : spaces.Space
        Primitive action space of the wrapped environment.
    termination_steps : int, default=1
        Number of primitive steps to execute before terminating.
    initiation_fn : callable | None, default=None
        Optional custom initiation predicate. If omitted, initiation is always
        allowed.
    """

    def __init__(
        self,
        action_space: spaces.Space,
        termination_steps: int = 1,
        initiation_fn: Optional[Callable[[Any], bool]] = None,
    ) -> None:
        super().__init__(policy=None)
        if termination_steps <= 0:
            raise ValueError("termination_steps must be positive.")

        self.action_space = action_space
        self.termination_steps = int(termination_steps)
        self.initiation_fn = initiation_fn
        self._num_steps = 0

    def initiation_set(self, obs: Any) -> bool:
        """Return whether this random option can be initiated."""
        if self.initiation_fn is None:
            return True
        return bool(self.initiation_fn(obs))

    def termination_condition(self, obs: Any) -> bool:
        """Terminate once ``termination_steps`` primitive actions were sampled."""
        del obs
        return self._num_steps >= self.termination_steps

    def intrinsic_reward(
        self,
        obs: Any,
        action: Any,
        next_obs: Any,
        external_reward: float,
        done: bool,
    ) -> float:
        """Use the external reward as a neutral intrinsic reward signal."""
        del obs, action, next_obs, done
        return float(external_reward)

    def reset_execution_state(self) -> None:
        """Reset internal primitive step counter."""
        self._num_steps = 0

    def predict(self, obs: Any, deterministic: bool = True) -> Any:
        """Sample one primitive action from the action space."""
        del obs, deterministic
        self._num_steps += 1
        return self.action_space.sample()
