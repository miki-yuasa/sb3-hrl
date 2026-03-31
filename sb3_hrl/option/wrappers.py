"""Gymnasium wrappers for option subpolicy and meta-controller training."""

from __future__ import annotations

from typing import Any, Literal, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .options import BaseOption, RandomOption


class SubpolicyTrainingWrapper(gym.Wrapper):
    """Wrap an environment to train one specific option subpolicy.

    This wrapper keeps the base environment dynamics unchanged but replaces
    extrinsic rewards with intrinsic rewards provided by the option.

    Parameters
    ----------
    env : gym.Env
        Wrapped base environment.
    option : BaseOption
        Option being trained in phase 1.

    Notes
    -----
    Typical SB3 usage:

    ``model = PPO("MlpPolicy", SubpolicyTrainingWrapper(env, option), ...)``
    """

    def __init__(self, env: gym.Env, option: BaseOption) -> None:
        super().__init__(env)
        self.option = option
        self._last_obs: Optional[np.ndarray] = None

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset wrapped env and option execution state."""
        obs, info = self.env.reset(**kwargs)
        self._last_obs = np.asarray(obs)
        self.option.reset_execution_state()
        return obs, info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Forward primitive action and return option-specific intrinsic reward."""
        if self._last_obs is None:
            raise RuntimeError(
                "Call reset() before step() in SubpolicyTrainingWrapper."
            )

        next_obs, external_reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        intrinsic = float(
            self.option.intrinsic_reward(
                self._last_obs,
                action,
                next_obs,
                float(external_reward),
                done,
            )
        )

        option_terminated = bool(self.option.termination_condition(next_obs))
        forced_terminated = bool(terminated or option_terminated)

        info = dict(info)
        info["subpolicy_external_reward"] = float(external_reward)
        info["subpolicy_intrinsic_reward"] = intrinsic
        info["option_terminated"] = option_terminated

        self._last_obs = np.asarray(next_obs)
        return next_obs, intrinsic, forced_terminated, bool(truncated), info


class MetaControllerEnvWrapper(gym.Wrapper):
    """Wrap an env so high-level actions select options.

    Parameters
    ----------
    env : gym.Env
        Wrapped base environment.
    options : list[BaseOption]
        Trained options available to the meta-controller.
    reward_type : Literal["smdp", "intra_option"], default="smdp"
        Reward semantics at the meta step.
        ``"smdp"`` returns discounted macro reward,
        ``"intra_option"`` returns undiscounted macro reward and exposes
        primitive transitions in info for custom learning logic.
    gamma : float, default=0.99
        Discount used to compute SMDP macro return.
    invalid_option_penalty : float, default=-1.0
        Reward emitted when selected option cannot be initiated.
    include_random_option : bool, default=True
        Whether to append a built-in random option to improve exploration.
    random_option_termination_steps : int, default=1
        Primitive step horizon for the random option.
    capture_primitive_transitions : bool | None, default=None
        Whether to include primitive transitions in the returned info dict.
        If ``None``, it is enabled automatically for ``reward_type='intra_option'``.
    max_option_steps : int | None, default=None
        Optional safety cap on primitive steps executed by one option.

    Notes
    -----
    Typical SB3 usage:

    ``model = DQN("MlpPolicy", MetaControllerEnvWrapper(env, options), ...)``
    """

    def __init__(
        self,
        env: gym.Env,
        options: list[BaseOption],
        reward_type: Literal["smdp", "intra_option"] = "smdp",
        gamma: float = 0.99,
        invalid_option_penalty: float = -1.0,
        include_random_option: bool = True,
        random_option_termination_steps: int = 1,
        capture_primitive_transitions: Optional[bool] = None,
        max_option_steps: Optional[int] = None,
    ) -> None:
        super().__init__(env)
        if reward_type not in {"smdp", "intra_option"}:
            raise ValueError("reward_type must be 'smdp' or 'intra_option'.")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1].")
        if max_option_steps is not None and max_option_steps <= 0:
            raise ValueError("max_option_steps must be positive when provided.")

        self.reward_type = reward_type
        self.gamma = float(gamma)
        self.invalid_option_penalty = float(invalid_option_penalty)
        self.max_option_steps = max_option_steps

        final_options = list(options)
        if include_random_option:
            final_options.append(
                RandomOption(
                    action_space=self.env.action_space,
                    termination_steps=random_option_termination_steps,
                )
            )
        if len(final_options) == 0:
            raise ValueError("MetaControllerEnvWrapper requires at least one option.")

        self.options = final_options
        if capture_primitive_transitions is None:
            capture_primitive_transitions = reward_type == "intra_option"
        self.capture_primitive_transitions = bool(capture_primitive_transitions)

        self.observation_space = self.env.observation_space
        self.action_space = spaces.Discrete(len(self.options))
        self._last_obs: Optional[np.ndarray] = None

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset wrapped env and internal observation cache."""
        obs, info = self.env.reset(**kwargs)
        self._last_obs = np.asarray(obs)
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one selected option and return one macro transition."""
        if self._last_obs is None:
            raise RuntimeError(
                "Call reset() before step() in MetaControllerEnvWrapper."
            )
        if not self.action_space.contains(action):
            raise ValueError("Action index out of range for MetaControllerEnvWrapper.")

        option = self.options[int(action)]
        current_obs = self._last_obs

        if not option.initiation_set(current_obs):
            info: dict[str, Any] = {
                "invalid_option": True,
                "option_index": int(action),
                "meta_option_steps": 0,
                "effective_gamma": 1.0,
                "reward_type": self.reward_type,
            }
            if self.capture_primitive_transitions:
                info["primitive_transitions"] = []
            return current_obs, self.invalid_option_penalty, False, False, info

        option.reset_execution_state()
        obs = current_obs
        total_reward = 0.0
        effective_gamma = 1.0
        steps = 0
        terminated = False
        truncated = False
        option_terminated = False
        last_info: dict[str, Any] = {}
        primitive_transitions: list[dict[str, Any]] = []

        while True:
            primitive_action = option.predict(obs)
            next_obs, reward, terminated, truncated, step_info = self.env.step(
                primitive_action
            )

            if self.reward_type == "smdp":
                total_reward += effective_gamma * float(reward)
            else:
                total_reward += float(reward)
            effective_gamma *= self.gamma

            if self.capture_primitive_transitions:
                primitive_transitions.append(
                    {
                        "obs": np.asarray(obs).copy(),
                        "action": np.asarray(primitive_action).copy(),
                        "reward": float(reward),
                        "next_obs": np.asarray(next_obs).copy(),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "info": dict(step_info),
                    }
                )

            steps += 1
            option_terminated = bool(option.termination_condition(next_obs))
            obs = np.asarray(next_obs)
            last_info = dict(step_info)

            if terminated or truncated or option_terminated:
                break
            if self.max_option_steps is not None and steps >= self.max_option_steps:
                truncated = True
                last_info.setdefault("TimeLimit.truncated", True)
                break

        self._last_obs = obs
        info = dict(last_info)
        info["invalid_option"] = False
        info["option_index"] = int(action)
        info["option_terminated"] = option_terminated
        info["meta_option_steps"] = steps
        info["effective_gamma"] = float(effective_gamma)
        info["reward_type"] = self.reward_type
        if self.capture_primitive_transitions:
            info["primitive_transitions"] = primitive_transitions

        return obs, float(total_reward), bool(terminated), bool(truncated), info


__all__ = ["SubpolicyTrainingWrapper", "MetaControllerEnvWrapper"]
