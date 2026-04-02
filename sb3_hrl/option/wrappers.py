"""Gymnasium wrappers for option subpolicy and meta-controller training."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, Literal, Optional, SupportsFloat, cast

import gymnasium as gym
import numpy as np
from gymnasium import Wrapper, spaces
from gymnasium.core import ActType, ObsType
from gymnasium.utils import RecordConstructorArgs

from sb3_hrl.typing import SB3ObsType

from .options import BaseIntrinsicReward, BaseOption, RandomOption


class SubpolicyTrainingWrapper(
    Wrapper[ObsType, ActType, ObsType, ActType], RecordConstructorArgs
):
    """Wrap an environment to train one specific option subpolicy.

    This wrapper keeps the base environment dynamics unchanged but replaces
    extrinsic rewards with intrinsic rewards provided by the option.

    Parameters
    ----------
    env : gym.Env
        Wrapped base environment.
    intrinsic_reward : BaseIntrinsicReward[ObsType, ActType]
        Intrinsic reward logic object used to compute phase-1 training reward.
    termination_condition : callable | None, default=None
        Optional termination predicate. If omitted and ``intrinsic_reward``
        object defines ``termination_condition(obs) -> bool``, it is used.
        Otherwise, option termination is disabled.

    Notes
    -----
    Typical SB3 usage:

    ``wrapped = SubpolicyTrainingWrapper(env, intrinsic_reward=MyIntrinsicReward(), termination_condition=...)``
    ``model = PPO("MlpPolicy", wrapped, ...)``
    """

    def __init__(
        self,
        env: gym.Env,
        intrinsic_reward_cls: type[BaseIntrinsicReward[ObsType, ActType]],
        intrinsic_reward_args: dict[str, Any] | None = None,
        termination_condition: Optional[Callable[[ObsType], bool]] = None,
    ) -> None:
        RecordConstructorArgs.__init__(
            self,
            intrinsic_reward_cls=intrinsic_reward_cls,
            intrinsic_reward_args=intrinsic_reward_args,
            termination_condition=termination_condition,
        )
        Wrapper.__init__(self, env)
        self._last_obs: Optional[ObsType] = None

        self._intrinsic_reward: BaseIntrinsicReward[ObsType, ActType] = (
            intrinsic_reward_cls(**(intrinsic_reward_args or {}))
        )
        self._termination_condition: Callable[[ObsType], bool] = (
            termination_condition
            if termination_condition is not None
            else lambda _obs: False
        )

    def reset(self, **kwargs: Any) -> tuple[ObsType, dict[str, Any]]:
        """Reset wrapped env and option execution state."""
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._intrinsic_reward.reset_execution_state()
        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Forward primitive action and return option-specific intrinsic reward."""
        if self._last_obs is None:
            raise RuntimeError(
                "Call reset() before step() in SubpolicyTrainingWrapper."
            )

        next_obs, external_reward, terminated, truncated, info = self.env.step(action)
        intrinsic = float(
            self._intrinsic_reward.intrinsic_reward(
                self._last_obs,
                action,
                next_obs,
                float(external_reward),
                terminated or truncated,
            )
        )

        option_terminated = bool(self._termination_condition(next_obs))
        forced_terminated = bool(terminated or option_terminated)

        info["subpolicy_external_reward"] = float(external_reward)
        info["subpolicy_intrinsic_reward"] = intrinsic
        info["option_terminated"] = option_terminated

        self._last_obs = next_obs
        return next_obs, intrinsic, forced_terminated, truncated, info


class MetaControllerEnvWrapper(
    Wrapper[SB3ObsType, int, SB3ObsType, ActType],
    RecordConstructorArgs,
):
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
    max_option_steps : int, default=50
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
        max_option_steps: int = 50,
        include_step_count_in_obs: bool = False,
    ) -> None:
        RecordConstructorArgs.__init__(
            self,
            env=env,
            options=options,
            reward_type=reward_type,
            gamma=gamma,
            invalid_option_penalty=invalid_option_penalty,
            include_random_option=include_random_option,
            random_option_termination_steps=random_option_termination_steps,
            capture_primitive_transitions=capture_primitive_transitions,
            max_option_steps=max_option_steps,
            include_step_count_in_obs=include_step_count_in_obs,
        )
        Wrapper.__init__(self, env)
        if reward_type not in {"smdp", "intra_option"}:
            raise ValueError("reward_type must be 'smdp' or 'intra_option'.")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1].")
        if max_option_steps is not None and max_option_steps <= 0:
            raise ValueError("max_option_steps must be positive when provided.")

        self.reward_type: Literal["smdp", "intra_option"] = reward_type
        self.gamma: float = float(gamma)
        self.invalid_option_penalty: float = float(invalid_option_penalty)
        self.max_option_steps: int = max_option_steps
        self.include_step_count_in_obs: bool = include_step_count_in_obs

        final_options: list[BaseOption] = options
        if include_random_option:
            final_options.append(
                RandomOption(
                    action_space=self.env.action_space,
                    termination_steps=random_option_termination_steps,
                )
            )
        if len(final_options) == 0:
            raise ValueError("MetaControllerEnvWrapper requires at least one option.")

        self.options: list[BaseOption] = final_options
        if capture_primitive_transitions is None:
            capture_primitive_transitions = reward_type == "intra_option"
        self.capture_primitive_transitions: bool = bool(capture_primitive_transitions)

        self.observation_space: spaces.Space = (
            self._build_observation_space_with_step_count(self.env.observation_space)
            if include_step_count_in_obs
            else self.env.observation_space
        )

        self.action_space: spaces.Discrete = spaces.Discrete(len(self.options))
        self._last_obs: SB3ObsType | None = None
        self._episode_primitive_steps: int = 0

    def _add_step_count_to_obs(self, obs: SB3ObsType, policy_step: int) -> SB3ObsType:
        """Add normalized low-level policy step count to dict observations."""

        denom = max(1, self.max_option_steps)
        step_ratio = np.float32(policy_step / denom)
        step_count = np.array([step_ratio], dtype=np.float32)

        if not isinstance(obs, dict):
            return {
                "observation": obs,
                "step_count": step_count,
            }
        obs_with_step = dict(obs)
        obs_with_step["step_count"] = step_count
        return obs_with_step

    def _build_observation_space_with_step_count(
        self,
        observation_space: spaces.Space,
    ) -> spaces.Space:
        """Extend Dict observation spaces with a normalized step_count field."""
        step_count_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )
        if not isinstance(observation_space, spaces.Dict):
            return spaces.Dict(
                observation=observation_space,
                step_count=step_count_space,
            )
        else:
            spaces_dict = dict(observation_space.spaces)
            spaces_dict["step_count"] = step_count_space
            return spaces.Dict(spaces_dict)

    def reset(self, **kwargs: Any) -> tuple[SB3ObsType, dict[str, Any]]:
        """Reset wrapped env and internal observation cache."""
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._episode_primitive_steps = 0
        return obs, info

    def step(self, action: int) -> tuple[SB3ObsType, float, bool, bool, dict[str, Any]]:
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
        obs: SB3ObsType = current_obs
        total_reward: float = 0.0
        effective_gamma: float = 1.0
        steps: int = 0
        terminated: bool = False
        truncated: bool = False
        option_terminated: bool = False
        option_truncated: bool = False
        last_info: dict[str, Any] = {}
        primitive_transitions: list[dict[str, Any]] = []

        while True:
            if option.policy is None and option.has_policy_factory():
                option.ensure_policy_initialized()
            primitive_action: ActType = option.predict(obs)
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
                        "obs": copy.deepcopy(obs),
                        "action": copy.deepcopy(primitive_action),
                        "reward": float(reward),
                        "next_obs": copy.deepcopy(next_obs),
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": step_info,
                    }
                )

            steps += 1
            self._episode_primitive_steps += 1
            option_terminated = option.termination_condition(next_obs)
            option_truncated = steps >= self.max_option_steps
            obs: SB3ObsType = next_obs
            last_info: dict[str, Any] = dict(step_info)

            if terminated or truncated or option_terminated or option_truncated:
                break

        if option_terminated or option_truncated:
            option.remove_policy()

        self._last_obs = obs
        info = dict(last_info)
        info["invalid_option"] = False
        info["option_index"] = int(action)
        info["option_terminated"] = option_terminated
        info["meta_option_steps"] = steps
        info["cumulative_primitive_steps"] = self._episode_primitive_steps
        info["effective_gamma"] = float(effective_gamma)
        info["reward_type"] = self.reward_type
        if self.capture_primitive_transitions:
            info["primitive_transitions"] = primitive_transitions

        return obs, float(total_reward), bool(terminated), bool(truncated), info


class PrimitiveStepTimeLimit(
    Wrapper[SB3ObsType, int, SB3ObsType, int], RecordConstructorArgs
):
    """Wrap MetaControllerEnvWrapper to truncate episodes based on primitive step count.

    Unlike gymnasium's TimeLimit which counts macro steps (option selections), this wrapper
    truncates based on total primitive actions executed by the low-level policy. This provides
    clearer semantics when training hierarchical policies.

    Parameters
    ----------
    env : gym.Env
        Should be a MetaControllerEnvWrapper. Will work with other wrappers but semantics
        depend on whether they expose 'meta_option_steps' in the info dict.
    max_episode_steps : int
        Maximum number of primitive steps allowed per episode. Episode truncates when
        cumulative primitive steps >= max_episode_steps.

    Notes
    -----
    This wrapper should wrap MetaControllerEnvWrapper directly. Example:

        env = MetaControllerEnvWrapper(env, options=[...], include_step_count_in_obs=False)
        env = PrimitiveStepTimeLimit(env, max_episode_steps=500)

    The step count in the observation is separate from truncation logic and can be
    enabled independently via include_step_count_in_obs on MetaControllerEnvWrapper.
    """

    def __init__(self, env: gym.Env, max_episode_steps: int):
        """Initialize the wrapper.

        Parameters
        ----------
        env : gym.Env
            Environment to wrap (typically MetaControllerEnvWrapper).
        max_episode_steps : int
            Number of primitive steps before truncation.
        """
        assert isinstance(max_episode_steps, int) and max_episode_steps > 0, (
            f"max_episode_steps must be positive, got {max_episode_steps}"
        )

        RecordConstructorArgs.__init__(self, max_episode_steps=max_episode_steps)
        Wrapper.__init__(self, env)

        self._max_episode_steps = max_episode_steps
        self._cumulative_primitive_steps = 0

    def reset(self, **kwargs: Any) -> tuple[SB3ObsType, dict[str, Any]]:
        """Reset environment and step counter."""
        obs, info = self.env.reset(**kwargs)
        self._cumulative_primitive_steps = 0
        return obs, info

    def step(
        self, action: int
    ) -> tuple[SB3ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Step through environment and check primitive step limit.

        Parameters
        ----------
        action : int
            High-level action (option index).

        Returns
        -------
        tuple
            (observation, reward, terminated, truncated, info) with truncated=True
            if primitive step limit reached.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Get the number of primitive steps executed in this option
        meta_option_steps = info.get("meta_option_steps", 0)
        self._cumulative_primitive_steps += meta_option_steps

        # Check if we've exceeded the primitive step limit
        if self._cumulative_primitive_steps >= self._max_episode_steps:
            truncated = True
            # Mark in info that truncation was due to time limit
            info.setdefault("TimeLimit.truncated", True)

        return obs, reward, terminated, truncated, info


class MetaControllerPrimitiveStepTimeLimitWrapper(
    Wrapper[SB3ObsType, int, SB3ObsType, int],
    RecordConstructorArgs,
):
    """Compose MetaControllerEnvWrapper and PrimitiveStepTimeLimit in one class.

    This convenience wrapper is equivalent to:

        env = MetaControllerEnvWrapper(...)
        env = PrimitiveStepTimeLimit(env, max_episode_steps=...)

    Parameters
    ----------
    env : gym.Env
        Base environment.
    options : list[BaseOption]
        Trained options available to the meta-controller.
    max_episode_steps : int
        Primitive-step episode horizon used by PrimitiveStepTimeLimit.
    reward_type : Literal["smdp", "intra_option"], default="smdp"
        Reward semantics at each macro step.
    gamma : float, default=0.99
        Discount used for SMDP return accumulation.
    invalid_option_penalty : float, default=-1.0
        Reward emitted when selected option cannot be initiated.
    include_random_option : bool, default=True
        Whether to append a built-in random option.
    random_option_termination_steps : int, default=1
        Primitive-step horizon for the random option.
    capture_primitive_transitions : bool | None, default=None
        Whether primitive transitions are included in the info dict.
    max_option_steps : int, default=50
        Safety cap on primitive steps executed by one option.
    include_step_count_in_obs : bool, default=False
        Whether to include normalized option step count in observations.
    """

    def __init__(
        self,
        env: gym.Env,
        options: list[BaseOption],
        max_episode_steps: int,
        reward_type: Literal["smdp", "intra_option"] = "smdp",
        gamma: float = 0.99,
        invalid_option_penalty: float = -1.0,
        include_random_option: bool = True,
        random_option_termination_steps: int = 1,
        capture_primitive_transitions: Optional[bool] = None,
        max_option_steps: int = 50,
        include_step_count_in_obs: bool = False,
    ) -> None:
        RecordConstructorArgs.__init__(
            self,
            env=env,
            options=options,
            max_episode_steps=max_episode_steps,
            reward_type=reward_type,
            gamma=gamma,
            invalid_option_penalty=invalid_option_penalty,
            include_random_option=include_random_option,
            random_option_termination_steps=random_option_termination_steps,
            capture_primitive_transitions=capture_primitive_transitions,
            max_option_steps=max_option_steps,
            include_step_count_in_obs=include_step_count_in_obs,
        )

        meta_env = MetaControllerEnvWrapper(
            env=env,
            options=options,
            reward_type=reward_type,
            gamma=gamma,
            invalid_option_penalty=invalid_option_penalty,
            include_random_option=include_random_option,
            random_option_termination_steps=random_option_termination_steps,
            capture_primitive_transitions=capture_primitive_transitions,
            max_option_steps=max_option_steps,
            include_step_count_in_obs=include_step_count_in_obs,
        )
        wrapped_env = PrimitiveStepTimeLimit(
            env=meta_env,
            max_episode_steps=max_episode_steps,
        )
        Wrapper.__init__(self, wrapped_env)

    @property
    def meta_controller_env(self) -> MetaControllerEnvWrapper:
        """Return the inner MetaControllerEnvWrapper instance."""
        primitive_limit_env = cast(PrimitiveStepTimeLimit, self.env)
        return cast(MetaControllerEnvWrapper, primitive_limit_env.env)

    @property
    def primitive_step_time_limit_env(self) -> PrimitiveStepTimeLimit:
        """Return the inner PrimitiveStepTimeLimit instance."""
        return cast(PrimitiveStepTimeLimit, self.env)


__all__ = [
    "SubpolicyTrainingWrapper",
    "MetaControllerEnvWrapper",
]
