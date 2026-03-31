from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from sb3_hrl.option import BaseOption, MetaControllerEnvWrapper


class _CounterEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, terminate_at: int = 100) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1000.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.terminate_at = terminate_at
        self.count = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del seed, options
        self.count = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action: int):
        del action
        self.count += 1
        obs = np.array([float(self.count)], dtype=np.float32)
        reward = 1.0
        terminated = self.count >= self.terminate_at
        truncated = False
        return obs, reward, terminated, truncated, {"env_count": self.count}


class _FixedOption(BaseOption):
    def __init__(self, terminate_at_state: float, initiable: bool = True) -> None:
        super().__init__(policy=None)
        self.terminate_at_state = terminate_at_state
        self.initiable = initiable

    def initiation_set(self, obs: Any) -> bool:
        del obs
        return self.initiable

    def termination_condition(self, obs: Any) -> bool:
        return bool(np.asarray(obs)[0] >= self.terminate_at_state)

    def intrinsic_reward(
        self,
        obs: Any,
        action: Any,
        next_obs: Any,
        external_reward: float,
        done: bool,
    ) -> float:
        del obs, action, next_obs, done
        return float(external_reward)

    def predict(self, obs: Any, deterministic: bool = True):
        del obs, deterministic
        return 0


def test_meta_wrapper_invalid_option_penalty() -> None:
    env = _CounterEnv()
    invalid_option = _FixedOption(terminate_at_state=3.0, initiable=False)
    wrapped = MetaControllerEnvWrapper(
        env,
        options=[invalid_option],
        include_random_option=False,
        invalid_option_penalty=-1.0,
    )

    obs, _ = wrapped.reset()
    next_obs, reward, terminated, truncated, info = wrapped.step(0)

    assert np.allclose(next_obs, obs)
    assert reward == -1.0
    assert not terminated
    assert not truncated
    assert info["invalid_option"]
    assert info["meta_option_steps"] == 0


def test_meta_wrapper_smdp_discounted_return_and_effective_gamma() -> None:
    env = _CounterEnv(terminate_at=100)
    option = _FixedOption(terminate_at_state=3.0, initiable=True)
    wrapped = MetaControllerEnvWrapper(
        env,
        options=[option],
        include_random_option=False,
        reward_type="smdp",
        gamma=0.9,
    )

    obs, _ = wrapped.reset()
    assert float(obs[0]) == 0.0

    next_obs, reward, terminated, truncated, info = wrapped.step(0)
    assert float(next_obs[0]) == 3.0
    assert reward == pytest.approx(1.0 + 0.9 + 0.81)
    assert not terminated
    assert not truncated
    assert info["meta_option_steps"] == 3
    assert info["effective_gamma"] == pytest.approx(0.9**3)


def test_meta_wrapper_propagates_env_termination() -> None:
    env = _CounterEnv(terminate_at=2)
    option = _FixedOption(terminate_at_state=10.0, initiable=True)
    wrapped = MetaControllerEnvWrapper(
        env,
        options=[option],
        include_random_option=False,
        reward_type="smdp",
        gamma=0.9,
    )

    wrapped.reset()
    _, _, terminated, truncated, info = wrapped.step(0)
    assert terminated
    assert not truncated
    assert info["meta_option_steps"] == 2


def test_meta_wrapper_intra_option_exposes_primitive_transitions() -> None:
    env = _CounterEnv(terminate_at=100)
    option = _FixedOption(terminate_at_state=2.0, initiable=True)
    wrapped = MetaControllerEnvWrapper(
        env,
        options=[option],
        include_random_option=False,
        reward_type="intra_option",
        gamma=0.9,
    )

    wrapped.reset()
    _, reward, _, _, info = wrapped.step(0)
    assert reward == pytest.approx(2.0)
    assert info["reward_type"] == "intra_option"
    assert "primitive_transitions" in info
    assert len(info["primitive_transitions"]) == 2


def test_meta_wrapper_includes_random_option_in_action_space() -> None:
    env = _CounterEnv(terminate_at=100)
    option = _FixedOption(terminate_at_state=1.0, initiable=True)
    wrapped = MetaControllerEnvWrapper(
        env,
        options=[option],
        include_random_option=True,
        random_option_termination_steps=1,
    )

    assert wrapped.action_space.n == 2

    wrapped.reset()
    _, _, _, _, info = wrapped.step(1)
    assert not info["invalid_option"]
    assert info["meta_option_steps"] == 1
