from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sb3_hrl.option import BaseOption, SubpolicyTrainingWrapper


class _CounterEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, terminate_at: int = 100) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1000.0, shape=(1,), dtype=np.float32
        )
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


class _TerminatingOption(BaseOption):
    def initiation_set(self, obs: Any) -> bool:
        del obs
        return True

    def termination_condition(self, obs: Any) -> bool:
        return bool(np.asarray(obs)[0] >= 2.0)

    def intrinsic_reward(
        self,
        obs: Any,
        action: Any,
        next_obs: Any,
        external_reward: float,
        done: bool,
    ) -> float:
        del action, external_reward, done
        return float(np.asarray(next_obs)[0] - np.asarray(obs)[0])


def test_subpolicy_wrapper_replaces_reward_and_forces_termination() -> None:
    env = _CounterEnv(terminate_at=100)
    option = _TerminatingOption()
    wrapped = SubpolicyTrainingWrapper(env, option)

    obs, _ = wrapped.reset()
    assert float(obs[0]) == 0.0

    next_obs, reward, terminated, truncated, info = wrapped.step(0)
    assert float(next_obs[0]) == 1.0
    assert reward == 1.0
    assert not terminated
    assert not truncated
    assert info["subpolicy_external_reward"] == 1.0
    assert info["subpolicy_intrinsic_reward"] == 1.0

    next_obs, reward, terminated, truncated, info = wrapped.step(0)
    assert float(next_obs[0]) == 2.0
    assert reward == 1.0
    assert terminated
    assert not truncated
    assert info["option_terminated"]
