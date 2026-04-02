from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_hrl.option.callbacks import PrimitiveStepCountCallback
from sb3_hrl.option.policies import PrimitiveStepPPO


class PrimitiveInfoEnv(gym.Env[np.ndarray, int]):
    """Tiny env that emits configurable meta_option_steps in info."""

    metadata = {"render_modes": []}

    def __init__(self, primitive_step_sequence: Sequence[int], episode_len: int = 32):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)
        self._primitive_step_sequence = [int(v) for v in primitive_step_sequence]
        self._episode_len = int(episode_len)
        self._step_idx = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._step_idx = 0
        obs = np.zeros((1,), dtype=np.float32)
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        del action
        self._step_idx += 1

        idx = (self._step_idx - 1) % len(self._primitive_step_sequence)
        primitive_steps = int(self._primitive_step_sequence[idx])

        terminated = False
        truncated = self._step_idx >= self._episode_len
        obs = np.array([self._step_idx % 2], dtype=np.float32)
        reward = 1.0
        info = {"meta_option_steps": primitive_steps}
        return obs, reward, terminated, truncated, info


def _build_model(env: gym.Env | DummyVecEnv, n_steps: int) -> PrimitiveStepPPO:
    return PrimitiveStepPPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=2,
        n_epochs=1,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=0,
        policy_kwargs={"net_arch": [16, 16]},
    )


def test_learn_counts_primitive_steps_single_env() -> None:
    env = PrimitiveInfoEnv([3], episode_len=64)
    model = _build_model(env, n_steps=2)

    model.learn(total_timesteps=12)

    assert model.num_timesteps == 12


def test_learn_counts_primitive_steps_vec_env_sum() -> None:
    env = DummyVecEnv(
        [
            lambda: PrimitiveInfoEnv([2], episode_len=64),
            lambda: PrimitiveInfoEnv([5], episode_len=64),
        ]
    )
    model = _build_model(env, n_steps=1)

    model.learn(total_timesteps=14)

    assert model.num_timesteps == 14


def test_callback_does_not_double_count_with_primitive_step_ppo() -> None:
    env = PrimitiveInfoEnv([4], episode_len=64)
    model = _build_model(env, n_steps=2)
    callback = PrimitiveStepCountCallback()

    model.learn(total_timesteps=8, callback=callback)

    assert model.num_timesteps == 8


def test_sampling_budget_uses_meta_timesteps() -> None:
    env = PrimitiveInfoEnv([4], episode_len=64)
    model = _build_model(env, n_steps=3)

    model.learn(total_timesteps=10)

    # One rollout should collect exactly 3 macro steps (12 primitive steps)
    # before the outer learn loop checks the primitive total_timesteps budget.
    assert model.num_timesteps == 12
    assert model._n_updates == 1
