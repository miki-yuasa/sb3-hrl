"""Environment wrappers for ALLO-driven HRL training."""

from __future__ import annotations

from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from gymnasium.spaces import utils as space_utils
from stable_baselines3.common.base_class import BaseAlgorithm

from .allo import ALLOAlgorithm


class LaplacianRewardWrapper(gym.Wrapper):
    """Intrinsic reward wrapper based on Laplacian feature progress."""

    def __init__(
        self,
        env: gym.Env,
        allo: Union[ALLOAlgorithm, th.nn.Module],
        eigenvector_index: int,
        device: Union[str, th.device] = "cpu",
    ) -> None:
        super().__init__(env)
        self.allo = allo
        self.eigenvector_index = int(eigenvector_index)
        self.device = th.device(device)
        self._prev_obs: Optional[np.ndarray] = None

        if self.eigenvector_index < 0:
            raise ValueError("eigenvector_index must be non-negative.")

    def _encode(self, observation: np.ndarray) -> th.Tensor:
        """Encode a single observation into a feature vector."""
        flat = space_utils.flatten(self.observation_space, observation)
        flat = np.asarray(flat, dtype=np.float32).reshape(1, -1)
        with th.no_grad():
            if isinstance(self.allo, ALLOAlgorithm):
                features = self.allo.encode(flat)
            else:
                inputs = th.as_tensor(flat, dtype=th.float32, device=self.device)
                features = self.allo(inputs)
            return features[0]

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset wrapped environment and internal previous state."""
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = np.asarray(obs)
        return obs, info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step wrapped env and replace reward with intrinsic feature progress."""
        if self._prev_obs is None:
            raise RuntimeError("Call reset() before step() in LaplacianRewardWrapper.")

        obs_next, _reward, terminated, truncated, info = self.env.step(action)
        phi_prev = self._encode(self._prev_obs)
        phi_next = self._encode(np.asarray(obs_next))

        if self.eigenvector_index >= int(phi_prev.shape[0]):
            raise IndexError("eigenvector_index exceeds learned representation size.")

        intrinsic_reward = float(
            phi_next[self.eigenvector_index].item()
            - phi_prev[self.eigenvector_index].item()
        )

        self._prev_obs = np.asarray(obs_next)
        return obs_next, intrinsic_reward, terminated, truncated, info


class HRLMetaEnv(gym.Env[np.ndarray, int]):
    """Hierarchical environment where actions select option policies."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: gym.Env,
        subpolicies: list[BaseAlgorithm],
        option_horizon: int = 10,
    ) -> None:
        super().__init__()
        if len(subpolicies) == 0:
            raise ValueError("HRLMetaEnv requires at least one subpolicy.")
        if option_horizon <= 0:
            raise ValueError("option_horizon must be positive.")

        self.env = env
        self.subpolicies = subpolicies
        self.option_horizon = int(option_horizon)

        self.observation_space = env.observation_space
        self.action_space = spaces.Discrete(len(subpolicies))
        self._last_obs: Optional[np.ndarray] = None

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset base environment."""
        obs, info = self.env.reset(**kwargs)
        self._last_obs = np.asarray(obs)
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute selected option for option_horizon low-level steps."""
        if self._last_obs is None:
            raise RuntimeError("Call reset() before step() in HRLMetaEnv.")
        if not self.action_space.contains(action):
            raise ValueError("Action index out of range for HRLMetaEnv.")

        option = self.subpolicies[int(action)]
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        obs = self._last_obs

        for _ in range(self.option_horizon):
            primitive_action, _ = option.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(primitive_action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        self._last_obs = np.asarray(obs)
        info = dict(info)
        info["meta_option_steps"] = self.option_horizon
        return obs, total_reward, terminated, truncated, info

    def render(self):
        """Delegate rendering to wrapped env."""
        return self.env.render()

    def close(self) -> None:
        """Close wrapped env resources."""
        self.env.close()


__all__ = ["LaplacianRewardWrapper", "HRLMetaEnv"]
