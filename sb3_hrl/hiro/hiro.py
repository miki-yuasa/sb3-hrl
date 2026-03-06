"""HIRO algorithm implementation built on Stable-Baselines3 primitives."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union, cast

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from gymnasium.spaces import utils as space_utils
from stable_baselines3 import TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    ReplayBufferSamples,
)
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv

from .policies import (
    SubgoalProjectionWrapper,
    build_worker_observation_space,
    flatten_observation,
    make_worker_observation,
)


@dataclass
class _MacroTransitionAccumulator:
    """Mutable macro-transition state used while collecting rollouts.

    Attributes
    ----------
    start_obs : np.ndarray | None
            Flattened manager start observation at macro-step start.
    start_goal : np.ndarray | None
            Initial subgoal proposed by the manager for this macro-step.
    total_reward : float
            Accumulated extrinsic reward over the macro-transition.
    micro_obs : list[np.ndarray]
            Sequence of flattened states ``s_t, ..., s_{t+c-1}``.
    micro_next_obs : list[np.ndarray]
            Sequence of flattened next states ``s_{t+1}, ..., s_{t+c}``.
    micro_actions : list[np.ndarray]
            Sequence of scaled worker actions in ``[-1, 1]``.
        micro_projected_obs : list[np.ndarray]
            Sequence of projected states ``h(s_t), ..., h(s_{t+c-1})``.
        micro_projected_next_obs : list[np.ndarray]
            Sequence of projected next states ``h(s_{t+1}), ..., h(s_{t+c})``.
    """

    start_obs: Optional[np.ndarray] = None
    start_goal: Optional[np.ndarray] = None
    total_reward: float = 0.0
    micro_obs: list[np.ndarray] = None  # type: ignore[assignment]
    micro_next_obs: list[np.ndarray] = None  # type: ignore[assignment]
    micro_actions: list[np.ndarray] = None  # type: ignore[assignment]
    micro_projected_obs: list[np.ndarray] = None  # type: ignore[assignment]
    micro_projected_next_obs: list[np.ndarray] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset accumulator content for a new macro-step."""
        self.start_obs = None
        self.start_goal = None
        self.total_reward = 0.0
        self.micro_obs = []
        self.micro_next_obs = []
        self.micro_actions = []
        self.micro_projected_obs = []
        self.micro_projected_next_obs = []


class _SpaceOverrideEnv(gym.Env[np.ndarray, np.ndarray]):
    """Minimal environment proxy with overridden spaces.

    Parameters
    ----------
    base_env : gym.Env
            Backing environment used for ``step`` and ``reset`` calls.
    observation_space : spaces.Space
            Observation space exposed by the proxy.
    action_space : spaces.Box
            Action space exposed by the proxy.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_env: gym.Env[Any, Any],
        observation_space: spaces.Space,
        action_space: spaces.Box,
    ) -> None:
        super().__init__()
        self.base_env = base_env
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ):
        """Delegate reset to the base environment."""
        return self.base_env.reset(seed=seed, options=options)

    def step(self, action: np.ndarray):
        """Delegate step to the base environment."""
        return self.base_env.step(action)


class HIROReplayBuffer(ReplayBuffer):
    """Replay buffer for HIRO manager transitions with off-policy correction.

    Parameters
    ----------
    subgoal_freq : int
            High-level action period ``c``.
        state_to_goal_proj_fn : callable
            Projection function ``h(s)`` used only for API compatibility.
    worker_action_dim : int
            Low-level action dimension (scaled action stored in manager buffer metadata).
    correction_candidate_count : int, default=10
            Number of candidate subgoals evaluated during relabeling.
    correction_noise_scale : float, default=0.5
            Fraction of subgoal range used as candidate Gaussian std.
    correction_action_sigma : float, default=0.2
            Std used to score deterministic low-level actions as Gaussian log-likelihood.

    Notes
    -----
    SB3 TD3 uses deterministic actors. To compute HIRO's correction score
    ``sum log pi(a|s,g)``, this buffer models the low-level policy as an
    isotropic Gaussian centered at the current deterministic action mean.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        *,
        subgoal_freq: int,
        state_to_goal_proj_fn: Callable[[np.ndarray], np.ndarray],
        worker_action_dim: int,
        correction_candidate_count: int = 10,
        correction_noise_scale: float = 0.5,
        correction_action_sigma: float = 0.2,
    ) -> None:
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        if not isinstance(action_space, spaces.Box):
            raise TypeError("HIROReplayBuffer requires Box action space for subgoals.")
        self._subgoal_action_space = action_space

        self.subgoal_freq = subgoal_freq
        self.state_to_goal_proj_fn = state_to_goal_proj_fn
        self.worker_action_dim = worker_action_dim
        self.correction_candidate_count = correction_candidate_count
        self.correction_noise_scale = correction_noise_scale
        self.correction_action_sigma = correction_action_sigma

        self.micro_obs = np.zeros(
            (self.buffer_size, self.n_envs, subgoal_freq, self.obs_shape[0]),
            dtype=np.float32,
        )
        self.micro_next_obs = np.zeros(
            (self.buffer_size, self.n_envs, subgoal_freq, self.obs_shape[0]),
            dtype=np.float32,
        )
        self.micro_actions = np.zeros(
            (self.buffer_size, self.n_envs, subgoal_freq, worker_action_dim),
            dtype=np.float32,
        )
        goal_dim = int(self._subgoal_action_space.shape[0])
        self.micro_projected_obs = np.zeros(
            (self.buffer_size, self.n_envs, subgoal_freq, goal_dim),
            dtype=np.float32,
        )
        self.micro_projected_next_obs = np.zeros(
            (self.buffer_size, self.n_envs, subgoal_freq, goal_dim),
            dtype=np.float32,
        )
        self.micro_lengths = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)

        self._low_level_action_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def set_low_level_action_fn(
        self, action_fn: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        """Set callable used to evaluate low-level action likelihoods.

        Parameters
        ----------
        action_fn : callable
                Function mapping worker observations to scaled actions in ``[-1, 1]``.
        """
        self._low_level_action_fn = action_fn

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Add one manager transition and attached low-level trajectory.

        Parameters
        ----------
        obs : np.ndarray
                Manager observation at macro-step start with shape ``(n_envs, obs_dim)``.
        next_obs : np.ndarray
                Manager next observation at macro-step end with shape ``(n_envs, obs_dim)``.
        action : np.ndarray
                Scaled manager action/subgoal with shape ``(n_envs, goal_dim)``.
        reward : np.ndarray
                Aggregated extrinsic reward.
        done : np.ndarray
                Done flags.
        infos : list[dict[str, Any]]
                Info dicts from env steps.
        micro_observations : np.ndarray
                Flattened state sequence with shape ``(T, obs_dim)``, ``T <= c``.
        micro_next_observations : np.ndarray
                Flattened next-state sequence with shape ``(T, obs_dim)``.
        micro_actions : np.ndarray
                Scaled worker actions with shape ``(T, worker_action_dim)``.
        micro_projected_observations : np.ndarray
            Projected state sequence with shape ``(T, goal_dim)``.
        micro_projected_next_observations : np.ndarray
            Projected next-state sequence with shape ``(T, goal_dim)``.
        """
        micro_observations = cast(np.ndarray, kwargs["micro_observations"])
        micro_next_observations = cast(np.ndarray, kwargs["micro_next_observations"])
        micro_actions = cast(np.ndarray, kwargs["micro_actions"])
        micro_projected_observations = cast(
            np.ndarray, kwargs["micro_projected_observations"]
        )
        micro_projected_next_observations = cast(
            np.ndarray, kwargs["micro_projected_next_observations"]
        )

        length = int(micro_observations.shape[0])
        if length == 0:
            raise ValueError(
                "Manager transition must include at least one low-level step."
            )
        if length > self.subgoal_freq:
            raise ValueError("micro_observations length cannot exceed subgoal_freq.")

        pos = self.pos
        self.micro_lengths[pos] = length
        self.micro_obs[pos, 0, :length] = micro_observations.astype(np.float32)
        self.micro_next_obs[pos, 0, :length] = micro_next_observations.astype(
            np.float32
        )
        self.micro_actions[pos, 0, :length] = micro_actions.astype(np.float32)
        self.micro_projected_obs[pos, 0, :length] = micro_projected_observations.astype(
            np.float32
        )
        self.micro_projected_next_obs[pos, 0, :length] = (
            micro_projected_next_observations.astype(np.float32)
        )

        super().add(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=reward,
            done=done,
            infos=infos,
        )

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> ReplayBufferSamples:
        """Sample manager transitions and apply HIRO off-policy correction."""
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :],
                env,
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, env_indices, :], env
            )

        obs = self._normalize_obs(self.observations[batch_inds, env_indices, :], env)
        actions = self.actions[batch_inds, env_indices, :].copy()
        rewards = self._normalize_reward(
            self.rewards[batch_inds, env_indices].reshape(-1, 1), env
        )
        dones = (
            self.dones[batch_inds, env_indices]
            * (1 - self.timeouts[batch_inds, env_indices])
        ).reshape(-1, 1)

        if self._low_level_action_fn is not None:
            actions = self._relabel_goals(
                batch_inds=batch_inds, env_indices=env_indices, current_actions=actions
            )

        return ReplayBufferSamples(
            observations=self.to_torch(cast(np.ndarray, obs)),
            actions=self.to_torch(actions),
            next_observations=self.to_torch(cast(np.ndarray, next_obs)),
            dones=self.to_torch(dones),
            rewards=self.to_torch(rewards),
        )

    def _relabel_goals(
        self,
        batch_inds: np.ndarray,
        env_indices: np.ndarray,
        current_actions: np.ndarray,
    ) -> np.ndarray:
        """Relabel sampled goals using HIRO off-policy correction.

        Parameters
        ----------
        batch_inds : np.ndarray
                Sampled buffer indices.
        env_indices : np.ndarray
                Sampled environment indices.
        current_actions : np.ndarray
                Currently stored scaled manager actions.

        Returns
        -------
        np.ndarray
                Relabeled scaled manager actions.
        """
        if self._low_level_action_fn is None:
            return current_actions

        relabeled = current_actions.copy()
        sigma = max(self.correction_action_sigma, 1e-6)

        for sample_i, (buffer_idx, env_idx) in enumerate(zip(batch_inds, env_indices)):
            length = int(self.micro_lengths[buffer_idx, env_idx])
            if length <= 0:
                continue

            states = self.micro_obs[buffer_idx, env_idx, :length]
            actions = self.micro_actions[buffer_idx, env_idx, :length]
            projected_states = self.micro_projected_obs[buffer_idx, env_idx, :length]
            projected_next_states = self.micro_projected_next_obs[
                buffer_idx, env_idx, :length
            ]

            old_goal = self._unscale_action(current_actions[sample_i])
            start_proj = projected_states[0]
            end_proj = projected_next_states[length - 1]
            delta_goal = end_proj - start_proj

            candidates = [old_goal.astype(np.float32), delta_goal.astype(np.float32)]

            goal_std = self.correction_noise_scale * np.maximum(
                self._subgoal_action_space.high - self._subgoal_action_space.low,
                1e-6,
            )
            remaining = max(self.correction_candidate_count - 2, 0)
            if remaining > 0:
                noise = np.random.normal(
                    loc=0.0, scale=goal_std, size=(remaining, old_goal.shape[0])
                ).astype(np.float32)
                sampled = delta_goal[None, :] + noise
                sampled = np.clip(
                    sampled,
                    self._subgoal_action_space.low,
                    self._subgoal_action_space.high,
                )
                candidates.extend(list(sampled))

            best_goal = old_goal
            best_score = -np.inf

            for candidate in candidates:
                score = 0.0
                current_goal = candidate.astype(np.float32, copy=True)

                for step_idx in range(length):
                    worker_obs = np.concatenate(
                        [states[step_idx], current_goal], axis=0
                    ).astype(np.float32, copy=False)
                    predicted_action = self._low_level_action_fn(worker_obs[None, :])[0]
                    diff = actions[step_idx] - predicted_action
                    score += float(-0.5 * np.sum((diff / sigma) ** 2))

                    projected_state = projected_states[step_idx]
                    projected_next_state = projected_next_states[step_idx]
                    current_goal = projected_state + current_goal - projected_next_state

                if score > best_score:
                    best_score = score
                    best_goal = candidate

            relabeled[sample_i] = self._scale_action(best_goal)

        return relabeled

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale action from environment range to ``[-1, 1]``."""
        low = self._subgoal_action_space.low
        high = self._subgoal_action_space.high
        return (2.0 * (action - low) / (high - low) - 1.0).astype(np.float32)

    def _unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """Unscale action from ``[-1, 1]`` to environment range."""
        low = self._subgoal_action_space.low
        high = self._subgoal_action_space.high
        return (low + (0.5 * (scaled_action + 1.0) * (high - low))).astype(np.float32)


class HIRO(BaseAlgorithm):
    """Hierarchical Reinforcement Learning with Off-policy correction (HIRO).

    Parameters
    ----------
    policy : str | type
            SB3 policy used for both manager and worker TD3 agents.
    env : GymEnv | str
            Training environment with continuous action space.
    learning_rate : float, default=1e-3
            Learning rate used by both TD3 agents.
    buffer_size : int, default=1_000_000
            Replay buffer size for both worker and manager.
    learning_starts : int, default=1000
            Number of environment steps collected before gradient updates start.
    batch_size : int, default=256
            Minibatch size.
    tau : float, default=0.005
            Polyak averaging coefficient.
    gamma : float, default=0.99
            Discount factor.
    train_freq : int, default=1
            Number of env steps between training calls.
    gradient_steps : int, default=1
            Number of gradient updates per training call.
    subgoal_freq : int, default=10
            Subgoal period ``c``.
    subgoal_space : spaces.Box, optional
            Manager action space. If omitted and projection is identity, observation
            bounds are reused.
    state_to_goal_proj_fn : callable, optional
            Projection function ``h(s)`` from flattened state to subgoal.
    manager_exploration_noise : float, default=0.1
            Gaussian exploration std for manager actions.
    worker_exploration_noise : float, default=0.1
            Gaussian exploration std for worker actions.
    correction_candidate_count : int, default=10
            Candidate goals evaluated in manager replay correction.
    correction_noise_scale : float, default=0.5
            Candidate sampling std scale based on subgoal range.
    correction_action_sigma : float, default=0.2
            Low-level Gaussian action model std for correction scoring.
    manager_kwargs : dict, optional
            Extra kwargs for manager TD3.
    worker_kwargs : dict, optional
            Extra kwargs for worker TD3.
    stats_window_size : int, default=100
            Logging window size.
    tensorboard_log : str, optional
            TensorBoard log path.
    verbose : int, default=0
            Verbosity level.
    device : str | th.device, default="auto"
            Torch device.
    seed : int, optional
            RNG seed.

    Notes
    -----
    This class keeps two SB3 TD3 instances internally and handles rollout
    collection manually to enforce HIRO's two time scales and intrinsic reward.
    """

    policy_aliases = TD3.policy_aliases

    def __init__(
        self,
        policy: Union[str, type],
        env: Union[GymEnv, str],
        learning_rate: float = 1e-3,
        buffer_size: int = 1_000_000,
        learning_starts: int = 1_000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        subgoal_freq: int = 10,
        subgoal_space: Optional[spaces.Box] = None,
        state_to_goal_proj_fn: Optional[
            Callable[[Union[np.ndarray, dict[str, np.ndarray]]], np.ndarray]
        ] = None,
        manager_exploration_noise: float = 0.1,
        worker_exploration_noise: float = 0.1,
        correction_candidate_count: int = 10,
        correction_noise_scale: float = 0.5,
        correction_action_sigma: float = 0.2,
        manager_kwargs: Optional[dict[str, Any]] = None,
        worker_kwargs: Optional[dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        seed: Optional[int] = None,
    ) -> None:
        base_policy: Union[str, type] = policy
        if (
            isinstance(policy, str)
            and policy in {"MlpPolicy", "CnnPolicy"}
            and not isinstance(env, str)
            and isinstance(env.observation_space, spaces.Dict)
        ):
            # BaseAlgorithm validates policy/obs-space consistency for direct policy creation.
            # HIRO creates internal TD3 models with flattened observations, so this guard
            # should not block dict-observation environments.
            base_policy = "MultiInputPolicy"

        super().__init__(
            policy=base_policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=None,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=False,
            monitor_wrapper=True,
            seed=seed,
            use_sde=False,
            sde_sample_freq=-1,
            supported_action_spaces=(spaces.Box,),
        )

        if not isinstance(self.action_space, spaces.Box):
            raise TypeError(
                "HIRO requires a continuous (Box) environment action space."
            )
        self._env_action_space = self.action_space

        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.subgoal_freq = subgoal_freq
        self.manager_exploration_noise = manager_exploration_noise
        self.worker_exploration_noise = worker_exploration_noise

        self._manager_kwargs = manager_kwargs or {}
        self._worker_kwargs = worker_kwargs or {}
        self._td3_policy = policy

        self._projection = SubgoalProjectionWrapper(
            state_to_goal_proj_fn,
            observation_space=self.observation_space,
        )
        self._flat_obs_space = space_utils.flatten_space(self.observation_space)
        if not isinstance(self._flat_obs_space, spaces.Box):
            raise TypeError(
                "HIRO currently supports observation spaces that flatten to Box."
            )

        if subgoal_space is None:
            if state_to_goal_proj_fn is not None:
                raise ValueError(
                    "subgoal_space must be provided when using a custom projection function."
                )
            subgoal_space = spaces.Box(
                low=self._flat_obs_space.low.astype(np.float32),
                high=self._flat_obs_space.high.astype(np.float32),
                dtype=np.float32,
            )

        self.subgoal_space = subgoal_space
        self.worker_observation_space = build_worker_observation_space(
            self.observation_space, self.subgoal_space
        )

        self.correction_candidate_count = correction_candidate_count
        self.correction_noise_scale = correction_noise_scale
        self.correction_action_sigma = correction_action_sigma

        self.manager: TD3
        self.worker: TD3

        self._active_goal: Optional[np.ndarray] = None
        self._goal_step = 0
        self._macro = _MacroTransitionAccumulator()

        self._setup_model()

    def _setup_model(self) -> None:
        """Initialize manager and worker TD3 models and replay buffers."""
        if self.env is None:
            raise ValueError("Environment must be defined before model setup.")
        if not isinstance(self.env, VecEnv):
            raise TypeError("HIRO expects a vectorized environment internally.")

        vec_env_any = cast(Any, self.env)
        if not hasattr(vec_env_any, "envs"):
            raise TypeError("HIRO expects a DummyVecEnv-like vectorized environment.")
        raw_env = vec_env_any.envs[0]
        manager_env = _SpaceOverrideEnv(
            base_env=raw_env,
            observation_space=self._flat_obs_space,
            action_space=self.subgoal_space,
        )
        worker_env = _SpaceOverrideEnv(
            base_env=raw_env,
            observation_space=self.worker_observation_space,
            action_space=self._env_action_space,
        )

        shared_td3_kwargs: dict[str, Any] = {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "gamma": self.gamma,
            "train_freq": 1,
            "gradient_steps": 1,
            "verbose": self.verbose,
            "device": self.device,
            "seed": self.seed,
        }

        manager_td3_kwargs = shared_td3_kwargs | self._manager_kwargs
        manager_td3_kwargs["replay_buffer_class"] = HIROReplayBuffer
        manager_td3_kwargs["replay_buffer_kwargs"] = {
            "subgoal_freq": self.subgoal_freq,
            "state_to_goal_proj_fn": self._project_state,
            "worker_action_dim": int(self._env_action_space.shape[0]),
            "correction_candidate_count": self.correction_candidate_count,
            "correction_noise_scale": self.correction_noise_scale,
            "correction_action_sigma": self.correction_action_sigma,
        }

        worker_td3_kwargs = shared_td3_kwargs | self._worker_kwargs

        self.manager = TD3(
            self._td3_policy, manager_env, _init_setup_model=True, **manager_td3_kwargs
        )
        self.worker = TD3(
            self._td3_policy, worker_env, _init_setup_model=True, **worker_td3_kwargs
        )

        assert isinstance(self.manager.replay_buffer, HIROReplayBuffer)
        self.manager.replay_buffer.set_low_level_action_fn(
            self._predict_worker_scaled_action
        )

        # Expose manager policy for BaseAlgorithm.predict compatibility.
        self.policy = self.manager.policy

    def _project_state(
        self, observation: Union[np.ndarray, dict[str, np.ndarray]]
    ) -> np.ndarray:
        """Project observation into subgoal coordinates."""
        projected = self._projection(observation)
        if projected.shape != self.subgoal_space.shape:
            raise ValueError(
                f"Projected goal shape {projected.shape} does not match subgoal space {self.subgoal_space.shape}."
            )
        return projected

    def _extract_single_env_observation(
        self,
        vec_obs: Union[np.ndarray, dict[str, np.ndarray]],
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        """Extract one-environment observation from VecEnv output."""
        if isinstance(vec_obs, dict):
            return {key: value[0] for key, value in vec_obs.items()}
        return vec_obs[0]

    def _predict_worker_scaled_action(
        self, worker_observation: np.ndarray
    ) -> np.ndarray:
        """Predict scaled worker actions in ``[-1, 1]`` for correction scoring."""
        with th.no_grad():
            obs_tensor = th.as_tensor(
                worker_observation, dtype=th.float32, device=self.worker.device
            )
            action = self.worker.actor(obs_tensor)
        return action.detach().cpu().numpy().astype(np.float32)

    def _sample_manager_goal(
        self, flat_obs: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Sample current manager goal with warmup and exploration handling."""
        if self.num_timesteps < self.learning_starts:
            return self.subgoal_space.sample().astype(np.float32)

        goal, _ = self.manager.predict(flat_obs[None, :], deterministic=deterministic)
        goal = goal[0].astype(np.float32)
        if not deterministic and self.manager_exploration_noise > 0.0:
            noise = np.random.normal(
                0.0, self.manager_exploration_noise, size=goal.shape
            ).astype(np.float32)
            goal = goal + noise
        return np.clip(goal, self.subgoal_space.low, self.subgoal_space.high).astype(
            np.float32
        )

    def _sample_worker_action(
        self, worker_obs: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Sample low-level environment action with warmup and exploration."""
        if self.num_timesteps < self.learning_starts:
            return self._env_action_space.sample().astype(np.float32)

        action, _ = self.worker.predict(
            worker_obs[None, :], deterministic=deterministic
        )
        action = action[0].astype(np.float32)
        if not deterministic and self.worker_exploration_noise > 0.0:
            noise = np.random.normal(
                0.0, self.worker_exploration_noise, size=action.shape
            ).astype(np.float32)
            action = action + noise
        return np.clip(
            action, self._env_action_space.low, self._env_action_space.high
        ).astype(np.float32)

    def _extract_transition_next_obs(
        self,
        vec_next_obs: Union[np.ndarray, dict[str, np.ndarray]],
        done: bool,
        info: dict[str, Any],
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        """Get true next observation, handling VecEnv terminal observation semantics."""
        if done and info.get("terminal_observation") is not None:
            return cast(
                Union[np.ndarray, dict[str, np.ndarray]], info["terminal_observation"]
            )
        return self._extract_single_env_observation(vec_next_obs)

    def _finalize_macro_transition(
        self, next_flat_obs: np.ndarray, done: bool, info: dict[str, Any]
    ) -> None:
        """Store one manager transition in replay buffer."""
        if self._macro.start_obs is None or self._macro.start_goal is None:
            return
        if len(self._macro.micro_obs) == 0:
            return

        manager_obs = self._macro.start_obs[None, :].astype(np.float32)
        manager_next_obs = next_flat_obs[None, :].astype(np.float32)
        manager_action = self.manager.policy.scale_action(
            self._macro.start_goal[None, :]
        ).astype(np.float32)
        manager_reward = np.array([self._macro.total_reward], dtype=np.float32)
        manager_done = np.array([float(done)], dtype=np.float32)

        micro_obs = np.asarray(self._macro.micro_obs, dtype=np.float32)
        micro_next_obs = np.asarray(self._macro.micro_next_obs, dtype=np.float32)
        micro_actions = np.asarray(self._macro.micro_actions, dtype=np.float32)
        micro_projected_obs = np.asarray(
            self._macro.micro_projected_obs, dtype=np.float32
        )
        micro_projected_next_obs = np.asarray(
            self._macro.micro_projected_next_obs, dtype=np.float32
        )

        assert isinstance(self.manager.replay_buffer, HIROReplayBuffer)
        self.manager.replay_buffer.add(
            obs=manager_obs,
            next_obs=manager_next_obs,
            action=manager_action,
            reward=manager_reward,
            done=manager_done,
            infos=[info],
            micro_observations=micro_obs,
            micro_next_observations=micro_next_obs,
            micro_actions=micro_actions,
            micro_projected_observations=micro_projected_obs,
            micro_projected_next_observations=micro_projected_next_obs,
        )

        self._macro.reset()

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """Train worker and manager TD3 components.

        Parameters
        ----------
        gradient_steps : int
                Number of gradient updates per model.
        batch_size : int
                Batch size for replay sampling.
        """
        if isinstance(self.manager.replay_buffer, HIROReplayBuffer):
            self.manager.replay_buffer.set_low_level_action_fn(
                self._predict_worker_scaled_action
            )

        worker_buffer_size = (
            self.worker.replay_buffer.size()
            if self.worker.replay_buffer is not None
            else 0
        )
        if worker_buffer_size > 0:
            self.worker._current_progress_remaining = self._current_progress_remaining
            self.worker.train(
                gradient_steps=gradient_steps,
                batch_size=min(batch_size, worker_buffer_size),
            )

        manager_buffer_size = (
            self.manager.replay_buffer.size()
            if self.manager.replay_buffer is not None
            else 0
        )
        if manager_buffer_size > 0:
            self.manager._current_progress_remaining = self._current_progress_remaining
            self.manager.train(
                gradient_steps=gradient_steps,
                batch_size=min(batch_size, manager_buffer_size),
            )

    def dump_logs(self) -> None:
        """Write rollout and training logs in SB3 format."""
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max(
            (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon
        )
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record(
                "rollout/ep_rew_mean",
                safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
            )
            self.logger.record(
                "rollout/ep_len_mean",
                safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
            )
        self.logger.record("time/fps", fps)
        self.logger.record(
            "time/time_elapsed", int(time_elapsed), exclude="tensorboard"
        )
        self.logger.record(
            "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
        )
        self.logger.record(
            "train/manager_n_updates", self.manager._n_updates, exclude="tensorboard"
        )
        self.logger.record(
            "train/worker_n_updates", self.worker._n_updates, exclude="tensorboard"
        )
        if len(self.ep_success_buffer) > 0:
            self.logger.record(
                "rollout/success_rate", safe_mean(self.ep_success_buffer)
            )
        self.logger.dump(step=self.num_timesteps)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "HIRO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "HIRO":
        """Train the HIRO hierarchy.

        Parameters
        ----------
        total_timesteps : int
                Number of environment steps to collect.
        callback : callable | BaseCallback | list[BaseCallback], optional
                SB3 callback(s).
        log_interval : int, default=4
                Dump logs every ``log_interval`` completed episodes.
        tb_log_name : str, default="HIRO"
                TensorBoard run name.
        reset_num_timesteps : bool, default=True
                Whether to reset counters before training.
        progress_bar : bool, default=False
                Whether to display SB3 progress bar.

        Returns
        -------
        HIRO
                The trained instance.
        """
        total_timesteps, callback = self._setup_learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=tb_log_name,
            progress_bar=progress_bar,
        )

        # Reuse HIRO logger for nested TD3 optimizers.
        self.manager.set_logger(self.logger)
        self.worker.set_logger(self.logger)

        callback.on_training_start(locals(), globals())
        assert self.env is not None
        assert self._last_obs is not None
        assert self.worker.replay_buffer is not None

        while self.num_timesteps < total_timesteps:
            vec_obs = cast(Union[np.ndarray, dict[str, np.ndarray]], self._last_obs)
            obs = self._extract_single_env_observation(vec_obs)
            flat_obs = flatten_observation(self.observation_space, obs)

            if self._active_goal is None:
                self._active_goal = self._sample_manager_goal(
                    flat_obs, deterministic=False
                )
                self._goal_step = 0
                self._macro.start_obs = flat_obs.copy()
                self._macro.start_goal = self._active_goal.copy()

            worker_obs = make_worker_observation(flat_obs, self._active_goal)
            env_action = self._sample_worker_action(worker_obs, deterministic=False)

            new_obs, rewards, dones, infos = self.env.step(env_action[None, :])
            new_obs = cast(Union[np.ndarray, dict[str, np.ndarray]], new_obs)
            reward = float(rewards[0])
            done = bool(dones[0])
            info = infos[0]

            next_obs = self._extract_transition_next_obs(new_obs, done, info)
            next_flat_obs = flatten_observation(self.observation_space, next_obs)
            projected_obs = self._project_state(obs)
            projected_next_obs = self._project_state(next_obs)

            transitioned_goal = projected_obs + self._active_goal - projected_next_obs
            intrinsic_reward = -float(np.linalg.norm(transitioned_goal, ord=2))

            worker_next_obs = make_worker_observation(next_flat_obs, transitioned_goal)
            scaled_worker_action = self.worker.policy.scale_action(env_action[None, :])[
                0
            ].astype(np.float32)
            self.worker.replay_buffer.add(
                obs=worker_obs[None, :].astype(np.float32),
                next_obs=worker_next_obs[None, :].astype(np.float32),
                action=scaled_worker_action[None, :],
                reward=np.array([intrinsic_reward], dtype=np.float32),
                done=np.array([float(done)], dtype=np.float32),
                infos=[info],
            )

            self._macro.total_reward += reward
            self._macro.micro_obs.append(flat_obs.copy())
            self._macro.micro_next_obs.append(next_flat_obs.copy())
            self._macro.micro_actions.append(scaled_worker_action.copy())
            self._macro.micro_projected_obs.append(projected_obs.copy())
            self._macro.micro_projected_next_obs.append(projected_next_obs.copy())

            self.num_timesteps += 1
            self._update_info_buffer(infos, dones)
            self._last_obs = new_obs

            callback.update_locals(locals())
            if not callback.on_step():
                break

            macro_done = done or (self._goal_step + 1 >= self.subgoal_freq)
            if macro_done:
                self._finalize_macro_transition(
                    next_flat_obs=next_flat_obs, done=done, info=info
                )

            if done:
                self._episode_num += 1
                self._active_goal = None
                self._goal_step = 0
                self._macro.reset()
                if log_interval > 0 and self._episode_num % log_interval == 0:
                    self.dump_logs()
            else:
                if macro_done:
                    self._active_goal = None
                    self._goal_step = 0
                else:
                    self._active_goal = transitioned_goal.astype(np.float32)
                    self._goal_step += 1

            if (
                self.num_timesteps > self.learning_starts
                and self.train_freq > 0
                and self.num_timesteps % self.train_freq == 0
            ):
                self._update_current_progress_remaining(
                    self.num_timesteps, total_timesteps
                )
                effective_gradient_steps = (
                    self.gradient_steps if self.gradient_steps >= 0 else self.train_freq
                )
                if effective_gradient_steps > 0:
                    self.train(
                        gradient_steps=effective_gradient_steps,
                        batch_size=self.batch_size,
                    )

        callback.on_training_end()
        return self

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """Predict a manager subgoal for an environment observation.

        Parameters
        ----------
        observation : np.ndarray
                Environment observation.
        state : tuple[np.ndarray, ...], optional
                Unused recurrent state placeholder for SB3 compatibility.
        episode_start : np.ndarray, optional
                Unused recurrent mask placeholder for SB3 compatibility.
        deterministic : bool, default=False
                Whether to use deterministic manager action.

        Returns
        -------
        tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]
                Predicted manager subgoal and unchanged recurrent state.
        """
        if isinstance(observation, dict):
            raise TypeError("HIRO.predict currently expects ndarray observations.")

        obs_shape = self.observation_space.shape
        if obs_shape is None:
            raise ValueError(
                "Observation space shape must be defined for HIRO.predict."
            )

        if observation.ndim == len(obs_shape):
            flat = flatten_observation(self.observation_space, observation)
            subgoal, _ = self.manager.predict(
                flat[None, :], deterministic=deterministic
            )
            return subgoal, state

        flat_batch = np.stack(
            [flatten_observation(self.observation_space, obs) for obs in observation],
            axis=0,
        )
        subgoals, _ = self.manager.predict(flat_batch, deterministic=deterministic)
        return subgoals, state

    def _excluded_save_params(self) -> list[str]:
        """Exclude nested TD3 models from pickle payload."""
        return super()._excluded_save_params() + ["manager", "worker"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        """Register nested model parameters for SB3 save/load."""
        return ["manager", "worker"], []
