"""ALLO representation learning algorithm implementation."""

from __future__ import annotations

from typing import Optional, Union, cast

import numpy as np
import torch as th
from gymnasium import spaces
from gymnasium.spaces import utils as space_utils
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback


class _LaplacianFeatureNet(th.nn.Module):
    """Neural feature extractor used by :class:`ALLOAlgorithm`."""

    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        layers: list[th.nn.Module] = []
        prev = input_dim
        for hidden_dim in hidden_dims:
            layers.append(th.nn.Linear(prev, hidden_dim))
            layers.append(th.nn.ReLU())
            prev = hidden_dim
        layers.append(th.nn.Linear(prev, feature_dim))
        self.model = th.nn.Sequential(*layers)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Compute Laplacian feature vectors.

        Parameters
        ----------
        observations : torch.Tensor
            Flattened observations of shape ``[batch_size, input_dim]``.

        Returns
        -------
        torch.Tensor
            Feature tensor of shape ``[batch_size, feature_dim]``.
        """
        return self.model(observations)


class ALLOAlgorithm(BaseAlgorithm):
    """ALLO pretrainer for Laplacian representation learning.

    Notes
    -----
    This algorithm does not learn an action policy. It collects transitions
    with a random behavior policy to optimize a Laplacian feature network
    using the ALLO max-min objective.
    """

    def __init__(
        self,
        env: Union[GymEnv, str],
        representation_dim: int,
        learning_rate: float = 3e-4,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        train_freq: int = 256,
        gradient_steps: int = 1,
        gamma_pairs: float = 0.95,
        pair_horizon: int = 20,
        lr_duals: float = 5e-3,
        dual_momentum: float = 0.9,
        dual_min: float = 0.0,
        dual_max: float = 100.0,
        lr_barrier_coeff: float = 1e-3,
        barrier_min: float = 1.0,
        barrier_max: float = 100.0,
        use_barrier_for_duals: bool = True,
        grad_clip_norm: float = 10.0,
        hidden_dims: tuple[int, ...] = (256, 256),
        seed: Optional[int] = None,
        device: Union[str, th.device] = "auto",
        verbose: int = 0,
    ) -> None:
        super().__init__(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            verbose=verbose,
            device=device,
            seed=seed,
            support_multi_env=True,
            monitor_wrapper=True,
            supported_action_spaces=(spaces.Box, spaces.Discrete, spaces.MultiDiscrete),
        )
        if representation_dim <= 0:
            raise ValueError("representation_dim must be positive.")

        self.representation_dim = int(representation_dim)
        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.train_freq = int(train_freq)
        self.gradient_steps = int(gradient_steps)
        self.gamma_pairs = float(gamma_pairs)
        self.pair_horizon = int(pair_horizon)

        self.lr_duals = float(lr_duals)
        self.dual_momentum = float(dual_momentum)
        self.dual_min = float(dual_min)
        self.dual_max = float(dual_max)

        self.lr_barrier_coeff = float(lr_barrier_coeff)
        self.barrier_min = float(barrier_min)
        self.barrier_max = float(barrier_max)
        self.use_barrier_for_duals = bool(use_barrier_for_duals)
        self.grad_clip_norm = float(grad_clip_norm)
        self.hidden_dims = hidden_dims

        self._flat_obs_space = space_utils.flatten_space(self.observation_space)
        if not isinstance(self._flat_obs_space, spaces.Box):
            raise TypeError(
                "ALLOAlgorithm requires a flattenable Box observation space."
            )
        self._obs_dim = int(np.prod(self._flat_obs_space.shape))
        obs_shape = self.observation_space.shape
        self._obs_ndim = len(obs_shape) if obs_shape is not None else 1

        self.feature_net: _LaplacianFeatureNet
        self.optimizer: th.optim.Optimizer
        self.replay_buffer: ReplayBuffer

        self.dual_variables: th.Tensor
        self.dual_velocities: th.Tensor
        self.barrier_coeffs: th.Tensor

        self._lag_values = np.arange(1, self.pair_horizon + 1, dtype=np.int64)
        lag_weights = (1.0 - self.gamma_pairs) * np.power(
            self.gamma_pairs, self._lag_values - 1
        )
        self._lag_probs = lag_weights / np.sum(lag_weights)

        self._allo_last_obs: Optional[np.ndarray] = None

        self._setup_model()

    def _setup_model(self) -> None:
        """Create replay buffer, feature network, and ALLO state tensors."""
        self.replay_buffer = ReplayBuffer(
            buffer_size=self.buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            n_envs=self.n_envs,
        )

        self.feature_net = _LaplacianFeatureNet(
            input_dim=self._obs_dim,
            feature_dim=self.representation_dim,
            hidden_dims=self.hidden_dims,
        ).to(self.device)
        lr_value = self.learning_rate
        learning_rate = float(lr_value(1.0)) if callable(lr_value) else float(lr_value)
        self.optimizer = th.optim.Adam(self.feature_net.parameters(), lr=learning_rate)

        shape = (self.representation_dim, self.representation_dim)
        self.dual_variables = th.zeros(shape, dtype=th.float32, device=self.device)
        self.dual_velocities = th.zeros(shape, dtype=th.float32, device=self.device)
        self.barrier_coeffs = th.ones(shape, dtype=th.float32, device=self.device)

    def _excluded_save_params(self) -> list[str]:
        """Exclude raw env handle from pickled data."""
        return [*super()._excluded_save_params(), "env"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        """Return torch state objects used by SB3 save/load."""
        state_dicts = ["feature_net", "optimizer"]
        tensors = ["dual_variables", "dual_velocities", "barrier_coeffs"]
        return state_dicts, tensors

    def _flatten_observations(self, observations: np.ndarray) -> np.ndarray:
        """Flatten observations into ``[batch_size, obs_dim]``.

        Parameters
        ----------
        observations : np.ndarray
            Observation batch with shape ``[batch_size, *obs_shape]``.

        Returns
        -------
        np.ndarray
            Flattened float32 observations with shape ``[batch_size, obs_dim]``.
        """
        batch_size = observations.shape[0]
        return observations.reshape(batch_size, -1).astype(np.float32, copy=False)

    def encode(self, observations: Union[np.ndarray, th.Tensor]) -> th.Tensor:
        """Encode observations into Laplacian features."""
        if isinstance(observations, np.ndarray):
            obs_np = observations
            if obs_np.ndim == self._obs_ndim:
                obs_np = np.expand_dims(obs_np, axis=0)
            flat = self._flatten_observations(obs_np)
            obs_tensor = th.as_tensor(flat, dtype=th.float32, device=self.device)
        else:
            obs_tensor = observations.to(self.device)
            if obs_tensor.ndim == self._obs_ndim:
                obs_tensor = obs_tensor.unsqueeze(0)
            obs_tensor = obs_tensor.view(obs_tensor.shape[0], -1)
        return self.feature_net(obs_tensor)

    def _sample_uniform_states(self, batch_size: int) -> np.ndarray:
        """Sample uncorrelated states from replay buffer."""
        current_size = self.replay_buffer.size()
        indices = np.random.randint(0, current_size, size=batch_size)
        env_indices = np.random.randint(0, self.n_envs, size=batch_size)
        return self.replay_buffer.observations[indices, env_indices]

    def _sample_discounted_pairs(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample temporally-correlated state pairs using truncated geometric lags."""
        current_size = self.replay_buffer.size()
        if current_size < 2:
            raise RuntimeError("Replay buffer does not contain enough transitions.")

        lags = np.random.choice(self._lag_values, size=batch_size, p=self._lag_probs)
        max_lag = int(np.max(lags))

        if self.replay_buffer.full:
            first_indices = np.random.randint(0, self.buffer_size, size=batch_size)
            second_indices = (first_indices + lags) % self.buffer_size
        else:
            usable = max(current_size - max_lag, 1)
            first_indices = np.random.randint(0, usable, size=batch_size)
            second_indices = np.minimum(first_indices + lags, current_size - 1)

        env_indices = np.random.randint(0, self.n_envs, size=batch_size)

        first = self.replay_buffer.observations[first_indices, env_indices]
        second = self.replay_buffer.next_observations[second_indices, env_indices]
        return first, second

    def _collect_random_transitions(self, n_steps: int) -> None:
        """Collect transitions with a random behavior policy."""
        if self.env is None:
            raise RuntimeError("ALLOAlgorithm environment is not initialized.")
        if self._allo_last_obs is None:
            reset_obs = self.env.reset()
            if isinstance(reset_obs, dict):
                raise TypeError(
                    "ALLOAlgorithm currently does not support Dict observations."
                )
            self._allo_last_obs = np.asarray(reset_obs)

        for _ in range(n_steps):
            actions = np.array([self.action_space.sample() for _ in range(self.n_envs)])
            new_obs, rewards, dones, infos = self.env.step(actions)
            if isinstance(new_obs, dict):
                raise TypeError(
                    "ALLOAlgorithm currently does not support Dict observations."
                )
            new_obs_array = np.asarray(new_obs)

            self.replay_buffer.add(
                obs=cast(np.ndarray, self._allo_last_obs),
                next_obs=new_obs_array,
                action=actions,
                reward=rewards,
                done=dones,
                infos=infos,
            )

            self._allo_last_obs = new_obs_array
            self.num_timesteps += self.n_envs

    def train_step(self) -> dict[str, float]:
        """Run one ALLO optimization step."""
        if self.replay_buffer.size() < max(self.batch_size, self.pair_horizon + 1):
            raise RuntimeError("Not enough replay data to run ALLO training step.")

        states_1_np, states_2_np = self._sample_discounted_pairs(self.batch_size)
        uncorr_1_np = self._sample_uniform_states(self.batch_size)
        uncorr_2_np = self._sample_uniform_states(self.batch_size)

        states_1 = th.as_tensor(
            self._flatten_observations(states_1_np),
            dtype=th.float32,
            device=self.device,
        )
        states_2 = th.as_tensor(
            self._flatten_observations(states_2_np),
            dtype=th.float32,
            device=self.device,
        )
        uncorr_1 = th.as_tensor(
            self._flatten_observations(uncorr_1_np),
            dtype=th.float32,
            device=self.device,
        )
        uncorr_2 = th.as_tensor(
            self._flatten_observations(uncorr_2_np),
            dtype=th.float32,
            device=self.device,
        )

        phi_1 = self.feature_net(states_1)
        phi_2 = self.feature_net(states_2)
        graph_loss = ((phi_1 - phi_2) ** 2).mean(dim=0).sum()

        phi_unc_1 = self.feature_net(uncorr_1)
        phi_unc_2 = self.feature_net(uncorr_2)

        norm = float(self.batch_size)
        m1 = (phi_unc_1.T @ phi_unc_1.detach()) / norm
        m2 = (phi_unc_2.T @ phi_unc_2.detach()) / norm

        identity = th.eye(self.representation_dim, device=self.device, dtype=th.float32)
        e1 = th.tril(m1 - identity)
        e2 = th.tril(m2 - identity)
        e = 0.5 * (e1 + e2)
        e_quad = e1 * e2

        dual_loss = (self.dual_variables.detach() * e).sum()
        barrier_coeff_val = self.barrier_coeffs[0, 0].detach()
        barrier_loss = (barrier_coeff_val * e_quad).sum()

        total_loss = graph_loss + dual_loss + barrier_loss

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        th.nn.utils.clip_grad_norm_(self.feature_net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        with th.no_grad():
            barrier_scalar = float(self.barrier_coeffs[0, 0].item())
            effective_lr = self.lr_duals * (
                1.0 + float(self.use_barrier_for_duals) * (barrier_scalar - 1.0)
            )

            proposed = self.dual_variables + effective_lr * e
            proposed = th.clamp(proposed, min=self.dual_min, max=self.dual_max)

            delta = proposed - self.dual_variables
            self.dual_velocities = self.dual_momentum * self.dual_velocities + delta
            self.dual_variables = th.tril(
                th.clamp(
                    self.dual_variables + self.dual_velocities,
                    min=self.dual_min,
                    max=self.dual_max,
                )
            )

            positive_quad = th.clamp(e_quad, min=0.0)
            barrier_increment = self.lr_barrier_coeff * positive_quad.mean()
            self.barrier_coeffs = th.tril(
                th.clamp(
                    self.barrier_coeffs + barrier_increment,
                    min=self.barrier_min,
                    max=self.barrier_max,
                )
            )

        return {
            "loss/total": float(total_loss.item()),
            "loss/graph": float(graph_loss.item()),
            "loss/dual": float(dual_loss.item()),
            "loss/barrier": float(barrier_loss.item()),
            "diagnostics/barrier_coeff": float(self.barrier_coeffs[0, 0].item()),
            "diagnostics/mean_constraint": float(e.abs().mean().item()),
        }

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 10,
        tb_log_name: str = "ALLO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "ALLOAlgorithm":
        """Train ALLO representation network."""
        total_timesteps, callback = self._setup_learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=tb_log_name,
            progress_bar=progress_bar,
        )

        iteration = 0
        while self.num_timesteps < total_timesteps:
            collect_steps = min(self.train_freq, total_timesteps - self.num_timesteps)
            self._collect_random_transitions(collect_steps)

            if self.replay_buffer.size() >= max(self.batch_size, self.pair_horizon + 1):
                for _ in range(self.gradient_steps):
                    stats = self.train_step()
                    for key, value in stats.items():
                        self.logger.record(key, value)

            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            iteration += 1
            if log_interval > 0 and iteration % log_interval == 0:
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            if not callback.on_step():
                break

        callback.on_training_end()
        return self


__all__ = ["ALLOAlgorithm"]
