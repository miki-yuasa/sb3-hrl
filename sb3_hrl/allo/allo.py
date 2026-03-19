"""Augmented Lagrangian Laplacian Objective (ALLO) representation learner.

This module implements the ALLO pretraining component used for hierarchical
reinforcement learning (HRL) option discovery.

Algorithm context
-----------------
The objective is to learn Laplacian coordinates of an MDP state space so that
temporally-extended actions (options) can be discovered and optimized. In
practice, the learned coordinates are later used to define intrinsic rewards for
low-level policies, while a high-level controller selects among those
subpolicies.

Compared with graph-drawing style formulations that can converge to arbitrary
rotations of eigenvectors, ALLO introduces an augmented Lagrangian max-min
objective with stop-gradient operators to break rotational symmetry in the
optimization dynamics.

Mathematical objective
----------------------
ALLO optimizes the following saddle objective:

max_beta min_u
     sum_i <u_i, L u_i>
     + sum_j sum_{k<=j} beta_{jk} ( <u_j, [u_k]> - delta_{jk} )
     + b * sum_j sum_{k<=j} ( <u_j, [u_k]> - delta_{jk} )^2

where:

- u are neural-network features (dimension d), approximating Laplacian modes.
- L is the graph Laplacian induced by environment transitions.
- beta is a lower-triangular matrix of dual variables.
- [u_k] denotes stop-gradient (implemented via ``detach()``).
- b is a barrier coefficient that is increased during training.
- delta_{jk} is the Kronecker delta.

Implementation details in this module
-------------------------------------
This implementation follows the ALLO pretrainer stage:

1. Collect transitions into an SB3 ``ReplayBuffer`` using random actions.
2. Sample discounted temporal state pairs from a truncated geometric lag
    distribution and minimize temporal smoothness:
    ``graph_loss = mean((phi(s1) - phi(s2))^2, batch).sum(features)``.
3. Sample two independent uncorrelated state batches, compute detached
    inner-product constraints, and form:
    - dual loss using current dual variables,
    - barrier loss using element-wise quadratic constraint errors.
4. Update network parameters with gradient descent, then update dual variables
    and barrier coefficients in a no-grad block using clamped ascent dynamics
    and momentum.

Intrinsic reward wrappers, subpolicy training helpers, and hierarchical
meta-environment logic are implemented in sibling modules.
"""

from __future__ import annotations

import importlib
from typing import Optional, Union, cast

import numpy as np
import torch as th
from gymnasium import spaces
from gymnasium.spaces import utils as space_utils
from stable_baselines3.common.base_class import (
    BaseAlgorithm,
    BaseCallback,
    CallbackList,
    ConvertCallback,
)
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from tqdm.rich import tqdm

from .utils import ALLOProgressBarCallback


class _LaplacianFeatureNet(th.nn.Module):
    """Neural feature extractor used by :class:`ALLO`.

    Parameters
    ----------
    input_dim : int
        Flattened observation dimension.
    feature_dim : int
        Output Laplacian feature dimension.
    hidden_dims : tuple[int, ...], default=(256, 256)
        Hidden MLP layer widths.
    """

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


class _ALLOPolicy(BasePolicy):
    """Lightweight random policy used to satisfy SB3 BaseAlgorithm API.

    ALLO does not optimize a control policy, but SB3's ``predict()`` method
    requires ``self.policy`` to exist.
    """

    def _predict(
        self,
        observation: Union[th.Tensor, dict[str, th.Tensor]],
        deterministic: bool = False,
    ) -> th.Tensor:
        del deterministic

        if isinstance(observation, dict):
            first_tensor = next(iter(observation.values()))
            batch_size = int(first_tensor.shape[0])
        else:
            batch_size = int(observation.shape[0])

        sampled_actions = np.array(
            [self.action_space.sample() for _ in range(batch_size)]
        )
        return th.as_tensor(sampled_actions, device=self.device)


class ALLO(BaseAlgorithm):
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
        gradient_steps: int = 1,
        gamma_pairs: float = 0.95,
        pair_horizon: int = 20,
        lr_duals: float = 5e-3,
        dual_momentum: float = 0.9,
        dual_min: float = 0.0,
        dual_max: float = 100.0,
        lr_barrier_coeff: float = 1e-3,
        barrier_init: float = 1.0,
        barrier_min: float = 1.0,
        barrier_max: float = 100.0,
        use_barrier_for_duals: bool = True,
        grad_clip_norm: float = 10.0,
        hidden_dims: tuple[int, ...] = (256, 256),
        auto_collect_if_needed: bool = True,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[str, th.device] = "auto",
        seed: Optional[int] = None,
    ) -> None:
        """Initialize ALLO pretrainer.

        Parameters
        ----------
        env : GymEnv | str
            Environment instance or Gymnasium environment id.
        representation_dim : int
            Number of Laplacian coordinates to learn.
        learning_rate : float, default=3e-4
            Optimizer learning rate for the feature network.
        buffer_size : int, default=100_000
            Replay buffer capacity.
        batch_size : int, default=256
            Mini-batch size for ALLO objective updates.
        gradient_steps : int, default=1
            Number of optimization steps per epoch.
        gamma_pairs : float, default=0.95
            Geometric factor for discounted temporal pair sampling.
        pair_horizon : int, default=20
            Maximum temporal offset when sampling state pairs.
        lr_duals : float, default=5e-3
            Learning rate for dual-variable ascent updates.
        dual_momentum : float, default=0.9
            Momentum factor used for dual updates.
        dual_min : float, default=0.0
            Minimum clamp value for dual variables.
        dual_max : float, default=100.0
            Maximum clamp value for dual variables.
        lr_barrier_coeff : float, default=1e-3
            Learning rate for barrier coefficient updates.
        barrier_init : float, default=1.0
            Initial value for barrier coefficients.
        barrier_min : float, default=1.0
            Minimum clamp value for barrier coefficients.
        barrier_max : float, default=100.0
            Maximum clamp value for barrier coefficients.
        use_barrier_for_duals : bool, default=True
            Whether dual learning rate scales with barrier value.
        grad_clip_norm : float, default=10.0
            Gradient clipping norm.
        hidden_dims : tuple[int, ...], default=(256, 256)
            Hidden MLP layer widths for feature network.
        auto_collect_if_needed : bool, default=False
            Whether ``learn()`` should automatically collect random transitions
            to fill the replay buffer when it is not full.
        stats_window_size : int, default=100
            Window size for running statistics.
        tensorboard_log : str | None, default=None
            TensorBoard log directory.
        verbose : int, default=0
            SB3 verbosity level.
        seed : int | None, default=None
            Random seed.
        device : str | torch.device, default="auto"
            Torch device for model and tensors.

        Returns
        -------
        None
            Initializes the algorithm in place.
        """
        super().__init__(
            policy=_ALLOPolicy,
            env=env,
            learning_rate=learning_rate,
            verbose=verbose,
            device=device,
            seed=seed,
            support_multi_env=True,
            monitor_wrapper=True,
            supported_action_spaces=(spaces.Box, spaces.Discrete, spaces.MultiDiscrete),
            tensorboard_log=tensorboard_log,
            stats_window_size=stats_window_size,
        )
        if representation_dim <= 0:
            raise ValueError("representation_dim must be positive.")

        self.representation_dim = int(representation_dim)
        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.gradient_steps = int(gradient_steps)
        self.gamma_pairs = float(gamma_pairs)
        self.pair_horizon = int(pair_horizon)

        self.lr_duals = float(lr_duals)
        self.dual_momentum = float(dual_momentum)
        self.dual_min = float(dual_min)
        self.dual_max = float(dual_max)

        self.lr_barrier_coeff = float(lr_barrier_coeff)
        self.barrier_int = float(barrier_init)
        self.barrier_min = float(barrier_min)
        self.barrier_max = float(barrier_max)
        self.use_barrier_for_duals = bool(use_barrier_for_duals)
        self.grad_clip_norm = float(grad_clip_norm)
        self.hidden_dims = hidden_dims
        self.auto_collect_if_needed = bool(auto_collect_if_needed)
        self._flat_obs_space = space_utils.flatten_space(self.observation_space)
        if not isinstance(self._flat_obs_space, spaces.Box):
            raise TypeError("ALLO requires a flattenable Box observation space.")
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

        self._allo_last_obs: Optional[Union[np.ndarray, dict[str, np.ndarray]]] = None

        self._setup_model()

    def _setup_model(self) -> None:
        """Create replay buffer, feature network, and ALLO state tensors.

        Parameters
        ----------
        None
            Uses instance attributes set at initialization.

        Returns
        -------
        None
            Allocates internal model, optimizer, and constrained variables.
        """
        self.replay_buffer = ReplayBuffer(
            buffer_size=self.buffer_size,
            observation_space=self._flat_obs_space,
            action_space=self.action_space,
            device=self.device,
            n_envs=self.n_envs,
        )

        self._setup_lr_schedule()
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
        )
        self.policy = self.policy.to(self.device)

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
        self.barrier_coeffs = (
            th.ones(shape, dtype=th.float32, device=self.device) * self.barrier_int
        )

    def _excluded_save_params(self) -> list[str]:
        """Exclude raw env handle from pickled data.

        Parameters
        ----------
        None
            Method has no external parameters.

        Returns
        -------
        list[str]
            Attribute names excluded from standard SB3 pickling.
        """
        return [*super()._excluded_save_params(), "env"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        """Return torch state objects used by SB3 save/load.

        Parameters
        ----------
        None
            Method has no external parameters.

        Returns
        -------
        tuple[list[str], list[str]]
            Tuple of state-dict names and tensor attribute names.
        """
        state_dicts = ["feature_net", "optimizer"]
        tensors = ["dual_variables", "dual_velocities", "barrier_coeffs"]
        return state_dicts, tensors

    def _flatten_single_observation(
        self, observation: Union[np.ndarray, dict[str, np.ndarray]]
    ) -> np.ndarray:
        """Flatten one environment observation.

        Parameters
        ----------
        observation : np.ndarray | dict[str, np.ndarray]
            Single observation compatible with ``self.observation_space``.

        Returns
        -------
        np.ndarray
            Flattened 1D observation of shape ``[obs_dim]``.
        """
        if isinstance(observation, dict):
            flat = space_utils.flatten(self.observation_space, observation)
        else:
            flat = observation.reshape(-1)
        return np.asarray(flat, dtype=np.float32)

    def _numpy_to_torch(
        self, arrays: Union[np.ndarray, tuple[np.ndarray, ...]]
    ) -> Union[th.Tensor, tuple[th.Tensor, ...]]:
        """Convert numpy array(s) to torch tensor(s) on device.

        Parameters
        ----------
        arrays : np.ndarray | tuple[np.ndarray, ...]
            Numpy array or tuple of arrays to convert.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, ...]
            Tensor(s) on self.device with dtype float32.
        """
        if isinstance(arrays, tuple):
            return tuple(
                th.as_tensor(arr, dtype=th.float32, device=self.device)
                for arr in arrays
            )
        return th.as_tensor(arrays, dtype=th.float32, device=self.device)

    def _get_barrier_coeff(self) -> float:
        """Get the current barrier coefficient scalar value.

        Returns
        -------
        float
            Current barrier coefficient (from [0, 0] element).
        """
        return float(self.barrier_coeffs[0, 0].item())

    def _flatten_vec_observations(
        self, observations: Union[np.ndarray, dict[str, np.ndarray]]
    ) -> np.ndarray:
        """Flatten single or vectorized observations into ``[batch_size, obs_dim]``.

        Parameters
        ----------
        observations : np.ndarray | dict[str, np.ndarray]
            Observation returned by env reset/step.

        Returns
        -------
        np.ndarray
            Flattened observations with shape ``[batch_size, obs_dim]``.
        """
        if isinstance(observations, dict):
            # VecEnv dict observations are key->array with first dim as n_envs.
            first_key = next(iter(observations))
            first_val = np.asarray(observations[first_key])
            is_batched = first_val.ndim >= 1 and first_val.shape[0] == self.n_envs
            if not is_batched:
                return self._flatten_single_observation(observations)[None, :]

            flattened = []
            for i in range(self.n_envs):
                single_obs = {
                    key: np.asarray(value)[i] for key, value in observations.items()
                }
                flattened.append(self._flatten_single_observation(single_obs))
            return np.stack(flattened, axis=0)

        obs_array = np.asarray(observations)
        batch_size = obs_array.shape[0] if obs_array.ndim > self._obs_ndim else 1
        if obs_array.ndim == self._obs_ndim:
            obs_array = obs_array[None, ...]
        return obs_array.reshape(batch_size, -1).astype(np.float32, copy=False)

    def encode(
        self, observations: Union[np.ndarray, dict[str, np.ndarray], th.Tensor]
    ) -> th.Tensor:
        """Encode observations into Laplacian features.

        Parameters
        ----------
        observations : np.ndarray | dict[str, np.ndarray] | torch.Tensor
            Single observation of shape ``[*obs_shape]`` or batch of shape
            ``[batch_size, *obs_shape]``.

        Returns
        -------
        torch.Tensor
            Encoded features with shape ``[batch_size, representation_dim]``.
        """
        if isinstance(observations, (np.ndarray, dict)):
            flat = self._flatten_vec_observations(observations)
            obs_tensor = self._numpy_to_torch(flat)
        else:
            obs_tensor = observations.to(self.device)
            if obs_tensor.ndim == self._obs_ndim:
                obs_tensor = obs_tensor.unsqueeze(0)
            obs_tensor = obs_tensor.view(obs_tensor.shape[0], -1)
        return self.feature_net(obs_tensor)

    def _sample_uniform_states(self, batch_size: int) -> np.ndarray:
        """Sample uncorrelated states from replay buffer.

        Parameters
        ----------
        batch_size : int
            Number of observations to sample.

        Returns
        -------
        np.ndarray
            Observation batch of shape ``[batch_size, *obs_shape]``.
        """
        current_size = self.replay_buffer.size()
        indices = np.random.randint(0, current_size, size=batch_size)
        env_indices = np.random.randint(0, self.n_envs, size=batch_size)
        return self.replay_buffer.observations[indices, env_indices]

    def _sample_discounted_pairs(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample temporally-correlated state pairs using truncated geometric lags.

        Parameters
        ----------
        batch_size : int
            Number of paired samples.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Pair ``(s_1, s_2)`` each with shape ``[batch_size, *obs_shape]``.
        """
        current_size = self.replay_buffer.size()
        if current_size < 2:
            raise RuntimeError("Replay buffer does not contain enough transitions.")

        lags = np.random.choice(self._lag_values, size=batch_size, p=self._lag_probs)
        max_lag = int(np.max(lags))

        if self.replay_buffer.full:
            # SB3 ReplayBuffer stores per-env time slots on axis 0.
            rb_capacity = int(self.replay_buffer.buffer_size)
            first_indices = np.random.randint(0, rb_capacity, size=batch_size)
            second_indices = (first_indices + lags) % rb_capacity
        else:
            usable = max(current_size - max_lag, 1)
            first_indices = np.random.randint(0, usable, size=batch_size)
            second_indices = np.minimum(first_indices + lags, current_size - 1)

        env_indices = np.random.randint(0, self.n_envs, size=batch_size)

        first = self.replay_buffer.observations[first_indices, env_indices]
        second = self.replay_buffer.next_observations[second_indices, env_indices]
        return first, second

    def collect_random_transitions(
        self, num_steps: Optional[int] = None, progress_bar: bool = False
    ) -> None:
        """Collect transitions into the replay buffer before offline training.

        Parameters
        ----------
        num_steps : int | None, default=None
            Number of vectorized environment steps to collect. When ``None``,
            collects exactly ``buffer_size`` vectorized steps, which fills the
            replay buffer once.
        progress_bar : bool, default=False
            Whether to show a tqdm progress bar while collecting transitions.

        Returns
        -------
        None
            Populates the replay buffer with random-policy transitions.
        """
        if self.env is None:
            raise RuntimeError("ALLO environment is not initialized.")

        steps_to_collect = (
            int(self.replay_buffer.buffer_size) if num_steps is None else int(num_steps)
        )
        if steps_to_collect <= 0:
            raise ValueError("num_steps must be a positive integer.")

        if self._allo_last_obs is None:
            reset_obs = self.env.reset()
            self._allo_last_obs = cast(
                Union[np.ndarray, dict[str, np.ndarray]], reset_obs
            )

        size_before = self.replay_buffer.size()
        collected_steps = 0
        pbar = None

        if progress_bar:
            try:
                progress_total_samples = steps_to_collect * self.n_envs
                pbar = tqdm(
                    total=progress_total_samples,
                    desc="ALLO sample collection",
                    unit="samples",
                    leave=False,
                )
            except ImportError:
                if self.verbose >= 1:
                    print(
                        "[ALLO] tqdm is not installed. Falling back to logger/print progress."
                    )
                progress_bar = False

        try:
            while collected_steps < steps_to_collect:
                actions = np.array(
                    [self.action_space.sample() for _ in range(self.n_envs)]
                )
                new_obs, rewards, dones, infos = self.env.step(actions)
                obs_batch = self._flatten_vec_observations(
                    cast(Union[np.ndarray, dict[str, np.ndarray]], self._allo_last_obs)
                )
                next_obs_batch = self._flatten_vec_observations(
                    cast(Union[np.ndarray, dict[str, np.ndarray]], new_obs)
                )

                self.replay_buffer.add(
                    obs=obs_batch,
                    next_obs=next_obs_batch,
                    action=actions,
                    reward=rewards,
                    done=dones,
                    infos=infos,
                )

                self._allo_last_obs = cast(
                    Union[np.ndarray, dict[str, np.ndarray]], new_obs
                )
                collected_steps += 1

                if pbar is not None:
                    pbar.update(self.n_envs)
                elif (
                    self.verbose >= 1
                    and collected_steps % max(steps_to_collect // 20, 1) == 0
                ):
                    # Log progress ~20 times during collection if not using tqdm
                    progress_ratio = float(collected_steps) / float(steps_to_collect)
                    replay_slots = self.replay_buffer.size()
                    replay_samples = replay_slots * self.n_envs
                    print(
                        "[ALLO] Collection progress: "
                        f"{collected_steps}/{steps_to_collect} "
                        f"({100.0 * progress_ratio:.1f}%), "
                        f"replay_size={replay_samples} "
                        f"(slots={replay_slots}, n_envs={self.n_envs})"
                    )
        finally:
            if pbar is not None:
                pbar.close()

        size_after = self.replay_buffer.size()

        # Log collection stats when a logger is available; otherwise fall back to stdout.
        collected_slots = max(size_after - size_before, 0)
        replay_size_total = size_after * self.n_envs
        collected_samples = collected_slots * self.n_envs
        if hasattr(self, "_logger"):
            self.logger.record("collect/requested_steps", float(steps_to_collect))
            # Keep this key in total-transition units for easier interpretation.
            self.logger.record("collect/replay_size", float(replay_size_total))
            self.logger.record("collect/replay_slots", float(size_after))
            self.logger.record("collect/collected_samples", float(collected_samples))
            self.logger.record("collect/collected_slots", float(collected_slots))
            self.logger.record(
                "collect/replay_full", float(int(self.replay_buffer.full))
            )
            self.logger.dump(step=size_after)

    def train_step(self) -> dict[str, float]:
        """Run one ALLO optimization step.

        Parameters
        ----------
        None
            Uses replay-buffer samples and model state.

        Returns
        -------
        dict[str, float]
            Scalar diagnostics and loss components for logging.
        """
        if self.replay_buffer.size() < max(self.batch_size, self.pair_horizon + 1):
            raise RuntimeError("Not enough replay data to run ALLO training step.")

        states_1_np, states_2_np = self._sample_discounted_pairs(self.batch_size)
        uncorr_1_np = self._sample_uniform_states(self.batch_size)
        uncorr_2_np = self._sample_uniform_states(self.batch_size)

        # Convert all numpy arrays to tensors at once
        states_1, states_2, uncorr_1, uncorr_2 = self._numpy_to_torch(
            (
                self._flatten_vec_observations(states_1_np),
                self._flatten_vec_observations(states_2_np),
                self._flatten_vec_observations(uncorr_1_np),
                self._flatten_vec_observations(uncorr_2_np),
            )
        )

        # Compute features for graph loss (temporal smoothness)
        phi_1 = self.feature_net(states_1)
        phi_2 = self.feature_net(states_2)
        graph_loss = ((phi_1 - phi_2) ** 2).mean(dim=0).sum()

        # Compute features for orthogonality constraints
        phi_unc_1 = self.feature_net(uncorr_1)
        phi_unc_2 = self.feature_net(uncorr_2)

        # Compute constraint error matrices
        norm = float(self.batch_size)
        m1 = (phi_unc_1.T @ phi_unc_1.detach()) / norm
        m2 = (phi_unc_2.T @ phi_unc_2.detach()) / norm

        identity = th.eye(self.representation_dim, device=self.device, dtype=th.float32)
        e1 = th.tril(m1 - identity)
        e2 = th.tril(m2 - identity)
        e = 0.5 * (e1 + e2)
        e_quad = e1 * e2

        # Compute dual and barrier losses
        dual_loss = (self.dual_variables.detach() * e).sum()
        barrier_loss = (self._get_barrier_coeff() * e_quad).sum()
        total_loss = graph_loss + dual_loss + barrier_loss

        # Network update
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        th.nn.utils.clip_grad_norm_(self.feature_net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        # Update dual variables and barrier coefficients
        with th.no_grad():
            barrier_scalar = self._get_barrier_coeff()
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
            "diagnostics/barrier_coeff": barrier_scalar,
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
    ) -> "ALLO":
        """Train ALLO representation network offline.

        Parameters
        ----------
        total_timesteps : int
            Number of offline optimization epochs over the pre-filled replay
            buffer. In this offline ALLO implementation, this argument is
            interpreted as epochs (not environment interaction steps).
        callback : MaybeCallback, default=None
            Optional SB3 callback.
        log_interval : int, default=10
            Logger dump interval in outer-loop iterations.
        tb_log_name : str, default="ALLO"
            TensorBoard run label.
        reset_num_timesteps : bool, default=True
            Whether to reset timestep counter before training.
        progress_bar : bool, default=False
            Whether to show a progress bar.

        Returns
        -------
        ALLO
            Trained algorithm instance.
        """
        if total_timesteps <= 0:
            raise ValueError("total_timesteps must be a positive integer.")

        if not self.replay_buffer.full:
            if self.auto_collect_if_needed:
                missing_steps = (
                    int(self.replay_buffer.buffer_size) - self.replay_buffer.size()
                )
                if missing_steps > 0:
                    self.collect_random_transitions(
                        num_steps=missing_steps,
                        progress_bar=progress_bar,
                    )
            else:
                raise RuntimeError(
                    "Replay buffer must be pre-filled before offline training. "
                    "Call `collect_random_transitions()` first or set "
                    "`auto_collect_if_needed=True`."
                )

        total_epochs = int(total_timesteps)
        callback_total_timesteps = total_epochs

        total_timesteps, callback = self._setup_learn(
            total_timesteps=callback_total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=tb_log_name,
            progress_bar=progress_bar,
        )

        callback.on_training_start(locals(), globals())

        if self.replay_buffer.size() < max(self.batch_size, self.pair_horizon + 1):
            # Show the values of replay_buffer.size(), batch_size and pair_horizon for debugging
            raise RuntimeError(
                f"Not enough replay data to run ALLO training step. "
                f"Replay buffer size: {self.replay_buffer.size()}, "
                f"batch_size: {self.batch_size}, pair_horizon: {self.pair_horizon}."
                "Needs at least max(batch_size, pair_horizon + 1) transitions in the replay buffer."
            )

        for iteration in range(1, total_epochs + 1):
            for _ in range(self.gradient_steps):
                stats = self.train_step()
                for key, value in stats.items():
                    self.logger.record(key, value)

            # In offline ALLO, one outer iteration corresponds to one configured
            # training timestep (epoch), independent of vec-env parallelism.
            self.num_timesteps += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            if log_interval > 0 and iteration % log_interval == 0:
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            callback.update_locals(locals())
            if not callback.on_step():
                break

        callback.on_training_end()
        return self

    def _init_callback(
        self,
        callback: MaybeCallback,
        progress_bar: bool = False,
    ) -> BaseCallback:
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Add progress bar callback
        if progress_bar:
            callback = CallbackList([callback, ALLOProgressBarCallback()])

        callback.init_callback(self)
        return callback


__all__ = ["ALLO"]
