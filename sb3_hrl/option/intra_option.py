"""Intra-option learning integration scaffolding for SB3.

This module provides architecture hooks for users who want to combine
Sutton-Precup-Singh intra-option updates with SB3 training loops.

Why wrappers alone are insufficient
----------------------------------
SB3 algorithms consume one transition per ``env.step`` call and assume that
the returned tuple already matches the algorithm's update semantics.
Intra-option learning needs primitive transitions while an option is active:

.. math::

    U(s, \\omega) = r + \\gamma [(1-\\beta_\\omega(s'))Q(s',\\omega) + \\beta_\\omega(s')V(s')]

When the environment emits only a macro transition after option execution,
vanilla SB3 cannot perform this update without additional hooks.

Recommended integration
-----------------------
1. Wrap env with ``MetaControllerEnvWrapper(reward_type="intra_option",
   capture_primitive_transitions=True)``.
2. Use ``IntraOptionReplayBuffer`` (off-policy path) to store primitive traces
   alongside macro transitions.
3. Attach ``IntraOptionUpdateCallback`` and override
   ``process_intra_option_transitions`` to perform custom gradient steps.

Pseudo-code sketch
------------------
.. code-block:: python

    env = MetaControllerEnvWrapper(..., reward_type="intra_option")
    model = DQN(
        "MlpPolicy",
        env,
        replay_buffer_class=IntraOptionReplayBuffer,
    )

    class MyIntraOptionCallback(IntraOptionUpdateCallback):
        def process_intra_option_transitions(self, transitions, info):
            for tr in transitions:
                # 1) Evaluate beta_w(s') and Q/V estimates from your model.
                # 2) Build U(s,w) target from the equation above.
                # 3) Apply custom optimizer step.
                pass

    model.learn(..., callback=MyIntraOptionCallback())

This keeps ``model.learn()`` intact while exposing enough information for
custom intra-option updates.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback


class IntraOptionReplayBuffer(ReplayBuffer):
    """Replay buffer that stores primitive traces emitted by option wrapper.

    Notes
    -----
    The base ``ReplayBuffer`` stores macro transitions. This subclass additionally
    stores per-macro primitive traces from ``info['primitive_transitions']``.
    Users can consume these traces in custom training code.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
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
        self.primitive_traces: list[list[list[dict[str, Any]]]] = [
            [[] for _ in range(self.n_envs)] for _ in range(self.buffer_size)
        ]

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """Add one macro transition and cache primitive transition traces."""
        super().add(obs, next_obs, action, reward, done, infos)
        insert_idx = (self.pos - 1) % self.buffer_size

        for env_idx, info in enumerate(infos):
            traces = info.get("primitive_transitions", [])
            if isinstance(traces, list):
                self.primitive_traces[insert_idx][env_idx] = traces
            else:
                self.primitive_traces[insert_idx][env_idx] = []

    def get_primitive_traces(
        self,
        indices: np.ndarray,
        env_indices: Optional[np.ndarray] = None,
    ) -> list[list[dict[str, Any]]]:
        """Return primitive traces for sampled replay entries.

        Parameters
        ----------
        indices : np.ndarray
            Replay slot indices.
        env_indices : np.ndarray | None, default=None
            Environment indices for vectorized storage. If omitted, env index 0
            is used.
        """
        if env_indices is None:
            env_indices = np.zeros_like(indices)

        traces: list[list[dict[str, Any]]] = []
        for idx, env_idx in zip(indices.tolist(), env_indices.tolist()):
            traces.append(self.primitive_traces[idx][env_idx])
        return traces


class IntraOptionUpdateCallback(BaseCallback):
    """Callback hook to process primitive traces for intra-option updates.

    Override :meth:`process_intra_option_transitions` to inject custom update
    logic using transitions surfaced by ``MetaControllerEnvWrapper``.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.num_macro_steps = 0
        self.num_primitive_steps = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is None:
            return True

        for info in infos:
            self.num_macro_steps += 1
            transitions = info.get("primitive_transitions")
            if not isinstance(transitions, list):
                continue

            self.num_primitive_steps += len(transitions)
            self.process_intra_option_transitions(transitions, info)

        return True

    def process_intra_option_transitions(
        self,
        transitions: list[dict[str, Any]],
        info: dict[str, Any],
    ) -> None:
        """Handle primitive trace list for a single macro step.

        Parameters
        ----------
        transitions : list[dict[str, Any]]
            Primitive transitions produced while one option was executing.
        info : dict[str, Any]
            Macro-level info dictionary returned by the wrapped environment.

        Notes
        -----
        Subclasses should compute custom targets such as

        ``U(s,w)=r+gamma*((1-beta_w(s'))*Q(s',w)+beta_w(s')*V(s'))``

        and apply model-specific optimizer updates.
        """
        del transitions, info


__all__ = ["IntraOptionReplayBuffer", "IntraOptionUpdateCallback"]
