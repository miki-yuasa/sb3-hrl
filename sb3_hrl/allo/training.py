"""Training utilities for ALLO low-level and meta-level policies."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import gymnasium as gym
import torch as th
from gymnasium import spaces
from stable_baselines3 import DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm

from .allo import ALLOAlgorithm
from .wrappers import HRLMetaEnv, LaplacianRewardWrapper


def train_subpolicies(
    env_id: str,
    allo: Union[ALLOAlgorithm, th.nn.Module],
    num_eigenvectors: int,
    total_timesteps: int = 100_000,
    save_dir: Union[str, Path] = "subpolicies",
    algorithm: str = "auto",
    device: Union[str, th.device] = "auto",
) -> list[Path]:
    """Train one low-level subpolicy per Laplacian eigenvector.

    Parameters
    ----------
    env_id : str
        Gymnasium environment id used for subpolicy training.
    allo : ALLOAlgorithm | torch.nn.Module
        Trained ALLO encoder or compatible feature model.
    num_eigenvectors : int
        Number of option policies to train.
    total_timesteps : int, default=100_000
        Timesteps per option policy.
    save_dir : str | pathlib.Path, default="subpolicies"
        Directory where option checkpoints are saved.
    algorithm : str, default="auto"
        Training algorithm selector: "auto", "ppo", or "sac".
    device : str | torch.device, default="auto"
        Device passed to SB3 algorithms.

    Returns
    -------
    list[pathlib.Path]
        Saved checkpoint paths for all trained options.
    """
    if num_eigenvectors <= 0:
        raise ValueError("num_eigenvectors must be positive.")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    saved_models: list[Path] = []

    for z in range(num_eigenvectors):
        env = gym.make(env_id)
        wrapped_env = LaplacianRewardWrapper(env=env, allo=allo, eigenvector_index=z)

        algo_name = algorithm.lower()
        if algo_name not in {"auto", "ppo", "sac"}:
            wrapped_env.close()
            raise ValueError("algorithm must be one of {'auto', 'ppo', 'sac'}.")

        if algo_name == "auto":
            use_sac = isinstance(wrapped_env.action_space, spaces.Box)
        else:
            use_sac = algo_name == "sac"

        if use_sac:
            model = SAC("MlpPolicy", wrapped_env, verbose=0, device=device)
        else:
            model = PPO("MlpPolicy", wrapped_env, verbose=0, device=device)

        model.learn(total_timesteps=total_timesteps)
        model_path = save_path / f"subpolicy_{z}.zip"
        model.save(model_path)
        saved_models.append(model_path)
        wrapped_env.close()

    return saved_models


def _load_single_subpolicy(path: Union[str, Path], env: gym.Env) -> BaseAlgorithm:
    """Load a single subpolicy checkpoint with fallback across common SB3 classes.

    Parameters
    ----------
    path : str | pathlib.Path
        Path to subpolicy checkpoint.
    env : gym.Env
        Environment bound to the loaded model.

    Returns
    -------
    stable_baselines3.common.base_class.BaseAlgorithm
        Loaded SB3 policy instance.
    """
    path_str = str(path)
    errors: list[str] = []
    for cls in (PPO, SAC, TD3, DQN):
        try:
            return cls.load(path_str, env=env)
        except Exception as exc:
            errors.append(f"{cls.__name__}: {exc}")
    details = " | ".join(errors)
    raise ValueError(f"Unable to load subpolicy '{path_str}'. Details: {details}")


def train_meta_policy(
    env_id: str,
    subpolicy_paths: list[Union[str, Path]],
    option_horizon: int = 10,
    total_timesteps: int = 100_000,
    device: Union[str, th.device] = "auto",
) -> PPO:
    """Train high-level PPO policy on HRLMetaEnv.

    Parameters
    ----------
    env_id : str
        Gymnasium environment id used for high-level training.
    subpolicy_paths : list[str | pathlib.Path]
        Paths to pretrained low-level option checkpoints.
    option_horizon : int, default=10
        Number of low-level steps per high-level decision.
    total_timesteps : int, default=100_000
        Total training timesteps for the high-level PPO policy.
    device : str | torch.device, default="auto"
        Device passed to PPO.

    Returns
    -------
    stable_baselines3.PPO
        Trained high-level PPO model.
    """
    if len(subpolicy_paths) == 0:
        raise ValueError("subpolicy_paths must contain at least one checkpoint path.")

    base_env = gym.make(env_id)
    subpolicies = [
        _load_single_subpolicy(path, env=base_env) for path in subpolicy_paths
    ]
    meta_env = HRLMetaEnv(
        env=base_env, subpolicies=subpolicies, option_horizon=option_horizon
    )
    meta_policy = PPO("MlpPolicy", meta_env, verbose=0, device=device)
    meta_policy.learn(total_timesteps=total_timesteps)
    return meta_policy


__all__ = ["train_subpolicies", "train_meta_policy"]
