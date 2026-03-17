# ALLO Usage

This guide shows a practical workflow for using `ALLO` to learn a Laplacian representation, train subpolicies with intrinsic rewards, and train a high-level meta-policy.

## 1. Pretrain ALLO Representation

```python
from sb3_hrl.allo import ALLO

# Use any Gymnasium env id or an instantiated env
allo = ALLO(
    env="Pendulum-v1",
    representation_dim=8,
    learning_rate=3e-4,
    batch_size=256,
    train_freq=256,
    gradient_steps=1,
    verbose=1,
)

allo.learn(total_timesteps=100_000)
allo.save("checkpoints/allo_pretrainer")
```

## 2. Train Low-Level Subpolicies

Each subpolicy optimizes intrinsic reward along one eigenvector coordinate.

```python
from sb3_hrl.allo import train_subpolicies

subpolicy_paths = train_subpolicies(
    env_id="Pendulum-v1",
    allo=allo,
    num_eigenvectors=allo.representation_dim,
    total_timesteps=200_000,
    save_dir="checkpoints/subpolicies",
    algorithm="auto",  # auto, ppo, or sac
)
```

## 3. Train Meta-Policy

The high-level policy selects which subpolicy to execute for `option_horizon` steps.

```python
from sb3_hrl.allo import train_meta_policy

meta_policy = train_meta_policy(
    env_id="Pendulum-v1",
    subpolicy_paths=[str(p) for p in subpolicy_paths],
    option_horizon=10,
    total_timesteps=200_000,
)

meta_policy.save("checkpoints/meta_policy")
```

## 4. Manual Meta Environment Construction

If you want full control, create `HRLMetaEnv` yourself and train with your own SB3 setup.

```python
import gymnasium as gym
from stable_baselines3 import PPO
from sb3_hrl.allo import HRLMetaEnv

# Assume `loaded_subpolicies` is a list of SB3 models
base_env = gym.make("Pendulum-v1")
meta_env = HRLMetaEnv(env=base_env, subpolicies=loaded_subpolicies, option_horizon=10)

meta = PPO("MlpPolicy", meta_env, verbose=1)
meta.learn(total_timesteps=200_000)
```

## Notes

- `ALLO` currently expects flattenable Box observations.
- Dict observations are not supported by the current implementation.
- For continuous action spaces, `train_subpolicies(..., algorithm="auto")` uses SAC.
- For discrete action spaces, `train_subpolicies(..., algorithm="auto")` uses PPO.
