# HIRO Usage Guide

This guide explains how to train and use the `HIRO` algorithm in this package.

## What You Get

`HIRO` is implemented as a Stable-Baselines3-compatible algorithm that internally trains:

- a high-level `manager` policy (proposes subgoals)
- a low-level `worker` policy (produces environment actions)

The implementation follows the architecture in `hiro_guide.md` and uses TD3 for both levels.

## Installation

From the project root:

```bash
pip install -e .
```

## Quick Start

```python
import gymnasium as gym
from sb3_hrl import HIRO

env = gym.make("Pendulum-v1")

model = HIRO(
    policy="MlpPolicy",
    env=env,
    learning_starts=1_000,
    subgoal_freq=10,
    buffer_size=1_000_000,
    batch_size=256,
)

model.learn(total_timesteps=200_000)
```

## Minimal Working Example

```python
import gymnasium as gym
from sb3_hrl import HIRO

env = gym.make("Pendulum-v1")

model = HIRO("MlpPolicy", env, subgoal_freq=5)
model.learn(total_timesteps=50_000)

obs, _ = env.reset()
for _ in range(200):
    goal, _ = model.predict(obs, deterministic=True)
    # HIRO.predict returns the manager subgoal.
    # For evaluation, use a rollout loop with the worker policy if needed.
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, _ = env.reset()
```

## Core Constructor Arguments

Commonly used parameters:

- `policy`: TD3 policy name, usually `"MlpPolicy"`
- `env`: Gymnasium environment (continuous action space required)
- `learning_rate`, `buffer_size`, `batch_size`, `tau`, `gamma`
- `learning_starts`: warmup before gradient updates
- `train_freq`: optimization trigger in env steps
- `gradient_steps`: updates per training trigger
- `subgoal_freq`: manager subgoal period `c`
- `subgoal_space`: manager action space (subgoal space)
- `state_to_goal_proj_fn`: projection `h(s)`
- `manager_kwargs`: extra kwargs passed to manager TD3
- `worker_kwargs`: extra kwargs passed to worker TD3

## Using a Custom Projection Function

If your subgoal is not the full state, pass both:

- `state_to_goal_proj_fn`
- `subgoal_space`

Example:

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sb3_hrl import HIRO

def project_state(flat_state: np.ndarray) -> np.ndarray:
    # Example: use first two coordinates as goal representation
    return flat_state[:2].astype(np.float32)

env = gym.make("Pendulum-v1")
subgoal_space = spaces.Box(
    low=np.array([-2.0, -8.0], dtype=np.float32),
    high=np.array([2.0, 8.0], dtype=np.float32),
    dtype=np.float32,
)

model = HIRO(
    policy="MlpPolicy",
    env=env,
    state_to_goal_proj_fn=project_state,
    subgoal_space=subgoal_space,
    subgoal_freq=10,
)

model.learn(total_timesteps=200_000)
```

## Tuning Tips

- Start with `subgoal_freq` in the range `5` to `20`.
- Increase `learning_starts` if early training is unstable.
- Keep manager and worker network sizes similar first, then tune separately with `manager_kwargs` and `worker_kwargs`.
- If relabeling appears noisy, tune:
  - `correction_candidate_count`
  - `correction_noise_scale`
  - `correction_action_sigma`

## Saving and Loading

```python
model.save("hiro_pendulum")

loaded = HIRO.load("hiro_pendulum", env=env)
loaded.learn(total_timesteps=50_000, reset_num_timesteps=False)
```

## Current Constraints

- Environment action space must be `gymnasium.spaces.Box`.
- Observation space must be flattenable to a `Box`.
- Current implementation targets single-environment training (`n_envs = 1`).
- `HIRO.predict` returns manager subgoals (not direct env actions).

## Troubleshooting

- Error about policy name: use `"MlpPolicy"` for TD3-based setup.
- Error about `subgoal_space`: provide it whenever you pass `state_to_goal_proj_fn`.
- Shape mismatch in projection: ensure your projection output exactly matches `subgoal_space.shape`.
