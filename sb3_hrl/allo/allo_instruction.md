# Role and Context

You are an Expert Reinforcement Learning Engineer specializing in PyTorch and Stable Baselines 3 (SB3). Your task is to implement a Hierarchical Reinforcement Learning (HRL) framework using the "Augmented Lagrangian Laplacian Objective" (ALLO) for option discovery.

The goal is to learn the Laplacian representation of an MDP's state space to discover temporally-extended actions (options). Previous methods like the Generalized Graph Drawing Objective (GGDO) suffer from severe hyperparameter sensitivity and converge to arbitrary rotations of eigenvectors. ALLO solves this using a max-min objective augmented with stop-gradient operators. This breaks the symmetry of eigenvector rotations, making the true eigenvectors and eigenvalues the unique stable equilibrium under gradient ascent-descent dynamics.

Your implementation must act as an offline feature extractor/representation learner that first ingests a massive dataset of transitions into an SB3 `ReplayBuffer`, and then optimizes the representations, which will subsequently be used to train hierarchical options.

# High-Level Architecture

You will implement a 4-step HRL pipeline:

1. **ALLO Offline Pretrainer:** A pre-training module that first collects a large dataset of transitions using a random policy, and then trains a neural network offline to approximate the graph Laplacian eigenvectors using the ALLO max-min objective.

2. **Intrinsic Reward Wrapper:** A Gym `RewardWrapper` that uses the trained ALLO network to generate intrinsic rewards based on moving along specific inferred eigenvectors.

3. **Subpolicy Training:** A script that trains $d$ low-level subpolicies (using standard SB3 agents like SAC or PPO), where each subpolicy is optimized to maximize the intrinsic reward of one specific eigenvector.

4. **High-Level Meta-Policy:** A custom hierarchical Gym environment where a high-level SB3 agent chooses which of the pre-trained subpolicies (options) to execute for $k$ steps to maximize the external environmental reward.

# Mathematical Formulation

The ALLO max-min objective is defined as:

$$
\max_{\beta} \min_{u} \sum_{i=1}^{d} \langle u_i, L u_i \rangle + \sum_{j=1}^{d} \sum_{k=1}^{j} \beta_{jk} (\langle u_j, [u_k] \rangle - \delta_{jk}) + b \sum_{j=1}^{d} \sum_{k=1}^{j} (\langle u_j, [u_k] \rangle - \delta_{jk})^2
$$

Where:

* $u$ is the parameterized neural network output (dimension $d$) representing the eigenvectors.

* $L$ is the graph Laplacian.

* $\beta_{jk}$ are the dual variables (a lower triangular matrix).

* $[u_k]$ denotes the stop-gradient operator applied to $u_k$ (implemented via `.detach()`).

* $b$ is a barrier coefficient that increases over time.

* $\delta_{jk}$ is the Kronecker delta.

# Implementation Instructions

## Step 1: The ALLO Feature Extractor (`ALLO`)

Create a class `ALLO` that inherits from `stable_baselines3.common.base_class.BaseAlgorithm`. While it uses SB3 components, it should be designed for an **Offline Collect-Then-Train** workflow.

### A. Architecture & Initialization

* **Buffer:** Use an SB3 `ReplayBuffer`. You must provide a method `collect_random_transitions(env, num_steps)` to completely fill this buffer before any training begins.

* **Network:** An SB3 `BaseFeaturesExtractor` or a standard MLP/CNN (depending on state shape) that outputs a feature vector of dimension $d$.

* **Parameters to maintain:**

  * `self.dual_variables`: A lower triangular matrix of shape $d \times d$, initialized to zeros, with `requires_grad=False`.

  * `self.dual_velocities`: Matrix of shape $d \times d$, initialized to zeros.

  * `self.barrier_coeffs`: Lower triangular matrix, tracking the barrier coefficient $b$.

### B. Optimization & Update Rules

Implement the `learn(total_timesteps, ...)` method inherited from `BaseAlgorithm`. **Crucially, clarify that in this offline context, the `total_timesteps` argument should be treated as `epochs`.** The method should iterate over the pre-filled replay buffer for `total_timesteps` (epochs). Each iteration (epoch) should perform the following steps:

**1. Graph Loss (Temporal Smoothness):**

* Sample a batch of discounted state pairs $(s_1, s_2)$ from the SB3 replay buffer using inverse transform sampling from a truncated geometric distribution.

* Compute features: $\phi_1 = \text{network}(s_1)$ and $\phi_2 = \text{network}(s_2)$.

* Compute the graph loss: $\frac{1}{B} \sum (\phi_1 - \phi_2)^2$ (mean over batch, sum over dimensions).

**2. Orthogonality & Barrier Loss:**

* Sample two sets of uncorrelated states ($s_{\text{uncorr1}}$ and $s_{\text{uncorr2}}$) uniformly from the completely filled replay buffer.

* Compute features: $\phi_{\text{uncorr1}}$ and $\phi_{\text{uncorr2}}$.

* Compute inner product matrices (the `.detach()` operation serves as the stop-gradient $[u_k]$ to break symmetry):

  * $M_1 = \frac{1}{B} (\phi_{\text{uncorr1}}^T \phi_{\text{uncorr1}}\text{.detach()})$

  * $M_2 = \frac{1}{B} (\phi_{\text{uncorr2}}^T \phi_{\text{uncorr2}}\text{.detach()})$

* Compute error matrices:

  * $E_1 = \text{tril}(M_1 - I)$

  * $E_2 = \text{tril}(M_2 - I)$

  * $E = 0.5 * (E_1 + E_2)$

* Calculate quadratic error: $E_{\text{quad}} = E_1 * E_2$ (element-wise).

* **Dual Loss:** sum of `self.dual_variables.detach() * E`.

* **Barrier Loss:** sum of $E_{\text{quad}}$ weighted by `self.barrier_coeffs[0, 0].detach()`.

**3. Total Loss & Network Update:**

* Total Loss = Graph Loss + Dual Loss + Barrier Loss.

* Perform standard PyTorch optimization step on the network parameters (`optimizer.zero_grad()`, `loss.backward()`, `torch.nn.utils.clip_grad_norm_`, `optimizer.step()`).

**4. Dual Variable & Barrier Updates (No Grad):**
Inside a `with torch.no_grad():` block, update the dual variables via gradient ascent:

* `effective_lr = lr_duals * (1 + use_barrier_for_duals * (barrier_coeff_val - 1))`

* `updated_duals = self.dual_variables + effective_lr * E`

* Clamp `updated_duals` between specified min/max bounds (e.g., 0.0 to 100.0).

* Apply momentum/velocity to the dual updates using `self.dual_velocities`.

* Set `self.dual_variables` to the lower triangular part of `updated_duals`.

* Update `self.barrier_coeffs` by adding the mean of the clamped positive quadratic errors ($E_{\text{quad}}$) multiplied by a barrier learning rate (`lr_barrier_coeff`). Clamp the new barrier coefficients.

## Step 2: Intrinsic Reward Wrapper (`LaplacianRewardWrapper`)

Create a standard `gymnasium.RewardWrapper` (or `gymnasium.Wrapper`).

* **Initialization:** Takes the trained `ALLO` network and a target `eigenvector_index` $z \in \{0 \dots d-1\}$.

* **Reward Logic:** `reward = phi(s_next)[z] - phi(s_current)[z]`. (This encourages the agent to transition to states that maximize the value of the $z$-th eigenvector, effectively discovering a bottleneck or specific room/region).

## Step 3: Training the Subpolicies (Options)

Provide a clear, reusable function `train_subpolicies(env_id, allo_network, num_eigenvectors)`.

* It should loop $z$ from $0$ to $d-1$.

* For each $z$, wrap the environment with `LaplacianRewardWrapper(..., eigenvector_index=z)`.

* Instantiate an SB3 `PPO` or `SAC` agent.

* Call `agent.learn()` and save the resulting subpolicy as `subpolicy_{z}.zip`.

## Step 4: The High-Level Hierarchical Environment (`HRLMetaEnv`)

Create a `gymnasium.Env` that wraps the original environment.

* **Observation Space:** Same as the original environment.

* **Action Space:** `Discrete(d)` (where $d$ is the number of trained subpolicies).

* **Initialization:** Loads a list of the trained SB3 subpolicies.

* **Step Function (`step(action)`)**:

  * The `action` selects which subpolicy to run.

  * Run the selected subpolicy in the underlying environment for a fixed temporal abstraction parameter $k$ steps (or until the subpolicy reaches a terminal state/bottleneck).

  * Accumulate the *external* (true) environment rewards during these $k$ steps.

  * Return the final state, accumulated external reward, and done flags.

* Provide a snippet demonstrating how to train a top-level SB3 `PPO` agent on this `HRLMetaEnv`.

# Coding Standards

1. Use `gymnasium` instead of the old `gym`.

2. Ensure strict adherence to the Stable Baselines 3 class architectures where applicable.

3. Include comprehensive Numpy-style docstrings outlining the shapes of tensors (e.g., `[batch_size, state_dim]`).

4. Keep the PyTorch device management robust (use `self.device`).

Please output the complete Python implementation containing the 4 steps described above.