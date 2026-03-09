# Avalanche Drone SAR Simulator

Multi-agent avalanche search-and-rescue environment with two backends:

| Backend | Package | Physics | Use case |
|---|---|---|---|
| **Kinematic (legacy)** | `avalanche_sim` | Pure-JAX 3-DOF kinematics | Fast prototyping |
| **MuJoCo/MJX (new)** | `avalanche_mujoco` | Real 6-DOF rigid-body dynamics | Physics fidelity, GPU training |

---

## Setup

```bash
# Minimal (JAX only — kinematic sim)
uv sync

# With MuJoCo CPU physics + RL wrappers
uv sync --extra dev
```

Dependencies installed by `--extra dev`: `mujoco`, `gymnasium`, `pettingzoo`, `matplotlib`, `pytest`.

For GPU-batched simulation via MJX (requires CUDA):
```bash
pip install "mujoco-mjx>=3.1" "jax[cuda12]>=0.4.30"
```

---

## MuJoCo simulator (`avalanche_mujoco`)

### Use Case 1 — Simulate & Visualize (CPU)

```python
import jax
from avalanche_mujoco import AvalancheConfig, make_physics_env

cfg = AvalancheConfig(num_drones=4, num_victims=6)
env = make_physics_env(cfg, viewer=True)   # viewer=True opens the MuJoCo GUI

key = jax.random.PRNGKey(0)
obs, mission = env.reset(key)

for step in range(250):
    key, k = jax.random.split(key)
    actions = jax.random.uniform(k, (cfg.num_drones, 4), minval=-1.0, maxval=1.0)
    obs, mission, rewards, dones, info = env.step(k, mission, actions)
    if mission.done:
        break

env.close()
```

Or run the demo script:

```bash
uv run python scripts/demo_visualize.py
```

### Use Case 2 — GPU-Batched Simulation (MJX)

```python
import jax
import jax.numpy as jnp
from avalanche_mujoco import AvalancheConfig, AvalancheMJXEnv

cfg = AvalancheConfig(num_drones=4)
env = AvalancheMJXEnv(cfg)

N = 1000
keys = jax.random.split(jax.random.PRNGKey(0), N)
states = jax.jit(jax.vmap(env.reset))(keys)

@jax.jit
def step_batch(states, key):
    step_keys = jax.random.split(key, N)
    actions = jax.random.uniform(key, (N, cfg.num_drones, 4), minval=-1.0, maxval=1.0)
    return jax.vmap(env.step)(states, actions, step_keys)

states, rewards, info = step_batch(states, jax.random.PRNGKey(1))
print(f"Mean reward: {float(jnp.mean(rewards)):.4f}")
```

Or run the demo script:

```bash
uv run python scripts/demo_batch.py --n-envs 64 --n-steps 50
```

### Use Case 3 — RL / MARL Training

**Single-agent (Gymnasium):**

```python
from avalanche_mujoco import AvalancheConfig
from avalanche_mujoco.wrappers import AvalancheGymnasiumEnv

env = AvalancheGymnasiumEnv(AvalancheConfig(num_drones=1), backend="cpu")
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

**Multi-agent (PettingZoo):**

```python
from avalanche_mujoco import AvalancheConfig
from avalanche_mujoco.wrappers import AvalanchePettingZooEnv

env = AvalanchePettingZooEnv(AvalancheConfig(num_drones=4), backend="cpu")
obs, info = env.reset(seed=0)
actions = {a: env.action_space(a).sample() for a in env.agents}
obs, rewards, terminated, truncated, info = env.step(actions)
```

Or run the training demo (REINFORCE baseline, no extra deps; pass `--algo ppo` for SB3):

```bash
uv run python scripts/demo_train.py --steps 50000
uv run python scripts/demo_train.py --steps 100000 --algo ppo   # requires stable-baselines3
```

### Action & Observation spaces

**Actions:** `Box(-1, 1, shape=(4,))` per drone — `[vx, vy, vz, yaw_rate]` in world frame.
An internal PID controller converts velocity commands to rotor thrusts.

**Observations** (per drone):

| Key | Shape | Description |
|---|---|---|
| `drone_features` | `(8,)` | pos (norm), heading cos/sin, battery, payload, neighbour count |
| `victim_features` | `(num_victims, 5)` | known/confirmed/aided/survival/distance per victim |
| `team_features` | `(5,)` | time, coverage, mean battery, mean survival, known victims |
| `coverage_map` | `(res_y, res_x)` | binary scanned-cell map |
| `action_mask` | `(4,)` | 0 if drone is dead, 1 otherwise (gymnasium only) |

---

## Kinematic simulator (`avalanche_sim`, legacy)

```python
import jax
from avalanche_sim import EnvConfig, make_env
from avalanche_sim.policies import random_policy

env = make_env(EnvConfig())
key = jax.random.PRNGKey(0)
obs, state = env.reset(key)
actions = random_policy(jax.random.PRNGKey(1), state, env.config)
obs, state, rewards, dones, info = env.step(jax.random.PRNGKey(2), state, actions)
```

Run the rollout check (saves a GIF/PNG/HTML):

```bash
uv run python scripts/final_check.py --steps 30 --output artifacts/final-check.gif
uv run python scripts/final_check.py --steps 30 --output artifacts/final-check.html  # 3D viewer
```

---

## Tests

```bash
# All tests (requires mujoco, gymnasium, pettingzoo)
uv run pytest

# Just pure-JAX mission tests (no mujoco needed)
uv run pytest tests/test_mission.py

# Physics tests only
uv run pytest tests/test_physics.py

# Wrapper conformance (check_env + parallel_api_test)
uv run pytest tests/test_wrappers.py
```

---

## Package structure

```
src/
  avalanche_sim/          # Legacy kinematic simulator
  avalanche_mujoco/       # MuJoCo/MJX simulator
    models/quadrotor.xml  # Skydio X2-class MJCF drone model
    config.py             # AvalancheConfig
    types.py              # MissionState, MJXBatchState, Observation
    terrain_mesh.py       # Heightmap generation + virtual terrain constraint
    drone.py              # Multi-drone MJCF composition
    scene.py              # Full scene assembler (terrain + drones + markers)
    pid.py                # Velocity cmd → rotor thrust PID (pure JAX)
    wind.py               # Wind force injection
    mission.py            # Sensing, comms, delivery, rewards (pure JAX)
    obs.py                # Observation construction (pure JAX)
    physics_env.py        # CPU MuJoCo environment
    mjx_env.py            # MJX batched environment
    wrappers/
      gymnasium_wrapper.py
      pettingzoo_wrapper.py
scripts/
  demo_visualize.py       # Use case 1: live viewer
  demo_batch.py           # Use case 2: GPU batch rollout
  demo_train.py           # Use case 3: PPO/REINFORCE baseline
  final_check.py          # Legacy kinematic sim check
```
