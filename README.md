# Avalanche Drone SAR Simulator

JAX-native multi-agent avalanche search-and-rescue environment with a simplified 3-DOF drone model.

## Highlights

- Pure JAX `reset`/`step` API
- Batched multi-agent state tensors
- Procedural alpine terrain and avalanche debris generation
- Probabilistic victim detection, local communications, and aid delivery
- Scripted and random baseline policies
- Headless confirmation visualization for fast manual checks

## Setup

```bash
uv sync --extra dev
```

This creates `.venv/` and installs the simulator, test dependencies, and plotting support used by the confirmation check.

## Run

```bash
uv run python
```

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

## Tests

```bash
uv run pytest
```

## Final Confirmation Check

Run a short simulation and write a quick top-down plot:

```bash
uv run python scripts/final_check.py --steps 30 --output artifacts/final-check.png
```

The image marks terrain, debris, scanned cells, drones, and victim states so you can confirm the sim is stepping and updating state coherently.
