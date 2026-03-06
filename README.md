# Avalanche Drone SAR Simulator

JAX-native multi-agent avalanche search-and-rescue environment with a simplified 3-DOF drone model.

## Highlights

- Pure JAX `reset`/`step` API
- Batched multi-agent state tensors
- Procedural alpine terrain and avalanche debris generation
- Probabilistic victim detection, local communications, and aid delivery
- Scripted and random baseline policies
- Headless confirmation visualization for fast manual checks, including animated rollouts
- Interactive browser-based 3D rollout viewer with playback controls
- Pluggable capability modules for movement, sensing, communication, and aid delivery

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

Run a short simulation and write an animated top-down rollout:

```bash
uv run python scripts/final_check.py --steps 30 --output artifacts/final-check.gif
```

The GIF shows terrain, debris, scanned cells, drones, victim states, and the current timestep so you can confirm temporal progression instead of only the final frame.

If you only want the last frame, use a `.png` output path:

```bash
uv run python scripts/final_check.py --steps 30 --output artifacts/final-check.png
```

For the interactive 3D viewer, write an `.html` output:

```bash
uv run python scripts/final_check.py --steps 30 --output artifacts/final-check.html
```

If you want the raw rollout payload for other tooling, write a `.json` output:

```bash
uv run python scripts/final_check.py --steps 30 --output artifacts/final-check.json
```

## Modular Capabilities

The environment still defaults to the built-in drone behavior, but you can swap capability models independently:

```python
from dataclasses import replace

from avalanche_sim import EnvConfig, make_env
from avalanche_sim.capabilities import ProbabilisticSensingModel, default_capabilities

capabilities = replace(default_capabilities(), sensing=ProbabilisticSensingModel())
env = make_env(EnvConfig(), capabilities=capabilities)
```
