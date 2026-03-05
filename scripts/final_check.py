from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from avalanche_sim import EnvConfig, make_env
from avalanche_sim.policies import lawnmower_policy
from avalanche_sim.visualization import save_overview


def run(seed: int, steps: int, output: Path) -> Path:
    env = make_env(EnvConfig())
    key = jax.random.PRNGKey(seed)
    _, state = env.reset(key)

    for _ in range(steps):
        key, step_key = jax.random.split(key)
        actions = lawnmower_policy(state, env.config)
        _, state, _, dones, _ = env.step(step_key, state, actions)
        if bool(jnp.all(dones)):
            break

    return save_overview(env.config, state, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short simulation and save a confirmation plot.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("artifacts/final-check.png"))
    args = parser.parse_args()

    output = run(args.seed, args.steps, args.output)
    print(output)


if __name__ == "__main__":
    main()
