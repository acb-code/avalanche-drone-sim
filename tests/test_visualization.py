from __future__ import annotations

import jax
import jax.numpy as jnp

from avalanche_sim import EnvConfig, make_env
from avalanche_sim.visualization import save_overview, save_rollout_gif


def test_save_overview_writes_png(tmp_path) -> None:
    env = make_env(EnvConfig())
    _, state = env.reset(jax.random.PRNGKey(0))

    output = save_overview(env.config, state, tmp_path / "overview.png")

    assert output.exists()
    assert output.stat().st_size > 0


def test_save_rollout_gif_writes_gif(tmp_path) -> None:
    env = make_env(EnvConfig())
    _, state = env.reset(jax.random.PRNGKey(1))
    states = [state]

    for step in range(3):
        _, state, _, _, _ = env.step(
            jax.random.PRNGKey(step + 2),
            state,
            jnp.zeros((env.config.num_drones, 4)),
        )
        states.append(state)

    output = save_rollout_gif(env.config, states, tmp_path / "rollout.gif", fps=4)

    assert output.exists()
    assert output.stat().st_size > 0
