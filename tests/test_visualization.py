from __future__ import annotations

import jax

from avalanche_sim import EnvConfig, make_env
from avalanche_sim.visualization import save_overview


def test_save_overview_writes_png(tmp_path) -> None:
    env = make_env(EnvConfig())
    _, state = env.reset(jax.random.PRNGKey(0))

    output = save_overview(env.config, state, tmp_path / "overview.png")

    assert output.exists()
    assert output.stat().st_size > 0
