from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from avalanche_sim import Action, EnvConfig, make_env


def test_reset_returns_expected_shapes() -> None:
    env = make_env(EnvConfig())
    obs, state = env.reset(jax.random.PRNGKey(0))

    assert state.drone_positions.shape == (env.config.num_drones, 3)
    assert state.victim_positions.shape == (env.config.num_victims, 3)
    assert obs.drone_features.shape[0] == env.config.num_drones
    assert obs.coverage_map.shape == (
        env.config.num_drones,
        env.config.coverage_resolution_y,
        env.config.coverage_resolution_x,
    )


def test_step_advances_time_and_marks_coverage() -> None:
    env = make_env(EnvConfig())
    _, state = env.reset(jax.random.PRNGKey(1))

    zero_actions = Action(control=jnp.zeros((env.config.num_drones, 4), dtype=jnp.float32))
    _, next_state, rewards, dones, info = env.step(jax.random.PRNGKey(2), state, zero_actions)

    assert int(next_state.time) == 1
    assert float(next_state.scanned_cells.mean()) > 0.0
    assert rewards.shape == (env.config.num_drones,)
    assert dones.shape == (env.config.num_drones,)
    assert "coverage" in info


def test_delivery_consumes_payload_when_drone_is_in_range() -> None:
    config = EnvConfig(num_drones=1, num_victims=1, payload_per_drone=1, delivery_range=10.0)
    env = make_env(config)
    _, state = env.reset(jax.random.PRNGKey(3))

    state = replace(
        state,
        drone_positions=state.victim_positions + jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        drone_knowledge=jnp.array([[True]]),
        drone_payload=jnp.array([1], dtype=jnp.int32),
    )

    _, next_state, _, _, info = env.step(
        jax.random.PRNGKey(4),
        state,
        Action(control=jnp.zeros((1, 4), dtype=jnp.float32)),
    )

    assert bool(next_state.victim_aided[0])
    assert int(next_state.drone_payload[0]) == 0
    assert int(info["deliveries"]) == 1
