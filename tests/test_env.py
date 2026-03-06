from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from avalanche_sim import Action, EnvConfig, make_env
from avalanche_sim.capabilities import SensingResult, default_capabilities


class AlwaysDetectSensingModel:
    def sense(self, key, config, state, positions) -> SensingResult:
        del key, config, positions
        detections = jnp.ones((state.drone_positions.shape[0], state.victim_positions.shape[0]), dtype=jnp.bool_)
        return SensingResult(detections=detections, confirmations=detections)


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


def test_custom_sensing_capability_can_be_injected() -> None:
    capabilities = replace(default_capabilities(), sensing=AlwaysDetectSensingModel())
    env = make_env(EnvConfig(num_drones=2, num_victims=3), capabilities=capabilities)
    _, state = env.reset(jax.random.PRNGKey(5))

    _, next_state, _, _, _ = env.step(
        jax.random.PRNGKey(6),
        state,
        Action(control=jnp.zeros((env.config.num_drones, 4), dtype=jnp.float32)),
    )

    assert bool(jnp.all(next_state.victim_found))
    assert bool(jnp.all(next_state.victim_confirmed))


def test_terrain_blocks_horizontal_motion_without_clearance() -> None:
    config = EnvConfig(
        num_drones=1,
        num_victims=1,
        map_size_x=40.0,
        map_size_y=40.0,
        max_xy_speed=20.0,
        altitude_min=8.0,
        altitude_max=80.0,
        coverage_resolution_x=4,
        coverage_resolution_y=4,
    )
    env = make_env(config)
    _, state = env.reset(jax.random.PRNGKey(9))

    terrain = jnp.array(
        [
            [0.0, 25.0, 25.0, 25.0],
            [0.0, 25.0, 25.0, 25.0],
            [0.0, 25.0, 25.0, 25.0],
            [0.0, 25.0, 25.0, 25.0],
        ],
        dtype=jnp.float32,
    )
    state = replace(
        state,
        terrain_height=terrain,
        drone_positions=jnp.array([[5.0, 5.0, 10.0]], dtype=jnp.float32),
        drone_heading=jnp.array([0.0], dtype=jnp.float32),
    )

    _, next_state, _, _, _ = env.step(
        jax.random.PRNGKey(10),
        state,
        Action(control=jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)),
    )

    assert float(next_state.drone_positions[0, 0]) < 10.0
