"""Tests for pure-JAX mission logic (no MuJoCo dependency)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from avalanche_mujoco.config import AvalancheConfig
from avalanche_mujoco.mission import (
    communicate,
    compute_rewards,
    detect_collisions,
    deliver,
    sense,
    update_coverage,
    update_battery,
)
from avalanche_mujoco.terrain_mesh import generate_scene, terrain_height_at
from avalanche_mujoco.types import MissionState


def _make_mission(cfg: AvalancheConfig, key: jax.Array) -> MissionState:
    terrain_height, debris_mask, _, wind_field, victim_positions, severity = generate_scene(cfg, key)
    return MissionState(
        victim_found=jnp.zeros(cfg.num_victims, dtype=jnp.bool_),
        victim_confirmed=jnp.zeros(cfg.num_victims, dtype=jnp.bool_),
        victim_aided=jnp.zeros(cfg.num_victims, dtype=jnp.bool_),
        victim_survival=jnp.ones(cfg.num_victims),
        victim_positions=victim_positions,
        victim_severity=severity,
        drone_knowledge=jnp.zeros((cfg.num_drones, cfg.num_victims), dtype=jnp.bool_),
        shared_known_victims=jnp.zeros(cfg.num_victims, dtype=jnp.bool_),
        drone_battery=jnp.full(cfg.num_drones, cfg.base_battery),
        drone_payload=jnp.full(cfg.num_drones, cfg.payload_per_drone, dtype=jnp.int32),
        scanned_cells=jnp.zeros(
            (cfg.coverage_resolution_y, cfg.coverage_resolution_x), dtype=jnp.bool_
        ),
        terrain_height=terrain_height,
        debris_mask=debris_mask,
        wind_field=wind_field,
        time=jnp.array(0, dtype=jnp.int32),
        done=jnp.array(False),
        metrics={},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Terrain
# ──────────────────────────────────────────────────────────────────────────────

def test_generate_scene_shapes():
    cfg = AvalancheConfig(num_drones=2, num_victims=3)
    terrain, debris, unsafe, wind, victims, severity = generate_scene(cfg, jax.random.PRNGKey(0))
    assert terrain.shape == (cfg.coverage_resolution_y, cfg.coverage_resolution_x)
    assert debris.shape == terrain.shape
    assert wind.shape == terrain.shape + (2,)
    assert victims.shape == (cfg.num_victims, 3)
    assert severity.shape == (cfg.num_victims,)


def test_terrain_height_at_in_bounds():
    cfg = AvalancheConfig()
    terrain, *_ = generate_scene(cfg, jax.random.PRNGKey(1))
    xy = jnp.array([[0.0, 0.0], [cfg.map_size_x, cfg.map_size_y], [120.0, 80.0]])
    heights = terrain_height_at(cfg, terrain, xy)
    assert heights.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(heights)))


# ──────────────────────────────────────────────────────────────────────────────
# Coverage
# ──────────────────────────────────────────────────────────────────────────────

def test_coverage_marks_cells():
    cfg = AvalancheConfig(num_drones=2, coverage_resolution_x=4, coverage_resolution_y=4)
    cells = jnp.zeros((4, 4), dtype=jnp.bool_)
    positions = jnp.array([[20.0, 10.0, 15.0], [180.0, 130.0, 15.0]])
    updated, new_count = update_coverage(cfg, cells, positions)
    assert bool(jnp.any(updated))
    assert float(jnp.sum(new_count)) > 0


def test_coverage_no_double_count():
    cfg = AvalancheConfig(num_drones=1, coverage_resolution_x=4, coverage_resolution_y=4)
    cells = jnp.zeros((4, 4), dtype=jnp.bool_)
    pos = jnp.array([[60.0, 40.0, 15.0]])
    cells, _ = update_coverage(cfg, cells, pos)
    _, new_count2 = update_coverage(cfg, cells, pos)
    assert float(new_count2[0]) == 0.0  # already scanned


# ──────────────────────────────────────────────────────────────────────────────
# Sensing
# ──────────────────────────────────────────────────────────────────────────────

def test_sensing_out_of_range_no_detection():
    cfg = AvalancheConfig(num_drones=1, num_victims=1, sensor_range=10.0)
    mission = _make_mission(cfg, jax.random.PRNGKey(5))
    # Place drone far from all victims
    drone_pos = jnp.array([[0.0, 0.0, 100.0]])
    # Force victim position to be far away
    from dataclasses import replace as dc_replace
    mission = dc_replace(mission, victim_positions=jnp.array([[230.0, 150.0, 5.0]]))
    detections, confirmations = sense(jax.random.PRNGKey(0), cfg, mission, drone_pos, jnp.array([0.0]))
    assert not bool(jnp.any(detections))


def test_sensing_always_detects_when_close():
    cfg = AvalancheConfig(num_drones=1, num_victims=1, sensor_range=50.0, detection_decay=0.001)
    mission = _make_mission(cfg, jax.random.PRNGKey(7))
    from dataclasses import replace as dc_replace
    mission = dc_replace(
        mission,
        victim_positions=jnp.array([[10.0, 10.0, 5.0]]),
        debris_mask=jnp.zeros((cfg.coverage_resolution_y, cfg.coverage_resolution_x), dtype=jnp.bool_),
    )
    drone_pos = jnp.array([[10.0, 10.0, 12.0]])
    # Run many times and check at least some detections
    detected_any = False
    for i in range(20):
        detections, _ = sense(jax.random.PRNGKey(i), cfg, mission, drone_pos, jnp.array([0.0]))
        if bool(jnp.any(detections)):
            detected_any = True
            break
    assert detected_any


# ──────────────────────────────────────────────────────────────────────────────
# Communication
# ──────────────────────────────────────────────────────────────────────────────

def test_communication_propagates_knowledge():
    cfg = AvalancheConfig(num_drones=3, num_victims=2, communication_range=50.0)
    # All drones close together → knowledge spreads
    knowledge = jnp.array([[True, False], [False, False], [False, False]])
    positions = jnp.array([[10.0, 10.0, 15.0]] * 3)
    next_know, shared = communicate(cfg, knowledge, positions)
    # Drone 0 knows victim 0, all drones in range → all should know
    assert bool(jnp.all(next_know[:, 0]))
    assert bool(shared[0])


def test_communication_blocked_out_of_range():
    cfg = AvalancheConfig(num_drones=2, num_victims=1, communication_range=10.0)
    knowledge = jnp.array([[True], [False]])
    # Drones far apart
    positions = jnp.array([[0.0, 0.0, 15.0], [200.0, 150.0, 15.0]])
    next_know, _ = communicate(cfg, knowledge, positions)
    assert not bool(next_know[1, 0])  # drone 1 should NOT get the knowledge


# ──────────────────────────────────────────────────────────────────────────────
# Delivery
# ──────────────────────────────────────────────────────────────────────────────

def test_delivery_in_range_reduces_payload():
    cfg = AvalancheConfig(num_drones=1, num_victims=1, delivery_range=10.0, payload_per_drone=1)
    mission = _make_mission(cfg, jax.random.PRNGKey(3))
    from dataclasses import replace as dc_replace
    mission = dc_replace(
        mission,
        victim_positions=jnp.array([[50.0, 50.0, 5.0]]),
        drone_payload=jnp.array([1], dtype=jnp.int32),
        drone_knowledge=jnp.array([[True]]),
    )
    drone_pos = jnp.array([[50.0, 50.0, 10.0]])
    payload, victim_aided, delivery_flags, _ = deliver(cfg, mission, drone_pos, jnp.array([[True]]))
    assert bool(delivery_flags[0])
    assert bool(victim_aided[0])
    assert int(payload[0]) == 0


def test_delivery_out_of_range_no_delivery():
    cfg = AvalancheConfig(num_drones=1, num_victims=1, delivery_range=5.0)
    mission = _make_mission(cfg, jax.random.PRNGKey(4))
    from dataclasses import replace as dc_replace
    mission = dc_replace(
        mission,
        victim_positions=jnp.array([[50.0, 50.0, 5.0]]),
        drone_payload=jnp.array([1], dtype=jnp.int32),
        drone_knowledge=jnp.array([[True]]),
    )
    drone_pos = jnp.array([[100.0, 100.0, 10.0]])  # far away
    payload, victim_aided, delivery_flags, _ = deliver(cfg, mission, drone_pos, jnp.array([[True]]))
    assert not bool(delivery_flags[0])
    assert int(payload[0]) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Battery
# ──────────────────────────────────────────────────────────────────────────────

def test_battery_decreases_each_step():
    cfg = AvalancheConfig(num_drones=2, base_battery=100.0)
    battery = jnp.array([100.0, 100.0])
    speed = jnp.array([0.5, 0.0])
    new_battery = update_battery(cfg, battery, speed)
    assert bool(jnp.all(new_battery < battery))
    assert float(new_battery[0]) < float(new_battery[1])  # faster drone loses more


def test_battery_floors_at_zero():
    cfg = AvalancheConfig(num_drones=1)
    battery = jnp.array([0.001])
    speed = jnp.array([1.0])
    new_battery = update_battery(cfg, battery, speed)
    assert float(new_battery[0]) >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Collision detection
# ──────────────────────────────────────────────────────────────────────────────

def test_collision_detected_when_too_close():
    cfg = AvalancheConfig(num_drones=2, collision_distance=4.0)
    positions = jnp.array([[0.0, 0.0, 10.0], [2.0, 0.0, 10.0]])  # 2m apart → collision
    coll, near_coll = detect_collisions(cfg, positions)
    assert bool(jnp.any(coll))


def test_no_collision_when_far():
    cfg = AvalancheConfig(num_drones=2, collision_distance=4.0)
    positions = jnp.array([[0.0, 0.0, 10.0], [100.0, 0.0, 10.0]])
    coll, near_coll = detect_collisions(cfg, positions)
    assert not bool(jnp.any(coll))
    assert not bool(jnp.any(near_coll))


# ──────────────────────────────────────────────────────────────────────────────
# Observation
# ──────────────────────────────────────────────────────────────────────────────

def test_observation_shapes():
    from avalanche_mujoco.obs import build_observation
    cfg = AvalancheConfig(num_drones=3, num_victims=5)
    mission = _make_mission(cfg, jax.random.PRNGKey(10))
    drone_pos = jnp.zeros((cfg.num_drones, 3))
    heading = jnp.zeros(cfg.num_drones)
    obs = build_observation(cfg, mission, drone_pos, heading)
    assert obs.drone_features.shape == (cfg.num_drones, 8)
    assert obs.victim_features.shape == (cfg.num_drones, cfg.num_victims, 5)
    assert obs.team_features.shape == (cfg.num_drones, 5)
    assert obs.coverage_map.shape == (cfg.num_drones, cfg.coverage_resolution_y, cfg.coverage_resolution_x)
    assert obs.action_mask.shape == (cfg.num_drones, 4)
