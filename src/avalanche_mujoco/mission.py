"""Mission logic — pure JAX functions (sensing, comms, delivery, rewards, coverage).

Ported from avalanche_sim.capabilities and avalanche_sim.env as standalone
functions. Both CPU and MJX envs call the same code.

All functions:
  - Accept config and arrays as plain arguments (no `self`)
  - Are safe to jit / vmap
  - Have no side effects
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import AvalancheConfig
from .types import MissionState


# ──────────────────────────────────────────────────────────────────────────────
# Sensing
# ──────────────────────────────────────────────────────────────────────────────

def sense(
    key: jax.Array,
    cfg: AvalancheConfig,
    mission: MissionState,
    drone_positions: jnp.ndarray,   # (num_drones, 3)
    drone_heading: jnp.ndarray,     # (num_drones,) yaw radians
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Probabilistic sensing model.

    Returns
    -------
    detections    : (num_drones, num_victims) bool
    confirmations : (num_drones, num_victims) bool
    """
    diff = mission.victim_positions[None, :, :3] - drone_positions[:, None, :3]
    dist = jnp.linalg.norm(diff, axis=-1)
    horizontal = jnp.linalg.norm(diff[..., :2], axis=-1)
    forward = jnp.stack([jnp.cos(drone_heading), jnp.sin(drone_heading)], axis=-1)
    horizontal_unit = diff[..., :2] / jnp.maximum(horizontal[..., None], 1.0e-6)
    fov_score = jnp.sum(horizontal_unit * forward[:, None, :], axis=-1)
    altitude_factor = jnp.exp(
        -jnp.abs(drone_positions[:, None, 2] - cfg.sensor_altitude_scale) / cfg.sensor_altitude_scale
    )

    victim_cells_x = jnp.clip(
        jnp.floor(mission.victim_positions[:, 0] / cfg.map_size_x * cfg.coverage_resolution_x).astype(jnp.int32),
        0, cfg.coverage_resolution_x - 1,
    )
    victim_cells_y = jnp.clip(
        jnp.floor(mission.victim_positions[:, 1] / cfg.map_size_y * cfg.coverage_resolution_y).astype(jnp.int32),
        0, cfg.coverage_resolution_y - 1,
    )
    under_debris = mission.debris_mask[victim_cells_y, victim_cells_x]
    base_prob = jnp.exp(-cfg.detection_decay * dist) * altitude_factor
    base_prob = base_prob * (0.6 + 0.4 * (~under_debris)[None, :].astype(jnp.float32))
    base_prob = base_prob * (0.35 + 0.65 * (fov_score > cfg.sensor_fov_cos).astype(jnp.float32))
    known_bonus = cfg.rescan_bonus * mission.drone_knowledge.astype(jnp.float32)
    detect_prob = jnp.clip(base_prob + known_bonus, 0.0, 0.98)
    in_range = dist <= cfg.sensor_range

    key_detect, key_confirm = jax.random.split(key)
    detect_draw = jax.random.uniform(key_detect, detect_prob.shape)
    confirm_draw = jax.random.uniform(key_confirm, detect_prob.shape)
    detections = in_range & (detect_draw < detect_prob) & (~mission.victim_aided)[None, :]
    confirmations = detections & (confirm_draw < jnp.clip(detect_prob + 0.1, 0.0, 1.0))
    return detections, confirmations


# ──────────────────────────────────────────────────────────────────────────────
# Communication
# ──────────────────────────────────────────────────────────────────────────────

def communicate(
    cfg: AvalancheConfig,
    drone_knowledge: jnp.ndarray,   # (num_drones, num_victims) bool
    drone_positions: jnp.ndarray,   # (num_drones, 3)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Range-based knowledge sharing.

    Returns
    -------
    next_knowledge    : (num_drones, num_victims) bool
    shared_known      : (num_victims,) bool
    """
    disp = drone_positions[:, None, :2] - drone_positions[None, :, :2]
    dist = jnp.linalg.norm(disp, axis=-1)
    in_range = dist <= cfg.communication_range
    informed = jnp.einsum("ij,jv->iv", in_range.astype(jnp.int32), drone_knowledge.astype(jnp.int32)) > 0
    next_knowledge = jnp.logical_or(drone_knowledge, informed)
    shared_known = jnp.any(next_knowledge, axis=0)
    return next_knowledge, shared_known


# ──────────────────────────────────────────────────────────────────────────────
# Aid delivery
# ──────────────────────────────────────────────────────────────────────────────

def deliver(
    cfg: AvalancheConfig,
    mission: MissionState,
    drone_positions: jnp.ndarray,   # (num_drones, 3)
    drone_knowledge: jnp.ndarray,   # (num_drones, num_victims) bool
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Priority aid delivery.

    Returns
    -------
    payload           : (num_drones,) int32
    victim_aided      : (num_victims,) bool
    delivery_flags    : (num_victims,) bool   (new deliveries this step)
    duplicate_attempts: (num_victims,) int32
    """
    diff = mission.victim_positions[None, :, :] - drone_positions[:, None, :]
    dist = jnp.linalg.norm(diff, axis=-1)
    eligible = (
        (dist <= cfg.delivery_range)
        & drone_knowledge
        & (~mission.victim_aided)[None, :]
        & (mission.drone_payload[:, None] > 0)
    )
    inf_dist = jnp.where(eligible, dist, jnp.inf)
    chosen_drone = jnp.argmin(inf_dist, axis=0)
    victim_has_drone = jnp.any(eligible, axis=0)
    selected = jax.nn.one_hot(chosen_drone, cfg.num_drones, dtype=jnp.bool_).T & victim_has_drone[None, :]
    per_drone_deliveries = jnp.sum(selected, axis=1).astype(jnp.int32)
    payload = jnp.maximum(0, mission.drone_payload - per_drone_deliveries)
    delivery_flags = victim_has_drone
    victim_aided = jnp.logical_or(mission.victim_aided, delivery_flags)
    attempted = (dist <= cfg.delivery_range) & drone_knowledge & (mission.drone_payload[:, None] > 0)
    duplicate_attempts = jnp.maximum(
        0, jnp.sum(attempted, axis=0).astype(jnp.int32) - delivery_flags.astype(jnp.int32)
    )
    return payload, victim_aided, delivery_flags, duplicate_attempts


# ──────────────────────────────────────────────────────────────────────────────
# Coverage
# ──────────────────────────────────────────────────────────────────────────────

def update_coverage(
    cfg: AvalancheConfig,
    scanned_cells: jnp.ndarray,     # (res_y, res_x) bool
    drone_positions: jnp.ndarray,   # (num_drones, 3)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Mark grid cells beneath each drone as scanned.

    Returns
    -------
    updated_cells : (res_y, res_x) bool
    new_count     : (num_drones,) float32  — 1 for newly scanned, 0 otherwise
    """
    cx = jnp.clip(
        jnp.floor(drone_positions[:, 0] / cfg.map_size_x * cfg.coverage_resolution_x).astype(jnp.int32),
        0, cfg.coverage_resolution_x - 1,
    )
    cy = jnp.clip(
        jnp.floor(drone_positions[:, 1] / cfg.map_size_y * cfg.coverage_resolution_y).astype(jnp.int32),
        0, cfg.coverage_resolution_y - 1,
    )
    before = scanned_cells[cy, cx]
    updated = scanned_cells.at[cy, cx].set(True)
    new_count = (~before).astype(jnp.float32)
    return updated, new_count


# ──────────────────────────────────────────────────────────────────────────────
# Battery update
# ──────────────────────────────────────────────────────────────────────────────

def update_battery(
    cfg: AvalancheConfig,
    battery: jnp.ndarray,           # (num_drones,)
    speed_norm: jnp.ndarray,        # (num_drones,) in [0, 1]
) -> jnp.ndarray:
    """Drain battery proportionally to speed."""
    delta = cfg.battery_burn_per_step + cfg.battery_burn_per_speed * speed_norm
    return jnp.maximum(0.0, battery - delta)


# ──────────────────────────────────────────────────────────────────────────────
# Rewards
# ──────────────────────────────────────────────────────────────────────────────

def compute_rewards(
    cfg: AvalancheConfig,
    new_coverage_count: jnp.ndarray,   # (num_drones,)
    found: jnp.ndarray,                # (num_victims,) bool  — cumulative
    prev_found: jnp.ndarray,
    confirmed: jnp.ndarray,
    prev_confirmed: jnp.ndarray,
    delivery_flags: jnp.ndarray,       # (num_victims,) bool  — this step
    collisions: jnp.ndarray,           # (num_drones,) bool
    near_collisions: jnp.ndarray,      # (num_drones,) bool
    speed_norm: jnp.ndarray,           # (num_drones,)
    victim_survival: jnp.ndarray,      # (num_victims,)
    duplicate_attempts: jnp.ndarray,   # (num_victims,) int32
    timeout: jnp.ndarray,              # () bool
    alive_mask: jnp.ndarray,           # (num_drones,) bool
) -> jnp.ndarray:
    """Per-drone reward signal.

    Returns
    -------
    rewards : (num_drones,)
    """
    new_finds = jnp.any(found & ~prev_found).astype(jnp.float32)
    new_confirms = jnp.any(confirmed & ~prev_confirmed).astype(jnp.float32)
    deliveries = jnp.sum(delivery_flags.astype(jnp.float32))

    coverage_reward = cfg.reward_coverage * new_coverage_count
    team_bonus = (
        cfg.reward_find * new_finds
        + cfg.reward_confirm * new_confirms
        + cfg.reward_delivery * deliveries
        + cfg.reward_survival * jnp.sum(victim_survival)
    ) / cfg.num_drones

    penalties = (
        cfg.penalty_collision * collisions.astype(jnp.float32)
        + cfg.penalty_near_collision * near_collisions.astype(jnp.float32)
        + cfg.penalty_wasted_energy * speed_norm
        + cfg.penalty_duplicate_delivery * jnp.sum(duplicate_attempts.astype(jnp.float32)) / cfg.num_drones
        + cfg.penalty_timeout * timeout.astype(jnp.float32) / cfg.num_drones
    )
    rewards = coverage_reward + team_bonus - penalties
    return rewards * alive_mask.astype(jnp.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Collision detection (world-frame drone positions)
# ──────────────────────────────────────────────────────────────────────────────

def detect_collisions(
    cfg: AvalancheConfig,
    drone_positions: jnp.ndarray,   # (num_drones, 3)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Drone-drone collision / near-collision detection.

    Returns
    -------
    collisions      : (num_drones,) bool
    near_collisions : (num_drones,) bool
    """
    disp = drone_positions[:, None, :] - drone_positions[None, :, :]
    pair_dist = jnp.linalg.norm(disp, axis=-1) + jnp.eye(cfg.num_drones) * 1.0e6
    min_dist = jnp.min(pair_dist, axis=1)
    collisions = min_dist < cfg.collision_distance
    near_collisions = (min_dist >= cfg.collision_distance) & (min_dist < cfg.near_collision_distance)
    return collisions, near_collisions


# ──────────────────────────────────────────────────────────────────────────────
# Mission state step (the "logic" half of env.step)
# ──────────────────────────────────────────────────────────────────────────────

def mission_step(
    key: jax.Array,
    cfg: AvalancheConfig,
    mission: MissionState,
    drone_positions: jnp.ndarray,   # (num_drones, 3)
    drone_heading: jnp.ndarray,     # (num_drones,) yaw
    speed_norm: jnp.ndarray,        # (num_drones,) normalised [0,1]
    collisions: jnp.ndarray,        # (num_drones,) bool
    near_collisions: jnp.ndarray,   # (num_drones,) bool
) -> tuple[MissionState, jnp.ndarray, dict]:
    """Apply one round of SAR mission logic.

    Returns
    -------
    next_mission : MissionState
    rewards      : (num_drones,)
    info         : dict
    """
    # Coverage
    scanned_cells, new_coverage_count = update_coverage(cfg, mission.scanned_cells, drone_positions)

    # Sensing
    detections, confirmations = sense(key, cfg, mission, drone_positions, drone_heading)

    # Communication
    post_detect_knowledge = jnp.logical_or(mission.drone_knowledge, detections)
    drone_knowledge, shared_known = communicate(cfg, post_detect_knowledge, drone_positions)

    # Victim found / confirmed
    found = jnp.logical_or(mission.victim_found, jnp.any(detections, axis=0))
    confirmed = jnp.logical_or(mission.victim_confirmed, jnp.any(confirmations, axis=0))

    # Survival decay
    victim_survival = jnp.maximum(
        0.0,
        mission.victim_survival
        - cfg.severity_decay * mission.victim_severity * (1.0 - mission.victim_aided.astype(jnp.float32)),
    )

    # Delivery
    payload, victim_aided, delivery_flags, duplicate_attempts = deliver(
        cfg, mission,
        drone_positions, drone_knowledge,
    )

    # Battery
    battery = update_battery(cfg, mission.drone_battery, speed_norm)
    alive_mask = battery > 0.0

    # Episode termination
    timeout = (mission.time + 1) >= cfg.horizon
    mission_complete = jnp.all(victim_aided)
    done = jnp.logical_or(timeout, mission_complete)

    # Rewards
    rewards = compute_rewards(
        cfg,
        new_coverage_count,
        found, mission.victim_found,
        confirmed, mission.victim_confirmed,
        delivery_flags,
        collisions, near_collisions,
        speed_norm,
        victim_survival,
        duplicate_attempts,
        timeout,
        alive_mask,
    )

    next_mission = MissionState(
        victim_found=found,
        victim_confirmed=confirmed,
        victim_aided=victim_aided,
        victim_survival=victim_survival,
        victim_positions=mission.victim_positions,
        victim_severity=mission.victim_severity,
        drone_knowledge=drone_knowledge,
        shared_known_victims=shared_known,
        drone_battery=battery,
        drone_payload=payload,
        scanned_cells=scanned_cells,
        terrain_height=mission.terrain_height,
        debris_mask=mission.debris_mask,
        wind_field=mission.wind_field,
        time=mission.time + 1,
        done=done,
        metrics={
            "coverage": scanned_cells.mean(),
            "find_events": jnp.sum(found.astype(jnp.int32)),
            "confirm_events": jnp.sum(confirmed.astype(jnp.int32)),
            "delivery_events": jnp.sum(victim_aided.astype(jnp.int32)),
        },
    )
    info = {
        "team_done": done,
        "coverage": scanned_cells.mean(),
        "new_detections": jnp.any(found & ~mission.victim_found),
        "new_confirmations": jnp.any(confirmed & ~mission.victim_confirmed),
        "deliveries": jnp.sum(delivery_flags.astype(jnp.int32)),
        "collisions": collisions,
        "near_collisions": near_collisions,
    }
    return next_mission, rewards, info


