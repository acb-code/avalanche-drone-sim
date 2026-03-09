"""Observation construction — pure JAX, backward-compatible with old sim.

build_observation(cfg, mission, drone_positions, drone_heading) → Observation
"""
from __future__ import annotations

import jax.numpy as jnp

from .config import AvalancheConfig
from .types import MissionState, Observation


def build_observation(
    cfg: AvalancheConfig,
    mission: MissionState,
    drone_positions: jnp.ndarray,   # (num_drones, 3)
    drone_heading: jnp.ndarray,     # (num_drones,) yaw in radians
) -> Observation:
    """Build the per-drone observation dict.

    Output shapes mirror avalanche_sim.types.Observation for drop-in compat.
    """
    # ── Drone self-features ────────────────────────────────────────────────
    pos_norm = drone_positions / jnp.array([
        cfg.map_size_x, cfg.map_size_y, cfg.altitude_max
    ])
    own = jnp.concatenate([
        pos_norm,
        jnp.stack([
            jnp.cos(drone_heading),
            jnp.sin(drone_heading),
            mission.drone_battery / cfg.base_battery,
            mission.drone_payload / jnp.maximum(1, cfg.payload_per_drone),
        ], axis=-1),
    ], axis=-1)   # (num_drones, 7)

    # ── Neighbour connectivity feature ────────────────────────────────────
    rel = drone_positions[None, :, :] - drone_positions[:, None, :]
    rel_dist = jnp.linalg.norm(rel[..., :2], axis=-1)
    neighbor_mask = (rel_dist <= cfg.communication_range).astype(jnp.float32)
    neighbor_count = jnp.sum(neighbor_mask, axis=-1, keepdims=True) / jnp.maximum(1, cfg.num_drones - 1)
    drone_features = jnp.concatenate([own, neighbor_count], axis=-1)   # (num_drones, 8)

    # ── Victim features ────────────────────────────────────────────────────
    victim_rel = mission.victim_positions[None, :, :] - drone_positions[:, None, :]
    victim_dist = jnp.linalg.norm(victim_rel, axis=-1) / jnp.sqrt(
        cfg.map_size_x**2 + cfg.map_size_y**2
    )
    victim_visible = mission.drone_knowledge.astype(jnp.float32)
    victim_features = jnp.stack([
        victim_visible,
        jnp.broadcast_to(mission.victim_confirmed.astype(jnp.float32), victim_visible.shape),
        jnp.broadcast_to(mission.victim_aided.astype(jnp.float32), victim_visible.shape),
        jnp.broadcast_to(mission.victim_survival, victim_visible.shape),
        victim_dist,
    ], axis=-1)   # (num_drones, num_victims, 5)

    # ── Global team features ───────────────────────────────────────────────
    team_features = jnp.tile(
        jnp.array([
            mission.time / cfg.horizon,
            mission.scanned_cells.mean(),
            jnp.mean(mission.drone_battery) / cfg.base_battery,
            jnp.mean(mission.victim_survival),
            jnp.mean(mission.shared_known_victims.astype(jnp.float32)),
        ]),
        (cfg.num_drones, 1),
    )   # (num_drones, 5)

    # ── Coverage map ───────────────────────────────────────────────────────
    coverage_map = jnp.broadcast_to(
        mission.scanned_cells[None, :, :].astype(jnp.float32),
        (cfg.num_drones, cfg.coverage_resolution_y, cfg.coverage_resolution_x),
    )

    # ── Action mask (dead drones have zero mask) ───────────────────────────
    action_mask = (mission.drone_battery[:, None] > 0.0).astype(jnp.float32) * jnp.ones(
        (cfg.num_drones, 4), dtype=jnp.float32
    )

    return Observation(
        drone_features=drone_features,
        victim_features=victim_features,
        team_features=team_features,
        coverage_map=coverage_map,
        action_mask=action_mask,
    )
