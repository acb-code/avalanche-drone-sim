"""Types for the MuJoCo-backed avalanche simulator.

MissionState  — pure-JAX mission bookkeeping (vmappable pytree).
MJXBatchState — full batched env state (mjx.Data + MissionState + scene).
Observation   — per-drone observation (backward-compatible with old sim).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
from jax.tree_util import register_dataclass


@register_dataclass
@dataclass
class MissionState:
    """Pure-JAX mission state — safe to vmap / jit over."""
    # victim status
    victim_found: jnp.ndarray           # (num_victims,) bool
    victim_confirmed: jnp.ndarray       # (num_victims,) bool
    victim_aided: jnp.ndarray           # (num_victims,) bool
    victim_survival: jnp.ndarray        # (num_victims,) float32 [0,1]
    victim_positions: jnp.ndarray       # (num_victims, 3) float32 — world coords
    victim_severity: jnp.ndarray        # (num_victims,) float32

    # drone bookkeeping
    drone_knowledge: jnp.ndarray        # (num_drones, num_victims) bool
    shared_known_victims: jnp.ndarray   # (num_victims,) bool
    drone_battery: jnp.ndarray          # (num_drones,) float32
    drone_payload: jnp.ndarray          # (num_drones,) int32

    # coverage
    scanned_cells: jnp.ndarray          # (coverage_res_y, coverage_res_x) bool

    # scene parameters (per-instance, vmapped over batch)
    terrain_height: jnp.ndarray         # (coverage_res_y, coverage_res_x) float32
    debris_mask: jnp.ndarray            # (coverage_res_y, coverage_res_x) bool
    wind_field: jnp.ndarray             # (coverage_res_y, coverage_res_x, 2) float32

    # episode bookkeeping
    time: jnp.ndarray                   # () int32
    done: jnp.ndarray                   # () bool

    # episode metrics (aggregated scalars)
    metrics: dict = field(default_factory=dict)


@register_dataclass
@dataclass
class Observation:
    """Per-drone observation — mirrors old avalanche_sim.types.Observation."""
    drone_features: jnp.ndarray   # (num_drones, 8)
    victim_features: jnp.ndarray  # (num_drones, num_victims, 5)
    team_features: jnp.ndarray    # (num_drones, 5)
    coverage_map: jnp.ndarray     # (num_drones, res_y, res_x)
    action_mask: jnp.ndarray      # (num_drones, 4)


@dataclass
class DronePhysicsState:
    """Transient per-step physics quantities extracted from MuJoCo data.

    Not a JAX pytree — used only in physics_env.py (CPU path).
    For MJX path, extract directly from mjx_data arrays.
    """
    positions: jnp.ndarray    # (num_drones, 3) world XYZ
    quats: jnp.ndarray        # (num_drones, 4) wxyz quaternion
    lin_vel: jnp.ndarray      # (num_drones, 3) world-frame linear velocity
    ang_vel: jnp.ndarray      # (num_drones, 3) body-frame angular velocity
    heading: jnp.ndarray      # (num_drones,) yaw angle in radians


@register_dataclass
@dataclass
class MJXBatchState:
    """Full batched environment state for the MJX backend.

    mjx_data is a mjx.Data pytree (batched via vmap).
    mission is a MissionState pytree (batched).
    """
    mission: MissionState
    # mjx_data stored as an opaque field — registered with JAX via mjx pytree
    # We cannot type-annotate it here without importing mjx at module level.
    mjx_data: object = field(default=None)
