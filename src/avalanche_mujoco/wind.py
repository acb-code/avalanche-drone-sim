"""Wind force application — works on both mujoco.MjData and mjx.Data.

CPU path:  apply_wind_cpu(data, model, cfg, mission, drone_body_ids)
MJX path:  apply_wind_mjx(mjx_data, cfg, mission, drone_body_ids)

Both sample the per-cell wind field at the current drone XY positions and
write an XY force (Z=0) into xfrc_applied for each drone body.
"""
from __future__ import annotations

import jax.numpy as jnp

from .config import AvalancheConfig
from .terrain_mesh import terrain_height_at    # reuse grid-lookup helper
from .types import MissionState


# ──────────────────────────────────────────────────────────────────────────────
# Shared pure-JAX wind sampling
# ──────────────────────────────────────────────────────────────────────────────

def _sample_wind(
    cfg: AvalancheConfig,
    mission: MissionState,
    drone_xy: jnp.ndarray,   # (num_drones, 2)
) -> jnp.ndarray:
    """Sample wind field at drone positions → (num_drones, 2) force vectors."""
    cx = jnp.clip(
        jnp.floor(drone_xy[:, 0] / cfg.map_size_x * cfg.coverage_resolution_x).astype(jnp.int32),
        0, cfg.coverage_resolution_x - 1,
    )
    cy = jnp.clip(
        jnp.floor(drone_xy[:, 1] / cfg.map_size_y * cfg.coverage_resolution_y).astype(jnp.int32),
        0, cfg.coverage_resolution_y - 1,
    )
    # wind_field: (res_y, res_x, 2)
    wind_xy = mission.wind_field[cy, cx]   # (num_drones, 2)
    # Scale from m/s wind velocity to force (F = drag * vel, simple linear model)
    # Using an approximate drag coefficient; drone mass ~1.7 kg, Cd~0.5.
    drag_coeff = 0.85  # N/(m/s)
    return wind_xy * drag_coeff   # (num_drones, 2)


# ──────────────────────────────────────────────────────────────────────────────
# CPU path — mutates mujoco.MjData in-place
# ──────────────────────────────────────────────────────────────────────────────

def apply_wind_cpu(
    data,                          # mujoco.MjData
    cfg: AvalancheConfig,
    mission: MissionState,
    drone_body_ids: list[int],
) -> None:
    """Write wind forces into data.xfrc_applied for each drone body."""
    import numpy as np
    drone_xy = jnp.array(data.xpos[drone_body_ids, :2])
    forces = _sample_wind(cfg, mission, drone_xy)   # (num_drones, 2)
    forces_np = np.array(forces)
    for i, bid in enumerate(drone_body_ids):
        data.xfrc_applied[bid, 0] = forces_np[i, 0]   # Fx
        data.xfrc_applied[bid, 1] = forces_np[i, 1]   # Fy
        data.xfrc_applied[bid, 2] = 0.0                # Fz (no vertical wind)


# ──────────────────────────────────────────────────────────────────────────────
# MJX path — returns new mjx.Data with updated xfrc_applied (immutable)
# ──────────────────────────────────────────────────────────────────────────────

def apply_wind_mjx(
    mjx_data,                      # mjx.Data
    cfg: AvalancheConfig,
    mission: MissionState,
    drone_body_ids: jnp.ndarray,   # (num_drones,) int32
) -> object:
    """Return updated mjx_data with wind forces in xfrc_applied.

    mjx.Data is an immutable pytree; we use .replace() to update.
    xfrc_applied shape: (num_bodies, 6)  [Fx, Fy, Fz, tx, ty, tz]
    """
    drone_xy = mjx_data.xpos[drone_body_ids, :2]   # (num_drones, 2)
    forces = _sample_wind(cfg, mission, drone_xy)   # (num_drones, 2)

    # Build full xfrc_applied: only update drone rows, zero elsewhere
    xfrc = mjx_data.xfrc_applied   # (num_bodies, 6)

    # Zero out previous wind forces for drone bodies
    xfrc = xfrc.at[drone_body_ids, 0].set(forces[:, 0])
    xfrc = xfrc.at[drone_body_ids, 1].set(forces[:, 1])
    xfrc = xfrc.at[drone_body_ids, 2].set(0.0)

    return mjx_data.replace(xfrc_applied=xfrc)
