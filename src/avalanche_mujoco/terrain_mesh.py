"""Terrain generation — pure JAX, shared by CPU and MJX backends.

CPU backend: can also generate an hfield PNG for MuJoCo native terrain.
MJX backend: stores terrain as a JAX array; enforces altitude constraints
             virtually after each mjx.step (no hfield collision support in MJX).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import AvalancheConfig


# ──────────────────────────────────────────────────────────────────────────────
# Scene generation (terrain + debris + wind + victims)
# ──────────────────────────────────────────────────────────────────────────────

def generate_scene(
    config: AvalancheConfig, key: jax.Array
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate randomised scene parameters.

    Returns
    -------
    terrain_height : (res_y, res_x)
    debris_mask    : (res_y, res_x) bool
    unsafe_mask    : (res_y, res_x) bool
    wind_field     : (res_y, res_x, 2)
    victim_positions : (num_victims, 3)
    victim_severity  : (num_victims,)
    """
    key_center, key_angle, key_spread, key_victims, key_wind = jax.random.split(key, 5)

    gx = jnp.linspace(0.0, config.map_size_x, config.coverage_resolution_x)
    gy = jnp.linspace(0.0, config.map_size_y, config.coverage_resolution_y)
    xx, yy = jnp.meshgrid(gx, gy)

    terrain_height = (
        config.terrain_slope * (config.map_size_y - yy)
        + config.ridge_amplitude * jnp.sin(xx * config.terrain_frequency_x)
        + 0.5 * config.ridge_amplitude * jnp.cos(yy * config.terrain_frequency_y)
    )

    # ── Debris zone (avalanche deposit) ────────────────────────────────────
    center = jnp.array([
        jax.random.uniform(key_center, (), minval=config.map_size_x * 0.35, maxval=config.map_size_x * 0.65),
        jax.random.uniform(key_angle,  (), minval=config.map_size_y * 0.35, maxval=config.map_size_y * 0.75),
    ])
    angle = jax.random.uniform(key_spread, (), minval=-0.45, maxval=0.45)
    direction = jnp.array([jnp.cos(angle), -jnp.sin(angle)])
    normal = jnp.array([-direction[1], direction[0]])
    rel = jnp.stack([xx - center[0], yy - center[1]], axis=-1)
    longitudinal = rel @ direction
    lateral = rel @ normal
    spread = jax.random.uniform(key_wind, (), minval=0.85, maxval=1.25)
    debris_mask = (
        (jnp.abs(longitudinal) < config.debris_length * spread * 0.5)
        & (jnp.abs(lateral) < config.debris_width * spread)
        & (yy > config.map_size_y * 0.18)
    )

    unsafe_mask = (terrain_height > jnp.percentile(terrain_height, 84.0)) | (yy < config.map_size_y * 0.08)

    # ── Victims (clustered in debris zone) ─────────────────────────────────
    victim_keys = jax.random.split(key_victims, config.num_victims)
    cluster_offsets = jax.vmap(
        lambda k: jax.random.normal(k, (2,))
        * jnp.array([config.debris_length * 0.18, config.debris_width * 0.35])
    )(victim_keys)
    victim_xy = center + cluster_offsets
    victim_xy = victim_xy.at[:, 0].set(jnp.clip(victim_xy[:, 0], 0.0, config.map_size_x))
    victim_xy = victim_xy.at[:, 1].set(jnp.clip(victim_xy[:, 1], 0.0, config.map_size_y))
    victim_z = terrain_height_at(config, terrain_height, victim_xy) + 1.0
    victim_positions = jnp.concatenate([victim_xy, victim_z[:, None]], axis=-1)

    severity = jax.random.uniform(key_victims, (config.num_victims,), minval=0.45, maxval=1.0)

    # ── Wind field ──────────────────────────────────────────────────────────
    wind_base = jnp.stack([
        jnp.sin(xx * 0.03 + 0.7) + 0.4 * jnp.cos(yy * 0.05),
        0.6 * jnp.cos(xx * 0.04) - jnp.sin(yy * 0.025 + 1.1),
    ], axis=-1)
    wind_scale = jax.random.uniform(key_center, (), minval=0.3, maxval=1.0) if config.randomize_wind else 1.0
    wind_field = config.wind_strength * wind_scale * wind_base / 1.8

    return (
        terrain_height,
        debris_mask.astype(jnp.bool_),
        unsafe_mask.astype(jnp.bool_),
        wind_field,
        victim_positions,
        severity,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Terrain query helpers
# ──────────────────────────────────────────────────────────────────────────────

def terrain_height_at(
    config: AvalancheConfig,
    terrain_height: jnp.ndarray,
    xy: jnp.ndarray,
) -> jnp.ndarray:
    """Bilinear-nearest lookup of terrain height at world-frame XY positions.

    Parameters
    ----------
    xy : (N, 2) world coordinates
    Returns
    -------
    heights : (N,)
    """
    cx = jnp.clip(
        jnp.floor(xy[:, 0] / config.map_size_x * config.coverage_resolution_x).astype(jnp.int32),
        0, config.coverage_resolution_x - 1,
    )
    cy = jnp.clip(
        jnp.floor(xy[:, 1] / config.map_size_y * config.coverage_resolution_y).astype(jnp.int32),
        0, config.coverage_resolution_y - 1,
    )
    return terrain_height[cy, cx]


def constrain_by_terrain(
    config: AvalancheConfig,
    terrain_height: jnp.ndarray,
    start_positions: jnp.ndarray,
    candidate_positions: jnp.ndarray,
) -> jnp.ndarray:
    """Clip drone motion that would enter terrain (virtual terrain enforcement).

    Identical logic to the old kinematic sim — safe to jit/vmap.
    """
    start_xy = start_positions[:, :2]
    target_xy = candidate_positions[:, :2]
    target_z = candidate_positions[:, 2]

    sample_fracs = jnp.linspace(0.0, 1.0, 9, dtype=jnp.float32)
    path_xy = start_xy[:, None, :] + (target_xy - start_xy)[:, None, :] * sample_fracs[None, :, None]
    flat_path_xy = path_xy.reshape((-1, 2))
    path_terrain = terrain_height_at(config, terrain_height, flat_path_xy).reshape(
        (start_positions.shape[0], sample_fracs.shape[0])
    )
    min_clearance_z = path_terrain + config.altitude_min
    safe_prefix = jnp.cumprod(
        (target_z[:, None] >= min_clearance_z).astype(jnp.int32), axis=1
    ).astype(jnp.float32)
    allowed_frac = jnp.max(sample_fracs[None, :] * safe_prefix, axis=1)

    constrained_xy = start_xy + (target_xy - start_xy) * allowed_frac[:, None]
    terrain_under = terrain_height_at(config, terrain_height, constrained_xy)
    constrained_z = jnp.maximum(target_z, terrain_under + config.altitude_min)
    constrained_z = jnp.clip(constrained_z, terrain_under + config.altitude_min, config.altitude_max)
    return jnp.concatenate([constrained_xy, constrained_z[:, None]], axis=-1)


# ──────────────────────────────────────────────────────────────────────────────
# hfield export (CPU mode only — not used by MJX)
# ──────────────────────────────────────────────────────────────────────────────

def terrain_to_hfield_bytes(terrain_height: jnp.ndarray) -> bytes:
    """Convert terrain array to MuJoCo hfield binary (uint16, row-major).

    The hfield asset expects values in [0, 1] scaled to uint16.
    MuJoCo scales by the hfield size attribute at runtime.
    """
    import numpy as np
    h = np.array(terrain_height, dtype=np.float32)
    h_min, h_max = h.min(), h.max()
    if h_max > h_min:
        h_norm = (h - h_min) / (h_max - h_min)
    else:
        h_norm = np.zeros_like(h)
    h_uint16 = (h_norm * 65535).astype(np.uint16)
    return h_uint16.tobytes()
