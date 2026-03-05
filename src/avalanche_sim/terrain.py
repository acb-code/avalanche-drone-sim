from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import EnvConfig


def generate_scene(config: EnvConfig, key: jax.Array) -> tuple[jnp.ndarray, ...]:
    key_center, key_angle, key_spread, key_victims, key_wind = jax.random.split(key, 5)
    gx = jnp.linspace(0.0, config.map_size_x, config.coverage_resolution_x)
    gy = jnp.linspace(0.0, config.map_size_y, config.coverage_resolution_y)
    xx, yy = jnp.meshgrid(gx, gy)

    terrain_height = (
        config.terrain_slope * (config.map_size_y - yy)
        + config.ridge_amplitude * jnp.sin(xx * config.terrain_frequency_x)
        + 0.5 * config.ridge_amplitude * jnp.cos(yy * config.terrain_frequency_y)
    )

    center = jnp.array(
        [
            jax.random.uniform(key_center, (), minval=config.map_size_x * 0.35, maxval=config.map_size_x * 0.65),
            jax.random.uniform(key_angle, (), minval=config.map_size_y * 0.35, maxval=config.map_size_y * 0.75),
        ]
    )
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

    severity = jax.random.uniform(
        key_victims,
        (config.num_victims,),
        minval=0.45,
        maxval=1.0,
    )

    wind_base = jnp.stack(
        [
            jnp.sin(xx * 0.03 + 0.7) + 0.4 * jnp.cos(yy * 0.05),
            0.6 * jnp.cos(xx * 0.04) - jnp.sin(yy * 0.025 + 1.1),
        ],
        axis=-1,
    )
    wind_scale = jax.random.uniform(key_center, (), minval=0.3, maxval=1.0) if config.randomize_wind else 1.0
    wind_field = config.wind_strength * wind_scale * wind_base / 1.8
    return terrain_height, debris_mask.astype(jnp.bool_), unsafe_mask.astype(jnp.bool_), wind_field, victim_positions, severity


def terrain_height_at(config: EnvConfig, terrain_height: jnp.ndarray, xy: jnp.ndarray) -> jnp.ndarray:
    cx = jnp.clip(
        jnp.floor(xy[:, 0] / config.map_size_x * config.coverage_resolution_x).astype(jnp.int32),
        0,
        config.coverage_resolution_x - 1,
    )
    cy = jnp.clip(
        jnp.floor(xy[:, 1] / config.map_size_y * config.coverage_resolution_y).astype(jnp.int32),
        0,
        config.coverage_resolution_y - 1,
    )
    return terrain_height[cy, cx]