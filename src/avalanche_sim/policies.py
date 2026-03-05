from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import Action, EnvConfig, EnvState


def random_policy(key: jax.Array, state: EnvState, config: EnvConfig) -> Action:
    del config
    return Action(
        control=jax.random.uniform(key, (state.drone_positions.shape[0], 4), minval=-1.0, maxval=1.0)
    )


def lawnmower_policy(state: EnvState, config: EnvConfig) -> Action:
    num_drones = state.drone_positions.shape[0]
    lane_targets_x = jnp.linspace(config.map_size_x * 0.1, config.map_size_x * 0.9, num_drones)
    lane_targets_y = jnp.where(
        (state.time // 20) % 2 == 0,
        config.map_size_y * 0.2,
        config.map_size_y * 0.85,
    )
    target_xy = jnp.stack([lane_targets_x, jnp.full((num_drones,), lane_targets_y)], axis=-1)

    known_idx = jnp.argmax(state.shared_known_victims.astype(jnp.int32))
    has_known = jnp.any(state.shared_known_victims)
    victim_targets = jnp.broadcast_to(state.victim_positions[known_idx, :2], target_xy.shape)
    target_xy = jnp.where(has_known, victim_targets, target_xy)

    delta = target_xy - state.drone_positions[:, :2]
    planar = delta / jnp.maximum(jnp.linalg.norm(delta, axis=-1, keepdims=True), 1.0)
    desired_alt = jnp.where(has_known, config.altitude_min + 8.0, config.altitude_min + 18.0)
    vertical = jnp.clip((desired_alt - state.drone_positions[:, 2]) / config.max_z_speed, -1.0, 1.0)
    yaw_target = jnp.arctan2(delta[:, 1], delta[:, 0])
    yaw_err = jnp.arctan2(jnp.sin(yaw_target - state.drone_heading), jnp.cos(yaw_target - state.drone_heading))
    yaw_rate = jnp.clip(yaw_err / config.max_yaw_rate, -1.0, 1.0)
    control = jnp.concatenate([planar, vertical[:, None], yaw_rate[:, None]], axis=-1)
    return Action(control=jnp.clip(control, -1.0, 1.0))


def spiral_search_policy(state: EnvState, config: EnvConfig) -> Action:
    center = jnp.array([config.map_size_x * 0.5, config.map_size_y * 0.5])
    rel = state.drone_positions[:, :2] - center
    tangent = jnp.stack([-rel[:, 1], rel[:, 0]], axis=-1)
    tangent = tangent / jnp.maximum(jnp.linalg.norm(tangent, axis=-1, keepdims=True), 1.0)
    radial = -rel / jnp.maximum(jnp.linalg.norm(rel, axis=-1, keepdims=True), 1.0)
    blend = 0.75 * tangent + 0.25 * radial
    vertical = jnp.clip((config.altitude_min + 20.0 - state.drone_positions[:, 2]) / config.max_z_speed, -1.0, 1.0)
    yaw_target = jnp.arctan2(blend[:, 1], blend[:, 0])
    yaw_err = jnp.arctan2(jnp.sin(yaw_target - state.drone_heading), jnp.cos(yaw_target - state.drone_heading))
    yaw_rate = jnp.clip(yaw_err / config.max_yaw_rate, -1.0, 1.0)
    control = jnp.concatenate([blend, vertical[:, None], yaw_rate[:, None]], axis=-1)
    return Action(control=jnp.clip(control, -1.0, 1.0))