from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .types import DynamicsOutput, EnvConfig


class DynamicsBackend:
    def step(
        self,
        config: EnvConfig,
        positions: jnp.ndarray,
        heading: jnp.ndarray,
        battery: jnp.ndarray,
        actions: jnp.ndarray,
        wind_field: jnp.ndarray,
    ) -> DynamicsOutput:
        raise NotImplementedError


@dataclass
class Kinematic3DOFBackend(DynamicsBackend):
    def step(
        self,
        config: EnvConfig,
        positions: jnp.ndarray,
        heading: jnp.ndarray,
        battery: jnp.ndarray,
        actions: jnp.ndarray,
        wind_field: jnp.ndarray,
    ) -> DynamicsOutput:
        clipped = jnp.clip(actions, -1.0, 1.0)
        planar_cmd = clipped[:, :2] * config.max_xy_speed
        vertical_cmd = clipped[:, 2] * config.max_z_speed
        yaw_rate = clipped[:, 3] * config.max_yaw_rate

        cell_x = jnp.clip(
            jnp.floor(
                positions[:, 0] / config.map_size_x * config.coverage_resolution_x
            ).astype(jnp.int32),
            0,
            config.coverage_resolution_x - 1,
        )
        cell_y = jnp.clip(
            jnp.floor(
                positions[:, 1] / config.map_size_y * config.coverage_resolution_y
            ).astype(jnp.int32),
            0,
            config.coverage_resolution_y - 1,
        )
        local_wind = wind_field[cell_y, cell_x]

        delta_xy = (planar_cmd + local_wind) * config.dt
        delta_z = vertical_cmd * config.dt
        next_pos = positions + jnp.concatenate([delta_xy, delta_z[:, None]], axis=-1)
        next_pos = next_pos.at[:, 0].set(jnp.clip(next_pos[:, 0], 0.0, config.map_size_x))
        next_pos = next_pos.at[:, 1].set(jnp.clip(next_pos[:, 1], 0.0, config.map_size_y))
        next_pos = next_pos.at[:, 2].set(
            jnp.clip(next_pos[:, 2], config.altitude_min, config.altitude_max)
        )
        next_heading = jnp.arctan2(
            jnp.sin(heading + yaw_rate * config.dt),
            jnp.cos(heading + yaw_rate * config.dt),
        )

        disp = next_pos[:, None, :] - next_pos[None, :, :]
        pair_dist = jnp.linalg.norm(disp, axis=-1) + jnp.eye(config.num_drones) * 1.0e6
        min_dist = jnp.min(pair_dist, axis=1)
        collisions = min_dist < config.collision_distance
        near_collisions = jnp.logical_and(
            min_dist >= config.collision_distance,
            min_dist < config.near_collision_distance,
        )

        speed_norm = (
            jnp.linalg.norm(planar_cmd, axis=-1) / config.max_xy_speed
            + jnp.abs(vertical_cmd) / config.max_z_speed
            + jnp.abs(yaw_rate) / config.max_yaw_rate
        ) / 3.0
        battery_delta = config.battery_burn_per_step + config.battery_burn_per_speed * speed_norm
        next_battery = jnp.maximum(0.0, battery - battery_delta)

        return DynamicsOutput(
            positions=next_pos,
            heading=next_heading,
            battery=next_battery,
            collisions=collisions,
            near_collisions=near_collisions,
            speed_norm=speed_norm,
        )