from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
from jax.tree_util import register_dataclass


@register_dataclass
@dataclass(frozen=True)
class EnvConfig:
    num_drones: int = 4
    num_victims: int = 6
    map_size_x: float = 240.0
    map_size_y: float = 160.0
    altitude_min: float = 8.0
    altitude_max: float = 70.0
    max_xy_speed: float = 9.0
    max_z_speed: float = 4.0
    max_yaw_rate: float = 1.2
    dt: float = 1.0
    sensor_range: float = 22.0
    sensor_fov_cos: float = -0.15
    sensor_altitude_scale: float = 32.0
    communication_range: float = 60.0
    delivery_range: float = 6.5
    collision_distance: float = 4.0
    near_collision_distance: float = 8.0
    base_battery: float = 320.0
    battery_burn_per_step: float = 0.6
    battery_burn_per_speed: float = 0.07
    payload_per_drone: int = 2
    horizon: int = 250
    coverage_resolution_x: int = 24
    coverage_resolution_y: int = 16
    terrain_frequency_x: float = 0.045
    terrain_frequency_y: float = 0.035
    terrain_slope: float = 0.14
    ridge_amplitude: float = 7.5
    debris_width: float = 18.0
    debris_length: float = 90.0
    detection_decay: float = 0.11
    rescan_bonus: float = 0.16
    severity_decay: float = 0.0025
    wind_strength: float = 0.4
    randomize_wind: bool = True
    reward_find: float = 10.0
    reward_confirm: float = 3.0
    reward_delivery: float = 22.0
    reward_coverage: float = 0.15
    reward_survival: float = 0.04
    penalty_collision: float = 14.0
    penalty_near_collision: float = 3.0
    penalty_wasted_energy: float = 0.018
    penalty_duplicate_delivery: float = 2.5
    penalty_timeout: float = 8.0


@register_dataclass
@dataclass
class Action:
    control: jnp.ndarray


@register_dataclass
@dataclass
class Observation:
    drone_features: jnp.ndarray
    victim_features: jnp.ndarray
    team_features: jnp.ndarray
    coverage_map: jnp.ndarray
    action_mask: jnp.ndarray


@register_dataclass
@dataclass
class EnvState:
    drone_positions: jnp.ndarray
    drone_heading: jnp.ndarray
    drone_battery: jnp.ndarray
    drone_payload: jnp.ndarray
    terrain_height: jnp.ndarray
    debris_mask: jnp.ndarray
    unsafe_mask: jnp.ndarray
    wind_field: jnp.ndarray
    victim_positions: jnp.ndarray
    victim_severity: jnp.ndarray
    victim_found: jnp.ndarray
    victim_confirmed: jnp.ndarray
    victim_aided: jnp.ndarray
    victim_survival: jnp.ndarray
    drone_knowledge: jnp.ndarray
    shared_known_victims: jnp.ndarray
    scanned_cells: jnp.ndarray
    time: jnp.ndarray
    done: jnp.ndarray
    metrics: dict = field(default_factory=dict)


@register_dataclass
@dataclass
class DynamicsOutput:
    positions: jnp.ndarray
    heading: jnp.ndarray
    battery: jnp.ndarray
    collisions: jnp.ndarray
    near_collisions: jnp.ndarray
    speed_norm: jnp.ndarray