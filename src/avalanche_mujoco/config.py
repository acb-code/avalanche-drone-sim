"""AvalancheConfig — central configuration dataclass for the MuJoCo-backed sim.

Ports EnvConfig from avalanche_sim.types with physical-time parameters added.
All fields are frozen so the dataclass is safe to use as a JAX static argument.
"""
from __future__ import annotations

from dataclasses import dataclass

from jax.tree_util import register_dataclass


@register_dataclass
@dataclass(frozen=True)
class AvalancheConfig:
    # ── Fleet ──────────────────────────────────────────────────────────────
    num_drones: int = 4
    num_victims: int = 6

    # ── Map ────────────────────────────────────────────────────────────────
    map_size_x: float = 240.0       # metres
    map_size_y: float = 160.0       # metres
    altitude_min: float = 8.0       # AGL metres above terrain
    altitude_max: float = 70.0      # metres above sea level (approx)

    # ── Action interface (velocity commands, unchanged from kinematic sim) ─
    max_xy_speed: float = 9.0       # m/s
    max_z_speed: float = 4.0        # m/s
    max_yaw_rate: float = 1.2       # rad/s

    # ── Physics timing ─────────────────────────────────────────────────────
    sim_dt: float = 0.005           # MuJoCo internal timestep (s)
    control_dt: float = 0.05        # policy step duration (s)  → 10 substeps
    horizon: int = 250              # max policy steps per episode

    # ── Drone physics ──────────────────────────────────────────────────────
    drone_mass: float = 1.7         # kg
    rotor_arm: float = 0.175        # m (centre to rotor)
    k_drag: float = 0.016           # Nm/N yaw drag coefficient
    max_thrust_per_rotor: float = 10.0  # N

    # ── PID controller gains ───────────────────────────────────────────────
    pid_kp_vel_xy: float = 0.40
    pid_kp_vel_z: float = 0.55
    pid_kp_att: float = 9.0
    pid_kd_att: float = 2.5
    pid_kp_yaw: float = 3.5
    pid_kd_yaw: float = 0.8

    # ── Mission / sensing ──────────────────────────────────────────────────
    sensor_range: float = 22.0
    sensor_fov_cos: float = -0.15
    sensor_altitude_scale: float = 32.0
    communication_range: float = 60.0
    delivery_range: float = 6.5
    collision_distance: float = 4.0
    near_collision_distance: float = 8.0

    # ── Battery ────────────────────────────────────────────────────────────
    base_battery: float = 320.0
    battery_burn_per_step: float = 0.6
    battery_burn_per_speed: float = 0.07
    payload_per_drone: int = 2

    # ── Coverage grid ──────────────────────────────────────────────────────
    coverage_resolution_x: int = 24
    coverage_resolution_y: int = 16

    # ── Terrain generation ─────────────────────────────────────────────────
    terrain_frequency_x: float = 0.045
    terrain_frequency_y: float = 0.035
    terrain_slope: float = 0.14
    ridge_amplitude: float = 7.5
    debris_width: float = 18.0
    debris_length: float = 90.0

    # ── Victim dynamics ────────────────────────────────────────────────────
    detection_decay: float = 0.11
    rescan_bonus: float = 0.16
    severity_decay: float = 0.0025

    # ── Wind ───────────────────────────────────────────────────────────────
    wind_strength: float = 0.4
    randomize_wind: bool = True

    # ── Rewards / penalties ────────────────────────────────────────────────
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

    # ── Convenience ────────────────────────────────────────────────────────
    @property
    def substeps(self) -> int:
        """Number of MuJoCo sim steps per policy step."""
        return max(1, round(self.control_dt / self.sim_dt))
