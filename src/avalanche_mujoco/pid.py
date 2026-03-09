"""Velocity-command → rotor-thrust PID controller.

Pure JAX — compatible with jit and vmap.

Action interface: [-1, 1]^4  →  [vx, vy, vz, yaw_rate]  (world frame)
Output: rotor thrust array  (num_drones, 4)  in Newtons  [0, max_thrust]
        order: [FR, FL, RR, RL]

Cascade:
  1. Velocity error (world frame, Z world; XY rotated to body frame) → desired attitude
  2. Attitude error + angular rate → roll/pitch/yaw torques
  3. Torques + total thrust → 4 rotor thrusts via inversion of mixer matrix M
"""
from __future__ import annotations

import jax.numpy as jnp

from .config import AvalancheConfig

# ──────────────────────────────────────────────────────────────────────────────
# Quaternion utilities
# ──────────────────────────────────────────────────────────────────────────────

def _quat_to_rot(q: jnp.ndarray) -> jnp.ndarray:
    """Rotation matrix from unit quaternion [w, x, y, z] → (3, 3)."""
    w, x, y, z = q
    return jnp.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [  2*(x*y + z*w), 1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [  2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _quat_to_euler_zyx(q: jnp.ndarray) -> jnp.ndarray:
    """ZYX Euler angles (roll, pitch, yaw) from quaternion [w, x, y, z]."""
    w, x, y, z = q
    # roll (φ)
    sinr_cosp = 2.0 * (w*x + y*z)
    cosr_cosp = 1.0 - 2.0 * (x*x + y*y)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)
    # pitch (θ)
    sinp = 2.0 * (w*y - z*x)
    sinp = jnp.clip(sinp, -1.0, 1.0)
    pitch = jnp.arcsin(sinp)
    # yaw (ψ)
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)
    return jnp.array([roll, pitch, yaw])


# ──────────────────────────────────────────────────────────────────────────────
# Mixer matrix  (M maps [Fz, τx, τy, τz] → [T_FR, T_FL, T_RR, T_RL])
# ──────────────────────────────────────────────────────────────────────────────
# Rotor positions (arm = 0.175 m):
#   FR (+x, -y), FL (+x, +y), RR (-x, -y), RL (-x, +y)
# Torques from forces:
#   τx (roll)  = arm * (-T_FR + T_FL - T_RR + T_RL)
#   τy (pitch) = arm * (-T_FR - T_FL + T_RR + T_RL)
#   τz (yaw)   = k_drag * (-T_FR + T_FL + T_RR - T_RL)

def _build_mixer(arm: float, k_drag: float) -> jnp.ndarray:
    """Build 4×4 mixer matrix mapping wrench to rotor thrusts."""
    import numpy as np
    M = np.array([
        [ 1.0,      1.0,      1.0,      1.0    ],   # Fz
        [-arm,      arm,     -arm,      arm    ],   # τx (roll)
        [-arm,     -arm,      arm,      arm    ],   # τy (pitch)
        [-k_drag,   k_drag,   k_drag,  -k_drag],   # τz (yaw)
    ])
    return jnp.array(np.linalg.inv(M))


# ──────────────────────────────────────────────────────────────────────────────
# Per-drone thrust computation
# ──────────────────────────────────────────────────────────────────────────────

def _compute_single_drone_thrusts(
    cfg: AvalancheConfig,
    M_inv: jnp.ndarray,
    vel_cmd_world: jnp.ndarray,   # (3,) [vx, vy, vz] in world frame
    yaw_rate_cmd: float,          # scalar, rad/s
    pos: jnp.ndarray,             # (3,) world position (unused, future use)
    quat: jnp.ndarray,            # (4,) [w, x, y, z]
    vel_world: jnp.ndarray,       # (3,) current CoM velocity in world frame
    ang_vel_body: jnp.ndarray,    # (3,) current angular velocity in body frame
) -> jnp.ndarray:
    """Compute 4 rotor thrusts for a single drone. Pure-JAX, safe to vmap."""
    g = 9.81
    R = _quat_to_rot(quat)        # world ← body rotation
    euler = _quat_to_euler_zyx(quat)
    roll_cur, pitch_cur, _yaw_cur = euler[0], euler[1], euler[2]

    # ── Velocity control (world frame) ──────────────────────────────────────
    err_v = vel_cmd_world - vel_world                  # (3,)
    err_v_body = R.T @ err_v                           # rotate to body frame

    # Desired attitude from lateral velocity errors
    # body-X forward → pitch down (negative) for positive body-X error
    pitch_des = jnp.clip(-cfg.pid_kp_vel_xy * err_v_body[0], -0.45, 0.45)   # rad
    roll_des  = jnp.clip( cfg.pid_kp_vel_xy * err_v_body[1], -0.45, 0.45)   # rad

    # ── Attitude control ────────────────────────────────────────────────────
    err_roll  = roll_des  - roll_cur
    err_pitch = pitch_des - pitch_cur
    tau_x = cfg.pid_kp_att * err_roll  - cfg.pid_kd_att * ang_vel_body[0]
    tau_y = cfg.pid_kp_att * err_pitch - cfg.pid_kd_att * ang_vel_body[1]

    # ── Yaw rate control ────────────────────────────────────────────────────
    err_yaw_rate = yaw_rate_cmd - ang_vel_body[2]
    tau_z = cfg.pid_kp_yaw * err_yaw_rate - cfg.pid_kd_yaw * ang_vel_body[2]

    # ── Throttle (Z velocity in world frame) ────────────────────────────────
    hover_thrust = cfg.drone_mass * g
    F_z = hover_thrust + cfg.pid_kp_vel_z * err_v[2]      # world Z error
    F_z = jnp.clip(F_z, 0.0, 4.0 * cfg.max_thrust_per_rotor)

    # ── Mixer inverse ───────────────────────────────────────────────────────
    wrench = jnp.array([F_z, tau_x, tau_y, tau_z])
    thrusts = M_inv @ wrench
    return jnp.clip(thrusts, 0.0, cfg.max_thrust_per_rotor)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def compute_rotor_thrusts(
    cfg: AvalancheConfig,
    actions: jnp.ndarray,          # (num_drones, 4) in [-1, 1]
    quats: jnp.ndarray,            # (num_drones, 4) [w,x,y,z]
    lin_vel_world: jnp.ndarray,    # (num_drones, 3)
    ang_vel_body: jnp.ndarray,     # (num_drones, 3)
    positions: jnp.ndarray,        # (num_drones, 3) — reserved for future use
) -> jnp.ndarray:
    """Compute rotor thrusts for all drones.

    Returns
    -------
    thrusts : (num_drones, 4)  [T_FR, T_FL, T_RR, T_RL]  in Newtons
    """
    M_inv = _build_mixer(cfg.rotor_arm, cfg.k_drag)

    # Scale actions to physical commands
    vx = actions[:, 0] * cfg.max_xy_speed
    vy = actions[:, 1] * cfg.max_xy_speed
    vz = actions[:, 2] * cfg.max_z_speed
    yaw_rate = actions[:, 3] * cfg.max_yaw_rate
    vel_cmds = jnp.stack([vx, vy, vz], axis=-1)   # (num_drones, 3)

    # vmap over drones
    import jax
    thrusts = jax.vmap(
        lambda vc, yr, pos, q, lv, av: _compute_single_drone_thrusts(
            cfg, M_inv, vc, yr, pos, q, lv, av
        )
    )(vel_cmds, yaw_rate, positions, quats, lin_vel_world, ang_vel_body)
    return thrusts   # (num_drones, 4)
