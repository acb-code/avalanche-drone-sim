"""MJX batched environment — Use Case 2 (GPU-parallel simulation).

AvalancheMJXEnv runs thousands of envs in parallel via jax.vmap over MJXBatchState.

Key design:
- mjx.Model is shared (not batched) — created once from the flat-plane scene.
- mjx.Data + MissionState are batched via vmap.
- Terrain is virtual (JAX array in MissionState); altitude constraints applied
  after mjx.step by clipping and zeroing downward velocity.
- Wind applied to xfrc_applied before each step.

Usage (functional / vmap)
--------------------------
    env = AvalancheMJXEnv(config)
    keys = jax.random.split(jax.random.PRNGKey(0), N)
    states = jax.vmap(env.reset)(keys)
    actions = jnp.zeros((N, config.num_drones, 4))
    next_states, rewards, info = jax.vmap(env.step)(states, actions)
"""
from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .config import AvalancheConfig
from .mission import mission_step
from .obs import build_observation
from .pid import compute_rotor_thrusts
from .scene import build_scene
from .terrain_mesh import constrain_by_terrain, generate_scene
from .types import MJXBatchState, MissionState, Observation
from .wind import apply_wind_mjx

try:
    import mujoco
    import mujoco.mjx as mjx
    _MJX_AVAILABLE = True
except ImportError:
    _MJX_AVAILABLE = False


class AvalancheMJXEnv:
    """MJX-batched SAR environment.

    After construction, call reset / step via jax.vmap:

        states = jax.vmap(env.reset)(keys)          # keys: (N,)
        states, rewards, info = jax.vmap(env.step)(states, actions)
    """

    def __init__(self, config: AvalancheConfig):
        if not _MJX_AVAILABLE:
            raise ImportError(
                "mujoco-mjx is required. Install with: pip install mujoco-mjx>=3.1"
            )
        self.config = config

        # Build scene with flat terrain (no hfield) for MJX
        xml, self._body_names, self._actuator_names = build_scene(config, backend="mjx")
        mj_model: mujoco.MjModel = mujoco.MjModel.from_xml_string(xml)
        self._mjx_model = mjx.put_model(mj_model)

        # Cache body IDs
        self._drone_body_ids = np.array([
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self._body_names
        ], dtype=np.int32)

        # freejoint qpos addresses (7 per drone)
        self._drone_qpos_addrs = np.array([
            int(mj_model.jnt_qposadr[
                mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"drone{i}/root")
            ])
            for i in range(config.num_drones)
        ], dtype=np.int32)

        # freejoint qvel DOF addresses (6 per drone)
        self._drone_qvel_addrs = np.array([
            int(mj_model.jnt_dofadr[
                mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"drone{i}/root")
            ])
            for i in range(config.num_drones)
        ], dtype=np.int32)

        # Number of actuators per drone = 4
        self._num_act_per_drone = 4

        # Build a reference MjData to initialise
        self._ref_mj_data = mujoco.MjData(mj_model)
        mujoco.mj_resetData(mj_model, self._ref_mj_data)

        # Precompute spawn positions (numpy)
        from .drone import default_spawn_positions
        self._spawn_pos = default_spawn_positions(
            config.num_drones, config.map_size_x, config.map_size_y
        )

    # ──────────────────────────────────────────────────────────────────────
    # Reset (called per-env via vmap)
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, key: jax.Array) -> MJXBatchState:
        """Reset one environment. Call via jax.vmap(env.reset)(keys)."""
        terrain_height, debris_mask, _unsafe, wind_field, victim_positions, severity = (
            generate_scene(self.config, key)
        )

        # Initialise MJX data from reference + set drone qpos
        mjx_data = mjx.put_data(self._mjx_model.model if hasattr(self._mjx_model, 'model') else
                                 _get_mj_model_from_mjx(self._mjx_model),
                                 self._ref_mj_data)

        # Set drone positions in qpos
        for i, addr in enumerate(self._drone_qpos_addrs):
            mjx_data = mjx_data.replace(
                qpos=mjx_data.qpos.at[addr:addr+3].set(jnp.array(self._spawn_pos[i]))
            )
            # Identity quaternion
            mjx_data = mjx_data.replace(
                qpos=mjx_data.qpos.at[addr+3].set(1.0)
            )
            mjx_data = mjx_data.replace(
                qpos=mjx_data.qpos.at[addr+4:addr+7].set(jnp.zeros(3))
            )

        mjx_data = mjx.forward(self._mjx_model, mjx_data)

        mission = MissionState(
            victim_found=jnp.zeros(self.config.num_victims, dtype=jnp.bool_),
            victim_confirmed=jnp.zeros(self.config.num_victims, dtype=jnp.bool_),
            victim_aided=jnp.zeros(self.config.num_victims, dtype=jnp.bool_),
            victim_survival=jnp.ones(self.config.num_victims),
            victim_positions=victim_positions,
            victim_severity=severity,
            drone_knowledge=jnp.zeros((self.config.num_drones, self.config.num_victims), dtype=jnp.bool_),
            shared_known_victims=jnp.zeros(self.config.num_victims, dtype=jnp.bool_),
            drone_battery=jnp.full(self.config.num_drones, self.config.base_battery),
            drone_payload=jnp.full(self.config.num_drones, self.config.payload_per_drone, dtype=jnp.int32),
            scanned_cells=jnp.zeros(
                (self.config.coverage_resolution_y, self.config.coverage_resolution_x), dtype=jnp.bool_
            ),
            terrain_height=terrain_height,
            debris_mask=debris_mask,
            wind_field=wind_field,
            time=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False),
            metrics={
                "coverage": jnp.array(0.0),
                "find_events": jnp.array(0),
                "confirm_events": jnp.array(0),
                "delivery_events": jnp.array(0),
            },
        )
        return MJXBatchState(mission=mission, mjx_data=mjx_data)

    # ──────────────────────────────────────────────────────────────────────
    # Step (called per-env via vmap)
    # ──────────────────────────────────────────────────────────────────────

    def step(
        self,
        state: MJXBatchState,
        actions: jnp.ndarray,   # (num_drones, 4)
        key: jax.Array,
    ) -> tuple[MJXBatchState, jnp.ndarray, dict]:
        """Advance one env by one control step. Call via jax.vmap."""
        mjx_data = state.mjx_data
        mission = state.mission
        qpos_addrs = jnp.array(self._drone_qpos_addrs)
        qvel_addrs = jnp.array(self._drone_qvel_addrs)

        # ── Extract drone state from mjx.Data ─────────────────────────────
        positions = mjx_data.qpos[qpos_addrs[:, None] + jnp.arange(3)]     # (n, 3)
        quats = mjx_data.qpos[qpos_addrs[:, None] + jnp.arange(3, 7)]      # (n, 4) w,x,y,z
        lin_vel = mjx_data.qvel[qvel_addrs[:, None] + jnp.arange(3)]       # (n, 3)
        ang_vel = mjx_data.qvel[qvel_addrs[:, None] + jnp.arange(3, 6)]    # (n, 3)
        heading = jax.vmap(_quat_to_yaw_jax)(quats)                         # (n,)

        old_positions = positions

        # ── PID → rotor thrusts ───────────────────────────────────────────
        thrusts = compute_rotor_thrusts(
            self.config, actions, quats, lin_vel, ang_vel, positions
        )   # (n_drones, 4)

        # Write thrusts to ctrl
        ctrl = jnp.zeros_like(mjx_data.ctrl)
        for d in range(self.config.num_drones):
            base = d * self._num_act_per_drone
            ctrl = ctrl.at[base:base+4].set(thrusts[d])
        mjx_data = mjx_data.replace(ctrl=ctrl)

        # ── Apply wind ────────────────────────────────────────────────────
        mjx_data = apply_wind_mjx(
            mjx_data, self.config, mission,
            jnp.array(self._drone_body_ids)
        )

        # ── Physics step (multiple substeps) ─────────────────────────────
        def substep(carry, _):
            return mjx.step(self._mjx_model, carry), None

        mjx_data, _ = jax.lax.scan(substep, mjx_data, None, length=self.config.substeps)

        # ── Extract new positions ─────────────────────────────────────────
        new_positions = mjx_data.qpos[qpos_addrs[:, None] + jnp.arange(3)]
        new_quats = mjx_data.qpos[qpos_addrs[:, None] + jnp.arange(3, 7)]
        new_lin_vel = mjx_data.qvel[qvel_addrs[:, None] + jnp.arange(3)]
        new_heading = jax.vmap(_quat_to_yaw_jax)(new_quats)

        # ── Virtual terrain constraint ────────────────────────────────────
        constrained = constrain_by_terrain(
            self.config, mission.terrain_height, old_positions, new_positions
        )
        # Apply constrained positions back into qpos (immutable update)
        for i, addr in enumerate(self._drone_qpos_addrs):
            mjx_data = mjx_data.replace(
                qpos=mjx_data.qpos.at[addr:addr+3].set(constrained[i])
            )

        # ── Speed norm for battery / reward ──────────────────────────────
        speed_norm = (
            jnp.linalg.norm(new_lin_vel[:, :2], axis=-1) / self.config.max_xy_speed
            + jnp.abs(new_lin_vel[:, 2]) / self.config.max_z_speed
        ).clip(0.0, 1.0)

        # ── Mission logic ─────────────────────────────────────────────────
        next_mission, rewards, info = mission_step(
            key, self.config, mission,
            constrained,
            new_heading,
            speed_norm,
            jnp.zeros(self.config.num_drones, dtype=jnp.bool_),
            jnp.zeros(self.config.num_drones, dtype=jnp.bool_),
        )

        next_state = MJXBatchState(mission=next_mission, mjx_data=mjx_data)
        return next_state, rewards, info

    def get_obs(self, state: MJXBatchState) -> Observation:
        """Extract observation from a batch state (single env)."""
        qpos_addrs = jnp.array(self._drone_qpos_addrs)
        positions = state.mjx_data.qpos[qpos_addrs[:, None] + jnp.arange(3)]
        quats = state.mjx_data.qpos[qpos_addrs[:, None] + jnp.arange(3, 7)]
        heading = jax.vmap(_quat_to_yaw_jax)(quats)
        return build_observation(self.config, state.mission, positions, heading)


# ──────────────────────────────────────────────────────────────────────────────
# Batch rollout utility
# ──────────────────────────────────────────────────────────────────────────────

def batch_rollout(
    env: AvalancheMJXEnv,
    policy_fn,              # (state, key) → actions (num_drones, 4)
    n_envs: int,
    n_steps: int,
    seed: int = 0,
) -> dict:
    """Run N environments for T steps, return aggregate metrics.

    policy_fn is vmapped internally over the batch dimension.

    Returns dict with keys: mean_reward, total_deliveries, coverage, etc.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)
    states = jax.vmap(env.reset)(keys)

    @jax.jit
    def batched_step(states, key):
        step_keys = jax.random.split(key, n_envs)
        actions = jax.vmap(policy_fn)(states, step_keys)
        next_states, rewards, info = jax.vmap(env.step)(states, actions, step_keys)
        return next_states, rewards, info

    all_rewards = []
    rng = jax.random.PRNGKey(seed + 1)
    for _ in range(n_steps):
        rng, k = jax.random.split(rng)
        states, rewards, info = batched_step(states, k)
        all_rewards.append(rewards)

    all_rewards = jnp.stack(all_rewards, axis=0)   # (T, N, num_drones)
    return {
        "mean_reward_per_step": float(jnp.mean(all_rewards)),
        "total_reward": float(jnp.sum(all_rewards)),
        "final_coverage": float(jnp.mean(states.mission.metrics["coverage"])),
        "final_deliveries": float(jnp.mean(states.mission.metrics["delivery_events"])),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Quaternion yaw (JAX-compatible)
# ──────────────────────────────────────────────────────────────────────────────

def _quat_to_yaw_jax(q: jnp.ndarray) -> jnp.ndarray:
    """Extract yaw from [w, x, y, z] quaternion. Pure JAX, vmappable."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))


def _get_mj_model_from_mjx(mjx_model):
    """Compatibility helper — extract underlying MjModel if needed."""
    if hasattr(mjx_model, 'model'):
        return mjx_model.model
    return mjx_model


def make_mjx_env(config: AvalancheConfig | None = None) -> AvalancheMJXEnv:
    """Convenience factory."""
    return AvalancheMJXEnv(config or AvalancheConfig())
