"""CPU MuJoCo environment — Use Case 1 (simulate & visualize).

AvalanchePhysicsEnv wraps mujoco.MjModel + mujoco.MjData.
- Terrain is enforced virtually (constrain_by_terrain) after each step.
- Wind applied via xfrc_applied.
- Victim markers updated via mocap body positions.
- Optional non-blocking viewer via mujoco.viewer.launch_passive().
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
from .types import DronePhysicsState, MissionState, Observation
from .wind import apply_wind_cpu

try:
    import mujoco
    import mujoco.viewer
    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False


class AvalanchePhysicsEnv:
    """Single-episode MuJoCo CPU environment for SAR.

    Parameters
    ----------
    config  : AvalancheConfig
    viewer  : bool — if True, launch the passive MuJoCo viewer on reset().
    """

    def __init__(self, config: AvalancheConfig, viewer: bool = False):
        if not _MUJOCO_AVAILABLE:
            raise ImportError("mujoco is required. Install with: pip install mujoco>=3.1")
        self.config = config
        self._use_viewer = viewer
        self._viewer = None

        # Build the scene XML once (physics doesn't depend on random seed)
        xml, self._body_names, self._actuator_names = build_scene(config, backend="cpu")
        self._model: mujoco.MjModel = mujoco.MjModel.from_xml_string(xml)
        self._data: mujoco.MjData = mujoco.MjData(self._model)

        # Cache body IDs and joint addresses
        self._drone_body_ids: list[int] = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self._body_names
        ]
        self._victim_body_ids: list[int] = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, f"victim_{v}")
            for v in range(config.num_victims)
        ]
        # freejoint qpos addresses (7 per drone: pos[3] + quat[4])
        self._drone_qpos_addrs: list[int] = self._find_freejoint_addrs()

    def _find_freejoint_addrs(self) -> list[int]:
        addrs = []
        for i in range(self.config.num_drones):
            jname = f"drone{i}/root"
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            addrs.append(int(self._model.jnt_qposadr[jid]))
        return addrs

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, key: jax.Array) -> tuple[Observation, MissionState]:
        """Randomise scene, reset physics, return initial obs + mission state."""
        terrain_height, debris_mask, unsafe_mask, wind_field, victim_positions, severity = (
            generate_scene(self.config, key)
        )

        # Reset MuJoCo data
        mujoco.mj_resetData(self._model, self._data)

        # Set spawn positions
        from .drone import default_spawn_positions
        spawn_pos = default_spawn_positions(
            self.config.num_drones, self.config.map_size_x, self.config.map_size_y
        )
        for i, addr in enumerate(self._drone_qpos_addrs):
            self._data.qpos[addr:addr+3] = spawn_pos[i]
            # Unit quaternion (identity orientation)
            self._data.qpos[addr+3] = 1.0
            self._data.qpos[addr+4:addr+7] = 0.0

        # Place victim mocap bodies
        for v in range(self.config.num_victims):
            bid = self._victim_body_ids[v]
            if bid >= 0:
                mocap_id = self._model.body_mocapid[bid]
                if mocap_id >= 0:
                    self._data.mocap_pos[mocap_id] = np.array(victim_positions[v])
                    self._data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])

        mujoco.mj_forward(self._model, self._data)

        # Initialise mission state
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

        if self._use_viewer and self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

        drone_state = self._extract_drone_state()
        obs = build_observation(self.config, mission, drone_state.positions, drone_state.heading)
        return obs, mission

    def step(
        self,
        key: jax.Array,
        mission: MissionState,
        actions: jnp.ndarray,   # (num_drones, 4) in [-1, 1]
    ) -> tuple[Observation, MissionState, jnp.ndarray, jnp.ndarray, dict]:
        """Advance physics + mission by one control step."""
        # 1. Compute rotor thrusts via PID
        drone_state = self._extract_drone_state()
        thrusts = compute_rotor_thrusts(
            self.config,
            actions,
            drone_state.quats,
            drone_state.lin_vel,
            drone_state.ang_vel,
            drone_state.positions,
        )
        thrusts_np = np.array(thrusts)   # (num_drones, 4)

        # 2. Apply thrusts to actuator ctrl
        for d in range(self.config.num_drones):
            base = d * 4
            self._data.ctrl[base:base+4] = thrusts_np[d]

        # 3. Apply wind
        apply_wind_cpu(self._data, self.config, mission, self._drone_body_ids)

        # 4. Run MuJoCo substeps
        for _ in range(self.config.substeps):
            mujoco.mj_step(self._model, self._data)

        if self._viewer is not None:
            self._viewer.sync()

        # 5. Extract new drone positions
        new_drone_state = self._extract_drone_state()

        # 6. Virtual terrain constraint (clips Z, prevents tunnelling)
        old_positions = drone_state.positions
        candidate_positions = new_drone_state.positions
        constrained_positions = constrain_by_terrain(
            self.config, mission.terrain_height, old_positions, candidate_positions
        )

        # Update qpos with constrained positions
        for i, addr in enumerate(self._drone_qpos_addrs):
            self._data.qpos[addr:addr+3] = np.array(constrained_positions[i])
        mujoco.mj_forward(self._model, self._data)

        # Re-extract heading from constrained state
        final_state = self._extract_drone_state()

        # 7. Compute speed_norm for battery / reward
        vel = final_state.lin_vel
        speed_norm = (
            jnp.linalg.norm(vel[:, :2], axis=-1) / self.config.max_xy_speed
            + jnp.abs(vel[:, 2]) / self.config.max_z_speed
        ).clip(0.0, 1.0)

        # 8. Mission logic (pure JAX)
        next_mission, rewards, info = mission_step(
            key, self.config, mission,
            final_state.positions,
            final_state.heading,
            speed_norm,
            *_zero_collisions(self.config),  # MuJoCo handles real collisions
        )

        # 9. Update victim marker visuals
        self._update_victim_visuals(next_mission)

        obs = build_observation(self.config, next_mission, final_state.positions, final_state.heading)
        dones = jnp.full(self.config.num_drones, next_mission.done)
        return obs, next_mission, rewards, dones, info

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ──────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────

    def _extract_drone_state(self) -> DronePhysicsState:
        n = self.config.num_drones
        positions = np.zeros((n, 3))
        quats = np.zeros((n, 4))
        lin_vel = np.zeros((n, 3))
        ang_vel = np.zeros((n, 3))

        for i, addr in enumerate(self._drone_qpos_addrs):
            positions[i] = self._data.qpos[addr:addr+3]
            quats[i] = self._data.qpos[addr+3:addr+7]   # [w, x, y, z]

        # qvel: freejoint has 6 DOF — 3 lin vel (world), 3 ang vel (body)
        for i in range(n):
            jname = f"drone{i}/root"
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            vadr = int(self._model.jnt_dofadr[jid])
            lin_vel[i] = self._data.qvel[vadr:vadr+3]
            ang_vel[i] = self._data.qvel[vadr+3:vadr+6]

        # Heading = yaw from quaternion
        headings = np.array([_quat_to_yaw(quats[i]) for i in range(n)])
        return DronePhysicsState(
            positions=jnp.array(positions),
            quats=jnp.array(quats),
            lin_vel=jnp.array(lin_vel),
            ang_vel=jnp.array(ang_vel),
            heading=jnp.array(headings),
        )

    def _update_victim_visuals(self, mission: MissionState) -> None:
        """Move victim mocap bodies to their randomised positions."""
        for v in range(self.config.num_victims):
            bid = self._victim_body_ids[v]
            if bid < 0:
                continue
            mocap_id = self._model.body_mocapid[bid]
            if mocap_id >= 0:
                self._data.mocap_pos[mocap_id] = np.array(mission.victim_positions[v])

        # Update victim geom colours via geom_rgba
        for v in range(self.config.num_victims):
            gname = f"victim_geom_{v}"
            gid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid >= 0:
                if bool(mission.victim_aided[v]):
                    self._model.geom_rgba[gid] = [0.1, 0.75, 0.2, 0.9]
                elif bool(mission.victim_found[v]):
                    self._model.geom_rgba[gid] = [0.9, 0.6, 0.1, 0.9]
                else:
                    self._model.geom_rgba[gid] = [0.85, 0.1, 0.1, 0.9]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _quat_to_yaw(q: np.ndarray) -> float:
    """Extract yaw (Z-axis rotation) from [w, x, y, z] quaternion."""
    w, x, y, z = q
    return float(np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z)))


def _zero_collisions(cfg: AvalancheConfig):
    """Return zero collision arrays (MuJoCo handles real contacts)."""
    return (
        jnp.zeros(cfg.num_drones, dtype=jnp.bool_),
        jnp.zeros(cfg.num_drones, dtype=jnp.bool_),
    )


def make_physics_env(config: AvalancheConfig | None = None, **kwargs) -> AvalanchePhysicsEnv:
    """Convenience factory."""
    return AvalanchePhysicsEnv(config or AvalancheConfig(), **kwargs)
