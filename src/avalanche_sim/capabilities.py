from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp

from .dynamics import DynamicsBackend, Kinematic3DOFBackend
from .terrain import terrain_height_at
from .types import DynamicsOutput, EnvConfig, EnvState


@dataclass
class MovementResult:
    positions: jnp.ndarray
    heading: jnp.ndarray
    battery: jnp.ndarray
    collisions: jnp.ndarray
    near_collisions: jnp.ndarray
    speed_norm: jnp.ndarray


@dataclass
class SensingResult:
    detections: jnp.ndarray
    confirmations: jnp.ndarray


@dataclass
class CommunicationResult:
    drone_knowledge: jnp.ndarray
    shared_known_victims: jnp.ndarray


@dataclass
class DeliveryResult:
    payload: jnp.ndarray
    victim_aided: jnp.ndarray
    delivery_flags: jnp.ndarray
    duplicate_attempts: jnp.ndarray


class MovementModel(Protocol):
    def step(self, config: EnvConfig, state: EnvState, actions: jnp.ndarray) -> MovementResult:
        ...


class SensingModel(Protocol):
    def sense(
        self,
        key: jax.Array,
        config: EnvConfig,
        state: EnvState,
        positions: jnp.ndarray,
    ) -> SensingResult:
        ...


class CommunicationModel(Protocol):
    def share(
        self,
        config: EnvConfig,
        state: EnvState,
        drone_knowledge: jnp.ndarray,
        positions: jnp.ndarray,
    ) -> CommunicationResult:
        ...


class DeliveryModel(Protocol):
    def deliver(
        self,
        config: EnvConfig,
        state: EnvState,
        positions: jnp.ndarray,
        drone_knowledge: jnp.ndarray,
    ) -> DeliveryResult:
        ...


@dataclass(frozen=True)
class DroneCapabilities:
    movement: MovementModel
    sensing: SensingModel
    communication: CommunicationModel
    delivery: DeliveryModel


@dataclass
class DynamicsMovementModel:
    dynamics_backend: DynamicsBackend

    def step(self, config: EnvConfig, state: EnvState, actions: jnp.ndarray) -> MovementResult:
        dyn: DynamicsOutput = self.dynamics_backend.step(
            config,
            state.drone_positions,
            state.drone_heading,
            state.drone_battery,
            actions,
            state.wind_field,
        )
        positions = _constrain_motion_by_terrain(config, state, dyn.positions)
        return MovementResult(
            positions=positions,
            heading=dyn.heading,
            battery=dyn.battery,
            collisions=dyn.collisions,
            near_collisions=dyn.near_collisions,
            speed_norm=dyn.speed_norm,
        )


class ProbabilisticSensingModel:
    def sense(
        self,
        key: jax.Array,
        config: EnvConfig,
        state: EnvState,
        positions: jnp.ndarray,
    ) -> SensingResult:
        diff = state.victim_positions[None, :, :3] - positions[:, None, :3]
        dist = jnp.linalg.norm(diff, axis=-1)
        horizontal = jnp.linalg.norm(diff[..., :2], axis=-1)
        forward = jnp.stack([jnp.cos(state.drone_heading), jnp.sin(state.drone_heading)], axis=-1)
        horizontal_unit = diff[..., :2] / jnp.maximum(horizontal[..., None], 1.0e-6)
        fov_score = jnp.sum(horizontal_unit * forward[:, None, :], axis=-1)
        altitude_factor = jnp.exp(
            -jnp.abs(positions[:, None, 2] - config.sensor_altitude_scale) / config.sensor_altitude_scale
        )

        victim_cells_x = jnp.clip(
            jnp.floor(state.victim_positions[:, 0] / config.map_size_x * config.coverage_resolution_x).astype(jnp.int32),
            0,
            config.coverage_resolution_x - 1,
        )
        victim_cells_y = jnp.clip(
            jnp.floor(state.victim_positions[:, 1] / config.map_size_y * config.coverage_resolution_y).astype(jnp.int32),
            0,
            config.coverage_resolution_y - 1,
        )
        under_debris = state.debris_mask[victim_cells_y, victim_cells_x]
        base_prob = jnp.exp(-config.detection_decay * dist) * altitude_factor
        base_prob = base_prob * (0.6 + 0.4 * (~under_debris)[None, :].astype(jnp.float32))
        base_prob = base_prob * (0.35 + 0.65 * (fov_score > config.sensor_fov_cos).astype(jnp.float32))
        known_bonus = config.rescan_bonus * state.drone_knowledge.astype(jnp.float32)
        detect_prob = jnp.clip(base_prob + known_bonus, 0.0, 0.98)
        in_range = dist <= config.sensor_range
        key_detect, key_confirm = jax.random.split(key)
        detect_draw = jax.random.uniform(key_detect, detect_prob.shape)
        confirm_draw = jax.random.uniform(key_confirm, detect_prob.shape)
        detections = in_range & (detect_draw < detect_prob) & (~state.victim_aided)[None, :]
        confirmations = detections & (confirm_draw < jnp.clip(detect_prob + 0.1, 0.0, 1.0))
        return SensingResult(detections=detections, confirmations=confirmations)


class RangeCommunicationModel:
    def share(
        self,
        config: EnvConfig,
        state: EnvState,
        drone_knowledge: jnp.ndarray,
        positions: jnp.ndarray,
    ) -> CommunicationResult:
        del state
        disp = positions[:, None, :2] - positions[None, :, :2]
        dist = jnp.linalg.norm(disp, axis=-1)
        comms = dist <= config.communication_range
        informed_from_neighbors = jnp.einsum(
            "ij,jv->iv", comms.astype(jnp.int32), drone_knowledge.astype(jnp.int32)
        ) > 0
        next_knowledge = jnp.logical_or(drone_knowledge, informed_from_neighbors)
        shared_known = jnp.any(next_knowledge, axis=0)
        return CommunicationResult(
            drone_knowledge=next_knowledge,
            shared_known_victims=shared_known,
        )


class PriorityAidDeliveryModel:
    def deliver(
        self,
        config: EnvConfig,
        state: EnvState,
        positions: jnp.ndarray,
        drone_knowledge: jnp.ndarray,
    ) -> DeliveryResult:
        diff = state.victim_positions[None, :, :] - positions[:, None, :]
        dist = jnp.linalg.norm(diff, axis=-1)
        eligible = (
            (dist <= config.delivery_range)
            & drone_knowledge
            & (~state.victim_aided)[None, :]
            & (state.drone_payload[:, None] > 0)
        )
        inf_dist = jnp.where(eligible, dist, jnp.inf)
        chosen_drone = jnp.argmin(inf_dist, axis=0)
        victim_has_drone = jnp.any(eligible, axis=0)
        selected = jax.nn.one_hot(chosen_drone, config.num_drones, dtype=jnp.bool_).T & victim_has_drone[None, :]
        per_drone_deliveries = jnp.sum(selected, axis=1).astype(jnp.int32)
        payload = jnp.maximum(0, state.drone_payload - per_drone_deliveries)
        delivery_flags = victim_has_drone
        victim_aided = jnp.logical_or(state.victim_aided, delivery_flags)

        attempted = (dist <= config.delivery_range) & drone_knowledge & (state.drone_payload[:, None] > 0)
        duplicate_attempts = jnp.maximum(0, jnp.sum(attempted, axis=0) - delivery_flags.astype(jnp.int32))
        return DeliveryResult(
            payload=payload,
            victim_aided=victim_aided,
            delivery_flags=delivery_flags,
            duplicate_attempts=duplicate_attempts,
        )


def default_capabilities(dynamics_backend: DynamicsBackend | None = None) -> DroneCapabilities:
    backend = dynamics_backend or Kinematic3DOFBackend()
    return DroneCapabilities(
        movement=DynamicsMovementModel(backend),
        sensing=ProbabilisticSensingModel(),
        communication=RangeCommunicationModel(),
        delivery=PriorityAidDeliveryModel(),
    )


def _constrain_motion_by_terrain(
    config: EnvConfig,
    state: EnvState,
    candidate_positions: jnp.ndarray,
) -> jnp.ndarray:
    start_xy = state.drone_positions[:, :2]
    target_xy = candidate_positions[:, :2]
    target_z = candidate_positions[:, 2]

    sample_fracs = jnp.linspace(0.0, 1.0, 9, dtype=jnp.float32)
    path_xy = start_xy[:, None, :] + (target_xy - start_xy)[:, None, :] * sample_fracs[None, :, None]
    flat_path_xy = path_xy.reshape((-1, 2))
    path_terrain = terrain_height_at(config, state.terrain_height, flat_path_xy).reshape(
        (state.drone_positions.shape[0], sample_fracs.shape[0])
    )
    min_clearance_z = path_terrain + config.altitude_min
    safe_prefix = jnp.cumprod((target_z[:, None] >= min_clearance_z).astype(jnp.int32), axis=1).astype(jnp.float32)
    allowed_frac = jnp.max(sample_fracs[None, :] * safe_prefix, axis=1)

    constrained_xy = start_xy + (target_xy - start_xy) * allowed_frac[:, None]
    terrain_under_drones = terrain_height_at(config, state.terrain_height, constrained_xy)
    constrained_z = jnp.maximum(target_z, terrain_under_drones + config.altitude_min)
    constrained_z = jnp.clip(constrained_z, terrain_under_drones + config.altitude_min, config.altitude_max)
    return jnp.concatenate([constrained_xy, constrained_z[:, None]], axis=-1)
