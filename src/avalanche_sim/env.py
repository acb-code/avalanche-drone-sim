from __future__ import annotations

import jax
import jax.numpy as jnp

from .capabilities import DroneCapabilities, default_capabilities
from .dynamics import DynamicsBackend, Kinematic3DOFBackend
from .terrain import generate_scene
from .types import Action, EnvConfig, EnvState, Observation


def _default_positions(config: EnvConfig) -> tuple[jnp.ndarray, jnp.ndarray]:
    xs = jnp.linspace(config.map_size_x * 0.15, config.map_size_x * 0.85, config.num_drones)
    ys = jnp.full((config.num_drones,), config.map_size_y * 0.92)
    zs = jnp.linspace(config.altitude_min + 6.0, config.altitude_min + 12.0, config.num_drones)
    headings = jnp.linspace(-0.3, 0.3, config.num_drones)
    return jnp.stack([xs, ys, zs], axis=-1), headings


class AvalancheRescueEnv:
    def __init__(
        self,
        config: EnvConfig,
        dynamics_backend: DynamicsBackend | None = None,
        capabilities: DroneCapabilities | None = None,
    ):
        self.config = config
        self.dynamics_backend = dynamics_backend or Kinematic3DOFBackend()
        self.capabilities = capabilities or default_capabilities(self.dynamics_backend)

    def reset(self, key: jax.Array) -> tuple[Observation, EnvState]:
        terrain_height, debris_mask, unsafe_mask, wind_field, victim_positions, severity = generate_scene(
            self.config, key
        )
        drone_positions, drone_heading = _default_positions(self.config)
        scanned_cells = jnp.zeros(
            (self.config.coverage_resolution_y, self.config.coverage_resolution_x), dtype=jnp.bool_
        )
        state = EnvState(
            drone_positions=drone_positions,
            drone_heading=drone_heading,
            drone_battery=jnp.full((self.config.num_drones,), self.config.base_battery),
            drone_payload=jnp.full((self.config.num_drones,), self.config.payload_per_drone, dtype=jnp.int32),
            terrain_height=terrain_height,
            debris_mask=debris_mask,
            unsafe_mask=unsafe_mask,
            wind_field=wind_field,
            victim_positions=victim_positions,
            victim_severity=severity,
            victim_found=jnp.zeros((self.config.num_victims,), dtype=jnp.bool_),
            victim_confirmed=jnp.zeros((self.config.num_victims,), dtype=jnp.bool_),
            victim_aided=jnp.zeros((self.config.num_victims,), dtype=jnp.bool_),
            victim_survival=jnp.ones((self.config.num_victims,)),
            drone_knowledge=jnp.zeros((self.config.num_drones, self.config.num_victims), dtype=jnp.bool_),
            shared_known_victims=jnp.zeros((self.config.num_victims,), dtype=jnp.bool_),
            scanned_cells=scanned_cells,
            time=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False),
            metrics={
                "coverage": jnp.array(0.0),
                "find_events": jnp.array(0),
                "confirm_events": jnp.array(0),
                "delivery_events": jnp.array(0),
            },
        )
        obs = self._observe(state)
        return obs, state

    def step(
        self,
        key: jax.Array,
        state: EnvState,
        actions: Action | jnp.ndarray,
    ) -> tuple[Observation, EnvState, jnp.ndarray, jnp.ndarray, dict]:
        action_arr = actions.control if isinstance(actions, Action) else actions
        movement = self.capabilities.movement.step(self.config, state, action_arr)

        scanned_cells, new_coverage_count = self._update_coverage(state.scanned_cells, movement.positions)
        sensing = self.capabilities.sensing.sense(key, self.config, state, movement.positions)
        communication = self.capabilities.communication.share(
            self.config,
            state,
            jnp.logical_or(state.drone_knowledge, sensing.detections),
            movement.positions,
        )
        found = jnp.logical_or(state.victim_found, jnp.any(sensing.detections, axis=0))
        confirmed = jnp.logical_or(state.victim_confirmed, jnp.any(sensing.confirmations, axis=0))

        victim_survival = jnp.maximum(
            0.0,
            state.victim_survival
            - self.config.severity_decay
            * state.victim_severity
            * (1.0 - state.victim_aided.astype(jnp.float32)),
        )
        delivery = self.capabilities.delivery.deliver(
            self.config,
            state,
            movement.positions,
            communication.drone_knowledge,
        )

        alive_mask = movement.battery > 0.0
        mission_complete = jnp.all(delivery.victim_aided)
        timeout = (state.time + 1) >= self.config.horizon
        done = jnp.logical_or(timeout, mission_complete)

        rewards = self._compute_rewards(
            new_coverage_count,
            found,
            state.victim_found,
            confirmed,
            state.victim_confirmed,
            delivery.delivery_flags,
            movement.collisions,
            movement.near_collisions,
            movement.speed_norm,
            victim_survival,
            delivery.duplicate_attempts,
            timeout,
        )
        rewards = rewards * alive_mask.astype(jnp.float32)

        next_state = EnvState(
            drone_positions=movement.positions,
            drone_heading=movement.heading,
            drone_battery=movement.battery,
            drone_payload=delivery.payload,
            terrain_height=state.terrain_height,
            debris_mask=state.debris_mask,
            unsafe_mask=state.unsafe_mask,
            wind_field=state.wind_field,
            victim_positions=state.victim_positions,
            victim_severity=state.victim_severity,
            victim_found=found,
            victim_confirmed=confirmed,
            victim_aided=delivery.victim_aided,
            victim_survival=victim_survival,
            drone_knowledge=communication.drone_knowledge,
            shared_known_victims=communication.shared_known_victims,
            scanned_cells=scanned_cells,
            time=state.time + 1,
            done=done,
            metrics={
                "coverage": scanned_cells.mean(),
                "find_events": jnp.sum(found.astype(jnp.int32)),
                "confirm_events": jnp.sum(confirmed.astype(jnp.int32)),
                "delivery_events": jnp.sum(delivery.victim_aided.astype(jnp.int32)),
            },
        )
        obs = self._observe(next_state)
        dones = jnp.full((self.config.num_drones,), done)
        info = {
            "team_done": done,
            "coverage": scanned_cells.mean(),
            "new_detections": jnp.any(jnp.logical_and(found, ~state.victim_found)),
            "new_confirmations": jnp.any(jnp.logical_and(confirmed, ~state.victim_confirmed)),
            "deliveries": jnp.sum(delivery.delivery_flags.astype(jnp.int32)),
            "collisions": movement.collisions,
            "near_collisions": movement.near_collisions,
        }
        return obs, next_state, rewards, dones, info

    def _update_coverage(self, scanned_cells: jnp.ndarray, positions: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        cx = jnp.clip(
            jnp.floor(positions[:, 0] / self.config.map_size_x * self.config.coverage_resolution_x).astype(jnp.int32),
            0,
            self.config.coverage_resolution_x - 1,
        )
        cy = jnp.clip(
            jnp.floor(positions[:, 1] / self.config.map_size_y * self.config.coverage_resolution_y).astype(jnp.int32),
            0,
            self.config.coverage_resolution_y - 1,
        )
        before = scanned_cells[cy, cx]
        updated = scanned_cells.at[cy, cx].set(True)
        new_count = (~before).astype(jnp.float32)
        return updated, new_count

    def _compute_rewards(
        self,
        new_coverage_count: jnp.ndarray,
        found: jnp.ndarray,
        prev_found: jnp.ndarray,
        confirmed: jnp.ndarray,
        prev_confirmed: jnp.ndarray,
        delivery_flags: jnp.ndarray,
        collisions: jnp.ndarray,
        near_collisions: jnp.ndarray,
        speed_norm: jnp.ndarray,
        victim_survival: jnp.ndarray,
        duplicate_attempts: jnp.ndarray,
        timeout: jnp.ndarray,
    ) -> jnp.ndarray:
        new_finds = jnp.any(jnp.logical_and(found, ~prev_found)).astype(jnp.float32)
        new_confirms = jnp.any(jnp.logical_and(confirmed, ~prev_confirmed)).astype(jnp.float32)
        deliveries = jnp.sum(delivery_flags.astype(jnp.float32))
        coverage_reward = self.config.reward_coverage * new_coverage_count
        team_bonus = (
            self.config.reward_find * new_finds
            + self.config.reward_confirm * new_confirms
            + self.config.reward_delivery * deliveries
            + self.config.reward_survival * jnp.sum(victim_survival)
        ) / self.config.num_drones
        penalties = (
            self.config.penalty_collision * collisions.astype(jnp.float32)
            + self.config.penalty_near_collision * near_collisions.astype(jnp.float32)
            + self.config.penalty_wasted_energy * speed_norm
            + self.config.penalty_duplicate_delivery
            * jnp.sum(duplicate_attempts.astype(jnp.float32))
            / self.config.num_drones
            + self.config.penalty_timeout * timeout.astype(jnp.float32) / self.config.num_drones
        )
        return coverage_reward + team_bonus - penalties

    def _observe(self, state: EnvState) -> Observation:
        own = jnp.concatenate(
            [
                state.drone_positions / jnp.array([self.config.map_size_x, self.config.map_size_y, self.config.altitude_max]),
                jnp.stack(
                    [
                        jnp.cos(state.drone_heading),
                        jnp.sin(state.drone_heading),
                        state.drone_battery / self.config.base_battery,
                        state.drone_payload / jnp.maximum(1, self.config.payload_per_drone),
                    ],
                    axis=-1,
                ),
            ],
            axis=-1,
        )

        rel = state.drone_positions[None, :, :] - state.drone_positions[:, None, :]
        rel_dist = jnp.linalg.norm(rel[..., :2], axis=-1)
        neighbor_mask = (rel_dist <= self.config.communication_range).astype(jnp.float32)
        drone_features = jnp.concatenate(
            [
                own,
                jnp.sum(neighbor_mask, axis=-1, keepdims=True) / jnp.maximum(1, self.config.num_drones - 1),
            ],
            axis=-1,
        )

        victim_rel = state.victim_positions[None, :, :] - state.drone_positions[:, None, :]
        victim_dist = jnp.linalg.norm(victim_rel, axis=-1) / jnp.sqrt(self.config.map_size_x**2 + self.config.map_size_y**2)
        victim_visible = state.drone_knowledge.astype(jnp.float32)
        victim_status = jnp.stack(
            [
                victim_visible,
                jnp.broadcast_to(state.victim_confirmed.astype(jnp.float32), victim_visible.shape),
                jnp.broadcast_to(state.victim_aided.astype(jnp.float32), victim_visible.shape),
                jnp.broadcast_to(state.victim_survival, victim_visible.shape),
                victim_dist,
            ],
            axis=-1,
        )
        team_features = jnp.tile(
            jnp.array(
                [
                    state.time / self.config.horizon,
                    state.scanned_cells.mean(),
                    jnp.mean(state.drone_battery) / self.config.base_battery,
                    jnp.mean(state.victim_survival),
                    jnp.mean(state.shared_known_victims.astype(jnp.float32)),
                ]
            ),
            (self.config.num_drones, 1),
        )
        coverage_map = jnp.broadcast_to(
            state.scanned_cells[None, :, :].astype(jnp.float32),
            (self.config.num_drones, self.config.coverage_resolution_y, self.config.coverage_resolution_x),
        )
        action_mask = (state.drone_battery[:, None] > 0.0).astype(jnp.float32) * jnp.ones((self.config.num_drones, 4), dtype=jnp.float32)
        return Observation(
            drone_features=drone_features,
            victim_features=victim_status,
            team_features=team_features,
            coverage_map=coverage_map,
            action_mask=action_mask,
        )


def make_env(
    config: EnvConfig,
    dynamics_backend: DynamicsBackend | None = None,
    capabilities: DroneCapabilities | None = None,
) -> AvalancheRescueEnv:
    return AvalancheRescueEnv(config=config, dynamics_backend=dynamics_backend, capabilities=capabilities)
