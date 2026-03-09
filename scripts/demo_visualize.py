#!/usr/bin/env python
"""demo_visualize.py — Use Case 1: live MuJoCo viewer with a lawnmower policy.

Run:
    uv run python scripts/demo_visualize.py
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from avalanche_mujoco import AvalancheConfig, make_physics_env


def lawnmower_policy(mission, config: AvalancheConfig) -> jnp.ndarray:
    """Simple lawnmower + victim-chase policy (adapted from old sim)."""
    from avalanche_mujoco.types import MissionState

    # Extract positions from MuJoCo (we need them from physics state,
    # but for the policy we'll use victim knowledge from mission)
    n = config.num_drones
    known_idx = jnp.argmax(mission.shared_known_victims.astype(jnp.int32))
    has_known = jnp.any(mission.shared_known_victims)

    # Default: spread out in y-lanes and sweep back/forth
    lane_xs = jnp.linspace(config.map_size_x * 0.1, config.map_size_x * 0.9, n)
    phase = (mission.time // 20) % 2
    target_y = jnp.where(phase == 0, config.map_size_y * 0.2, config.map_size_y * 0.85)
    target_xy = jnp.stack([lane_xs, jnp.full(n, target_y)], axis=-1)

    # If a victim is known, converge on it
    victim_xy = jnp.broadcast_to(mission.victim_positions[known_idx, :2], target_xy.shape)
    target_xy = jnp.where(has_known, victim_xy, target_xy)

    # Return placeholder velocity commands (actual positions handled by PID in env)
    # Here we return [vx, vy, vz, yaw_rate] in [-1, 1]
    return jnp.zeros((n, 4))   # hover in place for demo


def main():
    cfg = AvalancheConfig(num_drones=3, num_victims=4, horizon=200)
    print(f"Building scene for {cfg.num_drones} drones, {cfg.num_victims} victims...")

    try:
        env = make_physics_env(cfg, viewer=True)
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Install mujoco: pip install mujoco>=3.1")
        return

    key = jax.random.PRNGKey(42)
    obs, mission = env.reset(key)
    print("Environment reset. Starting rollout...")
    print(f"  Victim positions: {mission.victim_positions[:, :2]}")

    total_reward = 0.0
    for step in range(cfg.horizon):
        key, k = jax.random.split(key)
        # Lawnmower with slight forward motion
        n = cfg.num_drones
        actions = jnp.zeros((n, 4))
        # Slight forward motion (y-direction in map = vy, and vary altitude slightly)
        phase = (step // 30) % 2
        vy = 0.6 if phase == 0 else -0.6
        actions = actions.at[:, 1].set(vy)    # vy command
        actions = actions.at[:, 2].set(0.0)   # maintain altitude

        obs, mission, rewards, dones, info = env.step(k, mission, actions)
        total_reward += float(jnp.mean(rewards))

        if step % 25 == 0:
            cov = float(mission.scanned_cells.mean())
            aided = int(mission.victim_aided.sum())
            print(f"  step={step:4d}  coverage={cov:.2%}  aided={aided}/{cfg.num_victims}"
                  f"  reward={float(jnp.mean(rewards)):.3f}")

        if bool(mission.done):
            print(f"Episode done at step {step}")
            break

        time.sleep(0.02)  # ~20 Hz real-time

    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Final coverage: {float(mission.scanned_cells.mean()):.2%}")
    print(f"Victims aided: {int(mission.victim_aided.sum())}/{cfg.num_victims}")
    env.close()


if __name__ == "__main__":
    main()
