#!/usr/bin/env python
"""demo_batch.py — Use Case 2: GPU-batched simulation via MJX + jax.vmap.

Run:
    uv run python scripts/demo_batch.py [--n-envs 100] [--n-steps 50]
"""
from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from avalanche_mujoco import AvalancheConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs",  type=int, default=64,  help="Number of parallel envs")
    parser.add_argument("--n-steps", type=int, default=50,  help="Policy steps per env")
    parser.add_argument("--seed",    type=int, default=0)
    args = parser.parse_args()

    cfg = AvalancheConfig(num_drones=4, num_victims=6, horizon=args.n_steps)
    print(f"MJX batch demo: {args.n_envs} envs × {args.n_steps} steps")
    print(f"  num_drones={cfg.num_drones}, num_victims={cfg.num_victims}")

    try:
        from avalanche_mujoco import AvalancheMJXEnv
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Install: pip install 'mujoco-mjx>=3.1' 'jax[cuda12]>=0.4.30'")
        return

    print("Building MJX environment...")
    t0 = time.perf_counter()
    env = AvalancheMJXEnv(cfg)
    print(f"  Build time: {time.perf_counter()-t0:.2f}s")

    # ── Reset all envs ──────────────────────────────────────────────────
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_envs)
    print(f"Resetting {args.n_envs} envs...")
    t0 = time.perf_counter()
    states = jax.jit(jax.vmap(env.reset))(keys)
    jax.block_until_ready(states)
    print(f"  Reset time: {time.perf_counter()-t0:.3f}s")

    # ── Random policy rollout ───────────────────────────────────────────
    @jax.jit
    def batched_step(states, key):
        n = args.n_envs
        action_key, step_key = jax.random.split(key)
        # Random actions: (n_envs, num_drones, 4)
        actions = jax.random.uniform(
            action_key, (n, cfg.num_drones, 4), minval=-1.0, maxval=1.0
        )
        step_keys = jax.random.split(step_key, n)
        next_states, rewards, info = jax.vmap(env.step)(states, actions, step_keys)
        return next_states, jnp.mean(rewards, axis=-1)  # (n,)

    # Warm-up JIT
    print("Compiling JIT step...")
    rng = jax.random.PRNGKey(args.seed + 1)
    rng, k = jax.random.split(rng)
    states, _ = batched_step(states, k)
    jax.block_until_ready(states)
    print("  JIT compiled.")

    # Timed rollout
    print(f"Running {args.n_steps} steps...")
    t0 = time.perf_counter()
    all_rewards = []
    for step in range(args.n_steps - 1):   # -1 because we used 1 for warmup
        rng, k = jax.random.split(rng)
        states, step_rewards = batched_step(states, k)
        all_rewards.append(step_rewards)

    jax.block_until_ready(states)
    elapsed = time.perf_counter() - t0
    total_steps = (args.n_steps - 1) * args.n_envs
    print(f"\nResults:")
    print(f"  Wall time      : {elapsed:.3f}s")
    print(f"  Throughput     : {total_steps / elapsed:,.0f} env-steps/s")
    print(f"  Avg reward/step: {float(jnp.mean(jnp.stack(all_rewards))):.4f}")
    print(f"  Final coverage : {float(jnp.mean(states.mission.metrics['coverage'])):.2%}")
    print(f"  Final deliveries: {float(jnp.mean(states.mission.metrics['delivery_events'].astype(jnp.float32))):.1f}/{cfg.num_victims}")


if __name__ == "__main__":
    main()
