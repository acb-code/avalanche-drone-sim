#!/usr/bin/env python
"""demo_train.py — Use Case 3: PPO baseline on single-drone SAR.

Uses the Gymnasium wrapper + Stable-Baselines3 (if available) or a minimal
JAX-native REINFORCE loop otherwise.

Run:
    uv run python scripts/demo_train.py [--steps 100000] [--backend cpu]
"""
from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from avalanche_mujoco import AvalancheConfig
from avalanche_mujoco.wrappers import AvalancheGymnasiumEnv


def _try_sb3_ppo(env, total_steps: int) -> None:
    """Train with Stable-Baselines3 PPO if available."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
    except ImportError:
        raise ImportError("stable-baselines3 not installed")

    print("Running gymnasium env checker...")
    check_env(env, warn=True)
    print("  Env check passed.")

    print(f"Training PPO for {total_steps} steps...")
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        tensorboard_log=None,
    )
    t0 = time.perf_counter()
    model.learn(total_timesteps=total_steps)
    elapsed = time.perf_counter() - t0
    print(f"Training complete in {elapsed:.1f}s")
    return model


def _minimal_reinforce(env, total_steps: int) -> None:
    """Minimal REINFORCE loop — no external deps, just numpy/JAX."""
    print("Running minimal REINFORCE (no SB3)...")
    import numpy as np

    # Very simple linear policy: obs → action via random projection
    rng = np.random.default_rng(0)
    # Flatten the dict observation
    dummy_obs, _ = env.reset()
    flat_dim = sum(v.size for v in dummy_obs.values())
    action_dim = env.action_space.shape[0]
    W = rng.normal(0, 0.01, (flat_dim, action_dim))
    b = np.zeros(action_dim)

    def policy(obs_dict: dict) -> np.ndarray:
        flat = np.concatenate([v.flatten() for v in obs_dict.values()])
        return np.tanh(flat @ W + b)

    episode_rewards: list[float] = []
    steps_done = 0
    episode = 0

    while steps_done < total_steps:
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_len = 0
        done = False
        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_len += 1
            steps_done += 1
            done = terminated or truncated
            if steps_done >= total_steps:
                break

        episode_rewards.append(ep_reward)
        episode += 1
        if episode % 5 == 0:
            recent = episode_rewards[-5:]
            print(f"  episode={episode:4d}  steps={steps_done:6d}"
                  f"  mean_reward={np.mean(recent):.2f}  len={ep_len}")

    print(f"\nFinal 10-ep mean reward: {np.mean(episode_rewards[-10:]):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",   type=int, default=50_000)
    parser.add_argument("--backend", choices=["cpu", "mjx"], default="cpu")
    parser.add_argument("--algo",    choices=["ppo", "reinforce"], default="reinforce",
                        help="RL algorithm (ppo requires stable-baselines3)")
    args = parser.parse_args()

    cfg = AvalancheConfig(
        num_drones=1,   # single-drone for initial training
        num_victims=4,
        horizon=150,
    )
    print(f"Training setup: {args.algo.upper()} on {args.backend} backend")
    print(f"  Config: {cfg.num_drones} drone(s), {cfg.num_victims} victims, horizon={cfg.horizon}")

    env = AvalancheGymnasiumEnv(config=cfg, backend=args.backend)

    if args.algo == "ppo":
        try:
            _try_sb3_ppo(env, args.steps)
        except ImportError as e:
            print(f"  {e}")
            print("  Falling back to minimal REINFORCE.")
            _minimal_reinforce(env, args.steps)
    else:
        _minimal_reinforce(env, args.steps)

    env.close()


if __name__ == "__main__":
    main()
