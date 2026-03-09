"""Gymnasium wrapper — single-agent (centralised) interface.

Wraps AvalanchePhysicsEnv or AvalancheMJXEnv behind gymnasium.Env.

Observation space : Dict (matches Observation fields as Box arrays)
Action space      : Box(-1, 1, shape=(num_drones * 4,))  — flattened

Designed to pass gymnasium.utils.env_checker.check_env().
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
    _GymBase = gym.Env
except ImportError:
    _GYM_AVAILABLE = False
    _GymBase = object  # fallback for import without gymnasium

from ..config import AvalancheConfig
from ..types import MissionState, Observation


class AvalancheGymnasiumEnv(_GymBase):
    """Gymnasium-compatible wrapper for the SAR environment.

    Parameters
    ----------
    config  : AvalancheConfig
    backend : "cpu" (AvalanchePhysicsEnv) or "mjx" (AvalancheMJXEnv)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        config: AvalancheConfig | None = None,
        backend: str = "cpu",
        render_mode: str | None = None,
    ):
        if not _GYM_AVAILABLE:
            raise ImportError("gymnasium is required. Install with: pip install gymnasium>=0.29")
        self.config = config or AvalancheConfig()
        self.render_mode = render_mode
        self._rng = jax.random.PRNGKey(0)
        self._mission: MissionState | None = None

        if backend == "cpu":
            from ..physics_env import AvalanchePhysicsEnv
            self._env = AvalanchePhysicsEnv(self.config, viewer=(render_mode == "human"))
        elif backend == "mjx":
            from ..mjx_env import AvalancheMJXEnv
            self._env = AvalancheMJXEnv(self.config)
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

        self._backend = backend
        self._build_spaces()

    # ──────────────────────────────────────────────────────────────────────
    # Spaces
    # ──────────────────────────────────────────────────────────────────────

    def _build_spaces(self) -> None:
        cfg = self.config
        self.observation_space = spaces.Dict({
            "drone_features": spaces.Box(
                low=-2.0, high=2.0,
                shape=(cfg.num_drones, 8), dtype=np.float32
            ),
            "victim_features": spaces.Box(
                low=0.0, high=1.0,
                shape=(cfg.num_drones, cfg.num_victims, 5), dtype=np.float32
            ),
            "team_features": spaces.Box(
                low=0.0, high=1.0,
                shape=(cfg.num_drones, 5), dtype=np.float32
            ),
            "coverage_map": spaces.Box(
                low=0.0, high=1.0,
                shape=(cfg.num_drones, cfg.coverage_resolution_y, cfg.coverage_resolution_x),
                dtype=np.float32,
            ),
            "action_mask": spaces.Box(
                low=0.0, high=1.0,
                shape=(cfg.num_drones, 4), dtype=np.float32
            ),
        })
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(cfg.num_drones * 4,), dtype=np.float32
        )

    # ──────────────────────────────────────────────────────────────────────
    # gymnasium.Env API
    # ──────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)   # sets self.np_random (required by gym checker)
        if seed is not None:
            self._rng = jax.random.PRNGKey(seed)
        self._rng, key = jax.random.split(self._rng)

        if self._backend == "cpu":
            obs, self._mission = self._env.reset(key)
        else:
            self._state = self._env.reset(key)
            self._mission = self._state.mission
            obs = self._env.get_obs(self._state)

        return _obs_to_numpy(obs), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict, float, bool, bool, dict]:
        actions = jnp.array(action, dtype=jnp.float32).reshape(self.config.num_drones, 4)
        self._rng, key = jax.random.split(self._rng)

        if self._backend == "cpu":
            obs, self._mission, rewards, dones, info = self._env.step(key, self._mission, actions)
        else:
            self._state, rewards, info = self._env.step(self._state, actions, key)
            self._mission = self._state.mission
            obs = self._env.get_obs(self._state)

        # Centralised reward = mean over drones
        reward = float(jnp.mean(rewards))
        terminated = bool(self._mission.done)
        truncated = False
        gym_info = {k: _to_python(v) for k, v in info.items()}
        return _obs_to_numpy(obs), reward, terminated, truncated, gym_info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array" and self._mission is not None:
            # Render a simple matplotlib top-down view
            return _render_topdown(self.config, self._mission)
        return None

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _obs_to_numpy(obs: Observation) -> dict:
    return {
        "drone_features":  np.array(obs.drone_features,  dtype=np.float32),
        "victim_features": np.array(obs.victim_features, dtype=np.float32),
        "team_features":   np.array(obs.team_features,   dtype=np.float32),
        "coverage_map":    np.array(obs.coverage_map,    dtype=np.float32),
        "action_mask":     np.array(obs.action_mask,     dtype=np.float32),
    }


def _to_python(v: Any) -> Any:
    if hasattr(v, "tolist"):
        return v.tolist()
    if hasattr(v, "item"):
        return v.item()
    return v


def _render_topdown(config: AvalancheConfig, mission: MissionState) -> np.ndarray:
    """Minimal matplotlib render returning an RGB array."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=80)
    ax.set_xlim(0, config.map_size_x)
    ax.set_ylim(0, config.map_size_y)

    scanned = np.array(mission.scanned_cells, dtype=bool)
    if scanned.any():
        yi, xi = scanned.nonzero()
        xs = (xi + 0.5) * config.map_size_x / config.coverage_resolution_x
        ys = (yi + 0.5) * config.map_size_y / config.coverage_resolution_y
        ax.scatter(xs, ys, s=12, c="#80ed99", alpha=0.5, marker="s", linewidths=0)

    debris = np.array(mission.debris_mask, dtype=bool)
    if debris.any():
        yi, xi = debris.nonzero()
        xs = (xi + 0.5) * config.map_size_x / config.coverage_resolution_x
        ys = (yi + 0.5) * config.map_size_y / config.coverage_resolution_y
        ax.scatter(xs, ys, s=6, c="#f2e8c9", alpha=0.5, marker="s", linewidths=0)

    vpos = np.array(mission.victim_positions[:, :2])
    aided = np.array(mission.victim_aided, dtype=bool)
    found = np.array(mission.victim_found, dtype=bool)
    colors = ["#16a34a" if aided[i] else "#f97316" if found[i] else "#dc2626"
              for i in range(config.num_victims)]
    ax.scatter(vpos[:, 0], vpos[:, 1], c=colors, s=60, marker="x", linewidths=2)

    ax.set_title(f"t={int(mission.time)}  coverage={float(mission.scanned_cells.mean()):.1%}"
                 f"  aided={int(mission.victim_aided.sum())}/{config.num_victims}")
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return buf[:, :, :3]
