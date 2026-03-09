"""PettingZoo ParallelEnv wrapper — MARL interface.

Each drone is an independent agent: possible_agents = ["drone_0", ..., "drone_N"].
All agents act simultaneously (ParallelEnv, not AEC).

Designed to pass pettingzoo.test.parallel_api_test().
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

try:
    from pettingzoo import ParallelEnv
    _PZ_AVAILABLE = True
    _PZBase = ParallelEnv
except ImportError:
    _PZ_AVAILABLE = False
    _PZBase = object

try:
    import gymnasium
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False

from ..config import AvalancheConfig
from ..types import MissionState, Observation


class AvalanchePettingZooEnv(_PZBase):
    """PettingZoo ParallelEnv for multi-agent SAR.

    Each agent (drone) receives its own slice of the observation and
    issues its own 4-dimensional action.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "avalanche_sar_v1"}

    def __init__(
        self,
        config: AvalancheConfig | None = None,
        backend: str = "cpu",
        render_mode: str | None = None,
    ):
        if not _PZ_AVAILABLE:
            raise ImportError("pettingzoo is required. Install with: pip install pettingzoo>=1.24")
        if not _GYM_AVAILABLE:
            raise ImportError("gymnasium is required. Install with: pip install gymnasium>=0.29")
        self.config = config or AvalancheConfig()
        self.render_mode = render_mode
        self.possible_agents = [f"drone_{i}" for i in range(self.config.num_drones)]
        self.agents: list[str] = []
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
        self._last_obs: dict[str, dict] | None = None

    # ──────────────────────────────────────────────────────────────────────
    # Space definitions
    # ──────────────────────────────────────────────────────────────────────

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Dict:
        cfg = self.config
        # Note: action_mask is omitted here because pettingzoo.test.sample_action
        # would interpret it as discrete action indices. Battery state is already
        # encoded in drone_features[6] (battery/base_battery).
        return spaces.Dict({
            "drone_features": spaces.Box(low=-2.0, high=2.0, shape=(8,), dtype=np.float32),
            "victim_features": spaces.Box(low=0.0, high=1.0, shape=(cfg.num_victims, 5), dtype=np.float32),
            "team_features": spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32),
            "coverage_map": spaces.Box(
                low=0.0, high=1.0,
                shape=(cfg.coverage_resolution_y, cfg.coverage_resolution_x), dtype=np.float32
            ),
        })

    @lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # PettingZoo ParallelEnv API
    # ──────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        if seed is not None:
            self._rng = jax.random.PRNGKey(seed)
        self._rng, key = jax.random.split(self._rng)
        self.agents = list(self.possible_agents)

        if self._backend == "cpu":
            obs, self._mission = self._env.reset(key)
        else:
            self._state = self._env.reset(key)
            self._mission = self._state.mission
            obs = self._env.get_obs(self._state)

        self._last_obs = _split_obs(obs, self.agents)
        return self._last_obs, {a: {} for a in self.agents}

    def step(
        self,
        actions: dict[str, np.ndarray],
    ) -> tuple[dict, dict, dict, dict, dict]:
        """Step all agents simultaneously."""
        # Stack actions in agent order
        action_arr = jnp.stack([
            jnp.array(actions.get(a, np.zeros(4)), dtype=jnp.float32)
            for a in self.agents
        ], axis=0)   # (num_drones, 4)

        self._rng, key = jax.random.split(self._rng)

        if self._backend == "cpu":
            obs, self._mission, rewards, dones, info = self._env.step(key, self._mission, action_arr)
        else:
            self._state, rewards, info = self._env.step(self._state, action_arr, key)
            self._mission = self._state.mission
            obs = self._env.get_obs(self._state)

        obs_dict = _split_obs(obs, self.agents)
        rewards_np = np.array(rewards)
        done = bool(self._mission.done)

        reward_dict = {a: float(rewards_np[i]) for i, a in enumerate(self.agents)}
        terminated_dict = {a: done for a in self.agents}
        truncated_dict = {a: False for a in self.agents}
        info_dict = {a: {k: _to_python(v) for k, v in info.items()} for a in self.agents}

        if done:
            self.agents = []

        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array" and self._mission is not None:
            from .gymnasium_wrapper import _render_topdown
            return _render_topdown(self.config, self._mission)
        return None

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    def state(self) -> np.ndarray:
        """Global state (concatenated observations) for global critic."""
        if self._last_obs is None:
            return np.zeros(self.config.num_drones * 8, dtype=np.float32)
        return np.concatenate([
            self._last_obs[a]["drone_features"] for a in self.possible_agents
        ], axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _split_obs(obs: Observation, agents: list[str]) -> dict[str, dict]:
    """Split a batch Observation into per-agent dicts (no action_mask for PZ)."""
    result: dict[str, dict] = {}
    for i, agent in enumerate(agents):
        result[agent] = {
            "drone_features":  np.array(obs.drone_features[i],  dtype=np.float32),
            "victim_features": np.array(obs.victim_features[i], dtype=np.float32),
            "team_features":   np.array(obs.team_features[i],   dtype=np.float32),
            "coverage_map":    np.array(obs.coverage_map[i],    dtype=np.float32),
        }
    return result


def _to_python(v: Any) -> Any:
    if hasattr(v, "tolist"):
        return v.tolist()
    if hasattr(v, "item"):
        return v.item()
    return v
