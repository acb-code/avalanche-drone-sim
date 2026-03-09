"""Wrapper conformance tests.

gymnasium_wrapper: check_env() (requires gymnasium)
pettingzoo_wrapper: parallel_api_test() (requires pettingzoo)

Both skip gracefully when their deps are not installed.
"""
from __future__ import annotations

import numpy as np
import pytest

from avalanche_mujoco.config import AvalancheConfig


# Minimal config for fast testing
_CFG = AvalancheConfig(num_drones=2, num_victims=3, horizon=10)


# ──────────────────────────────────────────────────────────────────────────────
# Gymnasium wrapper
# ──────────────────────────────────────────────────────────────────────────────

gymnasium = pytest.importorskip("gymnasium", reason="gymnasium not installed")
mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")


class TestGymnasiumWrapper:
    def _make_env(self):
        from avalanche_mujoco.wrappers import AvalancheGymnasiumEnv
        return AvalancheGymnasiumEnv(config=_CFG, backend="cpu")

    def test_spaces_defined(self):
        env = self._make_env()
        assert env.observation_space is not None
        assert env.action_space is not None

    def test_reset_returns_obs_in_space(self):
        env = self._make_env()
        obs, info = env.reset(seed=0)
        assert isinstance(obs, dict)
        assert env.observation_space.contains(obs)

    def test_step_returns_correct_types(self):
        env = self._make_env()
        env.reset(seed=1)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_obs_in_space(self):
        env = self._make_env()
        env.reset(seed=2)
        for _ in range(3):
            obs, _, done, _, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
            if done:
                break

    def test_action_space_shape(self):
        env = self._make_env()
        assert env.action_space.shape == (_CFG.num_drones * 4,)

    def test_episode_terminates(self):
        env = self._make_env()
        env.reset(seed=3)
        done = False
        for _ in range(_CFG.horizon + 5):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                done = True
                break
        assert done

    def test_check_env(self):
        """gymnasium env_checker — most thorough conformance test."""
        from gymnasium.utils.env_checker import check_env
        env = self._make_env()
        check_env(env, warn=True, skip_render_check=True)

    def test_render_rgb_array(self):
        from avalanche_mujoco.wrappers import AvalancheGymnasiumEnv
        env = AvalancheGymnasiumEnv(config=_CFG, backend="cpu", render_mode="rgb_array")
        env.reset(seed=0)
        frame = env.render()
        if frame is not None:
            assert frame.ndim == 3
            assert frame.shape[2] == 3   # RGB

    def test_close(self):
        env = self._make_env()
        env.reset()
        env.close()   # should not raise


# ──────────────────────────────────────────────────────────────────────────────
# PettingZoo wrapper
# ──────────────────────────────────────────────────────────────────────────────

pettingzoo = pytest.importorskip("pettingzoo", reason="pettingzoo not installed")


class TestPettingZooWrapper:
    def _make_env(self):
        from avalanche_mujoco.wrappers import AvalanchePettingZooEnv
        return AvalanchePettingZooEnv(config=_CFG, backend="cpu")

    def test_agents_list(self):
        env = self._make_env()
        assert len(env.possible_agents) == _CFG.num_drones

    def test_reset_populates_agents(self):
        env = self._make_env()
        obs, info = env.reset(seed=0)
        assert len(env.agents) == _CFG.num_drones
        assert set(obs.keys()) == set(env.agents)

    def test_step_returns_correct_keys(self):
        env = self._make_env()
        env.reset(seed=1)
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terminated, truncated, info = env.step(actions)
        assert set(rewards.keys()) == set(env.agents) or set(rewards.keys()) == set()

    def test_obs_in_space(self):
        env = self._make_env()
        obs, _ = env.reset(seed=2)
        for agent in env.agents:
            assert env.observation_space(agent).contains(obs[agent])

    def test_agents_cleared_on_done(self):
        env = self._make_env()
        env.reset(seed=4)
        for _ in range(_CFG.horizon + 10):
            if not env.agents:
                break
            actions = {a: env.action_space(a).sample() for a in env.agents}
            env.step(actions)
        assert env.agents == []

    def test_parallel_api_test(self):
        """pettingzoo parallel_api_test — conformance test."""
        from pettingzoo.test import parallel_api_test
        env = self._make_env()
        parallel_api_test(env, num_cycles=5)
