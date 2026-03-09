"""avalanche_mujoco — MuJoCo/MJX-backed multi-drone SAR simulator.

Quick start
-----------
    from avalanche_mujoco import AvalancheConfig, make_physics_env

    cfg = AvalancheConfig(num_drones=4)
    env = make_physics_env(cfg)
    obs, mission = env.reset(jax.random.PRNGKey(0))
"""
from .config import AvalancheConfig
from .types import MissionState, MJXBatchState, Observation
from .physics_env import AvalanchePhysicsEnv, make_physics_env
from .terrain_mesh import generate_scene, terrain_height_at

__all__ = [
    "AvalancheConfig",
    "MissionState",
    "MJXBatchState",
    "Observation",
    "AvalanchePhysicsEnv",
    "make_physics_env",
    "generate_scene",
    "terrain_height_at",
]

# MJX env — optional, only importable if mujoco-mjx is installed
try:
    from .mjx_env import AvalancheMJXEnv, make_mjx_env
    __all__ += ["AvalancheMJXEnv", "make_mjx_env"]
except ImportError:
    pass
