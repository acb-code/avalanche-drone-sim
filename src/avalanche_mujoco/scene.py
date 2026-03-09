"""Top-level MJCF scene builder.

build_scene(config, spawn_positions, backend) → str

Assembles: terrain (hfield/plane) + N drones + victim markers + lighting.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Literal

import numpy as np

from .config import AvalancheConfig
from .drone import compose_multi_drone, default_spawn_positions, load_base_xml


# ──────────────────────────────────────────────────────────────────────────────
# Scene builder
# ──────────────────────────────────────────────────────────────────────────────

def build_scene(
    config: AvalancheConfig,
    spawn_positions: np.ndarray | None = None,
    spawn_yaws: np.ndarray | None = None,
    backend: Literal["cpu", "mjx"] = "cpu",
) -> tuple[str, list[str], list[str]]:
    """Build a complete MJCF scene string.

    Parameters
    ----------
    config          : AvalancheConfig
    spawn_positions : (n_drones, 3) initial drone world positions.
                      If None, default positions are used.
    spawn_yaws      : (n_drones,) initial yaw angles in radians.
    backend         : "cpu" adds visual hfield (no collision); "mjx" uses plane.

    Returns
    -------
    xml_string   : str  — full MJCF document
    body_names   : list[str]  — MuJoCo body names for each drone
    actuator_names : list[str] — 4*N actuator names
    """
    if spawn_positions is None:
        spawn_positions = default_spawn_positions(
            config.num_drones, config.map_size_x, config.map_size_y
        )
    if spawn_yaws is None:
        spawn_yaws = np.zeros(config.num_drones)

    # ── Root element ───────────────────────────────────────────────────────
    root = ET.Element("mujoco", model="avalanche_sar")

    # ── Compiler ───────────────────────────────────────────────────────────
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")

    # ── Options ────────────────────────────────────────────────────────────
    ET.SubElement(root, "option",
                  timestep=str(config.sim_dt),
                  gravity="0 0 -9.81",
                  integrator="RK4")

    # ── Defaults ───────────────────────────────────────────────────────────
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "motor", ctrlrange="0 10", ctrllimited="true")
    ET.SubElement(default, "geom", contype="0", conaffinity="0")

    # ── Assets ────────────────────────────────────────────────────────────
    asset = ET.SubElement(root, "asset")
    # Gradient sky texture
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient",
                  rgb1="0.53 0.8 0.92", rgb2="0.1 0.2 0.35",
                  width="512", height="512")
    # Ground plane texture
    ET.SubElement(asset, "texture", name="groundplane", type="2d", builtin="checker",
                  rgb1="0.45 0.42 0.38", rgb2="0.38 0.35 0.30",
                  width="512", height="512", mark="cross",
                  markrgb="0.7 0.7 0.7")
    ET.SubElement(asset, "material", name="groundplane", texture="groundplane",
                  texrepeat="16 16", reflectance="0.1")
    # Victim sphere material (red = unhelped, green = helped)
    ET.SubElement(asset, "material", name="victim_unhelped", rgba="0.85 0.1 0.1 0.9")
    ET.SubElement(asset, "material", name="victim_helped",   rgba="0.1 0.75 0.2 0.9")

    # ── Worldbody ─────────────────────────────────────────────────────────
    worldbody = ET.SubElement(root, "worldbody")

    # Lighting
    ET.SubElement(worldbody, "light", name="sun",
                  pos="0 0 200", dir="0.3 -0.5 -1",
                  diffuse="1 1 0.95", specular="0.2 0.2 0.2",
                  castshadow="true")
    ET.SubElement(worldbody, "light", name="fill",
                  pos="100 -100 150", dir="-0.3 0.5 -1",
                  diffuse="0.4 0.4 0.5", specular="0 0 0",
                  castshadow="false")

    # Ground / terrain
    cx = config.map_size_x / 2
    cy = config.map_size_y / 2
    if backend == "cpu":
        # Flat visual ground plane centred on the map
        ET.SubElement(worldbody, "geom",
                      name="ground", type="plane",
                      pos=f"{cx:.1f} {cy:.1f} 0",
                      size=f"{config.map_size_x/2:.1f} {config.map_size_y/2:.1f} 0.1",
                      material="groundplane",
                      contype="1", conaffinity="1")
    else:
        # MJX mode: flat plane + virtual terrain enforcement in mjx_env.py
        ET.SubElement(worldbody, "geom",
                      name="ground", type="plane",
                      pos=f"{cx:.1f} {cy:.1f} 0",
                      size=f"{config.map_size_x/2:.1f} {config.map_size_y/2:.1f} 0.1",
                      rgba="0.45 0.42 0.38 1",
                      contype="1", conaffinity="1")

    # Map boundary walls (visual only)
    wall_h = "25"
    wall_t = "0.5"
    for pos, size in [
        (f"{cx:.1f} 0 {wall_h}",            f"{cx:.1f} {wall_t} {wall_h}"),
        (f"{cx:.1f} {config.map_size_y:.1f} {wall_h}", f"{cx:.1f} {wall_t} {wall_h}"),
        (f"0 {cy:.1f} {wall_h}",            f"{wall_t} {cy:.1f} {wall_h}"),
        (f"{config.map_size_x:.1f} {cy:.1f} {wall_h}", f"{wall_t} {cy:.1f} {wall_h}"),
    ]:
        ET.SubElement(worldbody, "geom", type="box", pos=pos, size=size,
                      rgba="0.3 0.3 0.3 0.15", contype="0", conaffinity="0")

    # Victim markers (initial positions unknown at scene-build time →
    # set to map origin; runtime code updates xpos via mocap or geom_rgba)
    for v in range(config.num_victims):
        victim_body = ET.SubElement(worldbody, "body", name=f"victim_{v}",
                                    pos=f"{config.map_size_x*0.5:.1f} {config.map_size_y*0.5:.1f} 1.0",
                                    mocap="true")
        ET.SubElement(victim_body, "geom", name=f"victim_geom_{v}",
                      type="sphere", size="1.2",
                      material="victim_unhelped",
                      contype="0", conaffinity="0")

    # Camera for top-down overview
    ET.SubElement(worldbody, "camera", name="overview",
                  pos=f"{cx:.1f} {config.map_size_y*0.3:.1f} 180",
                  xyaxes="1 0 0 0 0.3 1")

    # ── Drones (injected into worldbody as sub-XML) ────────────────────────
    base_xml = load_base_xml()
    wb_xml, body_names, actuator_names = compose_multi_drone(
        base_xml, config.num_drones, spawn_positions, spawn_yaws
    )

    # Parse the drone worldbody fragment and graft children into our worldbody
    drone_wb = ET.fromstring(wb_xml)
    for child in drone_wb:
        worldbody.append(child)

    # ── Actuators ─────────────────────────────────────────────────────────
    actuator_el = ET.SubElement(root, "actuator")
    # Re-compose actuators from the drone XML (with prefixes)
    _, _, _ = _inject_actuators(config, base_xml, actuator_el)

    # ── Serialise ─────────────────────────────────────────────────────────
    ET.indent(root, space="  ")
    xml_string = '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")
    return xml_string, body_names, actuator_names


def _inject_actuators(
    config: AvalancheConfig,
    base_xml: str,
    actuator_el: ET.Element,
) -> tuple[ET.Element, list[str], list[str]]:
    """Copy prefixed actuators into the parent actuator element."""
    import copy
    import xml.etree.ElementTree as ET2
    tree = ET2.fromstring(base_xml)
    orig_act = tree.find("actuator")
    body_names_out: list[str] = []
    actuator_names_out: list[str] = []

    for i in range(config.num_drones):
        prefix = f"drone{i}/"
        if orig_act is not None:
            act_copy = copy.deepcopy(orig_act)
            for motor in act_copy:
                motor_copy = copy.deepcopy(motor)
                for attr in ("name", "site", "joint"):
                    val = motor_copy.get(attr)
                    if val is not None:
                        motor_copy.set(attr, f"{prefix}{val}")
                actuator_el.append(motor_copy)
                actuator_names_out.append(motor_copy.get("name", ""))

    return actuator_el, body_names_out, actuator_names_out
