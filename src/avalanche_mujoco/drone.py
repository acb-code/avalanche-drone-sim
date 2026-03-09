"""Multi-drone MJCF scene composition.

compose_multi_drone(base_xml, n_drones, spawn_positions) → MJCF XML string

Each drone is prefixed with "droneN/" so names remain unique.
The base model (models/quadrotor.xml) contains a single drone.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# Path to the bundled single-drone MJCF model
_MODEL_DIR = Path(__file__).parent / "models"
QUADROTOR_XML = _MODEL_DIR / "quadrotor.xml"


def _prefix_attr(element: ET.Element, attr: str, prefix: str) -> None:
    """Add prefix to a named attribute if it exists."""
    val = element.get(attr)
    if val is not None:
        element.set(attr, f"{prefix}{val}")


def _prefix_element(element: ET.Element, prefix: str) -> None:
    """Recursively prefix all name/joint/site/geom/actuator name attributes."""
    for attr in ("name", "joint", "site", "geom", "body"):
        _prefix_attr(element, attr, prefix)
    for child in element:
        _prefix_element(child, prefix)


def load_base_xml() -> str:
    """Return the base single-drone MJCF as a string."""
    return QUADROTOR_XML.read_text()


def compose_multi_drone(
    base_xml: str,
    n_drones: int,
    spawn_positions: np.ndarray,    # (n_drones, 3) world XYZ metres
    spawn_yaws: np.ndarray | None = None,  # (n_drones,) radians, default 0
) -> tuple[str, list[str], list[str]]:
    """Compose N drones from the base single-drone MJCF.

    Returns
    -------
    worldbody_xml : str  — <worldbody> XML fragment with N drones
    body_names    : list[str]  — e.g. ["drone0/drone", "drone1/drone", ...]
    actuator_names: list[str]  — 4*N actuator names in order
                                 [drone0/motor_fr, drone0/motor_fl, ...]
    """
    if spawn_yaws is None:
        spawn_yaws = np.zeros(n_drones)

    tree = ET.fromstring(base_xml)

    worldbody_out = ET.Element("worldbody")
    actuators_out: list[ET.Element] = []
    body_names: list[str] = []
    actuator_names: list[str] = []

    for i in range(n_drones):
        prefix = f"drone{i}/"

        # ── Copy and prefix the drone body ────────────────────────────────
        # Find the top-level body inside worldbody
        orig_wb = tree.find("worldbody")
        if orig_wb is None:
            raise ValueError("Base XML missing <worldbody>")
        orig_body = orig_wb.find("body")
        if orig_body is None:
            raise ValueError("Base XML worldbody missing <body>")

        import copy
        body = copy.deepcopy(orig_body)
        _prefix_element(body, prefix)

        # Set initial position and orientation
        px, py, pz = spawn_positions[i]
        yaw = float(spawn_yaws[i])
        body.set("pos", f"{px:.4f} {py:.4f} {pz:.4f}")
        # Encode yaw as quaternion [w, x, y, z] for z-axis rotation
        hw = float(np.cos(yaw / 2.0))
        hz = float(np.sin(yaw / 2.0))
        body.set("quat", f"{hw:.6f} 0 0 {hz:.6f}")

        worldbody_out.append(body)
        body_names.append(f"{prefix}drone")

        # ── Copy and prefix actuators ─────────────────────────────────────
        orig_act = tree.find("actuator")
        if orig_act is not None:
            import copy as _copy
            act_block = _copy.deepcopy(orig_act)
            _prefix_element(act_block, prefix)
            for motor in act_block:
                actuators_out.append(motor)
                actuator_names.append(motor.get("name", ""))

    # Serialise
    ET.indent(worldbody_out, space="  ")
    worldbody_xml = ET.tostring(worldbody_out, encoding="unicode")

    return worldbody_xml, body_names, actuator_names


def default_spawn_positions(
    n_drones: int,
    map_size_x: float,
    map_size_y: float,
    altitude: float = 15.0,
) -> np.ndarray:
    """Evenly-spaced spawn positions along the southern edge of the map."""
    xs = np.linspace(map_size_x * 0.15, map_size_x * 0.85, n_drones)
    ys = np.full(n_drones, map_size_y * 0.92)
    zs = np.linspace(altitude, altitude + 5.0, n_drones)
    return np.stack([xs, ys, zs], axis=-1)
