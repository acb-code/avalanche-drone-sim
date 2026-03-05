from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from .types import EnvConfig, EnvState


def _render_frame(
    config: EnvConfig,
    state: EnvState,
    title: str,
) -> Image.Image:
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.imshow(
        state.terrain_height,
        origin="lower",
        extent=(0.0, config.map_size_x, 0.0, config.map_size_y),
        cmap="gist_earth",
        alpha=0.9,
    )

    debris = state.debris_mask.astype(bool)
    if bool(debris.any()):
        y_idx, x_idx = debris.nonzero()
        x = (x_idx + 0.5) * config.map_size_x / config.coverage_resolution_x
        y = (y_idx + 0.5) * config.map_size_y / config.coverage_resolution_y
        ax.scatter(x, y, s=6, c="#f2e8c9", alpha=0.55, marker="s", linewidths=0)

    scanned = state.scanned_cells.astype(bool)
    if bool(scanned.any()):
        y_idx, x_idx = scanned.nonzero()
        x = (x_idx + 0.5) * config.map_size_x / config.coverage_resolution_x
        y = (y_idx + 0.5) * config.map_size_y / config.coverage_resolution_y
        ax.scatter(x, y, s=12, c="#80ed99", alpha=0.5, marker="s", linewidths=0)

    drone_xy = state.drone_positions[:, :2]
    ax.scatter(drone_xy[:, 0], drone_xy[:, 1], c="#0f172a", s=55, label="drones")

    victim_xy = state.victim_positions[:, :2]
    victim_colors = ["#16a34a" if aided else "#dc2626" for aided in state.victim_aided.tolist()]
    ax.scatter(victim_xy[:, 0], victim_xy[:, 1], c=victim_colors, s=45, marker="x", linewidths=2, label="victims")

    coverage = float(state.metrics.get("coverage", state.scanned_cells.mean()))
    status = f"t={int(state.time)} coverage={coverage:.2%} aided={int(state.victim_aided.sum())}/{state.victim_aided.shape[0]}"
    ax.set_title(f"{title}\n{status}")
    ax.set_xlabel("map x")
    ax.set_ylabel("map y")
    ax.add_patch(
        Rectangle((0.0, 0.0), config.map_size_x, config.map_size_y, fill=False, lw=1.2, ec="#0f172a")
    )
    ax.legend(loc="upper right")

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = Image.frombuffer(
        "RGBA",
        (width, height),
        fig.canvas.buffer_rgba(),
        "raw",
        "RGBA",
        0,
        1,
    ).copy()
    plt.close(fig)
    return image


def save_overview(
    config: EnvConfig,
    state: EnvState,
    output_path: str | Path,
    title: str = "Avalanche Drone Simulation",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    image = _render_frame(config, state, title)
    image.save(output)
    return output


def save_rollout_gif(
    config: EnvConfig,
    states: list[EnvState],
    output_path: str | Path,
    title: str = "Avalanche Drone Simulation",
    fps: int = 5,
) -> Path:
    if not states:
        raise ValueError("states must contain at least one frame")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame_ms = max(40, int(1000 / max(1, fps)))
    frames = [_render_frame(config, state, title).convert("P", palette=Image.Palette.ADAPTIVE) for state in states]
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
        optimize=False,
    )
    return output
