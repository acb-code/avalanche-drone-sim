from __future__ import annotations

import json
from pathlib import Path

from .types import EnvConfig, EnvState


def rollout_to_dict(
    config: EnvConfig,
    states: list[EnvState],
    title: str = "Avalanche Drone Simulation",
) -> dict:
    if not states:
        raise ValueError("states must contain at least one frame")

    first_state = states[0]
    frames = []
    for state in states:
        metrics = {
            "coverage": float(state.metrics.get("coverage", state.scanned_cells.mean())),
            "find_events": int(state.metrics.get("find_events", state.victim_found.sum())),
            "confirm_events": int(state.metrics.get("confirm_events", state.victim_confirmed.sum())),
            "delivery_events": int(state.metrics.get("delivery_events", state.victim_aided.sum())),
        }
        frames.append(
            {
                "time": int(state.time),
                "drone_positions": state.drone_positions.tolist(),
                "drone_heading": state.drone_heading.tolist(),
                "drone_battery": state.drone_battery.tolist(),
                "drone_payload": state.drone_payload.tolist(),
                "victim_found": state.victim_found.tolist(),
                "victim_confirmed": state.victim_confirmed.tolist(),
                "victim_aided": state.victim_aided.tolist(),
                "victim_survival": state.victim_survival.tolist(),
                "shared_known_victims": state.shared_known_victims.tolist(),
                "scanned_cells": state.scanned_cells.astype(bool).tolist(),
                "metrics": metrics,
            }
        )

    return {
        "title": title,
        "config": {
            "num_drones": config.num_drones,
            "num_victims": config.num_victims,
            "map_size_x": config.map_size_x,
            "map_size_y": config.map_size_y,
            "altitude_min": config.altitude_min,
            "altitude_max": config.altitude_max,
            "sensor_range": config.sensor_range,
            "sensor_fov_cos": config.sensor_fov_cos,
            "coverage_resolution_x": config.coverage_resolution_x,
            "coverage_resolution_y": config.coverage_resolution_y,
        },
        "terrain_height": first_state.terrain_height.tolist(),
        "debris_mask": first_state.debris_mask.astype(bool).tolist(),
        "unsafe_mask": first_state.unsafe_mask.astype(bool).tolist(),
        "victim_positions": first_state.victim_positions.tolist(),
        "victim_severity": first_state.victim_severity.tolist(),
        "frames": frames,
    }


def export_rollout_data(
    config: EnvConfig,
    states: list[EnvState],
    output_path: str | Path,
    title: str = "Avalanche Drone Simulation",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rollout_to_dict(config, states, title)), encoding="utf-8")
    return output


def save_interactive_rollout(
    config: EnvConfig,
    states: list[EnvState],
    output_path: str | Path,
    title: str = "Avalanche Drone Simulation",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rollout_to_dict(config, states, title))
    output.write_text(_build_viewer_html(payload), encoding="utf-8")
    return output


def _build_viewer_html(payload: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Avalanche Drone Viewer</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f1e8;
      --panel: rgba(255, 250, 242, 0.92);
      --ink: #172226;
      --accent: #b44f2c;
      --accent-2: #235789;
      --line: rgba(23, 34, 38, 0.14);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(180, 79, 44, 0.18), transparent 28%),
        linear-gradient(180deg, #f7f4ec 0%, #ebe2d3 100%);
      color: var(--ink);
    }}
    .layout {{
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(260px, 320px) 1fr;
    }}
    .panel {{
      padding: 24px 20px;
      background: var(--panel);
      backdrop-filter: blur(16px);
      border-right: 1px solid var(--line);
      display: flex;
      flex-direction: column;
      gap: 18px;
    }}
    h1 {{
      margin: 0;
      font-size: 1.45rem;
      line-height: 1.1;
    }}
    p {{
      margin: 0;
      color: rgba(23, 34, 38, 0.78);
      line-height: 1.45;
    }}
    .controls, .stats {{
      display: grid;
      gap: 10px;
    }}
    .row {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    button, select {{
      border: 0;
      border-radius: 999px;
      padding: 10px 14px;
      background: var(--ink);
      color: white;
      font: inherit;
      cursor: pointer;
    }}
    button.secondary {{
      background: #d8d0c1;
      color: var(--ink);
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: var(--accent);
    }}
    .stat {{
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255,255,255,0.62);
    }}
    .legend {{
      display: grid;
      gap: 8px;
      font-size: 0.92rem;
    }}
    .swatch {{
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 999px;
      margin-right: 8px;
    }}
    .viewer {{
      position: relative;
      overflow: hidden;
    }}
    canvas {{
      width: 100%;
      height: 100%;
      display: block;
      cursor: grab;
    }}
    canvas.dragging {{
      cursor: grabbing;
    }}
    .hud {{
      position: absolute;
      right: 18px;
      bottom: 18px;
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(255, 250, 242, 0.88);
      border: 1px solid var(--line);
      font-size: 0.9rem;
    }}
    @media (max-width: 900px) {{
      .layout {{
        grid-template-columns: 1fr;
        grid-template-rows: auto 1fr;
      }}
      .panel {{
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="panel">
      <div>
        <h1 id="title"></h1>
        <p>Interactive 3D playback for avalanche rescue rollouts. Drag to orbit, wheel to zoom, and scrub through time.</p>
      </div>
      <div class="controls">
        <div class="row">
          <button id="play">Play</button>
          <button id="reset" class="secondary">Reset View</button>
          <select id="speed">
            <option value="1">1x speed</option>
            <option value="2">2x speed</option>
            <option value="4">4x speed</option>
          </select>
        </div>
        <label for="frameRange">Frame</label>
        <input id="frameRange" type="range" min="0" max="0" value="0">
      </div>
      <div class="stats" id="stats"></div>
      <div class="legend">
        <div><span class="swatch" style="background:#1d3557"></span>Drones</div>
        <div><span class="swatch" style="background:rgba(35,87,137,0.35)"></span>Sensor cone</div>
        <div><span class="swatch" style="background:#d62828"></span>Victim not found</div>
        <div><span class="swatch" style="background:#f77f00"></span>Found / not confirmed</div>
        <div><span class="swatch" style="background:#fcbf49"></span>Confirmed / awaiting aid</div>
        <div><span class="swatch" style="background:#2a9d8f"></span>Aided victim</div>
      </div>
    </aside>
    <main class="viewer">
      <canvas id="canvas"></canvas>
      <div class="hud" id="hud"></div>
    </main>
  </div>
  <script>
    const payload = {payload};
    const titleEl = document.getElementById("title");
    const frameRange = document.getElementById("frameRange");
    const playButton = document.getElementById("play");
    const resetButton = document.getElementById("reset");
    const speedSelect = document.getElementById("speed");
    const statsEl = document.getElementById("stats");
    const hudEl = document.getElementById("hud");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    titleEl.textContent = payload.title;
    frameRange.max = String(payload.frames.length - 1);

    const camera = {{
      yaw: -0.9,
      pitch: 0.85,
      distance: Math.max(payload.config.map_size_x, payload.config.map_size_y) * 2.1,
      target: [payload.config.map_size_x / 2, payload.config.map_size_y / 2, 10],
    }};
    let frameIndex = 0;
    let playing = false;
    let lastTick = 0;
    let dragState = null;

    function resize() {{
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.floor(canvas.clientWidth * ratio);
      canvas.height = Math.floor(canvas.clientHeight * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }}

    function project(point) {{
      const dx = point[0] - camera.target[0];
      const dy = point[1] - camera.target[1];
      const dz = point[2] - camera.target[2];

      const cosYaw = Math.cos(camera.yaw);
      const sinYaw = Math.sin(camera.yaw);
      const x1 = cosYaw * dx - sinYaw * dy;
      const y1 = sinYaw * dx + cosYaw * dy;

      const cosPitch = Math.cos(camera.pitch);
      const sinPitch = Math.sin(camera.pitch);
      const y2 = cosPitch * y1 - sinPitch * dz;
      const z2 = sinPitch * y1 + cosPitch * dz;

      const focal = Math.min(canvas.clientWidth, canvas.clientHeight) * 0.78;
      const scale = focal / (camera.distance + z2 + 1e-6);
      return {{
        x: canvas.clientWidth / 2 + x1 * scale,
        y: canvas.clientHeight / 2 - y2 * scale,
        scale,
        depth: z2,
      }};
    }}

    function terrainPoint(ix, iy) {{
      const z = payload.terrain_height[iy][ix];
      return [
        (ix / (payload.config.coverage_resolution_x - 1)) * payload.config.map_size_x,
        (iy / (payload.config.coverage_resolution_y - 1)) * payload.config.map_size_y,
        z,
      ];
    }}

    function victimColor(frame, victimIndex) {{
      if (frame.victim_aided[victimIndex]) return "#2a9d8f";
      if (frame.victim_confirmed[victimIndex]) return "#fcbf49";
      if (frame.victim_found[victimIndex]) return "#f77f00";
      return "#d62828";
    }}

    function drawTerrain() {{
      const stepX = Math.max(1, Math.floor(payload.config.coverage_resolution_x / 12));
      const stepY = Math.max(1, Math.floor(payload.config.coverage_resolution_y / 10));
      ctx.lineWidth = 1;
      for (let y = 0; y < payload.config.coverage_resolution_y; y += stepY) {{
        ctx.beginPath();
        for (let x = 0; x < payload.config.coverage_resolution_x; x += stepX) {{
          const point = terrainPoint(x, y);
          const projected = project(point);
          if (x === 0) ctx.moveTo(projected.x, projected.y);
          else ctx.lineTo(projected.x, projected.y);
        }}
        ctx.strokeStyle = "rgba(55, 80, 66, 0.34)";
        ctx.stroke();
      }}
      for (let x = 0; x < payload.config.coverage_resolution_x; x += stepX) {{
        ctx.beginPath();
        for (let y = 0; y < payload.config.coverage_resolution_y; y += stepY) {{
          const point = terrainPoint(x, y);
          const projected = project(point);
          if (y === 0) ctx.moveTo(projected.x, projected.y);
          else ctx.lineTo(projected.x, projected.y);
        }}
        ctx.strokeStyle = "rgba(80, 103, 87, 0.26)";
        ctx.stroke();
      }}
    }}

    function drawOverlayCells(mask, color, size) {{
      const cells = [];
      for (let y = 0; y < mask.length; y += 1) {{
        for (let x = 0; x < mask[y].length; x += 1) {{
          if (!mask[y][x]) continue;
          const point = terrainPoint(x, y);
          cells.push(project([point[0], point[1], point[2] + 0.6]));
        }}
      }}
      ctx.fillStyle = color;
      for (const cell of cells) {{
        ctx.fillRect(cell.x - size / 2, cell.y - size / 2, size, size);
      }}
    }}

    function terrainHeightAt(x, y) {{
      const cx = Math.max(0, Math.min(
        payload.config.coverage_resolution_x - 1,
        Math.floor((x / payload.config.map_size_x) * payload.config.coverage_resolution_x)
      ));
      const cy = Math.max(0, Math.min(
        payload.config.coverage_resolution_y - 1,
        Math.floor((y / payload.config.map_size_y) * payload.config.coverage_resolution_y)
      ));
      return payload.terrain_height[cy][cx];
    }}

    function drawSensors(frame) {{
      const sensorHalfAngle = Math.acos(payload.config.sensor_fov_cos);
      for (let i = 0; i < frame.drone_positions.length; i += 1) {{
        const origin = frame.drone_positions[i];
        const heading = frame.drone_heading[i];
        const points = [[origin[0], origin[1], terrainHeightAt(origin[0], origin[1]) + 0.8]];
        const arcSteps = 12;
        for (let step = 0; step <= arcSteps; step += 1) {{
          const t = step / arcSteps;
          const angle = heading - sensorHalfAngle + t * sensorHalfAngle * 2;
          const x = Math.max(0, Math.min(payload.config.map_size_x, origin[0] + Math.cos(angle) * payload.config.sensor_range));
          const y = Math.max(0, Math.min(payload.config.map_size_y, origin[1] + Math.sin(angle) * payload.config.sensor_range));
          points.push([x, y, terrainHeightAt(x, y) + 0.8]);
        }}
        ctx.beginPath();
        points.forEach((point, index) => {{
          const projected = project(point);
          if (index === 0) ctx.moveTo(projected.x, projected.y);
          else ctx.lineTo(projected.x, projected.y);
        }});
        ctx.closePath();
        ctx.fillStyle = "rgba(35, 87, 137, 0.15)";
        ctx.fill();
        ctx.strokeStyle = "rgba(35, 87, 137, 0.38)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }}
    }}

    function drawActors(frame) {{
      const drawable = [];
      for (let i = 0; i < payload.victim_positions.length; i += 1) {{
        const point = payload.victim_positions[i];
        const projected = project(point);
        drawable.push({{
          type: "victim",
          depth: projected.depth,
          color: victimColor(frame, i),
          point: projected,
          label: `V${{i + 1}}`,
        }});
      }}
      for (let i = 0; i < frame.drone_positions.length; i += 1) {{
        const point = frame.drone_positions[i];
        const projected = project(point);
        drawable.push({{
          type: "drone",
          depth: projected.depth,
          color: "#1d3557",
          point: projected,
          label: `D${{i + 1}}`,
        }});
      }}
      drawable.sort((a, b) => a.depth - b.depth);
      for (const item of drawable) {{
        ctx.beginPath();
        const radius = item.type === "drone" ? 7 : 6;
        ctx.fillStyle = item.color;
        ctx.arc(item.point.x, item.point.y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "rgba(0,0,0,0.72)";
        ctx.font = "12px IBM Plex Sans, sans-serif";
        ctx.fillText(item.label, item.point.x + 8, item.point.y - 8);
      }}
    }}

    function updateStats(frame) {{
      const metrics = frame.metrics;
      statsEl.innerHTML = `
        <div class="stat">Frame ${{frameIndex + 1}} / ${{payload.frames.length}}</div>
        <div class="stat">Coverage ${{(metrics.coverage * 100).toFixed(1)}}%</div>
        <div class="stat">Found ${{frame.victim_found.filter(Boolean).length}} / ${{payload.config.num_victims}}</div>
        <div class="stat">Confirmed ${{frame.victim_confirmed.filter(Boolean).length}} / ${{payload.config.num_victims}}</div>
        <div class="stat">Aided ${{frame.victim_aided.filter(Boolean).length}} / ${{payload.config.num_victims}}</div>
      `;
      hudEl.textContent = `t=${{frame.time}}  drones=${{payload.config.num_drones}}  avg battery=${{average(frame.drone_battery).toFixed(1)}}`;
    }}

    function average(values) {{
      return values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
    }}

    function draw() {{
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      ctx.fillStyle = "#f1e8da";
      ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      drawTerrain();
      const frame = payload.frames[frameIndex];
      drawOverlayCells(payload.debris_mask, "rgba(120, 90, 60, 0.26)", 3);
      drawOverlayCells(frame.scanned_cells, "rgba(42, 157, 143, 0.22)", 4);
      drawSensors(frame);
      drawActors(frame);
      updateStats(frame);
      frameRange.value = String(frameIndex);
    }}

    function advanceFrame(timestamp) {{
      if (!playing) return;
      if (!lastTick) lastTick = timestamp;
      const speed = Number(speedSelect.value);
      if (timestamp - lastTick > 320 / speed) {{
        frameIndex = (frameIndex + 1) % payload.frames.length;
        lastTick = timestamp;
        draw();
      }}
      requestAnimationFrame(advanceFrame);
    }}

    playButton.addEventListener("click", () => {{
      playing = !playing;
      playButton.textContent = playing ? "Pause" : "Play";
      lastTick = 0;
      if (playing) requestAnimationFrame(advanceFrame);
    }});

    frameRange.addEventListener("input", (event) => {{
      frameIndex = Number(event.target.value);
      draw();
    }});

    resetButton.addEventListener("click", () => {{
      camera.yaw = -0.9;
      camera.pitch = 0.85;
      camera.distance = Math.max(payload.config.map_size_x, payload.config.map_size_y) * 2.1;
      draw();
    }});

    canvas.addEventListener("mousedown", (event) => {{
      dragState = {{ x: event.clientX, y: event.clientY }};
      canvas.classList.add("dragging");
    }});

    window.addEventListener("mousemove", (event) => {{
      if (!dragState) return;
      const dx = event.clientX - dragState.x;
      const dy = event.clientY - dragState.y;
      dragState = {{ x: event.clientX, y: event.clientY }};
      camera.yaw += dx * 0.01;
      camera.pitch = Math.max(0.15, Math.min(1.45, camera.pitch + dy * 0.01));
      draw();
    }});

    window.addEventListener("mouseup", () => {{
      dragState = null;
      canvas.classList.remove("dragging");
    }});

    canvas.addEventListener("wheel", (event) => {{
      event.preventDefault();
      camera.distance = Math.max(80, Math.min(1200, camera.distance + event.deltaY * 0.35));
      draw();
    }}, {{ passive: false }});

    window.addEventListener("resize", resize);
    resize();
  </script>
</body>
</html>
"""
