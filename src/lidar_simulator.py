"""
Vectorised 32-beam rotating LiDAR simulator.

Shoots rays from (0, 0, sensor_height) in 360° with configurable
vertical FOV and returns a labelled point cloud.
"""
import numpy as np
from typing import Tuple

from src.config import (
    LIDAR_NUM_BEAMS, LIDAR_V_FOV_DEG, LIDAR_H_RES_DEG, LIDAR_MAX_RANGE_M
)
from src.synthetic_scene import UrbanScene


def _build_ray_directions(num_beams: int, h_res_deg: float,
                           v_fov: Tuple[float, float]) -> np.ndarray:
    """
    Build (N_rays, 3) unit-direction array for all beam × azimuth combos.
    """
    v_angles = np.linspace(v_fov[0], v_fov[1], num_beams, dtype=np.float64)
    h_angles = np.arange(0.0, 360.0, h_res_deg, dtype=np.float64)

    V, H = np.meshgrid(np.radians(v_angles), np.radians(h_angles))  # (N_h, N_v)
    V = V.ravel()
    H = H.ravel()

    dirs = np.column_stack([
        np.cos(V) * np.cos(H),   # X forward
        np.cos(V) * np.sin(H),   # Y left
        np.sin(V),               # Z up
    ])
    return dirs   # (N_rays, 3)


def simulate_scan(
    scene: UrbanScene,
    sensor_height: float,
    num_beams: int       = LIDAR_NUM_BEAMS,
    h_res_deg: float     = LIDAR_H_RES_DEG,
    v_fov: Tuple[float, float] = LIDAR_V_FOV_DEG,
    max_range: float     = LIDAR_MAX_RANGE_M,
) -> dict:
    """
    Simulate one full 360° LiDAR scan from height `sensor_height`.

    Returns a dict with numpy arrays:
      points    (N, 3)  – hit coordinates in *sensor* frame (z=0 is sensor level)
      labels    (N,)    – integer feature label
      ranges    (N,)    – radial distance to hit point
      intensity (N,)    – surface reflectance
      world_pts (N, 3)  – hit coordinates in *world* frame (z=0 is ground)
    """
    origin = np.array([0.0, 0.0, sensor_height])
    dirs   = _build_ray_directions(num_beams, h_res_deg, v_fov)

    best_t, best_label, best_intens = scene.ray_intersect_all(origin, dirs)

    # Valid hits (within max range and actually hit something)
    valid = (best_t < max_range) & (best_label >= 0)

    world_pts = origin + best_t[valid, np.newaxis] * dirs[valid]   # (N, 3) world
    sensor_pts = world_pts - origin                                  # (N, 3) sensor

    return {
        "world_pts": world_pts,
        "points":    sensor_pts,
        "labels":    best_label[valid],
        "ranges":    best_t[valid],
        "intensity": best_intens[valid],
        "sensor_height": sensor_height,
        "num_rays": int(valid.sum()),
    }


def run_all_heights(scene: UrbanScene, heights: dict) -> dict:
    """Run simulate_scan for every height in the heights dict."""
    results = {}
    for name, h in heights.items():
        results[name] = simulate_scan(scene, sensor_height=h)
        print(f"  [{name:20s}] h={h:.1f}m  →  {results[name]['num_rays']:,} points")
    return results
