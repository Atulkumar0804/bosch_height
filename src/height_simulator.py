"""
Transform an existing point cloud to simulate a different sensor mounting height.

Math
----
LiDAR sensor frame: X forward, Y left, Z up (origin at sensor).

If the sensor moves UP by Δh = new_h − orig_h:
  • All scene objects are Δh lower relative to the new sensor origin.
  • In sensor coordinates: z_new = z_old − Δh
  • Ground (at z_world=0) was at z_sensor = −orig_h; now at z_sensor = −new_h.

The horizontal (x,y) coordinates are unchanged.
"""
import numpy as np
from typing import Dict


def transform_height(
    scan_result: dict,
    original_height: float,
    new_height: float,
) -> dict:
    """
    Shift a scan taken at `original_height` to appear as if taken at `new_height`.

    Args:
        scan_result:     dict returned by lidar_simulator.simulate_scan()
        original_height: sensor height at scan time (metres)
        new_height:      target sensor height (metres)

    Returns:
        New scan dict with adjusted coordinates.
    """
    delta_h = new_height - original_height

    old_sensor = scan_result["points"].copy()    # (N, 3) in sensor frame
    new_sensor = old_sensor.copy()
    new_sensor[:, 2] -= delta_h                 # shift Z

    # World frame: just re-add the new origin height
    new_origin    = np.array([0.0, 0.0, new_height])
    new_world_pts = new_sensor + new_origin

    # Recompute ranges
    new_ranges = np.linalg.norm(new_sensor, axis=1)

    # Filter to max range
    from src.config import LIDAR_MAX_RANGE_M
    valid = new_ranges < LIDAR_MAX_RANGE_M

    return {
        "world_pts":      new_world_pts[valid],
        "points":         new_sensor[valid],
        "labels":         scan_result["labels"][valid],
        "ranges":         new_ranges[valid],
        "intensity":      scan_result["intensity"][valid],
        "sensor_height":  new_height,
        "num_rays":       int(valid.sum()),
        "origin_height":  original_height,
    }


def adjust_camera_position(
    cam_translation_world: np.ndarray,
    original_height: float,
    new_height: float,
) -> np.ndarray:
    """
    Return camera position in world frame after raising sensor by Δh.
    Only the Z component changes.
    """
    delta_h = new_height - original_height
    new_t = cam_translation_world.copy()
    new_t[2] += delta_h
    return new_t


def build_height_variants(
    base_scan: dict,
    base_height: float,
    target_heights: Dict[str, float],
) -> Dict[str, dict]:
    """
    Build a dict of scan results for every target height by transforming
    the base scan. Includes the base scan keyed by its own name.
    """
    variants: Dict[str, dict] = {}
    for name, h in target_heights.items():
        if abs(h - base_height) < 1e-6:
            # No shift needed for the base height
            variants[name] = base_scan.copy()
            variants[name]["sensor_height"] = h
        else:
            variants[name] = transform_height(base_scan, base_height, h)
        print(f"  [{name:20s}] h={h:.1f}m  "
              f"→ ground z_sensor≈{-h:.1f}m  "
              f"pts={variants[name]['num_rays']:,}")
    return variants
