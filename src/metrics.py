"""
Quantitative metrics summarising the effect of sensor height on feature detection.

Exported function: compute_all_metrics(height_scans, comparison_df) → dict
"""
import numpy as np
import pandas as pd
from typing import Dict

from src.config import FEATURE_LABELS, LIDAR_V_FOV_DEG, LIDAR_MAX_RANGE_M
from src.feature_analyzer import _blind_spot_radius


ID_TO_NAME = FEATURE_LABELS


def ground_coverage_area(scan: dict) -> float:
    """Area (m²) of the ground plane covered by at least one LiDAR return."""
    gnd_id = next(k for k, v in FEATURE_LABELS.items() if v == "ground")
    mask = scan["labels"] == gnd_id
    pts  = scan["world_pts"][mask]
    if len(pts) == 0:
        return 0.0
    # Approximate by BEV bounding box
    return float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1]))


def traffic_light_elevation_angle(sensor_height: float,
                                  tl_height: float = 5.0,
                                  dist_m: float = 10.0) -> float:
    """
    Elevation angle (degrees) from sensor to top of a traffic light
    at `dist_m` ahead and `tl_height` above ground.
    Positive → looking up; negative → looking down.
    """
    dz = tl_height - sensor_height
    return float(np.degrees(np.arctan2(dz, dist_m)))


def traffic_light_in_fov(sensor_height: float,
                          tl_height: float = 5.0,
                          dist_m: float = 10.0) -> bool:
    """Return True if traffic light is within LiDAR vertical FOV."""
    angle = traffic_light_elevation_angle(sensor_height, tl_height, dist_m)
    return min(LIDAR_V_FOV_DEG) <= angle <= max(LIDAR_V_FOV_DEG)


def point_density_on_ground(scan: dict, near_m: float = 10.0) -> float:
    """Mean point density (pts/m²) on ground within `near_m` metres."""
    gnd_id = next(k for k, v in FEATURE_LABELS.items() if v == "ground")
    mask = (scan["labels"] == gnd_id) & (scan["ranges"] < near_m)
    pts  = scan["world_pts"][mask]
    if len(pts) < 2:
        return 0.0
    area = (np.ptp(pts[:, 0]) + 0.01) * (np.ptp(pts[:, 1]) + 0.01)
    return float(len(pts) / area)


def detection_rates(scan: dict) -> Dict[str, float]:
    """
    For each feature type, fraction of scene instances that have
    ≥ threshold points.  (Works on our single-scene synthetic data
    where each feature type has 1 representative; generalised below.)
    """
    from src.config import DETECTION_THRESHOLD
    rates = {}
    for label_id, name in FEATURE_LABELS.items():
        if name not in DETECTION_THRESHOLD:
            continue
        n     = int((scan["labels"] == label_id).sum())
        rates[name] = 1.0 if n >= DETECTION_THRESHOLD[name] else 0.0
    return rates


def max_detection_range(scan: dict, label_id: int) -> float:
    """Maximum range at which this feature is detected."""
    mask = scan["labels"] == label_id
    if not mask.any():
        return 0.0
    return float(scan["ranges"][mask].max())


def compute_all_metrics(
    height_scans: Dict[str, dict],
    comparison_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Compute a comprehensive metrics table (one row per height variant).
    """
    rows = []
    for name, scan in height_scans.items():
        h      = scan["sensor_height"]
        bs     = _blind_spot_radius(h)
        gca    = ground_coverage_area(scan)
        gnd_d  = point_density_on_ground(scan)
        tl_ang = traffic_light_elevation_angle(h)
        tl_fov = traffic_light_in_fov(h)
        det    = detection_rates(scan)

        # Total return count
        total = scan["num_rays"]

        # Per-feature max range
        feat_ranges = {}
        for lid, feat in FEATURE_LABELS.items():
            mr = max_detection_range(scan, lid)
            feat_ranges[f"max_range_{feat}"] = round(mr, 1)

        rows.append({
            "variant":                  name,
            "sensor_height_m":          h,
            "total_points":             total,
            "ground_coverage_m2":       round(gca, 1),
            "ground_density_near10m":   round(gnd_d, 2),
            "blind_spot_radius_m":      round(bs, 2),
            "tl_elevation_angle_deg":   round(tl_ang, 1),
            "traffic_light_in_fov":     tl_fov,
            **{f"det_{k}": v for k, v in det.items()},
            **feat_ranges,
        })

    return pd.DataFrame(rows)


def print_metrics(df: pd.DataFrame) -> None:
    """Pretty-print the metrics table."""
    print("\n" + "="*78)
    print("  QUANTITATIVE METRICS TABLE")
    print("="*78)

    key_cols = [
        "variant", "sensor_height_m", "total_points",
        "ground_coverage_m2", "ground_density_near10m",
        "blind_spot_radius_m", "tl_elevation_angle_deg", "traffic_light_in_fov",
    ]
    det_cols = [c for c in df.columns if c.startswith("det_")]
    range_cols = [c for c in df.columns if c.startswith("max_range_")]

    for col_group, label in [(key_cols, "OVERVIEW"),
                              (det_cols,   "DETECTION FLAGS"),
                              (range_cols, "MAX DETECTION RANGE (m)")]:
        available = ["variant"] + [c for c in col_group if c in df.columns and c != "variant"]
        sub = df[available].set_index("variant")
        print(f"\n  {label}")
        print(sub.to_string())

    print("="*78 + "\n")
