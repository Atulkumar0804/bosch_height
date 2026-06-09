"""
Per-feature detection quality analysis for different sensor heights.

For each feature type (traffic light, lane marking, car, …) and each
mounting height, computes:
  - point_count          total LiDAR returns on this feature
  - point_density        returns per m²  (approximated via BEV extent)
  - min_range_m          closest detected point distance
  - max_range_m          furthest detected point distance
  - mean_range_m         average range
  - z_coverage_m         vertical extent of hit points (how much of the
                          object height is sampled)
  - detected             True/False based on DETECTION_THRESHOLD
  - blind_spot_radius_m  for ground features: min horizontal range where
                          LiDAR beams can reach the ground at this height
"""
import numpy as np
import pandas as pd
from typing import Dict, List

from src.config import FEATURE_LABELS, DETECTION_THRESHOLD, LIDAR_V_FOV_DEG


# Inverse lookup
ID_TO_NAME = FEATURE_LABELS
GROUND_FEATURES = {"ground", "lane_marking", "road_symbol"}
ELEVATED_FEATURES = {"traffic_light", "traffic_sign", "pole"}
OBJECT_FEATURES = {"car", "truck", "pedestrian"}


def _blind_spot_radius(sensor_height: float) -> float:
    """
    Minimum horizontal range at which a ground-level LiDAR beam lands.
    The steepest downward beam angle is v_fov[0] (negative = downward).
    """
    steepest_down_deg = abs(min(LIDAR_V_FOV_DEG))   # e.g. 15°
    if steepest_down_deg == 0:
        return 0.0
    return sensor_height / np.tan(np.radians(steepest_down_deg))


def analyze_feature(
    scan: dict,
    label_id: int,
) -> dict:
    """
    Return detection metrics for one feature label from one scan.
    """
    name   = ID_TO_NAME.get(label_id, "unknown")
    mask   = scan["labels"] == label_id
    pts    = scan["points"][mask]     # sensor frame
    rng    = scan["ranges"][mask]
    n      = int(mask.sum())

    threshold = DETECTION_THRESHOLD.get(name, 1)

    if n == 0:
        return {
            "feature":          name,
            "point_count":      0,
            "point_density":    0.0,
            "min_range_m":      None,
            "max_range_m":      None,
            "mean_range_m":     None,
            "z_coverage_m":     0.0,
            "detected":         False,
            "threshold":        threshold,
        }

    # BEV footprint → density (points per m²)
    if n > 1:
        bev_area = (np.ptp(pts[:, 0]) + 0.01) * (np.ptp(pts[:, 1]) + 0.01)
    else:
        bev_area = 0.01
    density = n / bev_area

    z_cover = float(np.ptp(pts[:, 2])) if n > 1 else 0.0

    return {
        "feature":          name,
        "point_count":      n,
        "point_density":    round(float(density), 2),
        "min_range_m":      round(float(rng.min()), 2),
        "max_range_m":      round(float(rng.max()), 2),
        "mean_range_m":     round(float(rng.mean()), 2),
        "z_coverage_m":     round(z_cover, 3),
        "detected":         n >= threshold,
        "threshold":        threshold,
    }


def analyze_height_variant(
    scan: dict,
    feature_names: List[str] = None,
) -> pd.DataFrame:
    """
    Run analysis for all features in one height variant.
    Returns a DataFrame (one row per feature).
    """
    from src.config import NAME_TO_ID as NAME_TO_ID_local

    if feature_names is None:
        feature_names = list(DETECTION_THRESHOLD.keys())

    rows = []
    h = scan["sensor_height"]
    bs = _blind_spot_radius(h)

    from src.config import NAME_TO_ID as NAME_TO_ID_local
    for feat in feature_names:
        lid = NAME_TO_ID_local.get(feat)
        if lid is None:
            continue
        row = analyze_feature(scan, lid)
        row["sensor_height_m"]   = h
        row["blind_spot_radius_m"] = round(bs, 2)
        rows.append(row)

    return pd.DataFrame(rows)


def compare_heights(
    height_scans: Dict[str, dict],
    feature_names: List[str] = None,
) -> pd.DataFrame:
    """
    Run analysis for all heights and concatenate into one DataFrame.
    """
    frames = []
    for variant_name, scan in height_scans.items():
        df = analyze_height_variant(scan, feature_names)
        df.insert(0, "variant", variant_name)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined


def detection_summary(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot table: feature × height showing point_count and detected flag.
    """
    pivot = comparison_df.pivot_table(
        index="feature",
        columns="variant",
        values=["point_count", "detected", "mean_range_m"],
        aggfunc="first",
    )
    return pivot


def print_report(comparison_df: pd.DataFrame) -> None:
    """Print a nicely formatted summary to stdout."""
    print("\n" + "="*78)
    print("  FEATURE DETECTION REPORT  –  Effect of Sensor Mounting Height")
    print("="*78)

    for feat in comparison_df["feature"].unique():
        sub = comparison_df[comparison_df["feature"] == feat]
        print(f"\n  Feature: {feat.upper()}")
        print(f"  {'Height':20s} {'Points':>8} {'Density':>10} "
              f"{'Range(m)':>12} {'Z-cover(m)':>12} {'Detected':>10}")
        print("  " + "-"*74)
        for _, r in sub.iterrows():
            rng_str = (f"{r['min_range_m']:.1f}–{r['max_range_m']:.1f}"
                       if r["min_range_m"] is not None else "  N/A")
            det_str = "YES ✓" if r["detected"] else "NO  ✗"
            print(f"  {r['variant']:20s} {r['point_count']:>8d} "
                  f"{r['point_density']:>10.2f} "
                  f"{rng_str:>12s} "
                  f"{r['z_coverage_m']:>12.3f} "
                  f"{det_str:>10s}")

    print("\n" + "="*78)
    print("  BLIND-SPOT RADIUS (minimum ground-detection range)")
    seen = set()
    for _, r in comparison_df.iterrows():
        k = (r["variant"], r["sensor_height_m"])
        if k not in seen:
            print(f"  {r['variant']:20s}  h={r['sensor_height_m']:.1f}m  "
                  f"blind-spot ≥ {r['blind_spot_radius_m']:.1f} m")
            seen.add(k)
    print("="*78 + "\n")
