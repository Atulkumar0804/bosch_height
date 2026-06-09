"""
Unit tests for feature_analyzer.py

Tests: blind-spot formula, per-feature metric computation,
       detection flag threshold logic.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.feature_analyzer import (
    _blind_spot_radius,
    analyze_feature,
    compare_heights,
)
from src.config import FEATURE_LABELS, LIDAR_V_FOV_DEG, DETECTION_THRESHOLD


NAME_TO_ID = {v: k for k, v in FEATURE_LABELS.items()}


def _make_dummy_scan(feature_name, n_pts, sensor_height=1.0, ranges=None):
    """Build a minimal scan dict containing only points of one feature type."""
    lid = NAME_TO_ID[feature_name]
    rng = np.random.default_rng(7)
    pts = rng.uniform(0, 20, (n_pts, 3))
    if ranges is None:
        ranges_arr = np.linalg.norm(pts, axis=1)
    else:
        ranges_arr = np.full(n_pts, ranges)

    # Pad with ground points so scan has required keys
    total = n_pts + 10
    all_pts    = np.vstack([pts, rng.uniform(0, 5, (10, 3))])
    all_labels = np.concatenate([np.full(n_pts, lid, dtype=np.int32),
                                  np.zeros(10, dtype=np.int32)])
    all_ranges = np.concatenate([ranges_arr, np.ones(10)])
    all_intens = np.ones(total)

    return {
        "points":        all_pts,
        "world_pts":     all_pts + np.array([0, 0, sensor_height]),
        "labels":        all_labels,
        "ranges":        all_ranges,
        "intensity":     all_intens,
        "sensor_height": sensor_height,
        "num_rays":      total,
    }


class TestBlindSpot:
    def test_zero_height_gives_zero(self):
        bs = _blind_spot_radius(0.0)
        assert bs == pytest.approx(0.0, abs=1e-6)

    def test_higher_sensor_larger_blind_spot(self):
        bs1 = _blind_spot_radius(1.0)
        bs2 = _blind_spot_radius(2.0)
        bs3 = _blind_spot_radius(3.0)
        assert bs3 > bs2 > bs1 > 0

    def test_formula_correctness(self):
        """bs = height / tan(steepest_down_angle)."""
        h = 2.0
        steepest = abs(min(LIDAR_V_FOV_DEG))
        expected = h / np.tan(np.radians(steepest))
        assert _blind_spot_radius(h) == pytest.approx(expected, rel=1e-6)


class TestAnalyzeFeature:
    def test_no_points_gives_not_detected(self):
        scan = _make_dummy_scan("car", n_pts=0, sensor_height=1.0)
        # Overwrite to ensure no car points
        scan["labels"][:] = 0
        result = analyze_feature(scan, NAME_TO_ID["car"])
        assert result["detected"] is False
        assert result["point_count"] == 0

    def test_above_threshold_detected(self):
        thresh = DETECTION_THRESHOLD["car"]
        scan = _make_dummy_scan("car", n_pts=thresh + 5, sensor_height=1.0)
        result = analyze_feature(scan, NAME_TO_ID["car"])
        assert result["detected"] is True

    def test_below_threshold_not_detected(self):
        thresh = DETECTION_THRESHOLD["car"]
        scan = _make_dummy_scan("car", n_pts=max(thresh - 1, 0), sensor_height=1.0)
        if thresh > 1:
            result = analyze_feature(scan, NAME_TO_ID["car"])
            assert result["detected"] is False

    def test_mean_range_correct(self):
        scan = _make_dummy_scan("lane_marking", n_pts=20, sensor_height=1.0,
                                ranges=15.0)
        result = analyze_feature(scan, NAME_TO_ID["lane_marking"])
        assert result["mean_range_m"] == pytest.approx(15.0, abs=0.1)

    def test_returned_feature_name(self):
        scan = _make_dummy_scan("traffic_light", n_pts=10)
        result = analyze_feature(scan, NAME_TO_ID["traffic_light"])
        assert result["feature"] == "traffic_light"


class TestCompareHeights:
    def test_output_has_all_variants(self):
        variants = {
            "1m": _make_dummy_scan("car", 20, 1.0),
            "2m": _make_dummy_scan("car", 15, 2.0),
            "3m": _make_dummy_scan("car", 10, 3.0),
        }
        df = compare_heights(variants)
        assert set(df["variant"].unique()) == {"1m", "2m", "3m"}

    def test_output_has_expected_columns(self):
        variants = {"1m": _make_dummy_scan("car", 20, 1.0)}
        df = compare_heights(variants)
        required = {"variant", "feature", "point_count", "detected",
                    "sensor_height_m", "blind_spot_radius_m"}
        assert required.issubset(df.columns)

    def test_higher_sensor_larger_blind_spot_in_df(self):
        variants = {
            "1m": _make_dummy_scan("lane_marking", 20, 1.0),
            "3m": _make_dummy_scan("lane_marking", 20, 3.0),
        }
        df = compare_heights(variants)
        bs1 = df[df["variant"] == "1m"]["blind_spot_radius_m"].iloc[0]
        bs3 = df[df["variant"] == "3m"]["blind_spot_radius_m"].iloc[0]
        assert bs3 > bs1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
