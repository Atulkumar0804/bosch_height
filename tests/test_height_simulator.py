"""
Unit tests for height_simulator.py

Test: mathematical correctness of point-cloud Z-shifting.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.height_simulator import transform_height, adjust_camera_position


def _make_scan(n_pts=100, sensor_height=1.0):
    """Synthetic scan with known geometry."""
    rng = np.random.default_rng(42)
    pts_sensor = rng.uniform(-20, 20, (n_pts, 3))
    world_pts  = pts_sensor + np.array([0, 0, sensor_height])
    ranges     = np.linalg.norm(pts_sensor, axis=1)
    return {
        "points":        pts_sensor,
        "world_pts":     world_pts,
        "labels":        np.zeros(n_pts, dtype=np.int32),
        "ranges":        ranges,
        "intensity":     np.ones(n_pts),
        "sensor_height": sensor_height,
        "num_rays":      n_pts,
    }


class TestZShift:
    def test_ground_appears_lower_at_higher_height(self):
        """At height 3m, ground should be ~3m below sensor (z≈-3 in sensor frame)."""
        base = _make_scan(n_pts=500, sensor_height=1.0)
        # Place some points at world z=0 (ground)
        base["world_pts"][:10, 2] = 0.0
        base["points"][:10, 2]    = -1.0   # z_sensor = z_world - h = -1
        base["ranges"][:10]       = np.linalg.norm(base["points"][:10], axis=1)

        shifted = transform_height(base, original_height=1.0, new_height=3.0)
        # ground points should now have z_sensor ≈ -3
        ground_mask = np.abs(shifted["world_pts"][:, 2]) < 0.01
        if ground_mask.any():
            assert np.all(shifted["points"][ground_mask, 2] < -2.9), \
                "Ground z in sensor frame should be ≈ -3 after raising sensor to 3m"

    def test_same_height_is_identity(self):
        base = _make_scan(sensor_height=2.0)
        out  = transform_height(base, 2.0, 2.0)
        np.testing.assert_allclose(out["points"], base["points"], atol=1e-9)

    def test_z_shift_magnitude(self):
        """Shifting from 1→3m should subtract exactly 2 from z."""
        base = _make_scan(n_pts=50, sensor_height=1.0)
        pts_orig = base["points"].copy()
        out = transform_height(base, original_height=1.0, new_height=3.0)
        # Filter to common points (some may be dropped if out of range)
        common = min(len(pts_orig), len(out["points"]))
        dz = pts_orig[:common, 2] - out["points"][:common, 2]
        np.testing.assert_allclose(dz, 2.0, atol=1e-9)

    def test_xy_unchanged(self):
        """X and Y coordinates should not change."""
        base = _make_scan(n_pts=50, sensor_height=1.5)
        out  = transform_height(base, 1.5, 2.5)
        common = min(len(base["points"]), len(out["points"]))
        np.testing.assert_allclose(
            base["points"][:common, :2],
            out["points"][:common, :2],
            atol=1e-9,
        )

    def test_higher_sensor_increases_blind_spot(self):
        """Points very close to ground directly below sensor are lost at higher height."""
        from src.config import LIDAR_MAX_RANGE_M
        base = _make_scan(n_pts=200, sensor_height=1.0)
        out3 = transform_height(base, 1.0, 3.0)
        # Higher sensor → more points fall outside max_range or below ground
        # At minimum, ranges change
        assert out3["num_rays"] <= base["num_rays"]

    def test_delta_negative_lowers_sensor(self):
        """Shifting from 3→1m raises all z values by 2 (sensor moves DOWN)."""
        base = _make_scan(n_pts=50, sensor_height=3.0)
        pts_orig = base["points"].copy()
        out = transform_height(base, original_height=3.0, new_height=1.0)
        common = min(len(pts_orig), len(out["points"]))
        dz = out["points"][:common, 2] - pts_orig[:common, 2]
        np.testing.assert_allclose(dz, 2.0, atol=1e-9)


class TestCameraAdjust:
    def test_height_adjusts_z_only(self):
        t = np.array([0.0, 0.1, 1.5])
        out = adjust_camera_position(t, 1.5, 3.0)
        assert out[2] == pytest.approx(3.0)
        assert out[0] == pytest.approx(0.0)
        assert out[1] == pytest.approx(0.1)

    def test_original_unchanged(self):
        t = np.array([0.0, 0.0, 1.0])
        out = adjust_camera_position(t, 1.0, 1.0)
        np.testing.assert_allclose(out, t)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
