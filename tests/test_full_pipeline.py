"""
Integration / smoke test for the full pipeline.

Runs synthetic-scene simulation for all three heights, checks that
visualisations are saved and that the metrics / detection tables
contain the expected features.
"""
import sys, os, shutil, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.config import HEIGHTS, DETECTION_THRESHOLD, FEATURE_LABELS
from src.synthetic_scene import build_scene
from src.lidar_simulator import simulate_scan
from src.height_simulator import build_height_variants
from src.feature_analyzer import compare_heights, print_report
from src.metrics import compute_all_metrics


@pytest.fixture(scope="module")
def height_scans():
    """Simulate all heights once for the whole module."""
    scene = build_scene()
    scans = {}
    for name, h in HEIGHTS.items():
        scans[name] = simulate_scan(scene, sensor_height=h)
    return scans


@pytest.fixture(scope="module")
def comparison_df(height_scans):
    return compare_heights(height_scans)


@pytest.fixture(scope="module")
def metrics_df(height_scans, comparison_df):
    return compute_all_metrics(height_scans, comparison_df)


# ── Scan sanity ───────────────────────────────────────────────────────────────

class TestScanSanity:
    def test_all_heights_produce_points(self, height_scans):
        for name, scan in height_scans.items():
            assert scan["num_rays"] > 0, f"{name}: zero points"

    def test_labels_within_known_range(self, height_scans):
        valid_ids = set(FEATURE_LABELS.keys())
        for name, scan in height_scans.items():
            unknown = set(scan["labels"].tolist()) - valid_ids
            assert not unknown, f"{name}: unknown label ids {unknown}"

    def test_ranges_positive(self, height_scans):
        for name, scan in height_scans.items():
            assert np.all(scan["ranges"] > 0), f"{name}: non-positive range"

    def test_ground_points_present_all_heights(self, height_scans):
        gnd_id = next(k for k, v in FEATURE_LABELS.items() if v == "ground")
        for name, scan in height_scans.items():
            n_gnd = (scan["labels"] == gnd_id).sum()
            assert n_gnd > 100, f"{name}: too few ground points ({n_gnd})"

    def test_higher_sensor_has_farther_ground_z(self, height_scans):
        """At h=3m, ground pts should sit at z_sensor ≈ -3; at h=1m at z≈ -1."""
        gnd_id = next(k for k, v in FEATURE_LABELS.items() if v == "ground")
        scans_sorted = sorted(height_scans.items(),
                               key=lambda x: x[1]["sensor_height"])
        for i in range(len(scans_sorted) - 1):
            _, s_low  = scans_sorted[i]
            _, s_high = scans_sorted[i+1]
            gnd_mask_low  = s_low["labels"]  == gnd_id
            gnd_mask_high = s_high["labels"] == gnd_id
            mean_z_low  = s_low["points"][gnd_mask_low,   2].mean()
            mean_z_high = s_high["points"][gnd_mask_high, 2].mean()
            assert mean_z_high < mean_z_low, \
                "Ground should appear lower (more negative z) at higher sensor height"


# ── Detection analysis ────────────────────────────────────────────────────────

class TestDetectionAnalysis:
    def test_comparison_df_has_all_features(self, comparison_df):
        expected = set(DETECTION_THRESHOLD.keys())
        found    = set(comparison_df["feature"].unique())
        assert expected.issubset(found), f"Missing features: {expected - found}"

    def test_comparison_df_has_all_heights(self, comparison_df, height_scans):
        expected_variants = set(height_scans.keys())
        found_variants    = set(comparison_df["variant"].unique())
        assert expected_variants == found_variants

    def test_cars_detected_at_1m(self, comparison_df):
        row = comparison_df[
            (comparison_df["feature"] == "car") &
            (comparison_df["sensor_height_m"] == 1.0)
        ]
        assert len(row) > 0
        assert bool(row.iloc[0]["detected"]), "Cars should be detected at 1m"

    def test_traffic_light_detected(self, comparison_df):
        """Traffic light should be detectable at at least one height."""
        tl = comparison_df[comparison_df["feature"] == "traffic_light"]
        assert tl["detected"].any(), "Traffic light should be detected at some height"

    def test_blind_spot_increases_with_height(self, comparison_df):
        bs = (comparison_df.groupby("sensor_height_m")["blind_spot_radius_m"]
              .first().sort_index())
        for i in range(len(bs) - 1):
            assert bs.iloc[i+1] >= bs.iloc[i], \
                "Blind-spot radius must increase monotonically with height"

    def test_point_count_columns_non_negative(self, comparison_df):
        assert (comparison_df["point_count"] >= 0).all()


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_df_has_all_heights(self, metrics_df, height_scans):
        assert set(metrics_df["variant"]) == set(height_scans.keys())

    def test_ground_coverage_positive(self, metrics_df):
        assert (metrics_df["ground_coverage_m2"] > 0).all()

    def test_blind_spot_monotonic(self, metrics_df):
        df = metrics_df.sort_values("sensor_height_m")
        bs = df["blind_spot_radius_m"].values
        assert all(bs[i] <= bs[i+1] for i in range(len(bs)-1))

    def test_tl_angle_increases_as_sensor_rises(self, metrics_df):
        """As sensor height increases, elevation angle to TL (5m) should decrease."""
        df = metrics_df.sort_values("sensor_height_m")
        angles = df["tl_elevation_angle_deg"].values
        # Higher sensor → smaller upward angle (more level/negative)
        assert all(angles[i] >= angles[i+1] for i in range(len(angles)-1)), \
            "Elevation angle to TL should decrease as sensor height increases"

    def test_total_points_positive(self, metrics_df):
        assert (metrics_df["total_points"] > 0).all()


# ── Visualisation smoke test ──────────────────────────────────────────────────

class TestVisualisationSmoke:
    def test_camera_image_renders(self, height_scans):
        from src.visualizer import camera_with_lidar
        scan = list(height_scans.values())[0]
        img = camera_with_lidar(scan, color_mode="feature")
        assert img.shape == (900, 1600, 3)
        assert img.dtype.kind == 'u'

    def test_bev_renders(self, height_scans):
        from src.visualizer import bird_eye_view
        scan = list(height_scans.values())[0]
        bev = bird_eye_view(scan)
        assert bev.ndim == 3
        assert bev.shape[2] == 3

    def test_outputs_saved(self, height_scans, comparison_df, tmp_path, monkeypatch):
        """Patch OUTPUT_DIR and check that PNG files are written."""
        monkeypatch.setattr("src.visualizer.OUTPUT_DIR", str(tmp_path))
        monkeypatch.setattr("src.config.OUTPUT_DIR",     str(tmp_path))
        import src.visualizer as viz
        viz.OUTPUT_DIR = str(tmp_path)

        viz.save_camera_comparison(height_scans)
        viz.save_bev_comparison(height_scans)
        viz.save_feature_bar_chart(comparison_df)
        viz.save_detection_heatmap(comparison_df)

        pngs = list(tmp_path.glob("*.png"))
        assert len(pngs) >= 4, f"Expected ≥4 PNG files, got {len(pngs)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
