"""
Unit tests for projector.py

Tests: perspective projection math and colour mapping.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.projector import project_points, depth_to_color, label_to_color
from src.config import CAM_INTRINSIC, CAM_R_FROM_WORLD, CAM_WIDTH, CAM_HEIGHT, CAM_F


class TestProjectPoints:
    def test_point_on_optical_axis_projects_to_centre(self):
        """A point directly ahead (world-X axis) projects to image centre."""
        pts = np.array([[20.0, 0.0, 1.0]])   # 20m ahead, at sensor height
        pixels, depths, valid = project_points(pts, sensor_height=1.0)
        assert valid[0], "Point should be visible"
        assert pixels[0, 0] == pytest.approx(CAM_WIDTH / 2, abs=1.0),  "u should be cx"
        assert pixels[0, 1] == pytest.approx(CAM_HEIGHT / 2, abs=1.0), "v should be cy"

    def test_depth_equals_forward_distance(self):
        """Depth should equal the X-world coordinate of a centred point."""
        pts = np.array([[15.0, 0.0, 1.0]])
        _, depths, valid = project_points(pts, sensor_height=1.0)
        assert valid[0]
        assert depths[0] == pytest.approx(15.0, abs=0.1)

    def test_point_behind_camera_not_valid(self):
        """Negative X world = behind camera."""
        pts = np.array([[-5.0, 0.0, 1.0]])
        _, _, valid = project_points(pts, sensor_height=1.0)
        assert not valid[0]

    def test_higher_sensor_shifts_horizon(self):
        """At higher sensor height, a ground point projects lower in the image."""
        pts = np.array([[20.0, 0.0, 0.0]])   # ground point
        _, _, v1 = project_points(pts, sensor_height=1.0)
        _, _, v2 = project_points(pts, sensor_height=3.0)
        pix1, _, _ = project_points(pts, sensor_height=1.0)
        pix2, _, _ = project_points(pts, sensor_height=3.0)
        # Ground appears further below (larger v) at higher sensor
        assert pix2[0, 1] > pix1[0, 1], \
            "Ground point should project lower (larger v) at 3m than 1m"

    def test_left_point_projects_to_left_half(self):
        """A point to the left (world +Y) should appear on the left side of image."""
        pts = np.array([[10.0, 3.0, 1.0]])
        pixels, _, valid = project_points(pts, sensor_height=1.0)
        assert valid[0]
        assert pixels[0, 0] < CAM_WIDTH / 2, "Left-side point should have u < cx"

    def test_overhead_traffic_light_from_low_sensor(self):
        """Traffic light (z=5m) at 20m, 1m sensor: elevation ~11° → inside vertical FOV."""
        # At 10m the elevation angle is ~22° which clips the image edge;
        # at 20m it is ~11°, comfortably within the ±21° vertical half-FOV.
        pts = np.array([[20.0, 0.0, 5.0]])
        pixels, depths, valid = project_points(pts, sensor_height=1.0)
        assert valid[0], "Traffic light at 20m should be visible from 1m sensor"
        assert pixels[0, 1] < CAM_HEIGHT / 2, "TL should project above image centre"

    def test_batch_projection_shapes(self):
        """Output shapes must match input."""
        rng = np.random.default_rng(0)
        pts = rng.uniform(-10, 30, (500, 3))
        pts[:, 0] = np.abs(pts[:, 0]) + 1   # ensure some in front
        pixels, depths, valid = project_points(pts, sensor_height=1.0)
        assert pixels.shape == (500, 2)
        assert depths.shape == (500,)
        assert valid.shape  == (500,)


class TestDepthColor:
    def test_output_shape_and_dtype(self):
        d = np.linspace(0, 60, 100)
        c = depth_to_color(d)
        assert c.shape == (100, 3)
        assert c.dtype == np.uint8

    def test_near_is_blue(self):
        near = depth_to_color(np.array([0.0]))
        far  = depth_to_color(np.array([60.0]))
        # Jet: near→blue, far→red
        assert near[0, 2] > near[0, 0], "Near point should be more blue than red"
        assert far[0, 0] > far[0, 2],   "Far point should be more red than blue"

    def test_clipping(self):
        c = depth_to_color(np.array([-10.0, 200.0]))
        assert c.max() <= 255


class TestLabelColor:
    def test_known_labels(self):
        labels = np.array([0, 5, 3], dtype=np.int32)  # ground, car, traffic_light
        colors = label_to_color(labels)
        assert colors.shape == (3, 3)
        assert colors.dtype == np.uint8

    def test_unknown_label_black(self):
        labels = np.array([99], dtype=np.int32)
        colors = label_to_color(labels)
        assert colors[0].tolist() == [0, 0, 0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
