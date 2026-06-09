"""
Project LiDAR point clouds onto a pinhole camera image.

Camera frame convention (standard OpenCV / nuScenes):
  X = right, Y = down, Z = forward (into scene).

The camera is at (0, 0, sensor_height) in the world frame and looks
along the +X world axis.  See config.py for the rotation matrix.
"""
import numpy as np
from typing import Tuple

from src.config import (
    CAM_INTRINSIC, CAM_R_FROM_WORLD,
    CAM_WIDTH, CAM_HEIGHT,
)


def project_points(
    world_pts: np.ndarray,          # (N, 3)
    sensor_height: float,
    K: np.ndarray  = CAM_INTRINSIC,
    R: np.ndarray  = CAM_R_FROM_WORLD,
    cam_x: float   = 0.0,
    cam_y: float   = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project world points onto the image.

    Args:
        world_pts:      (N, 3) points in world frame
        sensor_height:  Z height of the camera in world frame
        K:              (3,3) intrinsic matrix
        R:              (3,3) rotation world→camera
        cam_x, cam_y:   horizontal camera position (usually 0)

    Returns:
        pixels  (N, 2)  – (u, v) pixel coordinates (may be outside image)
        depths  (N,)    – Z in camera frame (forward distance)
        valid   (N,)    – bool mask: in front of camera AND inside image
    """
    t_cam = np.array([cam_x, cam_y, sensor_height])          # camera world pos
    pts_cam = (R @ (world_pts - t_cam).T).T                  # (N, 3) camera frame

    # Keep only points in front of camera (Z_cam > 0)
    in_front = pts_cam[:, 2] > 0.1
    depths   = pts_cam[:, 2]

    with np.errstate(divide='ignore', invalid='ignore'):
        u = K[0, 0] * pts_cam[:, 0] / depths + K[0, 2]
        v = K[1, 1] * pts_cam[:, 1] / depths + K[1, 2]

    in_image = (
        (u >= 0) & (u < CAM_WIDTH) &
        (v >= 0) & (v < CAM_HEIGHT)
    )
    valid = in_front & in_image

    pixels = np.column_stack([u, v])
    return pixels, depths, valid


def depth_to_color(
    depths: np.ndarray,
    d_min: float = 0.0,
    d_max: float = 60.0,
) -> np.ndarray:
    """
    Map depth values to RGB colours using a jet-like colormap.
    Returns (N, 3) uint8 array (R, G, B).
    """
    norm = np.clip((depths - d_min) / (d_max - d_min + 1e-6), 0.0, 1.0)
    # Jet: blue→cyan→green→yellow→red
    r = np.clip(1.5 - np.abs(norm * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(norm * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(norm * 4 - 1), 0, 1)
    rgb = np.column_stack([r, g, b])
    return (rgb * 255).astype(np.uint8)


def label_to_color(labels: np.ndarray) -> np.ndarray:
    """Map feature label ids to RGB colours."""
    from src.config import FEATURE_COLORS
    rgb = np.zeros((len(labels), 3), dtype=np.uint8)
    for lid, color in FEATURE_COLORS.items():
        mask = labels == lid
        rgb[mask] = color
    return rgb


def overlay_points_on_image(
    image: np.ndarray,
    pixels: np.ndarray,
    colors: np.ndarray,
    valid: np.ndarray,
    radius: int = 3,
) -> np.ndarray:
    """
    Draw LiDAR points as filled circles on a copy of `image`.

    Args:
        image:  (H, W, 3) uint8 BGR image
        pixels: (N, 2) float pixel coords
        colors: (N, 3) uint8 RGB colours
        valid:  (N,) bool mask
        radius: circle radius in pixels
    """
    import cv2
    canvas = image.copy()
    px = pixels[valid].astype(np.int32)
    cl = colors[valid]
    # Sort by depth descending so closer points overwrite far ones
    for i in range(len(px)):
        c = (int(cl[i, 2]), int(cl[i, 1]), int(cl[i, 0]))  # BGR
        cv2.circle(canvas, (px[i, 0], px[i, 1]), radius, c, -1, cv2.LINE_AA)
    return canvas
