"""
NuScenes LiDAR-Based Image Reprojector
=======================================
Implements Option 2 from the ADASAdapt paper:
  "Use LiDAR-based reprojection to simulate elevated camera views."

For each frame:
  1. Load original camera image + calibrated LiDAR point cloud
  2. Raise virtual camera by delta_h in the ego Z direction
  3. Project LiDAR into the new camera pose → sparse but accurate depth map
  4. Densify depth via dilation
  5. Reproject original image using dense depth (Point Cloud Reprojection)
  6. Optionally inpaint holes

Key advantage over monocular depth:
  NuScenes provides synchronized LiDAR → exact 3D positions, no estimation error.

Note on limitation (from the paper):
  This captures GEOMETRIC shift effects only.
  Real 3m camera additionally loses texture (vehicle roofs/bonnets never seen
  by 1m camera) + occlusion changes + sensor placement effects.
"""

import os
import numpy as np
import cv2
from typing import Optional, Dict, List, Tuple

from src.real_image_transformer import point_cloud_reproject, inpaint_holes


# ─────────────────────────────────────────────────────────────────────────────
# Core reprojection
# ─────────────────────────────────────────────────────────────────────────────

def _build_extrinsic_4x4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4×4 ego→sensor transformation matrix."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.T            # ego→sensor rotation = sensor_R^T
    T[:3,  3] = -R.T @ t      # translation: -R^T @ t_sensor_in_ego
    return T


def create_lidar_depth_map(
    pts_lidar: np.ndarray,      # (N, 3) in LiDAR sensor frame
    R_lidar: np.ndarray,        # (3,3)  lidar rotation in ego
    t_lidar: np.ndarray,        # (3,)   lidar position in ego
    R_cam:   np.ndarray,        # (3,3)  camera rotation in ego
    t_cam:   np.ndarray,        # (3,)   camera position in ego (ALREADY RAISED if applicable)
    K:       np.ndarray,        # (3,3)  camera intrinsics
    img_h: int,
    img_w: int,
    min_depth: float = 0.5,
    max_depth: float = 80.0,
    dilation_ksize: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project LiDAR points into the camera frame and return a dense depth map.

    Returns:
        depth_map  (H, W) float32 – forward depth in metres (0 = no data)
        pts_uv     (M, 3) – valid projected points [u, v, depth]
    """
    # LiDAR → ego
    pts_ego = (R_lidar @ pts_lidar.T).T + t_lidar             # (N, 3)

    # ego → new camera
    pts_cam = (R_cam.T @ (pts_ego - t_cam).T).T               # (N, 3)

    # Forward-facing filter
    valid = pts_cam[:, 2] > min_depth
    pts_v = pts_cam[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    depths = pts_v[:, 2]

    u_f = fx * pts_v[:, 0] / depths + cx
    v_f = fy * pts_v[:, 1] / depths + cy
    u_i = np.round(u_f).astype(np.int32)
    v_i = np.round(v_f).astype(np.int32)

    in_img = (u_i >= 0) & (u_i < img_w) & (v_i >= 0) & (v_i < img_h) \
             & (depths < max_depth)

    # Build sparse depth (far-to-near so near values overwrite)
    order   = np.argsort(-depths[in_img])
    u_ok    = u_i[in_img][order]
    v_ok    = v_i[in_img][order]
    d_ok    = depths[in_img][order].astype(np.float32)

    depth_sparse = np.zeros((img_h, img_w), dtype=np.float32)
    depth_sparse[v_ok, u_ok] = d_ok

    # Densify via morphological dilation
    kernel = np.ones((dilation_ksize, dilation_ksize), np.uint8)
    depth_dilated = cv2.dilate(depth_sparse, kernel)
    depth_dense   = np.where(depth_sparse > 0, depth_sparse, depth_dilated)

    pts_uv = np.column_stack([u_f[in_img], v_f[in_img], depths[in_img]])
    return depth_dense, pts_uv


def reproject_frame(
    image:       np.ndarray,    # (H, W, 3) uint8 BGR
    R_cam:       np.ndarray,    # camera rotation in ego (from calibration)
    t_cam:       np.ndarray,    # camera position in ego (from calibration)
    R_lidar:     np.ndarray,
    t_lidar:     np.ndarray,
    pts_lidar:   np.ndarray,    # (N, 3) raw LiDAR points
    K:           np.ndarray,    # camera intrinsics
    delta_h:     float,         # height shift in metres
    pitch_deg:   float = 0.0,
    inpaint:     bool  = True,
) -> dict:
    """
    Core function: reproject one frame to a virtual camera at height + delta_h.

    Implements the paper's pseudo-code:
        T_cam_new = T_cam_orig.copy()
        T_cam_new[2, 3] += delta_h    # raise camera in ego Z
        points_new = T_cam_new @ lidar_points
        render(points_new)
    """
    H_img, W_img = image.shape[:2]

    # New camera position: raise Z in ego frame
    t_cam_new = t_cam + np.array([0.0, 0.0, delta_h])

    # Build LiDAR depth map at ORIGINAL camera position — point_cloud_reproject
    # reads depth_map[v,u] as depth at original pixel (u,v), so this must be
    # computed from t_cam (original), not t_cam_new.
    depth_map, _ = create_lidar_depth_map(
        pts_lidar, R_lidar, t_lidar,
        R_cam, t_cam,
        K, H_img, W_img,
    )

    # Point cloud reproject original image
    h_orig = float(t_cam[2])
    h_new  = float(t_cam_new[2])

    warped, hole_mask = point_cloud_reproject(
        image, depth_map, h_orig, h_new, K,
        pitch_deg=pitch_deg, R_cam=R_cam, t_cam=t_cam
    )

    if inpaint and hole_mask.any():
        inpainted = inpaint_holes(warped, hole_mask)
    else:
        inpainted = warped.copy()

    hole_pct = 100.0 * hole_mask.sum() / (H_img * W_img)

    return {
        "warped":     warped,
        "inpainted":  inpainted,
        "hole_mask":  hole_mask,
        "depth_map":  depth_map,
        "hole_pct":   hole_pct,
        "delta_h":    delta_h,
        "h_orig":     h_orig,
        "h_new":      h_new,
    }


def project_3d_box_to_camera(
    corners_world: np.ndarray,   # (8, 3) 3D bounding box corners in ego frame
    R_cam: np.ndarray,
    t_cam: np.ndarray,
    K: np.ndarray,
    delta_h: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D box corners into image at new height. Returns (pixels, valid)."""
    t_cam_new = t_cam + np.array([0.0, 0.0, delta_h])
    pts_cam   = (R_cam.T @ (corners_world - t_cam_new).T).T  # (8, 3)
    in_front  = pts_cam[:, 2] > 0.1
    fx, fy    = K[0, 0], K[1, 1]
    cx, cy    = K[0, 2], K[1, 2]
    depths    = pts_cam[:, 2]
    u = np.where(in_front, fx * pts_cam[:, 0] / depths + cx, -1)
    v = np.where(in_front, fy * pts_cam[:, 1] / depths + cy, -1)
    return np.column_stack([u, v]), in_front


# ─────────────────────────────────────────────────────────────────────────────
# NuScenes wrapper
# ─────────────────────────────────────────────────────────────────────────────

class NuScenesReprojector:
    """High-level wrapper that loads a nuScenes sample and reprojects it."""

    CATEGORIES = ["car", "truck", "pedestrian", "bicycle",
                  "traffic_cone", "motorcycle", "bus", "trailer"]

    def __init__(self, nusc, cam_name: str = "CAM_FRONT"):
        self.nusc     = nusc
        self.cam_name = cam_name

    def _load_sample_data(self, sample_token: str):
        from pyquaternion import Quaternion
        from nuscenes.utils.data_classes import LidarPointCloud

        sample   = self.nusc.get("sample", sample_token)

        # Camera
        cam_sd   = self.nusc.get("sample_data", sample["data"][self.cam_name])
        cam_path = os.path.join(self.nusc.dataroot, cam_sd["filename"])
        image    = cv2.imread(cam_path)
        cam_cal  = self.nusc.get("calibrated_sensor",
                                   cam_sd["calibrated_sensor_token"])
        K        = np.array(cam_cal["camera_intrinsic"], dtype=np.float64)
        R_cam    = Quaternion(cam_cal["rotation"]).rotation_matrix
        t_cam    = np.array(cam_cal["translation"], dtype=np.float64)

        # LiDAR
        lid_sd   = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lid_path = os.path.join(self.nusc.dataroot, lid_sd["filename"])
        pc       = LidarPointCloud.from_file(lid_path)
        pts_l    = pc.points[:3].T   # (N, 3)
        lid_cal  = self.nusc.get("calibrated_sensor",
                                   lid_sd["calibrated_sensor_token"])
        R_lidar  = Quaternion(lid_cal["rotation"]).rotation_matrix
        t_lidar  = np.array(lid_cal["translation"], dtype=np.float64)

        return image, K, R_cam, t_cam, pts_l, R_lidar, t_lidar, sample

    def reproject(self, sample_token: str, delta_h: float,
                  pitch_deg: float = 0.0) -> dict:
        img, K, R_c, t_c, pts_l, R_l, t_l, _ = \
            self._load_sample_data(sample_token)
        result = reproject_frame(img, R_c, t_c, R_l, t_l, pts_l,
                                  K, delta_h, pitch_deg)
        result["image_orig"] = img
        result["K"]          = K
        result["sample_token"] = sample_token
        return result

    def generate_height_series(
        self,
        sample_token: str,
        height_shifts: List[float],
        pitch_deg: float = 0.0,
    ) -> Dict[float, dict]:
        """Run reprojection for all height shifts in one call."""
        img, K, R_c, t_c, pts_l, R_l, t_l, _ = \
            self._load_sample_data(sample_token)

        results: Dict[float, dict] = {}
        for dh in height_shifts:
            res = reproject_frame(img, R_c, t_c, R_l, t_l, pts_l,
                                   K, dh, pitch_deg)
            res["image_orig"] = img
            res["K"]          = K
            results[dh]       = res
            print(f"    Δh={dh:+.1f}m  hole={res['hole_pct']:.1f}%")
        return results

    def get_annotations_for_sample(self, sample_token: str) -> List[dict]:
        """Return bounding-box annotations for all detected objects."""
        from nuscenes.utils.geometry_utils import BoxVisibility
        nusc = self.nusc
        sample = nusc.get("sample", sample_token)
        _, K, R_c, t_c, _, _, _, _ = self._load_sample_data(sample_token)

        boxes = nusc.get_boxes(sample["data"][self.cam_name])
        annots = []
        for box in boxes:
            corners = box.corners().T   # (8, 3) in ego frame
            category = box.name.split(".")[0]
            annots.append({
                "category": category,
                "corners":  corners,
                "center":   box.center,
                "size":     box.wlh,
            })
        return annots, K, R_c, t_c


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fallback (no nuScenes)
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_height_series(height_shifts: List[float]) -> Dict[float, dict]:
    """
    Reproduce the height series using our synthetic scene + exact renderer depths.
    Used when nuScenes is not available.
    """
    from src.scene_renderer import render_scene, RENDER_W, RENDER_H
    from src.synthetic_scene import build_scene
    from src.real_image_transformer import point_cloud_reproject, inpaint_holes, _scale_K

    scene  = build_scene()
    K      = _scale_K(RENDER_W, RENDER_H)

    image_1m, depth_1m = render_scene(sensor_height=1.0, scene=scene)
    results: Dict[float, dict] = {}

    for dh in height_shifts:
        if dh == 0.0:
            warped    = image_1m.copy()
            hole_mask = np.zeros((RENDER_H, RENDER_W), dtype=bool)
        else:
            warped, hole_mask = point_cloud_reproject(
                image_1m, depth_1m, 1.0, 1.0 + dh, K)

        inpainted = inpaint_holes(warped, hole_mask) if hole_mask.any() else warped
        hole_pct  = 100.0 * hole_mask.sum() / (RENDER_H * RENDER_W)

        # Also render from the actual new height (shows true geometry)
        true_render, _ = render_scene(sensor_height=1.0 + dh, scene=scene)

        results[dh] = {
            "warped":       warped,
            "inpainted":    inpainted,
            "true_render":  true_render,
            "hole_mask":    hole_mask,
            "depth_map":    depth_1m,
            "hole_pct":     hole_pct,
            "delta_h":      dh,
            "h_orig":       1.0,
            "h_new":        1.0 + dh,
            "image_orig":   image_1m,
            "K":            K,
        }
        print(f"    Δh={dh:+.1f}m  hole={hole_pct:.1f}%")

    return results
