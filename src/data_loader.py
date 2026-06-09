"""
Data loader: tries nuScenes first, falls back to synthetic.

When nuScenes is available the loader returns ONE sample's data in the
same dict format as the synthetic pipeline uses, so the rest of the
pipeline is agnostic to the data source.

nuScenes dict format
--------------------
  {
    "world_pts":      (N, 3)  float64  – LiDAR points in *ego* frame
    "points":         (N, 3)  float64  – same, treated as sensor frame
    "labels":         (N,)    int32    – all set to 0 (ground) for raw nuScenes
    "ranges":         (N,)    float64
    "intensity":      (N,)    float64
    "sensor_height":  float            – actual LIDAR_TOP height from calibration
    "camera_image":   (H,W,3) uint8    – front camera BGR image
    "cam_intrinsic":  (3,3)   float64
    "cam_R":          (3,3)   float64  – rotation world→camera
    "cam_t":          (3,)    float64  – camera world position
    "source":         "nuscenes" | "synthetic"
  }
"""
import os
import numpy as np
from typing import Optional


def _try_load_nuscenes(dataroot: str, version: str = "v1.0-mini",
                       sample_index: int = 0) -> Optional[dict]:
    """
    Attempt to load one sample from the nuScenes dataset.
    Returns None on any failure.
    """
    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.data_classes import LidarPointCloud
        from pyquaternion import Quaternion
        import cv2
    except ImportError:
        print("  [data_loader] nuscenes-devkit not installed. Using synthetic data.")
        return None

    if not os.path.isdir(dataroot):
        print(f"  [data_loader] nuScenes dataroot not found: {dataroot}")
        return None

    try:
        nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    except Exception as e:
        print(f"  [data_loader] Failed to load nuScenes: {e}")
        return None

    sample = nusc.sample[sample_index]

    # ── LiDAR ──────────────────────────────────────────────────────────────
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data  = nusc.get("sample_data", lidar_token)
    lidar_path  = os.path.join(dataroot, lidar_data["filename"])

    pc = LidarPointCloud.from_file(lidar_path)
    pts = pc.points[:3].T.astype(np.float64)         # (N, 3) in sensor frame

    lidar_calib = nusc.get("calibrated_sensor",
                            lidar_data["calibrated_sensor_token"])
    lidar_h = float(lidar_calib["translation"][2])   # height above ground

    # ── Camera ─────────────────────────────────────────────────────────────
    cam_token = sample["data"]["CAM_FRONT"]
    cam_data  = nusc.get("sample_data", cam_token)
    cam_path  = os.path.join(dataroot, cam_data["filename"])
    img_bgr   = cv2.imread(cam_path)

    cam_calib = nusc.get("calibrated_sensor",
                          cam_data["calibrated_sensor_token"])

    K = np.array(cam_calib["camera_intrinsic"], dtype=np.float64)
    Q_cam    = Quaternion(cam_calib["rotation"])
    R_cam    = Q_cam.rotation_matrix
    t_cam    = np.array(cam_calib["translation"], dtype=np.float64)

    Q_lidar  = Quaternion(lidar_calib["rotation"])
    R_lidar  = Q_lidar.rotation_matrix
    t_lidar  = np.array(lidar_calib["translation"], dtype=np.float64)

    # Transform LiDAR points to ego (vehicle) frame
    world_pts = (R_lidar @ pts.T).T + t_lidar     # (N, 3) ego frame

    ranges    = np.linalg.norm(pts, axis=1)
    intensity = pc.points[3] / 255.0 if pc.points.shape[0] > 3 else np.ones(len(pts))

    # Rotation from world (ego) to camera frame
    # P_cam = R_cam^T @ (P_world - t_cam)
    R_world_to_cam = R_cam.T

    return {
        "world_pts":     world_pts,
        "points":        pts,
        "labels":        np.zeros(len(pts), dtype=np.int32),
        "ranges":        ranges,
        "intensity":     intensity,
        "sensor_height": lidar_h,
        "num_rays":      len(pts),
        "camera_image":  img_bgr,
        "cam_intrinsic": K,
        "cam_R":         R_world_to_cam,
        "cam_t":         t_cam,
        "source":        "nuscenes",
    }


def load_data(
    dataroot: Optional[str] = None,
    version: str            = "v1.0-mini",
    sample_index: int       = 0,
) -> dict:
    """
    Primary entry point.  Returns a data dict in the standard format.
    Tries nuScenes first; falls back to synthetic if unavailable.
    """
    if dataroot is None:
        from src.config import NUSCENES_DATAROOT
        dataroot = NUSCENES_DATAROOT

    result = _try_load_nuscenes(dataroot, version, sample_index)
    if result is not None:
        print(f"  [data_loader] Loaded nuScenes sample {sample_index}  "
              f"(LiDAR height ≈ {result['sensor_height']:.2f} m)")
        return result

    # ── Synthetic fallback ─────────────────────────────────────────────────
    print("  [data_loader] Using SYNTHETIC urban scene.")
    from src.synthetic_scene import build_scene
    from src.lidar_simulator import simulate_scan
    from src.config import HEIGHTS

    scene = build_scene()
    base_height = list(HEIGHTS.values())[0]   # first height = reference
    scan = simulate_scan(scene, sensor_height=base_height)
    scan["source"] = "synthetic"
    scan["camera_image"] = None   # rendered on-the-fly by visualizer
    return scan
