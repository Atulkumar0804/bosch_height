"""
Core image-domain transformations for the viewpoint-shift pipeline.

Implements the three strategies described in the ADASAdapt paper:
  1. Point Cloud Reprojection  –  depth-based 3-D warping
  2. IPM Recalibration         –  Inverse Perspective Mapping (2-D homography)
  3. Inpainting                –  fill black-hole regions (Telea / Navier-Stokes)

All functions work on (H, W, 3) uint8 BGR images + float32 depth maps.
"""
import numpy as np
import cv2
from typing import Tuple

from src.config import CAM_INTRINSIC


# ─────────────────────────────────────────────────────────────────────────────
# 1. Point Cloud Reprojection  (Depth-Based Warping)
# ─────────────────────────────────────────────────────────────────────────────

def point_cloud_reproject(
    image: np.ndarray,
    depth_map: np.ndarray,
    original_height: float,
    new_height: float,
    K: np.ndarray  = None,
    pitch_deg: float = 0.0,
    R_cam: np.ndarray = None,
    t_cam: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate what the camera would see if raised from original_height to new_height.

    When R_cam and t_cam (NuScenes calibration) are supplied the function uses
    the exact sensor-to-ego rotation so it works for any camera orientation
    (CAM_BACK, CAM_FRONT_LEFT, etc.).  Without them it falls back to a simplified
    model that only works accurately for a forward-facing camera.

    Args:
        image:           (H, W, 3) uint8 BGR source image
        depth_map:       (H, W) float32 forward (Z_cam) depth in metres
        original_height: camera Z in ego frame (metres)
        new_height:      target camera Z in ego frame (metres)
        K:               (3,3) intrinsics
        pitch_deg:       downward pitch of new camera in degrees (legacy, R_cam path ignores)
        R_cam:           (3,3) camera rotation in ego (from NuScenes calibration)
        t_cam:           (3,)  camera translation in ego (from NuScenes calibration)

    Returns:
        warped   (H, W, 3) uint8
        hole_mask (H, W) bool
    """
    if K is None:
        K = _scale_K(image.shape[1], image.shape[0])

    H, W = image.shape[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    delta_h = new_height - original_height

    depth = depth_map.astype(np.float32)
    valid_depth = (depth > 0.1) & (depth < 200.0)

    u_grid = np.arange(W, dtype=np.float32)
    v_grid = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u_grid, v_grid)

    # Back-project each pixel to 3-D in camera frame
    X_cam = (uu - cx) * depth / fx   # (H, W)
    Y_cam = (vv - cy) * depth / fy
    Z_cam = depth

    if R_cam is not None and t_cam is not None:
        # ── Correct path: use actual sensor calibration ───────────────────────
        # Camera → ego:  pts_ego = R_cam @ pts_cam + t_cam
        pts_cam = np.stack([X_cam.ravel(), Y_cam.ravel(), Z_cam.ravel()])  # (3, HW)
        pts_ego = R_cam @ pts_cam + t_cam[:, None]                          # (3, HW)

        # Raise camera by delta_h in ego Z; keep same rotation
        t_cam_new = t_cam.copy()
        t_cam_new[2] += delta_h
        pts_cam_new = R_cam.T @ (pts_ego - t_cam_new[:, None])              # (3, HW)

        X_cam_new = pts_cam_new[0].reshape(H, W)
        Y_cam_new = pts_cam_new[1].reshape(H, W)
        Z_cam_new = pts_cam_new[2].reshape(H, W)
    else:
        # ── Simplified path (forward-facing camera only) ─────────────────────
        X_world = Z_cam
        Y_world = -X_cam
        Z_world = -Y_cam + original_height

        X_cam_new = -Y_world
        Y_cam_new = -(Z_world - new_height)
        Z_cam_new =  X_world

        if abs(pitch_deg) > 0.01:
            theta = np.radians(pitch_deg)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            Y_cam_pitch =  Y_cam_new * cos_t + Z_cam_new * sin_t
            Z_cam_pitch = -Y_cam_new * sin_t + Z_cam_new * cos_t
            Y_cam_new = Y_cam_pitch
            Z_cam_new = Z_cam_pitch

    # ── Project to new image ──────────────────────────────────────────────────
    in_front = Z_cam_new > 0.1
    with np.errstate(divide='ignore', invalid='ignore'):
        u_new = np.where(in_front, fx * X_cam_new / Z_cam_new + cx, -1.0)
        v_new = np.where(in_front, fy * Y_cam_new / Z_cam_new + cy, -1.0)

    u_int = np.round(u_new).astype(np.int32)
    v_int = np.round(v_new).astype(np.int32)

    paintable = (
        valid_depth & in_front &
        (u_int >= 0) & (u_int < W) &
        (v_int >= 0) & (v_int < H)
    )

    # ── Forward splat with Z-buffer (far → near to avoid Z-fighting) ─────────
    warped     = np.zeros_like(image)
    filled     = np.zeros((H, W), dtype=bool)
    z_buf      = np.full((H, W), np.inf)

    idx  = np.where(paintable.ravel())[0]
    # Sort by decreasing depth (paint far points first; near overwrite)
    order = np.argsort(-Z_cam_new.ravel()[idx])
    idx   = idx[order]

    src_v = idx // W
    src_u = idx %  W
    dst_v = v_int.ravel()[idx]
    dst_u = u_int.ravel()[idx]
    depths_sorted = Z_cam_new.ravel()[idx]

    # Vectorised Z-buffer check
    current_z = z_buf[dst_v, dst_u]
    closer    = depths_sorted < current_z
    dst_v_c   = dst_v[closer]
    dst_u_c   = dst_u[closer]
    src_v_c   = src_v[closer]
    src_u_c   = src_u[closer]
    depths_c  = depths_sorted[closer]

    z_buf[dst_v_c, dst_u_c]  = depths_c
    warped[dst_v_c, dst_u_c] = image[src_v_c, src_u_c]
    filled[dst_v_c, dst_u_c] = True

    # ── Sky / background (depth ≈ 0) ─────────────────────────────────────────
    # Sky is at effectively infinite distance → negligible parallax for any
    # realistic height shift.  Copy sky pixels as-is so they do not count
    # as texture-starvation holes (which are specifically about 3-D scene
    # content the 1m camera never observed).
    sky_mask = ~valid_depth
    warped[sky_mask] = image[sky_mask]
    filled[sky_mask] = True

    # Hole mask = only scene geometry regions without source data
    hole_mask = (~filled) & valid_depth
    return warped, hole_mask


# ─────────────────────────────────────────────────────────────────────────────
# 2. Inpainting
# ─────────────────────────────────────────────────────────────────────────────

def inpaint_holes(
    warped_image: np.ndarray,
    hole_mask: np.ndarray,
    method: str = "telea",
    inpaint_radius: int = 5,
) -> np.ndarray:
    """
    Fill black-hole regions using OpenCV inpainting.

    At large height shifts (+2m) the holes are so large that the result
    becomes a blurry smear — exactly as described in the paper.

    Args:
        warped_image: (H, W, 3) uint8 BGR (from point_cloud_reproject)
        hole_mask:    (H, W) bool – True = missing data
        method:       "telea" (fast) or "ns" (Navier-Stokes, slower)
        inpaint_radius: neighbourhood radius for inpainting

    Returns:
        (H, W, 3) uint8 inpainted image
    """
    mask_u8 = (hole_mask.astype(np.uint8)) * 255
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(warped_image, mask_u8, inpaint_radius, flag)


# ─────────────────────────────────────────────────────────────────────────────
# 3. IPM  –  Inverse Perspective Mapping
# ─────────────────────────────────────────────────────────────────────────────

def ipm_transform(
    image: np.ndarray,
    camera_height: float,
    target_height: float,
    K: np.ndarray = None,
    bev_scale: float = 10.0,
    bev_x_range: tuple = (0, 60),
    bev_y_range: tuple = (-15, 15),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse Perspective Mapping: flatten the road to a Bird's Eye View
    using the ground-plane homography for the given camera height.

    Works perfectly for 2-D flat objects (lane lines, road symbols).
    Fails for 3-D objects (causes vertical stretching artefacts).

    Args:
        image:         (H, W, 3) BGR
        camera_height: original camera height (metres)
        target_height: target camera height (metres) – changes ground-plane scale
        K:             camera intrinsics
        bev_scale:     pixels per metre in output BEV image
        bev_x_range:   (x_min, x_max) metres – forward range
        bev_y_range:   (y_min, y_max) metres – lateral range

    Returns:
        bev_image (BH, BW, 3) BGR Bird's Eye View
        H_mat (3,3) the homography used
    """
    if K is None:
        K = _scale_K(image.shape[1], image.shape[0])

    H_img, W_img = image.shape[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    BW = int((bev_x_range[1] - bev_x_range[0]) * bev_scale)
    BH = int((bev_y_range[1] - bev_y_range[0]) * bev_scale)

    # Ground-plane homography:
    # A ground point (x_world, y_world, 0) projected to image via:
    #   P_cam = R @ (P_world - t_cam)
    # where R = [[0,-1,0],[0,0,-1],[1,0,0]], t_cam=[0,0,h]
    # P_cam = [−y, −h, x]
    # u = fx*(−y)/(x) + cx    v = fy*(−h)/(x) + cy
    # Invert to get ground homography:
    # For 4 corner ground points, compute image coords and use getPerspectiveTransform

    h = target_height   # use target height for BEV recalibration

    def world_to_pixel(x_w, y_w):
        Z_c = x_w
        X_c = -y_w
        Y_c = -h
        if Z_c <= 0.01:
            return None
        u = fx * X_c / Z_c + cx
        v = fy * Y_c / Z_c + cy
        return (float(u), float(v))

    def bev_to_world(bx, by):
        x_w = bev_x_range[0] + bx / bev_scale
        y_w = bev_y_range[0] + (BH - by) / bev_scale
        return x_w, y_w

    # 4 BEV corners → corresponding image pixels
    bev_corners = np.array([
        [0, 0], [BW-1, 0], [BW-1, BH-1], [0, BH-1]
    ], dtype=np.float32)

    img_corners = []
    for bx, by in bev_corners:
        xw, yw = bev_to_world(float(bx), float(by))
        px = world_to_pixel(xw, yw)
        if px is None:
            px = (-1.0, -1.0)
        img_corners.append(px)
    img_corners = np.array(img_corners, dtype=np.float32)

    H_mat, _ = cv2.findHomography(img_corners, bev_corners)
    if H_mat is None:
        bev = np.zeros((BH, BW, 3), dtype=np.uint8)
        return bev, np.eye(3)

    bev = cv2.warpPerspective(image, H_mat, (BW, BH),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
    return bev, H_mat


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scale_K(width: int, height: int) -> np.ndarray:
    K = CAM_INTRINSIC.copy()
    K[0, 0] *= width  / 1600.0
    K[1, 1] *= height /  900.0
    K[0, 2] *= width  / 1600.0
    K[1, 2] *= height /  900.0
    return K


def geometric_depth_from_image(
    image: np.ndarray,
    camera_height: float,
    K: np.ndarray = None,
) -> np.ndarray:
    """
    Estimate a depth map from a real road image using the flat-ground assumption.

    For row v > horizon_v (road area):  depth = f * h / (v - horizon_v)
    For row v ≤ horizon_v (sky/objects): depth = large constant (100 m)

    This is an approximation that works well for open road scenes.
    Vehicles and buildings will have wrong depths, but the qualitative
    height-shift effect (texture starvation, horizon shift) will still appear.
    """
    if K is None:
        K = _scale_K(image.shape[1], image.shape[0])

    H_img, W_img = image.shape[:2]
    fy = K[1, 1]
    cy = K[1, 2]

    # Detect horizon: find the row with maximum horizontal gradient change
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, 31), 0)
    # Sobel in Y direction to find horizontal edges
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    row_energy = np.abs(sobel_y).sum(axis=1)
    # Look in middle third of image for horizon
    lo, hi = H_img // 6, 2 * H_img // 3
    horizon_v = int(lo + np.argmax(row_energy[lo:hi]))

    # Build depth map
    v_coords = np.arange(H_img, dtype=np.float32)[:, np.newaxis]   # (H, 1)
    dv = v_coords - horizon_v    # (H, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ground_depth = np.where(dv > 2.0, fy * camera_height / dv, 100.0)

    depth_map = np.tile(ground_depth, (1, W_img)).astype(np.float32)
    depth_map = np.clip(depth_map, 0.5, 120.0)
    return depth_map, horizon_v
