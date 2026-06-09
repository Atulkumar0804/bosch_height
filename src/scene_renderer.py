"""
Ray-cast the synthetic urban scene into a photorealistic camera image
plus an exact per-pixel depth map.

Resolution is configurable; defaults to 800×450 for speed.
"""
import numpy as np
import cv2
from src.synthetic_scene import build_scene, UrbanScene
from src.config import CAM_INTRINSIC, CAM_R_FROM_WORLD

# ── Per-label RGB colours for rendering ──────────────────────────────────────
RENDER_COLORS = {
    0: (110, 110, 105),   # asphalt – dark warm gray
    1: (240, 240, 240),   # lane marking – near white
    2: (255, 220,  40),   # road symbol – amber yellow
    3: ( 60,  60,  60),   # traffic light housing – dark
    4: ( 70, 130, 200),   # traffic sign – steel blue
    5: ( 40, 130, 220),   # car – Bosch blue
    6: (180,  90,  40),   # truck – brown-orange
    7: (220, 175, 140),   # pedestrian – skin tone
    8: (140, 140, 140),   # pole – gray
    9: ( 90,  75,  60),   # building – brown
}
SKY_TOP    = np.array([ 90, 140, 220], dtype=np.float32)  # deep sky blue
SKY_HORIZ  = np.array([175, 215, 245], dtype=np.float32)  # light hazy horizon

RENDER_W, RENDER_H = 800, 450


def _scaled_K(width: int, height: int) -> np.ndarray:
    K = CAM_INTRINSIC.copy()
    K[0, 0] *= width  / 1600.0   # fx
    K[1, 1] *= height /  900.0   # fy
    K[0, 2] *= width  / 1600.0   # cx
    K[1, 2] *= height /  900.0   # cy
    return K


def render_scene(
    sensor_height: float,
    width: int  = RENDER_W,
    height: int = RENDER_H,
    scene: UrbanScene = None,
    add_road_texture: bool = True,
    seed: int = 0,
) -> tuple:
    """
    Returns (bgr_image, depth_map):
      bgr_image : (H, W, 3) uint8
      depth_map : (H, W) float32 – forward (Z_cam) depth in metres;
                  0.0 where no hit (sky)
    """
    rng = np.random.default_rng(seed)
    if scene is None:
        scene = build_scene()

    K   = _scaled_K(width, height)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # ── Build ray directions for every pixel ─────────────────────────────────
    u_arr = np.arange(width,  dtype=np.float64)
    v_arr = np.arange(height, dtype=np.float64)
    uu, vv = np.meshgrid(u_arr, v_arr)          # (H, W)

    # Camera-frame ray (unnormalized): d_cam = [(u-cx)/fx, (v-cy)/fy, 1]
    d_cx = (uu - cx) / fx   # (H, W)
    d_cy = (vv - cy) / fy   # (H, W)

    # World-frame ray using R_cam_to_world = R_cam_from_world^T
    # R = [[0,-1,0],[0,0,-1],[1,0,0]]  →  R^T = [[0,0,1],[-1,0,0],[0,-1,0]]
    d_wx =  np.ones_like(d_cx)   # world-X (forward) = cam-Z = 1 (unnorm)
    d_wy = -d_cx                  # world-Y = -cam-X
    d_wz = -d_cy                  # world-Z = -cam-Y

    nrm  = np.sqrt(d_wx**2 + d_wy**2 + d_wz**2)
    d_wx /= nrm;  d_wy /= nrm;  d_wz /= nrm

    # Flatten → (N, 3)
    N    = height * width
    dirs = np.stack([d_wx.ravel(), d_wy.ravel(), d_wz.ravel()], axis=1)

    origin = np.array([0.0, 0.0, sensor_height])

    # ── Ray cast ─────────────────────────────────────────────────────────────
    best_t, best_label, best_intens = scene.ray_intersect_all(origin, dirs)

    best_t     = best_t.reshape(height, width)
    best_label = best_label.reshape(height, width)
    best_intens = best_intens.reshape(height, width)

    # ── Z-depth (camera forward = world-X direction) ─────────────────────────
    hit_world_x = origin[0] + best_t * d_wx
    depth_map = np.where(best_t < 1e9, hit_world_x, 0.0).astype(np.float32)

    # ── Sky background ────────────────────────────────────────────────────────
    # Horizon row ≈ cy for level camera
    t_sky   = np.clip(1.0 - vv / max(cy, 1), 0.0, 1.0)   # 0=horizon, 1=top
    sky_r   = (SKY_HORIZ[0] * (1-t_sky) + SKY_TOP[0] * t_sky)
    sky_g   = (SKY_HORIZ[1] * (1-t_sky) + SKY_TOP[1] * t_sky)
    sky_b   = (SKY_HORIZ[2] * (1-t_sky) + SKY_TOP[2] * t_sky)
    sky_bgr = np.stack([sky_b, sky_g, sky_r], axis=-1).astype(np.uint8)

    image = sky_bgr.copy()

    # ── Render scene hits ────────────────────────────────────────────────────
    valid = best_t < 79.0
    label_flat  = best_label.ravel()
    intens_flat = best_intens.ravel()
    valid_flat  = valid.ravel()

    # Colour lookup table  (shape: max_lid+1, 3) in RGB
    max_lid = max(RENDER_COLORS.keys())
    lut = np.zeros((max_lid + 2, 3), dtype=np.float32)
    for lid, rgb in RENDER_COLORS.items():
        lut[lid] = rgb

    colors_rgb = lut[np.clip(label_flat, 0, max_lid)]   # (N, 3) float32

    # Simple diffuse shading: brighter objects, perspective darkening for ground
    shade = np.clip(intens_flat * 1.1, 0.2, 1.0)

    # Road texture: add noise to ground (lid=0)
    road_mask = valid_flat & (label_flat == 0)
    if add_road_texture and road_mask.any():
        noise = rng.normal(0, 12, road_mask.sum()).astype(np.float32)
        shade[road_mask] *= 1.0
        colors_rgb[road_mask] += noise[:, np.newaxis]

    # Lane marking glow
    lane_mask = valid_flat & (label_flat == 1)
    if lane_mask.any():
        shade[lane_mask] = np.clip(shade[lane_mask] * 1.4, 0, 1)

    shaded_rgb = np.clip(colors_rgb * shade[:, np.newaxis], 0, 255).astype(np.uint8)
    shaded_bgr = shaded_rgb[:, ::-1]   # RGB → BGR

    image_flat = image.reshape(N, 3)
    image_flat[valid_flat] = shaded_bgr[valid_flat]
    image = image_flat.reshape(height, width, 3)

    # ── Light fog / distance haze ─────────────────────────────────────────────
    if valid.any():
        fog_factor = np.clip(depth_map / 80.0, 0.0, 0.7)       # (H, W)
        fog_color  = np.array([220, 225, 230], dtype=np.float32)  # BGR haze
        image = np.where(
            valid[:, :, np.newaxis],
            (image.astype(np.float32) * (1 - fog_factor[:, :, np.newaxis])
             + fog_color * fog_factor[:, :, np.newaxis]).clip(0, 255).astype(np.uint8),
            image,
        )

    return image, depth_map
