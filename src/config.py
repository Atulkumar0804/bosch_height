"""
Central configuration for the sensor-height comparison pipeline.
All scene/sensor/analysis parameters are defined here.
"""
import os
import numpy as np

# ── Sensor heights to compare (metres above ground) ──────────────────────────
HEIGHTS = {
    "1m_nuScenes": 1.0,
    "2m_mid":      2.0,
    "3m_elevated": 3.0,
}

# nuScenes – set NUSCENES_DATAROOT env var or edit this path
NUSCENES_DATAROOT = os.environ.get("NUSCENES_DATAROOT", "/data/nuscenes")
NUSCENES_VERSION  = "v1.0-mini"
# Actual LIDAR_TOP height in nuScenes (~1.84 m) — used when normalising
NUSCENES_LIDAR_HEIGHT = 1.84

# ── LiDAR simulation parameters ──────────────────────────────────────────────
LIDAR_NUM_BEAMS      = 32
LIDAR_V_FOV_DEG      = (-15.0, 15.0)   # vertical field-of-view per beam range
LIDAR_H_RES_DEG      = 0.25            # horizontal angular resolution
LIDAR_MAX_RANGE_M    = 80.0

# ── Camera parameters ─────────────────────────────────────────────────────────
CAM_WIDTH   = 1600
CAM_HEIGHT  = 900
CAM_FOV_H   = 70.0                     # horizontal FOV in degrees
# Derived intrinsics
CAM_F       = CAM_WIDTH / (2 * np.tan(np.radians(CAM_FOV_H / 2)))  # ~1143 px
CAM_CX      = CAM_WIDTH  / 2.0         # 800
CAM_CY      = CAM_HEIGHT / 2.0         # 450

CAM_INTRINSIC = np.array([
    [CAM_F,    0,    CAM_CX],
    [0,     CAM_F,  CAM_CY],
    [0,        0,       1 ],
], dtype=np.float64)

# Camera looks straight ahead (+X world) from sensor position
# Camera frame: X=right, Y=down, Z=forward
# Rotation: camera ← world
CAM_R_FROM_WORLD = np.array([
    [ 0, -1,  0],    # cam X = world -Y  (right in cam = –left in world)
    [ 0,  0, -1],    # cam Y = world -Z  (down in cam  = –up in world)
    [ 1,  0,  0],    # cam Z = world +X  (fwd  in cam  = +fwd in world)
], dtype=np.float64)

# ── Synthetic scene dimensions ────────────────────────────────────────────────
SCENE_LENGTH = 80.0      # metres ahead (+X)
SCENE_WIDTH  = 40.0      # metres left/right (±Y)
LANE_WIDTH   = 3.5       # metres

# ── Feature labels ────────────────────────────────────────────────────────────
FEATURE_LABELS = {
    0: "ground",
    1: "lane_marking",
    2: "road_symbol",
    3: "traffic_light",
    4: "traffic_sign",
    5: "car",
    6: "truck",
    7: "pedestrian",
    8: "pole",
    9: "building",
}
# Reverse lookup used by multiple modules
NAME_TO_ID = {v: k for k, v in FEATURE_LABELS.items()}
FEATURE_COLORS = {
    0: (128, 128, 128),  # ground   – grey
    1: (255, 255, 255),  # lane     – white
    2: (255, 220,  50),  # symbol   – yellow
    3: (255,  80,  80),  # t-light  – red
    4: (80,  180, 255),  # t-sign   – blue
    5: ( 50, 200,  50),  # car      – green
    6: (200, 100,  50),  # truck    – orange
    7: (255, 140, 200),  # pedestrian – pink
    8: (160, 160, 160),  # pole     – light grey
    9: (100,  80,  60),  # building – brown
}

# Minimum points required to consider a feature "detected"
DETECTION_THRESHOLD = {
    "lane_marking":  3,
    "road_symbol":   5,
    "traffic_light": 2,
    "traffic_sign":  2,
    "car":           8,
    "truck":        15,
    "pedestrian":    3,
}

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
