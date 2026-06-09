"""
Real NuScenes: 1 m vs 3 m Camera Height – Feature Comparison
=============================================================
Loads a real NuScenes camera image + LiDAR point cloud, reprojects
the scene to 3 m camera height, then produces 5 figures showing how
each feature changes:

  Figure 1  fc_01_lidar_overlay.png      LiDAR depth map overlaid on both images
  Figure 2  fc_02_3d_boxes.png           NuScenes 3D boxes projected at both heights
  Figure 3  fc_03_feature_crops.png      Per-category crop comparison (1m vs 3m)
  Figure 4  fc_04_lanes_road.png         Lane / road-marking detection comparison
  Figure 5  fc_05_metrics.png            Quantitative bar-chart summary

Usage
-----
  python run_feature_comparison.py \\
      --nuscenes /home/atul/Desktop/bosch/nuscenes_data

  # use a specific sample token
  python run_feature_comparison.py \\
      --nuscenes ... --sample <token>
"""

import os, sys, argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(__file__))
from src.nuscenes_reprojector import (
    create_lidar_depth_map,
    reproject_frame,
    project_3d_box_to_camera,
)

OUTPUT_DIR  = "outputs/feature_comparison"
TARGET_H    = 3.0     # metres — simulate mounting at 3 m

# ── Category map (NuScenes full name → short name) ──────────────────────────
CAT_MAP = {
    "vehicle.car":                     "car",
    "vehicle.truck":                   "truck",
    "vehicle.bus.rigid":               "bus",
    "vehicle.bus.bendy":               "bus",
    "human.pedestrian.adult":          "pedestrian",
    "human.pedestrian.child":          "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "vehicle.bicycle":                 "bicycle",
    "vehicle.motorcycle":              "motorcycle",
    "movable_object.trafficcone":      "traffic_cone",
    "movable_object.barrier":          "barrier",
    "vehicle.construction":            "construction",
    "static_object.bicycle_rack":      "bicycle_rack",
}

# BGR colours for each category
CAT_BGR = {
    "car":          ( 40, 210,  40),
    "truck":        ( 40, 140, 230),
    "bus":          (210, 100,  40),
    "pedestrian":   ( 40,  40, 230),
    "bicycle":      (210, 185,  40),
    "motorcycle":   (185,  40, 210),
    "traffic_cone": ( 40, 230, 230),
    "barrier":      (165, 165, 165),
    "construction": (165, 100,  40),
}
CAT_RGB = {k: (v[2]/255, v[1]/255, v[0]/255) for k, v in CAT_BGR.items()}

# NuScenes 3D-box corner connectivity
BOX_EDGES = [
    (0,1),(1,2),(2,3),(3,0),   # bottom face
    (4,5),(5,6),(6,7),(7,4),   # top face
    (0,4),(1,5),(2,6),(3,7),   # vertical edges
]

DARK_BG = "#1a1a2e"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sample(nusc, sample_token: str, cam_name: str = "CAM_FRONT") -> dict:
    """Load image, LiDAR, calibration, and annotations for one sample."""
    from pyquaternion import Quaternion
    from nuscenes.utils.data_classes import LidarPointCloud

    sample  = nusc.get("sample", sample_token)

    # ── Camera ───────────────────────────────────────────────────────────────
    cam_sd   = nusc.get("sample_data", sample["data"][cam_name])
    img_path = os.path.join(nusc.dataroot, cam_sd["filename"])
    image    = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(img_path)

    cam_cal  = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    K        = np.array(cam_cal["camera_intrinsic"], dtype=np.float64)
    R_cam    = Quaternion(cam_cal["rotation"]).rotation_matrix
    t_cam    = np.array(cam_cal["translation"], dtype=np.float64)

    # ── Ego pose (needed to convert global→ego for annotations) ─────────────
    ego_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])
    ego_t    = np.array(ego_pose["translation"], dtype=np.float64)
    ego_R    = Quaternion(ego_pose["rotation"]).rotation_matrix

    # ── LiDAR ────────────────────────────────────────────────────────────────
    lid_sd   = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lid_path = os.path.join(nusc.dataroot, lid_sd["filename"])
    pc       = LidarPointCloud.from_file(lid_path)
    pts_lidar = pc.points[:3].T                     # (N,3) in LiDAR sensor frame
    lid_cal  = nusc.get("calibrated_sensor", lid_sd["calibrated_sensor_token"])
    R_lidar  = Quaternion(lid_cal["rotation"]).rotation_matrix
    t_lidar  = np.array(lid_cal["translation"], dtype=np.float64)

    # ── Annotations: global → ego ────────────────────────────────────────────
    boxes_global = nusc.get_boxes(sample["data"][cam_name])
    annotations  = []
    for box in boxes_global:
        corners_global = box.corners().T              # (8,3) global frame
        corners_ego    = (ego_R.T @ (corners_global - ego_t).T).T
        center_ego     = ego_R.T @ (box.center - ego_t)
        short = CAT_MAP.get(box.name, box.name.split(".")[0])
        dist  = float(np.linalg.norm(center_ego[:2]))
        annotations.append({
            "category":    short,
            "full_name":   box.name,
            "corners_ego": corners_ego,
            "center_ego":  center_ego,
            "size":        box.wlh,
            "distance":    dist,
        })

    cam_h = float(t_cam[2])
    print(f"  Sample : {sample_token}")
    print(f"  Image  : {img_path.split('/')[-1]}  ({image.shape[1]}×{image.shape[0]})")
    print(f"  Cam h  : {cam_h:.3f} m  →  target {TARGET_H:.1f} m  (Δh={TARGET_H-cam_h:+.3f} m)")
    print(f"  LiDAR  : {len(pts_lidar):,} pts")
    print(f"  Annots : {len(annotations)}  ({len(set(a['category'] for a in annotations))} cats)")

    return dict(
        image=image, K=K, R_cam=R_cam, t_cam=t_cam,
        pts_lidar=pts_lidar, R_lidar=R_lidar, t_lidar=t_lidar,
        annotations=annotations, cam_height=cam_h,
        sample_token=sample_token,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reprojection
# ─────────────────────────────────────────────────────────────────────────────

def reproject_to_target(data: dict, target_h: float = TARGET_H) -> dict:
    delta_h = target_h - data["cam_height"]
    res = reproject_frame(
        data["image"],
        data["R_cam"], data["t_cam"],
        data["R_lidar"], data["t_lidar"],
        data["pts_lidar"], data["K"],
        delta_h=delta_h, inpaint=True,
    )
    res.update(K=data["K"], R_cam=data["R_cam"], t_cam=data["t_cam"],
               annotations=data["annotations"],
               pts_lidar=data["pts_lidar"],
               R_lidar=data["R_lidar"], t_lidar=data["t_lidar"],
               target_height=target_h)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# 3. LiDAR coloured overlay
# ─────────────────────────────────────────────────────────────────────────────

def lidar_overlay(
    image: np.ndarray,
    pts_lidar: np.ndarray,
    R_lidar: np.ndarray, t_lidar: np.ndarray,
    R_cam:   np.ndarray, t_cam:   np.ndarray,
    K:       np.ndarray,
    delta_h: float = 0.0,
    max_depth: float = 60.0,
    dot_r: int = 4,
) -> np.ndarray:
    """Overlay LiDAR points coloured by depth (blue=near, red=far)."""
    H, W = image.shape[:2]
    _, pts_uv = create_lidar_depth_map(
        pts_lidar, R_lidar, t_lidar,
        R_cam, t_cam + np.array([0.0, 0.0, delta_h]),
        K, H, W,
    )
    out = image.copy()
    if not len(pts_uv):
        return out

    # Sort far→near so near points draw on top
    order = np.argsort(-pts_uv[:, 2])
    pts_uv = pts_uv[order]

    depths_norm = np.clip(pts_uv[:, 2] / max_depth, 0, 1)
    rgb   = (plt.cm.jet(depths_norm)[:, :3] * 255).astype(np.uint8)
    bgr   = rgb[:, ::-1]

    u_i = np.round(pts_uv[:, 0]).astype(int)
    v_i = np.round(pts_uv[:, 1]).astype(int)
    valid = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
    for idx in np.where(valid)[0]:
        cv2.circle(out, (u_i[idx], v_i[idx]), dot_r,
                   tuple(int(c) for c in bgr[idx]), -1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. 3D bounding-box drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_boxes(
    image: np.ndarray,
    annotations: List[dict],
    R_cam: np.ndarray, t_cam: np.ndarray,
    K: np.ndarray,
    delta_h: float = 0.0,
    max_dist: float = 55.0,
    thickness: int = 2,
) -> Tuple[np.ndarray, List[dict]]:
    """Project NuScenes 3D boxes onto image; return annotated image + visible list."""
    out  = image.copy()
    H, W = image.shape[:2]
    visible = []

    for ann in sorted(annotations, key=lambda a: -a["distance"]):
        if ann["distance"] > max_dist:
            continue
        cat   = ann["category"]
        color = CAT_BGR.get(cat, (200, 200, 200))

        pix, front = project_3d_box_to_camera(
            ann["corners_ego"], R_cam, t_cam, K, delta_h=delta_h)

        n_drawn = 0
        for i, j in BOX_EDGES:
            if not (front[i] and front[j]):
                continue
            p1 = (int(round(pix[i, 0])), int(round(pix[i, 1])))
            p2 = (int(round(pix[j, 0])), int(round(pix[j, 1])))
            in_frame = ((-W//2 <= p1[0] <= W*3//2 and -H//2 <= p1[1] <= H*3//2) or
                        (-W//2 <= p2[0] <= W*3//2 and -H//2 <= p2[1] <= H*3//2))
            if in_frame:
                cv2.line(out, p1, p2, color, thickness, cv2.LINE_AA)
                n_drawn += 1

        if n_drawn:
            vis_pix = pix[front]
            if len(vis_pix):
                lx = int(np.clip(vis_pix[:, 0].mean() - 30, 0, W-1))
                ly = int(np.clip(vis_pix[:, 1].min() - 8, 10, H-1))
                label = f"{cat} {ann['distance']:.0f}m"
                cv2.putText(out, label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

            x1 = max(0, int(pix[front, 0].min()))
            y1 = max(0, int(pix[front, 1].min()))
            x2 = min(W, int(pix[front, 0].max()))
            y2 = min(H, int(pix[front, 1].max()))
            visible.append({**ann, "bbox2d": (x1, y1, x2-x1, y2-y1),
                             "corners2d": pix, "in_front": front})
    return out, visible


# ─────────────────────────────────────────────────────────────────────────────
# 5. Lane & road marking detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_road_features(image: np.ndarray) -> dict:
    """Detect white/yellow lanes, road symbols, and horizon line."""
    H, W = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # White markings
    white = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    # Yellow markings
    yellow = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))

    lane_mask = cv2.bitwise_or(white, yellow)
    road_top  = H // 2                  # only look below horizon
    lane_mask[:road_top] = 0

    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, k3)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN,  k3)

    # Hough lines
    edges = cv2.Canny(lane_mask, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25,
                             minLineLength=35, maxLineGap=25)

    # Road symbols: bright compact regions on road surface
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    road_gray = gray.copy(); road_gray[:road_top] = 0
    _, sym_bin = cv2.threshold(road_gray, 155, 255, cv2.THRESH_BINARY)
    sym_bin = cv2.morphologyEx(sym_bin, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    contours, _ = cv2.findContours(sym_bin, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    symbols = [c for c in contours if 150 < cv2.contourArea(c) < 60000]

    # Build annotated image
    ann = image.copy()
    overlay = ann.copy()
    overlay[lane_mask > 0] = (0, 200, 255)          # yellow-ish tint for lanes
    cv2.addWeighted(overlay, 0.4, ann, 0.6, 0, ann)
    if lines is not None:
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            cv2.line(ann, (x1,y1),(x2,y2), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.drawContours(ann, symbols, -1, (255, 0, 255), 2)
    # Horizon line
    cv2.line(ann, (0, road_top), (W, road_top), (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(ann, "horizon", (5, road_top-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1, cv2.LINE_AA)

    return dict(
        lane_mask=lane_mask, sym_bin=sym_bin, lines=lines, symbols=symbols,
        annotated=ann,
        lane_pixels=int(lane_mask.sum() // 255),
        n_lines=0 if lines is None else len(lines),
        n_symbols=len(symbols),
        sym_area=int(sum(cv2.contourArea(c) for c in symbols)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Feature crops
# ─────────────────────────────────────────────────────────────────────────────

def extract_feature_crops(
    image_1m: np.ndarray,
    image_3m: np.ndarray,
    annotations: List[dict],
    R_cam: np.ndarray, t_cam: np.ndarray,
    K: np.ndarray,
    delta_h: float,
    crop_sz: int = 200,
    max_per_cat: int = 2,
    max_dist: float = 50.0,
) -> Dict[str, list]:
    """
    For each annotation, project at Δh=0 and Δh=delta_h,
    extract matching crops from both images, annotate with metrics.
    Returns {category: [(crop_1m, crop_3m, metrics_dict), ...]}
    """
    H, W = image_1m.shape[:2]
    cat_counts: Dict[str, int] = defaultdict(int)
    results: Dict[str, list] = defaultdict(list)

    for ann in sorted(annotations, key=lambda a: a["distance"]):
        cat  = ann["category"]
        dist = ann["distance"]
        if cat_counts[cat] >= max_per_cat or dist > max_dist:
            continue

        color = CAT_BGR.get(cat, (200, 200, 200))

        def project_bbox(dh):
            pix, front = project_3d_box_to_camera(
                ann["corners_ego"], R_cam, t_cam, K, delta_h=dh)
            vis = pix[front]
            if len(vis) < 2:
                return None, pix, front
            x1 = max(0, int(vis[:,0].min()))
            y1 = max(0, int(vis[:,1].min()))
            x2 = min(W, int(vis[:,0].max()))
            y2 = min(H, int(vis[:,1].max()))
            if x2 <= x1 or y2 <= y1:
                return None, pix, front
            return (x1, y1, x2-x1, y2-y1), pix, front

        bbox_1m, pix_1m, front_1m = project_bbox(0.0)
        bbox_3m, pix_3m, front_3m = project_bbox(delta_h)

        if bbox_1m is None:
            continue

        def make_crop(image, bbox, pix, front, label, extra=""):
            if bbox is None:
                blank = np.zeros((crop_sz, crop_sz, 3), dtype=np.uint8)
                cv2.putText(blank, "out of view", (10, crop_sz//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)
                return blank
            x, y, w, h = bbox
            pad = max(15, int(max(w, h) * 0.25))
            x0 = max(0, x - pad); y0 = max(0, y - pad)
            x1c = min(W, x+w+pad); y1c = min(H, y+h+pad)
            roi = image[y0:y1c, x0:x1c]
            if roi.size == 0:
                return np.zeros((crop_sz, crop_sz, 3), dtype=np.uint8)
            crop = cv2.resize(roi, (crop_sz, crop_sz))
            # Scale bbox to crop coords
            sx = crop_sz / max(x1c - x0, 1)
            sy = crop_sz / max(y1c - y0, 1)
            bx1 = int((x - x0) * sx); by1 = int((y - y0) * sy)
            bx2 = int((x + w - x0) * sx); by2 = int((y + h - y0) * sy)
            cv2.rectangle(crop, (bx1, by1), (bx2, by2), color, 2)
            # Draw visible box corners
            for ci in range(8):
                if not front[ci]:
                    continue
                cx_ = int((pix[ci,0] - x0) * sx)
                cy_ = int((pix[ci,1] - y0) * sy)
                if 0 <= cx_ < crop_sz and 0 <= cy_ < crop_sz:
                    cv2.circle(crop, (cx_, cy_), 3, color, -1)
            cv2.putText(crop, label, (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            cv2.putText(crop, f"{dist:.1f}m", (4, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
            if extra:
                cv2.putText(crop, extra, (4, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,220,50), 1, cv2.LINE_AA)
            return crop

        area_1m = (bbox_1m[2] * bbox_1m[3]) if bbox_1m else 1
        area_3m = (bbox_3m[2] * bbox_3m[3]) if bbox_3m else 0
        ratio   = area_3m / max(area_1m, 1)
        visible_frac = front_3m.sum() / 8.0

        c1 = make_crop(image_1m, bbox_1m, pix_1m, front_1m, f"{cat} @ 1m")
        c3 = make_crop(image_3m, bbox_3m, pix_3m, front_3m, f"{cat} @ 3m",
                       extra=f"area:{ratio:.2f}x  vis:{visible_frac:.0%}")

        results[cat].append((c1, c3, {
            "distance": dist, "area_ratio": ratio,
            "visible_frac_3m": float(visible_frac),
            "bbox_1m": bbox_1m, "bbox_3m": bbox_3m,
            "size_wlh": ann["size"],
        }))
        cat_counts[cat] += 1

    return dict(results)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: LiDAR depth overlay
# ─────────────────────────────────────────────────────────────────────────────

def fig_lidar_overlay(data: dict, result_3m: dict, out_dir: str) -> None:
    img_1m = data["image"]
    img_3m = result_3m["inpainted"]
    delta_h = result_3m["delta_h"]
    h_orig  = result_3m["h_orig"]
    h_new   = result_3m["h_new"]

    ov_1m = lidar_overlay(img_1m,
                          data["pts_lidar"], data["R_lidar"], data["t_lidar"],
                          data["R_cam"], data["t_cam"], data["K"],
                          delta_h=0.0)
    ov_3m = lidar_overlay(img_3m,
                          data["pts_lidar"], data["R_lidar"], data["t_lidar"],
                          data["R_cam"], data["t_cam"], data["K"],
                          delta_h=delta_h)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.patch.set_facecolor(DARK_BG)

    for ax, img, title in [
        (axes[0], ov_1m, f"Original  (h = {h_orig:.2f} m)  –  LiDAR depth overlay"),
        (axes[1], ov_3m, f"Reprojected  (h = {h_new:.2f} m, Δh = {delta_h:+.2f} m)  –  LiDAR at new height"),
    ]:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, color="white", fontsize=12, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#555577")

    # Depth colour legend
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize
    cax = fig.add_axes([0.92, 0.12, 0.012, 0.76])
    cb  = ColorbarBase(cax, cmap=plt.cm.jet,
                       norm=Normalize(0, 60), orientation="vertical")
    cb.set_label("Depth (m)", color="white", fontsize=10)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    fig.suptitle("LiDAR Point Cloud: 1 m vs 3 m Camera View",
                 color="white", fontsize=14, y=0.99)
    plt.tight_layout(rect=[0, 0, 0.91, 0.97])
    path = os.path.join(out_dir, "fc_01_lidar_overlay.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: 3D bounding boxes
# ─────────────────────────────────────────────────────────────────────────────

def fig_3d_boxes(data: dict, result_3m: dict, out_dir: str) -> None:
    img_1m   = data["image"]
    img_3m   = result_3m["inpainted"]
    anns     = data["annotations"]
    R_cam    = data["R_cam"]
    t_cam    = data["t_cam"]
    K        = data["K"]
    delta_h  = result_3m["delta_h"]
    h_orig   = result_3m["h_orig"]
    h_new    = result_3m["h_new"]

    boxed_1m, vis_1m = draw_boxes(img_1m, anns, R_cam, t_cam, K, delta_h=0.0)
    boxed_3m, vis_3m = draw_boxes(img_3m, anns, R_cam, t_cam, K, delta_h=delta_h)

    # Count per category
    def cat_count(lst):
        c = defaultdict(int)
        for b in lst: c[b["category"]] += 1
        return dict(c)
    cnt_1m = cat_count(vis_1m)
    cnt_3m = cat_count(vis_3m)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.patch.set_facecolor(DARK_BG)

    for ax, img, cnt, title in [
        (axes[0], boxed_1m, cnt_1m, f"Height {h_orig:.2f} m  –  3D boxes from NuScenes annotations"),
        (axes[1], boxed_3m, cnt_3m, f"Height {h_new:.2f} m  –  same boxes reprojected to new camera pose"),
    ]:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, color="white", fontsize=12, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#555577")
        # Category legend inside axis
        patches = []
        for cat, n in sorted(cnt.items()):
            rgb = CAT_RGB.get(cat, (0.7, 0.7, 0.7))
            patches.append(mpatches.Patch(color=rgb, label=f"{cat} ({n})"))
        if patches:
            ax.legend(handles=patches, loc="upper right", fontsize=7,
                      framealpha=0.5, facecolor="#222244", labelcolor="white",
                      ncol=min(len(patches), 4))

    fig.suptitle("NuScenes 3D Bounding Boxes: 1 m vs 3 m Camera Height",
                 color="white", fontsize=14, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, "fc_02_3d_boxes.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")
    print(f"    Visible @ 1m: {cnt_1m}")
    print(f"    Visible @ 3m: {cnt_3m}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Per-category feature crops
# ─────────────────────────────────────────────────────────────────────────────

def fig_feature_crops(
    data: dict, result_3m: dict, crops: Dict[str, list], out_dir: str
) -> None:
    if not crops:
        print("  No feature crops to display.")
        return

    # Flatten to (label, crop_1m, crop_3m, metrics) rows
    rows = []
    for cat in sorted(crops.keys()):
        for crop_1m, crop_3m, meta in crops[cat]:
            rows.append((cat, crop_1m, crop_3m, meta))

    n_rows = len(rows)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, 2,
                              figsize=(9, 3.2 * n_rows),
                              squeeze=False)
    fig.patch.set_facecolor(DARK_BG)

    col_labels = [f"h = {result_3m['h_orig']:.2f} m  (original)",
                  f"h = {result_3m['h_new']:.2f} m  (reprojected)"]

    for ri, (cat, c1, c3, meta) in enumerate(rows):
        for ci, (ax, crop) in enumerate(zip(axes[ri], [c1, c3])):
            ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([])
            rgb = CAT_RGB.get(cat, (0.7, 0.7, 0.7))
            for sp in ax.spines.values():
                sp.set_edgecolor(rgb); sp.set_linewidth(2)
            if ri == 0:
                ax.set_title(col_labels[ci], color="white", fontsize=9, pad=4)
            if ci == 0:
                ax.set_ylabel(cat, color=rgb, fontsize=10, fontweight="bold")

        # Right-side annotation
        area_r = meta["area_ratio"]
        vis_f  = meta["visible_frac_3m"]
        status = ("OK" if area_r > 0.7 else
                  "REDUCED" if area_r > 0.4 else "CRITICAL")
        col = "#44dd44" if area_r > 0.7 else "#ffaa00" if area_r > 0.4 else "#ff4444"
        axes[ri][1].set_xlabel(
            f"area ratio: {area_r:.2f}x  |  corners visible: {vis_f:.0%}  |  {status}",
            color=col, fontsize=8, labelpad=3)

    fig.suptitle("Per-Category Object Appearance: 1 m vs 3 m Camera Height",
                 color="white", fontsize=12, y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.996])
    path = os.path.join(out_dir, "fc_03_feature_crops.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Lane & road marking detection
# ─────────────────────────────────────────────────────────────────────────────

def fig_lanes_road(data: dict, result_3m: dict, out_dir: str) -> None:
    img_1m  = data["image"]
    img_3m  = result_3m["inpainted"]
    h_orig  = result_3m["h_orig"]
    h_new   = result_3m["h_new"]
    hole_pct = result_3m["hole_pct"]

    rd_1m = detect_road_features(img_1m)
    rd_3m = detect_road_features(img_3m)

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(DARK_BG)
    gs  = GridSpec(3, 2, figure=fig, hspace=0.1, wspace=0.04)

    panels = [
        (0, 0, img_1m,              f"Original  h={h_orig:.2f}m  –  camera image"),
        (0, 1, img_3m,              f"Reprojected  h={h_new:.2f}m  –  hole={hole_pct:.1f}%"),
        (1, 0, rd_1m["annotated"],  "Lane markings + road symbols detected @ 1m"),
        (1, 1, rd_3m["annotated"],  "Lane markings + road symbols detected @ 3m"),
        (2, 0, _make_diff_img(img_1m, img_3m), "Pixel difference (|1m – 3m|)"),
        (2, 1, _make_hole_vis(img_3m, result_3m["hole_mask"]),
               "Texture-starvation holes (red = missing data)"),
    ]

    for r, c, img, title in panels:
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, color="white", fontsize=9, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#555577")

    # Metrics text box
    metrics = (
        f"1m:  lanes={rd_1m['lane_pixels']:,}px  lines={rd_1m['n_lines']}  "
        f"symbols={rd_1m['n_symbols']}  sym_area={rd_1m['sym_area']:,}px²\n"
        f"3m:  lanes={rd_3m['lane_pixels']:,}px  lines={rd_3m['n_lines']}  "
        f"symbols={rd_3m['n_symbols']}  sym_area={rd_3m['sym_area']:,}px²\n"
        f"Lane coverage change: "
        f"{100*(rd_3m['lane_pixels']-rd_1m['lane_pixels'])/max(rd_1m['lane_pixels'],1):+.1f}%  |  "
        f"Symbol count change: {rd_3m['n_symbols']-rd_1m['n_symbols']:+d}"
    )
    fig.text(0.5, 0.01, metrics, ha="center", va="bottom",
             color="#aaccff", fontsize=8.5,
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#12122a", alpha=0.8))

    fig.suptitle("Lane Lines & Road Markings: 1 m vs 3 m Camera Height",
                 color="white", fontsize=13, y=0.995)
    plt.tight_layout(rect=[0, 0.05, 1, 0.992])
    path = os.path.join(out_dir, "fc_04_lanes_road.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


def _make_diff_img(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(img1, img2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_norm = (diff_gray.astype(np.float32) * 3).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(diff_norm, cv2.COLORMAP_HOT)


def _make_hole_vis(image: np.ndarray, hole_mask: np.ndarray) -> np.ndarray:
    vis = image.copy()
    overlay = vis.copy()
    overlay[hole_mask] = (30, 30, 220)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
    return vis


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Quantitative metrics
# ─────────────────────────────────────────────────────────────────────────────

def fig_metrics(
    data: dict, result_3m: dict, crops: Dict[str, list], out_dir: str
) -> None:
    rd_1m = detect_road_features(data["image"])
    rd_3m = detect_road_features(result_3m["inpainted"])
    h_orig  = result_3m["h_orig"]
    h_new   = result_3m["h_new"]
    delta_h = result_3m["delta_h"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes.flat:
        ax.set_facecolor("#12122a")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_color("#555577")
        ax.yaxis.grid(True, color="#333355", linewidth=0.5)
        ax.set_axisbelow(True)

    # ── Panel A: per-category area ratio ─────────────────────────────────────
    ax = axes[0, 0]
    cats_sorted = sorted(crops.keys())
    if cats_sorted:
        ratios = []
        for cat in cats_sorted:
            r_list = [m["area_ratio"] for _, _, m in crops[cat]]
            ratios.append(np.mean(r_list))
        colors_bar = [("#44dd44" if r > 0.7 else "#ffaa00" if r > 0.4 else "#ff4444")
                      for r in ratios]
        bars = ax.bar(cats_sorted, ratios, color=colors_bar,
                      edgecolor="white", linewidth=0.5)
        ax.axhline(1.0, color="white", linewidth=0.8, linestyle="--")
        ax.axhline(0.5, color="#ff8888", linewidth=0.8, linestyle=":")
        ax.set_ylim(0, 1.4)
        ax.set_ylabel("Bbox area ratio (3m / 1m)", color="white")
        ax.set_title("Object Apparent Size Change", color="white", fontsize=11)
        ax.tick_params(axis="x", rotation=20)
        for bar, r in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{r:.2f}x", ha="center", color="white", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No crops", transform=ax.transAxes,
                ha="center", color="white")

    # ── Panel B: corner visibility ────────────────────────────────────────────
    ax = axes[0, 1]
    if cats_sorted:
        vis_fracs = []
        for cat in cats_sorted:
            vf_list = [m["visible_frac_3m"] for _, _, m in crops[cat]]
            vis_fracs.append(np.mean(vf_list) * 100)
        ax.bar(cats_sorted, vis_fracs, color="#4488ff",
               edgecolor="white", linewidth=0.5)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Box corners visible @ 3m (%)", color="white")
        ax.set_title("3D Box Visibility at 3 m", color="white", fontsize=11)
        ax.tick_params(axis="x", rotation=20)
        ax.axhline(50, color="#ffaa00", linewidth=0.8, linestyle=":")
        for i, v in enumerate(vis_fracs):
            ax.text(i, v + 1, f"{v:.0f}%", ha="center",
                    color="white", fontsize=9)

    # ── Panel C: road feature counts ─────────────────────────────────────────
    ax = axes[1, 0]
    feature_names = ["Lane pixels\n(÷100)", "Hough lines", "Road symbols"]
    v1 = [rd_1m["lane_pixels"]//100, rd_1m["n_lines"], rd_1m["n_symbols"]]
    v3 = [rd_3m["lane_pixels"]//100, rd_3m["n_lines"], rd_3m["n_symbols"]]
    x  = np.arange(len(feature_names))
    bw = 0.32
    b1 = ax.bar(x - bw/2, v1, bw, label=f"h={h_orig:.1f}m",
                color="#4488ff", edgecolor="white", linewidth=0.5)
    b3 = ax.bar(x + bw/2, v3, bw, label=f"h={h_new:.1f}m",
                color="#ff6644", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(feature_names, color="white")
    ax.set_ylabel("Count", color="white")
    ax.set_title("Road Feature Detection Comparison", color="white", fontsize=11)
    ax.legend(facecolor="#22224a", labelcolor="white", fontsize=9)
    for bars in [b1, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                        str(int(h)), ha="center", color="white", fontsize=8)

    # ── Panel D: object distance distribution ────────────────────────────────
    ax = axes[1, 1]
    dists = [a["distance"] for a in data["annotations"] if a["distance"] < 60]
    cat_dists = defaultdict(list)
    for a in data["annotations"]:
        if a["distance"] < 60:
            cat_dists[a["category"]].append(a["distance"])

    for cat, dd in sorted(cat_dists.items()):
        rgb = CAT_RGB.get(cat, (0.7, 0.7, 0.7))
        ax.scatter(dd, [cat] * len(dd), c=[rgb], s=60, alpha=0.8,
                   edgecolors="white", linewidths=0.4)
    ax.set_xlabel("Distance from ego (m)", color="white")
    ax.set_title("Annotation Distance Distribution", color="white", fontsize=11)
    ax.tick_params(axis="y", labelcolor="white", labelsize=8)

    fig.suptitle(
        f"Quantitative Feature Analysis: h={h_orig:.2f}m → {h_new:.2f}m  (Δh={delta_h:+.2f}m)",
        color="white", fontsize=13, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    path = os.path.join(out_dir, "fc_05_metrics.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(data: dict, result_3m: dict, crops: Dict[str, list]) -> None:
    h_orig  = result_3m["h_orig"]
    h_new   = result_3m["h_new"]
    delta_h = result_3m["delta_h"]
    hole_pct = result_3m["hole_pct"]

    rd_1m = detect_road_features(data["image"])
    rd_3m = detect_road_features(result_3m["inpainted"])

    print("\n" + "═"*68)
    print(f"  FEATURE COMPARISON:  h = {h_orig:.2f} m  →  {h_new:.2f} m  (Δh = {delta_h:+.2f} m)")
    print("═"*68)
    print(f"  Texture-starvation holes : {hole_pct:.1f} % of image")
    print()
    print(f"  {'Feature':<22}  {'@ 1m':>8}  {'@ 3m':>8}  {'Change':>10}")
    print("  " + "-"*54)
    print(f"  {'Lane pixels (×100)':<22}  "
          f"{rd_1m['lane_pixels']//100:>8}  "
          f"{rd_3m['lane_pixels']//100:>8}  "
          f"{100*(rd_3m['lane_pixels']-rd_1m['lane_pixels'])/max(rd_1m['lane_pixels'],1):>+9.1f}%")
    print(f"  {'Hough lines detected':<22}  "
          f"{rd_1m['n_lines']:>8}  "
          f"{rd_3m['n_lines']:>8}  "
          f"{rd_3m['n_lines']-rd_1m['n_lines']:>+10d}")
    print(f"  {'Road symbols':<22}  "
          f"{rd_1m['n_symbols']:>8}  "
          f"{rd_3m['n_symbols']:>8}  "
          f"{rd_3m['n_symbols']-rd_1m['n_symbols']:>+10d}")
    print()
    if crops:
        print(f"  {'Category':<18}  {'Dist(m)':>7}  {'AreaRatio':>9}  {'Vis@3m':>8}")
        print("  " + "-"*46)
        for cat in sorted(crops.keys()):
            for _, _, m in crops[cat]:
                flag = (" ✓" if m["area_ratio"] > 0.7 else
                        " !" if m["area_ratio"] > 0.4 else " ✗")
                print(f"  {cat:<18}  {m['distance']:>7.1f}  "
                      f"{m['area_ratio']:>9.3f}  "
                      f"{m['visible_frac_3m']:>7.0%} {flag}")
    print("═"*68 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NuScenes 1m vs 3m feature comparison")
    parser.add_argument("--nuscenes", default="nuscenes_data",
                        help="NuScenes dataroot")
    parser.add_argument("--version",  default="v1.0-mini")
    parser.add_argument("--sample",   default=None,
                        help="Sample token (default: first sample)")
    parser.add_argument("--cam",      default="CAM_FRONT")
    parser.add_argument("--target-height", type=float, default=TARGET_H,
                        help=f"Target camera height in metres (default: {TARGET_H})")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load NuScenes ─────────────────────────────────────────────────────────
    print("\n[1/6] Loading NuScenes …")
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.nuscenes, verbose=False)
    print(f"  {len(nusc.sample)} samples, {len(nusc.scene)} scenes")

    token = args.sample
    if token is None:
        # Pick a sample with more annotations for a better demo
        best_token, best_n = None, 0
        for s in nusc.sample[:40]:          # search first 40
            boxes = nusc.get_boxes(s["data"][args.cam])
            n_vis = sum(1 for b in boxes if CAT_MAP.get(b.name, "") in CAT_BGR)
            if n_vis > best_n:
                best_n, best_token = n_vis, s["token"]
        token = best_token or nusc.sample[0]["token"]

    # ── Load sample data ──────────────────────────────────────────────────────
    print("\n[2/6] Loading sample data …")
    data = load_sample(nusc, token, cam_name=args.cam)

    # ── Reproject to target height ────────────────────────────────────────────
    print(f"\n[3/6] Reprojecting to {args.target_height:.1f} m …")
    result_3m = reproject_to_target(data, target_h=args.target_height)
    print(f"  Hole coverage: {result_3m['hole_pct']:.1f} %")

    # ── Extract feature crops ─────────────────────────────────────────────────
    print("\n[4/6] Extracting per-category feature crops …")
    crops = extract_feature_crops(
        data["image"], result_3m["inpainted"],
        data["annotations"],
        data["R_cam"], data["t_cam"], data["K"],
        delta_h=result_3m["delta_h"],
    )
    print(f"  Crops: { {k: len(v) for k, v in crops.items()} }")

    # ── Generate figures ──────────────────────────────────────────────────────
    print("\n[5/6] Generating figures …")
    fig_lidar_overlay(data, result_3m, args.output_dir)
    fig_3d_boxes(data, result_3m, args.output_dir)
    fig_feature_crops(data, result_3m, crops, args.output_dir)
    fig_lanes_road(data, result_3m, args.output_dir)
    fig_metrics(data, result_3m, crops, args.output_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n[6/6] Summary …")
    print_summary(data, result_3m, crops)

    print(f"All outputs → {os.path.abspath(args.output_dir)}/\n")


if __name__ == "__main__":
    main()
