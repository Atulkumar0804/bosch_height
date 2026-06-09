"""
NuScenes: Render Scene from Both Camera Heights and Compare Features
=====================================================================
Proper pipeline:

  Step 1  — Load real NuScenes camera image (h ≈ 1.51 m) + LiDAR cloud
  Step 2  — Color each LiDAR point by sampling the original camera image
            → gives a 3-D coloured point cloud from real sensor data
  Step 3  — Project those coloured 3-D points from h = 3.0 m camera pose
            → produces the RENDERED view from 3 m using actual geometry
  Step 4  — Also run dense warp (inpainted) for a gap-free 3 m image
  Step 5  — Compare: original 1 m | sparse 3 m render | dense 3 m render
  Step 6  — Per-feature crops (traffic lights, road markings, vehicles,
            pedestrians, barriers, traffic cones) from both heights

Outputs  (outputs/height_render/)
  render_01_main.png          large side-by-side: 1m vs 3m renders
  render_02_lidar_views.png   LiDAR depth map at both camera heights
  render_03_feature_grid.png  every visible object at 1m vs 3m
  render_04_road_features.png lanes / road markings / symbols
  render_05_summary.png       geometric explanation + metrics

Usage
-----
  python run_height_render_compare.py \\
      --nuscenes /home/atul/Desktop/bosch/nuscenes_data
"""

import os, sys, argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from src.nuscenes_reprojector import (
    create_lidar_depth_map, reproject_frame, project_3d_box_to_camera,
)

OUTPUT_DIR  = "outputs/height_render"
H_ORIG_LABEL = "1.5 m"      # NuScenes camera ≈ 1.51 m
H_NEW_LABEL  = "3.0 m"
TARGET_H     = 3.0
DARK_BG      = "#1a1a2e"

CAT_MAP = {
    "vehicle.car":                          "car",
    "vehicle.truck":                        "truck",
    "vehicle.bus.rigid":                    "bus",
    "vehicle.bus.bendy":                    "bus",
    "human.pedestrian.adult":               "pedestrian",
    "human.pedestrian.child":               "pedestrian",
    "human.pedestrian.police_officer":      "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "vehicle.bicycle":                      "bicycle",
    "vehicle.motorcycle":                   "motorcycle",
    "movable_object.trafficcone":           "traffic_cone",
    "movable_object.barrier":               "barrier",
    "vehicle.construction":                 "construction",
}
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
CAT_RGB  = {k: (v[2]/255, v[1]/255, v[0]/255) for k, v in CAT_BGR.items()}
BOX_EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sample(nusc, token, cam_name="CAM_FRONT"):
    from pyquaternion import Quaternion
    from nuscenes.utils.data_classes import LidarPointCloud

    sample  = nusc.get("sample", token)
    cam_sd  = nusc.get("sample_data", sample["data"][cam_name])
    img_path = os.path.join(nusc.dataroot, cam_sd["filename"])
    image   = cv2.imread(img_path)

    cam_cal = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    K       = np.array(cam_cal["camera_intrinsic"], dtype=np.float64)
    R_cam   = Quaternion(cam_cal["rotation"]).rotation_matrix
    t_cam   = np.array(cam_cal["translation"], dtype=np.float64)

    ego_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])
    ego_t    = np.array(ego_pose["translation"], dtype=np.float64)
    ego_R    = Quaternion(ego_pose["rotation"]).rotation_matrix

    lid_sd  = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pc      = LidarPointCloud.from_file(
                  os.path.join(nusc.dataroot, lid_sd["filename"]))
    pts_l   = pc.points[:3].T
    lid_cal = nusc.get("calibrated_sensor", lid_sd["calibrated_sensor_token"])
    R_lid   = Quaternion(lid_cal["rotation"]).rotation_matrix
    t_lid   = np.array(lid_cal["translation"], dtype=np.float64)

    boxes_global = nusc.get_boxes(sample["data"][cam_name])
    anns = []
    for box in boxes_global:
        cg = box.corners().T
        ce = (ego_R.T @ (cg - ego_t).T).T
        short = CAT_MAP.get(box.name, box.name.split(".")[0])
        anns.append(dict(
            category=short, full_name=box.name,
            corners_ego=ce,
            center_ego=ego_R.T @ (box.center - ego_t),
            size=box.wlh,
            distance=float(np.linalg.norm((ego_R.T @ (box.center - ego_t))[:2])),
        ))

    cam_h = float(t_cam[2])
    delta_h = TARGET_H - cam_h
    print(f"  Camera height : {cam_h:.3f} m  →  target {TARGET_H:.1f} m  "
          f"(Δh = {delta_h:+.3f} m)")
    print(f"  LiDAR points  : {len(pts_l):,}")
    print(f"  Annotations   : {len(anns)}")
    return dict(image=image, K=K, R_cam=R_cam, t_cam=t_cam,
                pts_lidar=pts_l, R_lidar=R_lid, t_lidar=t_lid,
                annotations=anns, cam_height=cam_h, delta_h=delta_h,
                sample_token=token)


# ─────────────────────────────────────────────────────────────────────────────
# CORE: render scene from new height by colouring LiDAR with camera pixels
# ─────────────────────────────────────────────────────────────────────────────

def render_from_new_height(data: dict, target_h: float = TARGET_H) -> np.ndarray:
    """
    Render the scene as seen from target_h metres camera height.

    Method:
      1. Build a dense LiDAR depth map at the new camera position
         (create_lidar_depth_map with morphological dilation gives full coverage).
      2. Use point_cloud_reproject to warp original image pixels to their
         new 3-D positions as seen from target_h.
      3. Inpaint remaining holes.
      4. Overlay the sparse LiDAR colour points (direct sensor hits) on top
         for maximum geometric accuracy where the LiDAR has real data.

    The result is a full-resolution, geometrically correct view from target_h.
    """
    from src.real_image_transformer import point_cloud_reproject, inpaint_holes

    image   = data["image"]
    K       = data["K"]
    R_cam   = data["R_cam"]
    t_cam   = data["t_cam"]
    pts_l   = data["pts_lidar"]
    R_lid   = data["R_lidar"]
    t_lid   = data["t_lidar"]
    delta_h = target_h - data["cam_height"]

    H, W    = image.shape[:2]
    fx, fy  = K[0,0], K[1,1]
    cx, cy  = K[0,2], K[1,2]

    # ── Step 1: dense LiDAR depth map at ORIGINAL camera height ───────────
    # point_cloud_reproject reads depth_map[v,u] as depth at original pixel (u,v),
    # so the depth map must be built from the original camera pose (t_cam), not
    # the raised pose.  Using t_cam_new here caused the warp to be a near no-op.
    t_cam_new = t_cam + np.array([0.0, 0.0, delta_h])
    depth_map, pts_uv = create_lidar_depth_map(
        pts_l, R_lid, t_lid, R_cam, t_cam, K, H, W,
        dilation_ksize=11)

    # ── Step 2: depth-based image warp (pass R_cam/t_cam for correct transform)
    h_orig = float(t_cam[2])
    h_new  = float(t_cam_new[2])
    warped, hole_mask = point_cloud_reproject(
        image, depth_map, h_orig, h_new, K, R_cam=R_cam, t_cam=t_cam)

    # ── Step 3: inpaint holes ─────────────────────────────────────────────
    if hole_mask.any():
        rendered_base = inpaint_holes(warped, hole_mask)
    else:
        rendered_base = warped.copy()

    # ── Step 4: overlay direct LiDAR colour hits (sparse, accurate) ────────
    # Re-project LiDAR via original camera to get pixel colours
    pts_ego   = (R_lid @ pts_l.T).T + t_lid
    pts_cam_o = (R_cam.T @ (pts_ego - t_cam).T).T
    fwd_o     = pts_cam_o[:, 2] > 0.5
    d_o       = pts_cam_o[fwd_o, 2]
    u_o       = (fx * pts_cam_o[fwd_o,0]/d_o + cx).round().astype(int)
    v_o       = (fy * pts_cam_o[fwd_o,1]/d_o + cy).round().astype(int)
    in_o      = (u_o>=0)&(u_o<W)&(v_o>=0)&(v_o<H)

    pts_3d    = pts_ego[fwd_o][in_o]               # (M,3) ego
    colors    = image[v_o[in_o], u_o[in_o]]        # (M,3) BGR

    # Re-project those coloured points from new height
    pts_nc    = (R_cam.T @ (pts_3d - t_cam_new).T).T
    fwd_n     = pts_nc[:,2] > 0.5
    d_n       = pts_nc[fwd_n, 2]
    u_n       = (fx * pts_nc[fwd_n,0]/d_n + cx).round().astype(int)
    v_n       = (fy * pts_nc[fwd_n,1]/d_n + cy).round().astype(int)
    c_n       = colors[fwd_n]
    in_n      = (u_n>=0)&(u_n<W)&(v_n>=0)&(v_n<H)

    # Paint LiDAR hits with radius-3 dots (accurate geometry)
    rendered = rendered_base.copy()
    for i in np.where(in_n)[0]:
        cv2.circle(rendered, (u_n[i], v_n[i]), 3,
                   tuple(int(c) for c in c_n[i]), -1, cv2.LINE_AA)

    n_hits = int(in_n.sum())
    cov    = n_hits * 9 / (H * W) * 100          # ~dot area estimate
    print(f"  LiDAR direct hits : {n_hits:,}  "
          f"(~{cov:.1f}% coverage with r=3 dots)")
    # Return (lidar_coloured, raw_warp_with_holes, inpainted_base)
    # raw_warp_with_holes lets callers show the displacement visually
    raw_holes = warped.copy()
    raw_holes[hole_mask] = 0          # black holes = displaced-away pixels
    return rendered, raw_holes, rendered_base


# ─────────────────────────────────────────────────────────────────────────────
# Dense reprojection (gap-filled via inpainting)
# ─────────────────────────────────────────────────────────────────────────────

def dense_reproject(data: dict, target_h: float = TARGET_H) -> dict:
    res = reproject_frame(
        data["image"], data["R_cam"], data["t_cam"],
        data["R_lidar"], data["t_lidar"], data["pts_lidar"],
        data["K"], delta_h=target_h - data["cam_height"], inpaint=True,
    )
    return res


# ─────────────────────────────────────────────────────────────────────────────
# LiDAR depth-coloured overlay
# ─────────────────────────────────────────────────────────────────────────────

def lidar_depth_render(data: dict, delta_h: float = 0.0,
                       max_d: float = 60.0) -> np.ndarray:
    """Return a depth-colorised point cloud image (black background)."""
    H, W = data["image"].shape[:2]
    _, pts_uv = create_lidar_depth_map(
        data["pts_lidar"], data["R_lidar"], data["t_lidar"],
        data["R_cam"],
        data["t_cam"] + np.array([0., 0., delta_h]),
        data["K"], H, W)

    img = np.zeros((H, W, 3), dtype=np.uint8)
    if not len(pts_uv): return img

    order  = np.argsort(-pts_uv[:, 2])
    pts_uv = pts_uv[order]
    norm   = np.clip(pts_uv[:,2] / max_d, 0, 1)
    bgr    = (plt.cm.jet(norm)[:,:3][:,::-1] * 255).astype(np.uint8)

    u = np.round(pts_uv[:,0]).astype(int)
    v = np.round(pts_uv[:,1]).astype(int)
    ok = (u>=0)&(u<W)&(v>=0)&(v<H)
    for i in np.where(ok)[0]:
        cv2.circle(img, (u[i], v[i]), 4,
                   tuple(int(c) for c in bgr[i]), -1, cv2.LINE_AA)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# 3D boxes
# ─────────────────────────────────────────────────────────────────────────────

def draw_3d_boxes(image, anns, R_cam, t_cam, K, delta_h=0.0,
                  max_dist=55.0, thickness=2):
    out = image.copy()
    H, W = out.shape[:2]
    visible = []
    for ann in sorted(anns, key=lambda a: -a["distance"]):
        if ann["distance"] > max_dist: continue
        cat   = ann["category"]
        color = CAT_BGR.get(cat, (200,200,200))
        pix, front = project_3d_box_to_camera(
            ann["corners_ego"], R_cam, t_cam, K, delta_h=delta_h)
        n = 0
        for i, j in BOX_EDGES:
            if not (front[i] and front[j]): continue
            p1 = (int(round(pix[i,0])), int(round(pix[i,1])))
            p2 = (int(round(pix[j,0])), int(round(pix[j,1])))
            if ((-W//2<=p1[0]<=W*3//2 and -H//2<=p1[1]<=H*3//2) or
                (-W//2<=p2[0]<=W*3//2 and -H//2<=p2[1]<=H*3//2)):
                cv2.line(out, p1, p2, color, thickness, cv2.LINE_AA)
                n += 1
        if n:
            vp = pix[front]
            if len(vp):
                lx = int(np.clip(vp[:,0].mean()-30, 0, W-1))
                ly = int(np.clip(vp[:,1].min()-8, 10, H-1))
                cv2.putText(out, f"{cat} {ann['distance']:.0f}m",
                            (lx,ly), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, color, 1, cv2.LINE_AA)
            x1=max(0,int(pix[front,0].min()))
            y1=max(0,int(pix[front,1].min()))
            x2=min(W,int(pix[front,0].max()))
            y2=min(H,int(pix[front,1].max()))
            visible.append({**ann,"bbox2d":(x1,y1,x2-x1,y2-y1),
                             "pix2d":pix,"in_front":front})
    return out, visible


# ─────────────────────────────────────────────────────────────────────────────
# Feature crops
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_crops(img_1m, img_3m, anns, R_cam, t_cam, K, delta_h,
                      sz=200, max_per_cat=2, max_dist=50):
    H, W = img_1m.shape[:2]
    counts = defaultdict(int)
    result = defaultdict(list)

    for ann in sorted(anns, key=lambda a: a["distance"]):
        cat  = ann["category"]
        dist = ann["distance"]
        if counts[cat] >= max_per_cat or dist > max_dist: continue

        color = CAT_BGR.get(cat, (200,200,200))

        def bbox_from_proj(dh):
            pix, front = project_3d_box_to_camera(
                ann["corners_ego"], R_cam, t_cam, K, delta_h=dh)
            vis = pix[front]
            if len(vis) < 2: return None, pix, front
            x1=max(0,int(vis[:,0].min())); y1=max(0,int(vis[:,1].min()))
            x2=min(W,int(vis[:,0].max())); y2=min(H,int(vis[:,1].max()))
            if x2<=x1 or y2<=y1: return None, pix, front
            return (x1,y1,x2-x1,y2-y1), pix, front

        bb1, pix1, fr1 = bbox_from_proj(0.0)
        bb3, pix3, fr3 = bbox_from_proj(delta_h)
        if bb1 is None: continue

        def make_crop(image, bb, pix, front, label, extra=""):
            if bb is None:
                blank = np.zeros((sz,sz,3), dtype=np.uint8)
                cv2.putText(blank,"out of view",(10,sz//2),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,100,100),1)
                return blank
            x,y,w,h = bb
            pad  = max(20, int(max(w,h)*0.30))
            x0   = max(0,x-pad);  y0 = max(0,y-pad)
            x1c  = min(W,x+w+pad); y1c = min(H,y+h+pad)
            roi  = image[y0:y1c,x0:x1c]
            if roi.size == 0:
                return np.zeros((sz,sz,3),dtype=np.uint8)
            crop = cv2.resize(roi,(sz,sz))
            sx   = sz/max(x1c-x0,1);  sy = sz/max(y1c-y0,1)
            # Draw all visible 3D box edges on crop
            for i,j in BOX_EDGES:
                if not (front[i] and front[j]): continue
                p1=( int((pix[i,0]-x0)*sx), int((pix[i,1]-y0)*sy) )
                p2=( int((pix[j,0]-x0)*sx), int((pix[j,1]-y0)*sy) )
                in_c=((-5<=p1[0]<=sz+5 and -5<=p1[1]<=sz+5) or
                      (-5<=p2[0]<=sz+5 and -5<=p2[1]<=sz+5))
                if in_c:
                    cv2.line(crop,p1,p2,color,2,cv2.LINE_AA)
            # Labels
            cv2.putText(crop, label,(4,18),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,color,1,cv2.LINE_AA)
            cv2.putText(crop, f"dist {dist:.1f}m",(4,34),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1,cv2.LINE_AA)
            if extra:
                cv2.putText(crop, extra,(4,50),
                            cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,220,50),1,cv2.LINE_AA)
            return crop

        a1 = bb1[2]*bb1[3] if bb1 else 1
        a3 = bb3[2]*bb3[3] if bb3 else 0
        ratio = a3/max(a1,1)
        vis3  = float(fr3.sum()/8)

        c1 = make_crop(img_1m, bb1, pix1, fr1,
                       f"{cat} @ {H_ORIG_LABEL}")
        c3 = make_crop(img_3m, bb3, pix3, fr3,
                       f"{cat} @ {H_NEW_LABEL}",
                       extra=f"area {ratio:.2f}x | vis {vis3:.0%}")

        result[cat].append((c1, c3, dict(
            dist=dist, ratio=ratio, vis3=vis3,
            bb1=bb1, bb3=bb3, size=ann["size"])))
        counts[cat] += 1

    return dict(result)


# ─────────────────────────────────────────────────────────────────────────────
# Lane / road marking detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_lanes(image):
    H, W = image.shape[:2]
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white  = cv2.inRange(hsv, (0,0,170), (180,40,255))
    yellow = cv2.inRange(hsv, (15,70,70), (35,255,255))
    mask   = cv2.bitwise_or(white, yellow)
    mask[:H//2] = 0
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3)

    lines = cv2.HoughLinesP(cv2.Canny(mask,30,100),1,np.pi/180,
                             threshold=25, minLineLength=30, maxLineGap=20)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    road_g = gray.copy(); road_g[:H//2] = 0
    _, sym = cv2.threshold(road_g,155,255,cv2.THRESH_BINARY)
    sym = cv2.morphologyEx(sym,cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
    ctrs,_ = cv2.findContours(sym,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    symbols = [c for c in ctrs if 150 < cv2.contourArea(c) < 60000]

    ann = image.copy()
    ov  = ann.copy()
    ov[mask>0]=(0,200,255)
    cv2.addWeighted(ov,0.4,ann,0.6,0,ann)
    if lines is not None:
        for ln in lines:
            x1,y1,x2,y2 = ln[0]
            cv2.line(ann,(x1,y1),(x2,y2),(0,255,0),2,cv2.LINE_AA)
    cv2.drawContours(ann, symbols,-1,(255,0,255),2)
    cv2.line(ann,(0,H//2),(W,H//2),(0,200,255),1,cv2.LINE_AA)
    cv2.putText(ann,"horizon",(5,H//2-4),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,255),1,cv2.LINE_AA)
    return dict(mask=mask, symbols=symbols, lines=lines,
                annotated=ann,
                lane_px=int(mask.sum()//255),
                n_lines=0 if lines is None else len(lines),
                n_sym=len(symbols))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Main comparison  (1m original | 3m LiDAR render | 3m dense warp)
# ─────────────────────────────────────────────────────────────────────────────

def fig_main_comparison(data, rendered_3m, dense_3m, out_dir):
    img_1m    = data["image"]
    h_orig    = data["cam_height"]
    h_new     = TARGET_H
    hole_pct  = dense_3m["hole_pct"]
    delta_h   = dense_3m["delta_h"]

    raw_holes  = data.get("_raw_holes_3m_")    # warp before inpaint; holes = black
    inpainted  = data.get("_inpainted_3m_", dense_3m["inpainted"])

    # Difference image: |original – inpainted|×4, shown as heatmap
    diff_f = cv2.absdiff(img_1m, inpainted).astype(np.float32)
    diff_gray = cv2.cvtColor(diff_f.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    diff_amp  = np.clip(diff_gray.astype(np.float32) * 4, 0, 255).astype(np.uint8)
    diff_heat = cv2.applyColorMap(diff_amp, cv2.COLORMAP_JET)

    # Draw 3D boxes on original and inpainted
    img1_box, _ = draw_3d_boxes(img_1m,   data["annotations"],
                                 data["R_cam"], data["t_cam"], data["K"], delta_h=0.0)
    inp3_box, _ = draw_3d_boxes(inpainted, data["annotations"],
                                 data["R_cam"], data["t_cam"], data["K"], delta_h=delta_h)

    # 4-panel layout: original | raw warp (holes=black) | inpainted | diff heatmap
    use_raw = raw_holes is not None
    n_cols  = 4 if use_raw else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 8, 7))
    fig.patch.set_facecolor(DARK_BG)

    if use_raw:
        panels = [
            (img1_box,
             f"ORIGINAL  ·  h = {h_orig:.2f} m\n"
             f"Real NuScenes image"),
            (raw_holes,
             f"RAW WARP  ·  h = {h_new:.1f} m\n"
             f"Black = displaced pixels (no source data)"),
            (inp3_box,
             f"INPAINTED  ·  h = {h_new:.1f} m\n"
             f"Holes filled · hole% = {hole_pct:.1f}"),
            (diff_heat,
             f"CHANGE HEATMAP  (×4 amplified)\n"
             f"Blue=unchanged  Red=strong shift"),
        ]
        border_colors = ["#44dd44", "#ff4444", "#4488ff", "#ffaa00"]
    else:
        panels = [
            (img1_box,  f"ORIGINAL  ·  h = {h_orig:.2f} m"),
            (inp3_box,  f"INPAINTED  ·  h = {h_new:.1f} m  (hole={hole_pct:.1f}%)"),
            (diff_heat, f"CHANGE HEATMAP  (×4)"),
        ]
        border_colors = ["#44dd44", "#4488ff", "#ffaa00"]

    for ax, (img, title), bc in zip(axes, panels, border_colors):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, color="white", fontsize=9.5, pad=6, linespacing=1.4)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(bc); sp.set_linewidth(3)

    # Category legend
    patches = []
    for cat, rgb in sorted(CAT_RGB.items()):
        patches.append(mpatches.Patch(color=rgb, label=cat))
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=8, framealpha=0.4, facecolor="#22224a",
               labelcolor="white", bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        f"NuScenes Height Shift: {h_orig:.2f} m → {h_new:.1f} m  (Δh = {delta_h:+.2f} m)\n"
        f"Raw warp shows displaced pixels · heatmap shows where view changed most",
        color="white", fontsize=11, y=1.01)

    plt.tight_layout()
    path = os.path.join(out_dir, "render_01_main.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: LiDAR depth views at both heights
# ─────────────────────────────────────────────────────────────────────────────

def fig_lidar_views(data, out_dir):
    delta_h = TARGET_H - data["cam_height"]
    ld_1m   = lidar_depth_render(data, delta_h=0.0)
    ld_3m   = lidar_depth_render(data, delta_h=delta_h)
    img_1m  = data["image"]

    # Blend: 60% image + 40% LiDAR overlay
    blend_1m = cv2.addWeighted(img_1m, 0.55,
                                cv2.addWeighted(ld_1m, 1, img_1m, 0, 0), 0.45, 0)
    blend_3m_base = data.get("_rendered3m_", ld_3m)  # use LiDAR render if available
    blend_3m = cv2.addWeighted(blend_3m_base, 0.55, ld_3m, 0.45, 0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.patch.set_facecolor(DARK_BG)

    for ax, img, title in [
        (axes[0], blend_1m,
         f"LiDAR Depth Map  ·  h = {data['cam_height']:.2f} m  "
         f"(blue = near, red = far)"),
        (axes[1], blend_3m,
         f"LiDAR Depth Map  ·  h = {TARGET_H:.1f} m  "
         f"(same cloud, new camera position)"),
    ]:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, color="white", fontsize=10, pad=5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#555577")

    # Depth colorbar
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize
    cax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    cb  = ColorbarBase(cax, cmap=plt.cm.jet,
                       norm=Normalize(0, 60), orientation="vertical")
    cb.set_label("Depth (m)", color="white", fontsize=10)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    fig.suptitle("LiDAR Point Cloud View: How Sensor Perspective Changes with Height",
                 color="white", fontsize=12, y=1.0)
    plt.tight_layout(rect=[0,0,0.91,0.97])
    path = os.path.join(out_dir, "render_02_lidar_views.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Feature grid  (every detected category, 1m vs 3m)
# ─────────────────────────────────────────────────────────────────────────────

def fig_feature_grid(data, rendered_3m, dense_3m, crops, out_dir):
    if not crops:
        print("  No crops to display.")
        return

    delta_h = data["delta_h"]
    img_1m  = data["image"]
    img_3m  = rendered_3m                    # LiDAR render = primary 3m view
    img_3m_dense = dense_3m["inpainted"]

    # Add 3D boxes to both
    img1b, _ = draw_3d_boxes(img_1m, data["annotations"],
                              data["R_cam"], data["t_cam"], data["K"], 0.0)
    img3b, _ = draw_3d_boxes(img_3m, data["annotations"],
                              data["R_cam"], data["t_cam"], data["K"], delta_h)

    rows = []
    for cat in sorted(crops.keys()):
        for c1, c3, m in crops[cat]:
            rows.append((cat, c1, c3, m))
    n = len(rows)
    if n == 0: return

    fig = plt.figure(figsize=(22, 3.5 * (n + 1) + 1))
    fig.patch.set_facecolor(DARK_BG)
    gs = GridSpec(n + 1, 2, figure=fig, hspace=0.05, wspace=0.03,
                  height_ratios=[3.5] + [1.0] * n)

    # ── Row 0: full images with boxes ────────────────────────────────────────
    ax_full1 = fig.add_subplot(gs[0, 0])
    ax_full3 = fig.add_subplot(gs[0, 1])
    for ax, img, label in [
        (ax_full1, img1b,
         f"h = {data['cam_height']:.2f} m  —  original NuScenes image"),
        (ax_full3, img3b,
         f"h = {TARGET_H:.1f} m  —  LiDAR coloured render (re-projected)"),
    ]:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(label, color="white", fontsize=10, pad=5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#555577")

    # ── Rows 1…N: feature crops ───────────────────────────────────────────────
    for ri, (cat, c1, c3, meta) in enumerate(rows):
        ax_l = fig.add_subplot(gs[ri + 1, 0])
        ax_r = fig.add_subplot(gs[ri + 1, 1])
        ax_l.imshow(cv2.cvtColor(c1, cv2.COLOR_BGR2RGB))
        ax_r.imshow(cv2.cvtColor(c3, cv2.COLOR_BGR2RGB))

        for ax in (ax_l, ax_r):
            ax.set_xticks([]); ax.set_yticks([])
            rgb = CAT_RGB.get(cat, (0.7,0.7,0.7))
            for sp in ax.spines.values():
                sp.set_edgecolor(rgb); sp.set_linewidth(2.5)

        # Right label: impact assessment
        r = meta["ratio"]
        impact = ("size ↑ (roof visible)" if r > 1.05 else
                  "similar"               if r > 0.8  else
                  "size ↓ (elevated view)"if r > 0.5  else
                  "severely reduced")
        col = ("#44dd44" if r > 0.9 else "#ffaa00" if r > 0.5 else "#ff4444")
        ax_r.set_xlabel(
            f"dist {meta['dist']:.1f}m  |  area ratio: {r:.2f}x  |  {impact}",
            color=col, fontsize=8, labelpad=2)
        ax_l.set_ylabel(cat.upper(), color=CAT_RGB.get(cat,(0.7,0.7,0.7)),
                        fontsize=9, fontweight="bold")

    fig.suptitle(
        f"Per-Feature Comparison: Real NuScenes  ·  "
        f"h={data['cam_height']:.2f}m (original)  vs  h={TARGET_H:.1f}m (rendered)",
        color="white", fontsize=12, y=0.999)

    path = os.path.join(out_dir, "render_03_feature_grid.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Road features (lanes, symbols, horizon)
# ─────────────────────────────────────────────────────────────────────────────

def fig_road_features(data, rendered_3m, dense_3m, out_dir):
    img_1m = data["image"]
    img_3m_render = rendered_3m
    img_3m_dense  = dense_3m["inpainted"]
    h_orig = data["cam_height"]
    delta_h = data["delta_h"]

    rd1 = detect_lanes(img_1m)
    rd3 = detect_lanes(img_3m_render)   # on LiDAR render
    rd3d = detect_lanes(img_3m_dense)   # on dense warp

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(DARK_BG)
    gs = GridSpec(3, 3, figure=fig, hspace=0.08, wspace=0.04)

    panels = [
        (0,0, img_1m,         f"Original  h={h_orig:.2f}m"),
        (0,1, img_3m_render,  f"LiDAR render  h={TARGET_H:.1f}m"),
        (0,2, img_3m_dense,   f"Dense warp  h={TARGET_H:.1f}m"),
        (1,0, rd1["annotated"],  f"Lane detect @ {h_orig:.2f}m  "
                                  f"({rd1['lane_px']:,}px, {rd1['n_lines']} lines, "
                                  f"{rd1['n_sym']} symbols)"),
        (1,1, rd3["annotated"],  f"Lane detect @ {TARGET_H:.1f}m (LiDAR)  "
                                  f"({rd3['lane_px']:,}px, {rd3['n_lines']} lines, "
                                  f"{rd3['n_sym']} symbols)"),
        (1,2, rd3d["annotated"], f"Lane detect @ {TARGET_H:.1f}m (dense)  "
                                  f"({rd3d['lane_px']:,}px, {rd3d['n_lines']} lines, "
                                  f"{rd3d['n_sym']} symbols)"),
        (2,0, _diff_img(img_1m, img_3m_render),
              "Pixel diff: original vs LiDAR render (bright=changed)"),
        (2,1, _diff_img(img_1m, img_3m_dense),
              "Pixel diff: original vs dense warp"),
        (2,2, _hole_vis(img_3m_dense, dense_3m["hole_mask"]),
              f"Texture starvation holes (red) — {dense_3m['hole_pct']:.1f}%"),
    ]

    for r, c, img, title in panels:
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, color="white", fontsize=8, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#555577")

    fig.suptitle("Road Features: Lane Markings, Symbols, Horizon  —  "
                 f"{h_orig:.2f}m vs {TARGET_H:.1f}m",
                 color="white", fontsize=12, y=0.995)
    plt.tight_layout(rect=[0,0,1,0.99])
    path = os.path.join(out_dir, "render_04_road_features.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


def _diff_img(a, b):
    d = cv2.absdiff(a, b)
    return cv2.applyColorMap(
        np.clip(cv2.cvtColor(d,cv2.COLOR_BGR2GRAY).astype(np.float32)*3,0,255)
        .astype(np.uint8), cv2.COLORMAP_HOT)

def _hole_vis(img, mask):
    v = img.copy()
    ov = v.copy(); ov[mask]=(30,30,220)
    cv2.addWeighted(ov,0.6,v,0.4,0,v)
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Summary metrics + geometric explanation
# ─────────────────────────────────────────────────────────────────────────────

def fig_summary(data, dense_3m, crops, out_dir):
    h_orig  = data["cam_height"]
    delta_h = data["delta_h"]

    rd1 = detect_lanes(data["image"])
    rd3 = detect_lanes(dense_3m["inpainted"])

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(DARK_BG)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    style = dict(facecolor="#12122a", tick_params={"colors":"white"})

    # ── A: Bounding box area ratio per category ───────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor("#12122a"); ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_color("#555577")
    ax.yaxis.grid(True, color="#333355", lw=0.5); ax.set_axisbelow(True)

    if crops:
        cats = sorted(crops.keys())
        ratios = [np.mean([m["ratio"] for _,_,m in crops[c]]) for c in cats]
        colors_b = [("#44dd44" if r>0.9 else "#ffaa00" if r>0.5 else "#ff4444")
                    for r in ratios]
        bars = ax.bar(cats, ratios, color=colors_b, edgecolor="white", lw=0.5)
        ax.axhline(1.0, color="white", lw=0.8, ls="--")
        ax.set_ylim(0, 1.5)
        ax.set_ylabel("Bbox area ratio (3m/1m)", color="white")
        ax.set_title("Object Apparent Size Change", color="white", fontsize=10)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        for bar, r in zip(bars, ratios):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                    f"{r:.2f}x", ha="center", color="white", fontsize=8)

    # ── B: Road feature change ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor("#12122a"); ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_color("#555577")
    ax.yaxis.grid(True, color="#333355", lw=0.5); ax.set_axisbelow(True)

    feats = ["Lane pixels\n(÷100)", "Hough lines", "Road symbols"]
    v1 = [rd1["lane_px"]//100, rd1["n_lines"], rd1["n_sym"]]
    v3 = [rd3["lane_px"]//100, rd3["n_lines"], rd3["n_sym"]]
    x  = np.arange(len(feats)); bw = 0.35
    b1 = ax.bar(x-bw/2, v1, bw, label=f"h={h_orig:.1f}m",
                color="#4488ff", edgecolor="white", lw=0.4)
    b3 = ax.bar(x+bw/2, v3, bw, label=f"h={TARGET_H:.1f}m",
                color="#ff6644", edgecolor="white", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels(feats, color="white", fontsize=8)
    ax.set_title("Road Feature Detection", color="white", fontsize=10)
    ax.legend(facecolor="#22224a", labelcolor="white", fontsize=8)
    for bars in [b1, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x()+bar.get_width()/2, h+0.1,
                        str(int(h)), ha="center", color="white", fontsize=8)

    # ── C: Annotation distance scatter ───────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.set_facecolor("#12122a"); ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_color("#555577")
    ax.xaxis.grid(True, color="#333355", lw=0.5); ax.set_axisbelow(True)

    cat_dists = defaultdict(list)
    for a in data["annotations"]:
        if a["distance"] < 60:
            cat_dists[a["category"]].append(a["distance"])
    for cat, dd in sorted(cat_dists.items()):
        rgb = CAT_RGB.get(cat, (0.7,0.7,0.7))
        ax.scatter(dd, [cat]*len(dd), c=[rgb], s=55, alpha=0.85,
                   edgecolors="white", lw=0.3)
    ax.set_xlabel("Distance from ego (m)", color="white", fontsize=9)
    ax.set_title("Annotation Distance Distribution", color="white", fontsize=10)

    # ── D: Geometric explanation diagram ─────────────────────────────────────
    ax = fig.add_subplot(gs[1, :])
    ax.set_facecolor("#12122a")
    for sp in ax.spines.values(): sp.set_color("#555577")
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Geometric Effect of Height Change on Feature Perception",
                 color="white", fontsize=10, pad=5)

    # Ground line
    ax.axhline(0, color="#888888", lw=1.5)
    ax.text(0.1, -0.25, "Ground plane", color="#888888", fontsize=8)

    # Camera at 1m
    ax.plot(0.5, h_orig, "o", color="#44dd44", ms=12, zorder=5)
    ax.annotate(f"Camera @ {h_orig:.1f}m", xy=(0.5, h_orig),
                xytext=(0.5, h_orig+0.3), color="#44dd44",
                fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->", color="#44dd44"))

    # Camera at 3m
    ax.plot(0.5, TARGET_H, "o", color="#ffaa44", ms=12, zorder=5)
    ax.annotate(f"Camera @ {TARGET_H:.1f}m", xy=(0.5, TARGET_H),
                xytext=(0.5, TARGET_H+0.3), color="#ffaa44",
                fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->", color="#ffaa44"))

    # Arrow showing height increase
    ax.annotate("", xy=(0.5, TARGET_H), xytext=(0.5, h_orig),
                arrowprops=dict(arrowstyle="<->", color="white", lw=1.5))
    ax.text(0.8, (h_orig+TARGET_H)/2, f"Δh={data['delta_h']:+.2f}m",
            color="white", fontsize=8, va="center")

    # Objects at various distances
    objects = [
        (2.5, 0.5,  "barrier\n(low)"),
        (4.0, 0.9,  "traffic\ncone"),
        (5.0, 0.8,  "pedestrian\n(short)"),
        (6.5, 0.75, "car"),
        (8.0, 1.2,  "truck"),
    ]
    for ox, oh, olabel in objects:
        ax.fill_between([ox-0.2, ox+0.2], [0, 0], [oh, oh],
                        color="#4488ff", alpha=0.6)
        ax.text(ox, oh+0.1, olabel, ha="center", color="#88bbff", fontsize=7)

        # Sight line from 1m camera
        ax.plot([0.5, ox], [h_orig, oh], "--", color="#44dd44",
                lw=0.8, alpha=0.7)
        # Sight line from 3m camera
        ax.plot([0.5, ox], [TARGET_H, oh], "--", color="#ffaa44",
                lw=0.8, alpha=0.7)

    # Annotations
    notes = [
        (3.5, 3.5,
         "At 3m: camera looks DOWN on objects\n"
         "→ Roof/top visible (texture starvation for AI trained at 1m)\n"
         "→ Horizon line shifts UP in image\n"
         "→ Road markings appear wider (better top-down view)\n"
         "→ Low objects (barriers, cones) foreshortened more",
         "#ffaa00"),
    ]
    for tx, ty, txt, tc in notes:
        ax.text(tx, ty, txt, color=tc, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="#22224a", alpha=0.85))

    fig.suptitle(
        f"Summary: Height {h_orig:.2f}m → {TARGET_H:.1f}m Feature Impact",
        color="white", fontsize=13, y=0.99)
    plt.tight_layout(rect=[0,0,1,0.97])
    path = os.path.join(out_dir, "render_05_summary.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(data, dense_3m, crops):
    rd1 = detect_lanes(data["image"])
    rd3 = detect_lanes(dense_3m["inpainted"])
    print("\n" + "═"*66)
    print(f"  RENDER COMPARISON: {data['cam_height']:.2f}m → {TARGET_H:.1f}m")
    print("═"*66)
    print(f"  Hole coverage     : {dense_3m['hole_pct']:.1f}%")
    print(f"  Lane pixels       : {rd1['lane_px']:,} → {rd3['lane_px']:,}  "
          f"({100*(rd3['lane_px']-rd1['lane_px'])/max(rd1['lane_px'],1):+.1f}%)")
    print(f"  Hough lines       : {rd1['n_lines']} → {rd3['n_lines']}")
    print(f"  Road symbols      : {rd1['n_sym']} → {rd3['n_sym']}")
    print()
    if crops:
        print(f"  {'Category':<18} {'dist':>6} {'area ratio':>10} "
              f"{'vis@3m':>8}  Impact")
        print("  " + "-"*62)
        for cat in sorted(crops.keys()):
            for _,_,m in crops[cat]:
                r  = m["ratio"]
                imp = ("roof visible"    if r>1.05 else
                       "similar"         if r>0.85 else
                       "reduced view"    if r>0.5  else
                       "severely reduced")
                flag = "✓" if r>0.9 else "!" if r>0.5 else "✗"
                print(f"  {cat:<18} {m['dist']:>6.1f} {r:>10.3f}x "
                      f"{m['vis3']:>7.0%}  {flag} {imp}")
    print("═"*66 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuscenes",  default="nuscenes_data")
    ap.add_argument("--version",   default="v1.0-mini")
    ap.add_argument("--sample",    default=None)
    ap.add_argument("--cam",       default="CAM_FRONT")
    ap.add_argument("--target-height", type=float, default=TARGET_H)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[1/6] Loading NuScenes …")
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.nuscenes, verbose=False)
    print(f"  {len(nusc.sample)} samples, {len(nusc.scene)} scenes")

    token = args.sample
    if token is None:
        best_t, best_n = None, 0
        for s in nusc.sample[:40]:
            n = sum(1 for b in nusc.get_boxes(s["data"][args.cam])
                    if CAT_MAP.get(b.name,"") in CAT_BGR)
            if n > best_n: best_n, best_t = n, s["token"]
        token = best_t or nusc.sample[0]["token"]

    print("\n[2/6] Loading sample …")
    data = load_sample(nusc, token, args.cam)

    # ── Render from 3m (LiDAR coloured) ──────────────────────────────────────
    print(f"\n[3/6] Rendering scene from {TARGET_H:.1f}m using LiDAR colours …")
    rendered_3m, raw_holes_3m, inpainted_3m = render_from_new_height(data, target_h=TARGET_H)
    data["_rendered3m_"]  = rendered_3m
    data["_raw_holes_3m_"]    = raw_holes_3m
    data["_inpainted_3m_"]    = inpainted_3m

    # ── Dense reprojection (gap-filled) ───────────────────────────────────────
    print(f"\n[4/6] Dense warp (depth-based + inpainting) …")
    dense_3m = dense_reproject(data, target_h=TARGET_H)
    print(f"  Hole coverage: {dense_3m['hole_pct']:.1f}%")

    # ── Feature crops ─────────────────────────────────────────────────────────
    print("\n[5/6] Extracting feature crops …")
    crops = get_feature_crops(
        data["image"], rendered_3m,
        data["annotations"], data["R_cam"], data["t_cam"], data["K"],
        delta_h=data["delta_h"],
    )
    print(f"  Crops: { {k: len(v) for k,v in crops.items()} }")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n[6/6] Generating figures …")
    fig_main_comparison(data, rendered_3m, dense_3m, args.output_dir)
    fig_lidar_views(data, args.output_dir)
    fig_feature_grid(data, rendered_3m, dense_3m, crops, args.output_dir)
    fig_road_features(data, rendered_3m, dense_3m, args.output_dir)
    fig_summary(data, dense_3m, crops, args.output_dir)

    print_summary(data, dense_3m, crops)
    print(f"Outputs → {os.path.abspath(args.output_dir)}/\n")


if __name__ == "__main__":
    main()
