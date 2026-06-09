"""
Single-Image → Point Cloud → 3m Render → Feature Detection
============================================================

Pipeline (image-only, no LiDAR):
  1. Load NuScenes camera image + calibration
  2. MiDaS monocular depth estimation → relative depth map
  3. Scale relative → metric depth using camera height + ground-plane fit
  4. Build dense coloured point cloud (one point per valid pixel)
  5. Visualise point cloud (depth-coloured 2-D projection + simulated 3-D tilt)
  6. Reproject point cloud to 3 m camera height → rendered image
  7. Detect traffic lights (HSV + blob) and road symbols (colour + Hough)
     on BOTH the original and 3 m rendered views
  8. Save side-by-side comparison figures

Usage:
  python run_monocular_3m.py --nuscenes nuscenes_data
  python run_monocular_3m.py --nuscenes nuscenes_data --sample <tok> --cam CAM_FRONT
"""

import argparse, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

from src.real_image_transformer import point_cloud_reproject, inpaint_holes

TARGET_H   = 3.0
OUTPUT_DIR = "outputs/monocular_3m"
DARK_BG    = "#0a0a14"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# 1. MiDaS monocular depth
# ─────────────────────────────────────────────────────────────────────────────

def load_midas():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small",
                           trust_repo=True, verbose=False)
    model.to(DEVICE).eval()
    tf = torch.hub.load("intel-isl/MiDaS", "transforms",
                        trust_repo=True, verbose=False).small_transform
    return model, tf


def midas_relative_depth(model, tf, image_bgr):
    """Return (H, W) float32 relative inverse-depth (larger = closer)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inp = tf(rgb).to(DEVICE)
    with torch.no_grad():
        pred = model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=image_bgr.shape[:2],
            mode="bicubic", align_corners=False,
        ).squeeze()
    return pred.cpu().numpy().astype(np.float32)


def scale_to_metric(inv_depth_rel, K, cam_h, horizon_row=None):
    """
    Convert MiDaS relative inverse-depth to metric depth (metres).

    Strategy:
      • Detect horizon row from edge energy
      • For road pixels (v > horizon + margin), use flat-ground:
            metric_depth = f_y * cam_h / (v - horizon_v)
      • Fit a linear scale+offset [a, b] so that
            metric_depth ≈ a * inv_depth_rel + b   (least squares on road rows)
      • Apply the fitted scale globally

    Returns metric depth map (H,W) float32, horizon_row (int).
    """
    H, W  = inv_depth_rel.shape
    fy    = K[1, 1]
    cy    = K[1, 2]

    # ── horizon detection ────────────────────────────────────────────────────
    if horizon_row is None:
        gray    = np.zeros((H, W), dtype=np.float32)  # placeholder
        # Use the inverse-depth gradient: horizon is where depth changes sharply
        ddy     = np.abs(np.gradient(inv_depth_rel, axis=0))
        row_e   = ddy.mean(axis=1)
        lo, hi  = H // 5, 2 * H // 3
        horizon_row = int(lo + np.argmax(row_e[lo:hi]))

    # ── flat-ground metric depth for road band ───────────────────────────────
    margin       = 20
    road_rows    = np.arange(horizon_row + margin, H, dtype=np.float32)
    dv           = road_rows - horizon_row
    gt_depth_row = fy * cam_h / np.maximum(dv, 1.0)        # (n_road_rows,)

    rel_row_mean = inv_depth_rel[
        (horizon_row + margin):, W//3: 2*W//3].mean(axis=1)   # (n_road_rows,)

    # Least-squares:  gt ≈ a * rel + b
    valid = rel_row_mean > 0
    if valid.sum() > 10:
        A = np.stack([rel_row_mean[valid],
                      np.ones(valid.sum())], axis=1)
        coeffs, _, _, _ = np.linalg.lstsq(A, gt_depth_row[valid], rcond=None)
        a, b = float(coeffs[0]), float(coeffs[1])
    else:
        # Fallback: match median
        a = float(np.median(gt_depth_row) / max(np.median(rel_row_mean), 1e-3))
        b = 0.0

    metric = a * inv_depth_rel + b
    metric = np.clip(metric, 0.3, 120.0).astype(np.float32)
    return metric, horizon_row


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build dense coloured point cloud
# ─────────────────────────────────────────────────────────────────────────────

def build_point_cloud(image_bgr, depth_metric, K, stride=2):
    """
    Return arrays:
        pts_cam  (N, 3)  – camera-frame 3D (X_right, Y_down, Z_fwd)
        pts_rgb  (N, 3)  – uint8 RGB colours
    stride: sub-sampling stride (1=every pixel, 2=every other)
    """
    H, W   = image_bgr.shape[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    vs = np.arange(0, H, stride)
    us = np.arange(0, W, stride)
    uu, vv = np.meshgrid(us, vs)

    d = depth_metric[vv, uu]
    valid = (d > 0.3) & (d < 100.0)

    d   = d[valid]
    u   = uu[valid].astype(np.float32)
    v   = vv[valid].astype(np.float32)

    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d

    pts_cam = np.stack([X, Y, Z], axis=1)

    bgr = image_bgr[vv[valid], uu[valid]]
    rgb = bgr[:, ::-1]

    return pts_cam.astype(np.float32), rgb.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Reproject point cloud to new camera height
# ─────────────────────────────────────────────────────────────────────────────

def reproject_pc_to_height(pts_cam, pts_rgb, K, R_cam, t_cam, target_h, H, W):
    """
    Reproject dense point cloud to a camera raised by (target_h - t_cam[2]).

    ONLY the camera Z-position changes: t_new = t_cam + [0, 0, delta_h].
    Camera orientation R_cam is unchanged (no tilt, no pitch).
    """
    delta_h = target_h - float(t_cam[2])
    fx, fy  = K[0, 0], K[1, 1]
    cx, cy  = K[0, 2], K[1, 2]

    # Camera frame → ego frame  (pts_ego = R_cam @ pts_cam + t_cam)
    pts_ego = (R_cam @ pts_cam.T).T + t_cam[None, :]       # (N, 3)

    # Raise camera Z only — same orientation
    t_new  = t_cam.copy()
    t_new[2] += delta_h

    # Ego → new camera frame
    pts_nc = (R_cam.T @ (pts_ego - t_new[None, :]).T).T    # (N, 3)

    fwd    = pts_nc[:, 2] > 0.1
    d      = pts_nc[fwd, 2]
    u_f    = fx * pts_nc[fwd, 0] / d + cx
    v_f    = fy * pts_nc[fwd, 1] / d + cy
    u_i    = np.round(u_f).astype(np.int32)
    v_i    = np.round(v_f).astype(np.int32)
    cols   = pts_rgb[fwd]                                   # RGB

    in_img = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)

    # Sort far-to-near so near points overwrite
    order    = np.argsort(-d[in_img])
    u_ok     = u_i[in_img][order]
    v_ok     = v_i[in_img][order]
    c_ok     = cols[in_img][order]
    d_sorted = d[in_img][order]

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    z_buf  = np.full((H, W), np.inf)

    closer = d_sorted < z_buf[v_ok, u_ok]
    v_c = v_ok[closer]; u_c = u_ok[closer]
    z_buf[v_c, u_c]  = d_sorted[closer]
    canvas[v_c, u_c] = c_ok[closer, ::-1]   # RGB → BGR

    # Filled mask: dilate painted pixels to close sub-pixel gaps
    filled_bin = (z_buf < np.inf).astype(np.uint8)
    filled_dil = cv2.dilate(filled_bin, np.ones((5, 5), np.uint8))
    # Apply dilation-based fill: copy nearest painted neighbour colour
    canvas_dil = cv2.dilate(canvas, np.ones((5, 5), np.uint8))
    gap_mask   = (filled_dil > 0) & (filled_bin == 0)
    canvas[gap_mask] = canvas_dil[gap_mask]
    filled_bin[gap_mask] = 1

    hole_mask = filled_bin == 0
    hole_pct  = 100.0 * hole_mask.sum() / (H * W)

    raw_canvas = canvas.copy()
    if hole_mask.any():
        canvas = inpaint_holes(canvas, hole_mask)

    return canvas, raw_canvas, hole_mask, hole_pct


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature detection: traffic lights + road symbols
# ─────────────────────────────────────────────────────────────────────────────

def detect_traffic_lights(image_bgr):
    """
    Detect traffic light bounding boxes using HSV colour blobs.
    Returns list of (x, y, w, h, colour_label).
    """
    hsv   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    H, W  = image_bgr.shape[:2]
    boxes = []

    colour_ranges = {
        "RED":   [(np.array([0,120,100]),   np.array([10,255,255])),
                  (np.array([160,120,100]), np.array([180,255,255]))],
        "GREEN": [(np.array([35,60,60]),    np.array([90,255,255]))],
        "AMBER": [(np.array([10,120,150]),  np.array([30,255,255]))],
    }

    for label, ranges in colour_ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lo, hi) in ranges:
            mask |= cv2.inRange(hsv, lo, hi)

        # Restrict to upper 60% of image (traffic lights not on ground)
        mask[int(H * 0.6):] = 0

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 40:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w > W * 0.3 or h > H * 0.4:
                continue
            boxes.append((x, y, w, h, label))

    return boxes


def detect_road_symbols(image_bgr):
    """
    Detect lane lines, crosswalks, road text (BUS, STOP, arrows).
    Returns dict with keys: lanes, symbols, markings_mask.
    """
    H, W  = image_bgr.shape[:2]
    hsv   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # White markings
    white_mask = cv2.inRange(hsv,
                             np.array([0,   0, 160]),
                             np.array([180, 55, 255]))
    # Yellow markings
    yellow_mask = cv2.inRange(hsv,
                              np.array([15, 70, 70]),
                              np.array([40, 255, 255]))

    road_half = H // 2
    combined  = cv2.bitwise_or(white_mask, yellow_mask)
    combined[:road_half] = 0      # ignore upper half for road markings

    # Hough lines for lanes
    edges = cv2.Canny(gray, 50, 150)
    edges[:road_half] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=60, minLineLength=40, maxLineGap=30)
    lanes = lines if lines is not None else []

    # Contours for symbols (large blobs)
    kernel   = np.ones((5, 5), np.uint8)
    sym_mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    cnts, _  = cv2.findContours(sym_mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    symbols  = [cv2.boundingRect(c) for c in cnts
                if 200 < cv2.contourArea(c) < W * H * 0.05]

    return dict(lanes=lanes, symbols=symbols,
                markings_mask=combined, sym_mask=sym_mask)


TL_COLORS = {"RED": (0, 0, 220), "GREEN": (0, 200, 0), "AMBER": (0, 165, 255)}

def draw_traffic_lights(img, boxes, label_prefix=""):
    out = img.copy()
    for (x, y, w, h, col) in boxes:
        bgr = TL_COLORS.get(col, (200, 200, 200))
        cv2.rectangle(out, (x, y), (x+w, y+h), bgr, 2)
        cv2.rectangle(out, (x, y-16), (x+w, y), bgr, -1)
        cv2.putText(out, f"{label_prefix}{col}",
                    (x+2, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255,255,255), 1, cv2.LINE_AA)
    return out


def draw_road_symbols(img, result):
    out = img.copy()
    H, W = img.shape[:2]

    # Lane lines
    if len(result["lanes"]):
        for line in result["lanes"]:
            x1, y1, x2, y2 = line[0]
            cv2.line(out, (x1,y1), (x2,y2), (0,255,255), 2)

    # Marking blobs
    for (x,y,w,h) in result["symbols"]:
        cv2.rectangle(out, (x,y), (x+w,y+h), (255,100,0), 2)
        cv2.putText(out, "marking", (x,y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,100,0), 1)

    # Overlay marking mask in magenta (semi-transparent)
    mask_3ch = np.zeros_like(out)
    mask_3ch[result["markings_mask"] > 0] = (255, 0, 255)
    out = cv2.addWeighted(out, 0.85, mask_3ch, 0.35, 0)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Point cloud visualisation (depth-coloured 2-D)
# ─────────────────────────────────────────────────────────────────────────────

def depth_colormap(depth_metric):
    """Render depth as JET colourmap image."""
    d_norm = np.clip(depth_metric / 60.0, 0, 1)
    jet    = (plt.cm.jet(d_norm)[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(jet, cv2.COLOR_RGB2BGR)


def pointcloud_overlay(image_bgr, depth_metric, stride=4):
    """
    Draw coloured depth dots on top of image.
    Each dot coloured by depth (jet) with radius proportional to 1/depth.
    """
    H, W   = image_bgr.shape[:2]
    canvas = image_bgr.copy()
    jet_bg = depth_colormap(depth_metric)

    for v in range(0, H, stride):
        for u in range(0, W, stride):
            d = float(depth_metric[v, u])
            if d < 0.5 or d > 80:
                continue
            col = tuple(int(c) for c in jet_bg[v, u])
            r   = max(1, min(4, int(8 / max(d, 1))))
            cv2.circle(canvas, (u, v), r, col, -1, cv2.LINE_AA)
    return canvas


def simulated_3d_tilt(image_bgr, depth_metric, K, tilt_deg=25):
    """
    Warp the image to simulate looking at the scene from a tilted 3-D
    perspective — gives a pseudo-BEV feel for the point cloud visualisation.
    Uses a vertical homography based on depth gradient.
    """
    H, W  = image_bgr.shape[:2]
    theta = np.radians(tilt_deg)

    # Simple vertical projective warp:
    # source corners → destination corners (bottom stays, top compresses)
    src = np.float32([[0,0],[W,0],[W,H],[0,H]])
    d_top = W * np.tan(theta) * 0.3   # horizontal compression at top
    dst = np.float32([
        [d_top,     int(H*0.05)],
        [W - d_top, int(H*0.05)],
        [W,         H],
        [0,         H],
    ])
    M   = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(image_bgr, M, (W, H),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def fig_pipeline(image, depth_metric, pc_canvas, rendered,
                 horizon_row, hole_pct, cam_h, out_dir):
    """
    Figure 1:  Pipeline overview
      [Original] [Depth Map] [Point Cloud Overlay] [Rendered 3m]
    """
    pc_vis = pointcloud_overlay(image, depth_metric, stride=3)
    depth_vis = depth_colormap(depth_metric)

    # Mark horizon on depth vis
    cv2.line(depth_vis, (0, horizon_row), (depth_vis.shape[1], horizon_row),
             (255, 255, 255), 2)
    cv2.putText(depth_vis, "horizon", (10, horizon_row - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    panels = [
        (image,       f"ORIGINAL  ·  h = {cam_h:.2f} m"),
        (depth_vis,   f"METRIC DEPTH MAP\n(MiDaS + ground-plane scaling)"),
        (pc_vis,      f"DENSE POINT CLOUD OVERLAY\n(jet = depth, 1 pt per pixel)"),
        (rendered,    f"RENDERED FROM {TARGET_H:.1f} m\n(hole {hole_pct:.1f}% inpainted)"),
    ]
    border_cols = ["#44dd44", "#4488ff", "#ffaa22", "#ff5533"]

    fig, axes = plt.subplots(1, 4, figsize=(32, 7), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    for ax, (img, title), bc in zip(axes, panels, border_cols):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, color="white", fontsize=10, pad=6,
                     linespacing=1.6, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(bc); sp.set_linewidth(4)

    fig.suptitle(
        f"Single-Image → Point Cloud → {TARGET_H:.0f} m Render  ·  "
        f"NuScenes CAM_FRONT  ·  MiDaS depth estimation",
        color="white", fontsize=13, y=1.02, fontweight="bold")

    path = os.path.join(out_dir, "mono_01_pipeline.png")
    plt.tight_layout(pad=0.4)
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_traffic_lights(image, rendered, tl_orig, tl_3m, cam_h, out_dir):
    """
    Figure 2:  Traffic light detection comparison
      Top row:  detection on original | detection on 3m
      Bottom row: cropped zoom of each detected TL
    """
    ann_orig = draw_traffic_lights(image,    tl_orig, "")
    ann_3m   = draw_traffic_lights(rendered, tl_3m,   "")

    H, W     = image.shape[:2]

    fig = plt.figure(figsize=(20, 10), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.05)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_3m   = fig.add_subplot(gs[0, 1])
    ax_zo   = fig.add_subplot(gs[1, 0])
    ax_z3   = fig.add_subplot(gs[1, 1])

    ax_orig.imshow(cv2.cvtColor(ann_orig, cv2.COLOR_BGR2RGB))
    ax_orig.set_title(f"Traffic Light Detection  ·  h = {cam_h:.2f} m\n"
                      f"{len(tl_orig)} signals detected",
                      color="#88ff88", fontsize=11, pad=5, fontweight="bold")

    ax_3m.imshow(cv2.cvtColor(ann_3m, cv2.COLOR_BGR2RGB))
    ax_3m.set_title(f"Traffic Light Detection  ·  h = {TARGET_H:.1f} m\n"
                    f"{len(tl_3m)} signals detected",
                    color="#ffaa44", fontsize=11, pad=5, fontweight="bold")

    for ax in (ax_orig, ax_3m):
        ax.set_xticks([]); ax.set_yticks([])

    for sp in ax_orig.spines.values(): sp.set_edgecolor("#44dd44"); sp.set_linewidth(3)
    for sp in ax_3m.spines.values():   sp.set_edgecolor("#ff6633"); sp.set_linewidth(3)

    def make_zoom_strip(ax, img, boxes, label):
        crops = []
        for (x, y, w, h, col) in boxes[:6]:
            pad = max(20, int(max(w, h) * 0.6))
            x1  = max(0, x - pad); y1 = max(0, y - pad)
            x2  = min(W, x+w+pad); y2 = min(H, y+h+pad)
            cr  = img[y1:y2, x1:x2]
            if cr.size > 0:
                cr_r = cv2.resize(cr, (120, 120))
                crops.append((cr_r, col))
        if crops:
            strip = np.hstack([c for c, _ in crops])
            ax.imshow(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))
            ax.set_title(label, color="white", fontsize=9, pad=4)
        else:
            ax.text(0.5, 0.5, "No signals detected", color="gray",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label, color="white", fontsize=9, pad=4)
        ax.set_xticks([]); ax.set_yticks([])

    make_zoom_strip(ax_zo, image,    tl_orig,
                    f"Zoom: traffic lights at {cam_h:.2f} m")
    make_zoom_strip(ax_z3, rendered, tl_3m,
                    f"Zoom: traffic lights at {TARGET_H:.1f} m")

    fig.suptitle(
        "Traffic Light Detection: Original vs 3 m Rendered View\n"
        "Higher camera → signals appear lower in frame, perspective angle changes",
        color="white", fontsize=13, y=1.02, fontweight="bold")

    path = os.path.join(out_dir, "mono_02_traffic_lights.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_road_symbols(image, rendered, rs_orig, rs_3m, cam_h, out_dir):
    """
    Figure 3:  Road symbol detection comparison
      [original + markings] | [3m + markings]
      [marking mask orig]   | [marking mask 3m]
    """
    ann_orig = draw_road_symbols(image,    rs_orig)
    ann_3m   = draw_road_symbols(rendered, rs_3m)

    m_orig = cv2.cvtColor(rs_orig["markings_mask"], cv2.COLOR_GRAY2BGR)
    m_3m   = cv2.cvtColor(rs_3m["markings_mask"],   cv2.COLOR_GRAY2BGR)
    # colour masks
    m_orig[rs_orig["markings_mask"] > 0] = (0, 255, 200)
    m_3m  [rs_3m["markings_mask"]   > 0] = (0, 180, 255)

    panels = [
        (ann_orig, f"Road Symbols  ·  h = {cam_h:.2f} m\n"
                   f"Lanes: {len(rs_orig['lanes'])}  Symbols: {len(rs_orig['symbols'])}"),
        (ann_3m,   f"Road Symbols  ·  h = {TARGET_H:.1f} m\n"
                   f"Lanes: {len(rs_3m['lanes'])}  Symbols: {len(rs_3m['symbols'])}"),
        (m_orig,   f"Marking pixels @ {cam_h:.2f} m  (white+yellow)"),
        (m_3m,     f"Marking pixels @ {TARGET_H:.1f} m  (white+yellow)"),
    ]
    bcolors = ["#44dd44", "#ff6633", "#44dd44", "#ff6633"]

    fig, axes = plt.subplots(2, 2, figsize=(20, 10), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    axes = axes.ravel()

    for ax, (img, title), bc in zip(axes, panels, bcolors):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, color="white", fontsize=10, pad=5,
                     linespacing=1.5, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(bc); sp.set_linewidth(3)

    # Stat text
    orig_px = int((rs_orig["markings_mask"] > 0).sum())
    new_px  = int((rs_3m["markings_mask"]   > 0).sum())
    pct     = (new_px - orig_px) / max(orig_px, 1) * 100
    fig.text(0.5, -0.01,
             f"Marking pixel count:  {cam_h:.2f}m = {orig_px:,}  →  "
             f"{TARGET_H:.1f}m = {new_px:,}  ({pct:+.1f}%)\n"
             "From 3 m the camera looks down more steeply → road markings "
             "appear more foreshortened, AI trained at 1.5 m may miss them",
             ha="center", color="#aaaacc", fontsize=10)

    fig.suptitle(
        "Road Symbol Detection: Original vs 3 m Rendered View",
        color="white", fontsize=13, y=1.02, fontweight="bold")
    plt.tight_layout(pad=0.5)

    path = os.path.join(out_dir, "mono_03_road_symbols.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_side_by_side(image, rendered, cam_h, out_dir):
    """
    Figure 4:  Clean 3-panel matching reference comparison_300cm style.
    Original | Raw point-cloud warp holes visible | 3m rendered
    """
    # Create raw warp (before inpaint) with holes as black
    from src.real_image_transformer import geometric_depth_from_image
    H, W = image.shape[:2]

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)

    panels = [
        (image,    f"ORIGINAL  ·  h = {cam_h:.2f} m"),
        (rendered, f"RENDERED FROM {TARGET_H:.1f} m  (MiDaS depth)"),
    ]
    bcolors = ["#44dd44", "#ff5533"]

    for ax, (img, title), bc in zip(axes[:2], panels, bcolors):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, color="white", fontsize=12, pad=8, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(bc); sp.set_linewidth(4)

    # Difference heat
    diff   = cv2.absdiff(image, rendered)
    diff_g = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_a = np.clip(diff_g.astype(np.float32) * 5, 0, 255).astype(np.uint8)
    heat   = cv2.applyColorMap(diff_a, cv2.COLORMAP_JET)
    axes[2].imshow(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"CHANGE HEATMAP  (×5)\nblue=unchanged  red=max shift",
                      color="white", fontsize=12, pad=8, fontweight="bold")
    axes[2].set_xticks([]); axes[2].set_yticks([])
    for sp in axes[2].spines.values():
        sp.set_edgecolor("#ffaa00"); sp.set_linewidth(4)

    fig.suptitle(
        f"Camera Height Shift: {cam_h:.2f} m  →  {TARGET_H:.1f} m  "
        f"(Δh = {TARGET_H-cam_h:+.2f} m)  ·  MiDaS monocular depth",
        color="white", fontsize=14, y=1.02, fontweight="bold")
    plt.tight_layout(pad=0.3)

    path = os.path.join(out_dir, "mono_04_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_pointcloud(image, depth_metric, cam_h, out_dir):
    """
    Figure 5: Dense point cloud coloured by metric depth.
    Left: image  |  Right: point cloud overlay (jet by depth)
    """
    pc_vis = pointcloud_overlay(image, depth_metric, stride=3)
    depth_vis = depth_colormap(depth_metric)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)

    axes[0].imshow(cv2.cvtColor(image,     cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Original  ·  h = {cam_h:.2f} m",
                      color="#88ff88", fontsize=11, pad=6, fontweight="bold")

    axes[1].imshow(cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title("MiDaS Metric Depth Map\n(blue=far  red=near)",
                      color="#4488ff", fontsize=11, pad=6, fontweight="bold")

    axes[2].imshow(cv2.cvtColor(pc_vis,    cv2.COLOR_BGR2RGB))
    axes[2].set_title("Dense Point Cloud Overlay\n(jet=depth, 1 point per pixel)",
                      color="#ffaa22", fontsize=11, pad=6, fontweight="bold")

    border_cols = ["#44dd44", "#4488ff", "#ffaa22"]
    for ax, bc in zip(axes, border_cols):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(bc); sp.set_linewidth(3)

    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors  import Normalize
    cax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    cb  = ColorbarBase(cax, cmap=plt.cm.jet,
                       norm=Normalize(0, 60), orientation="vertical")
    cb.set_label("Depth (m)", color="white", fontsize=10)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    fig.suptitle(
        f"Point Cloud from Single Image (MiDaS)  ·  {len(image.ravel())//3:,} pixels → dense 3-D",
        color="white", fontsize=13, y=1.02, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.91, 1.0])

    path = os.path.join(out_dir, "mono_05_pointcloud.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(nusc, sample_token, cam_name):
    sample  = nusc.get('sample', sample_token)
    cam_sd  = nusc.get('sample_data', sample['data'][cam_name])
    cs      = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    R_cam   = Quaternion(cs['rotation']).rotation_matrix
    t_cam   = np.array(cs['translation'])
    K       = np.array(cs['camera_intrinsic'], dtype=np.float32)
    image   = cv2.imread(f"nuscenes_data/{cam_sd['filename']}")
    return image, K, R_cam, t_cam


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuscenes",   default="nuscenes_data")
    ap.add_argument("--version",    default="v1.0-mini")
    ap.add_argument("--sample",     default=None)
    ap.add_argument("--cam",        default="CAM_FRONT")
    ap.add_argument("--target-h",   type=float, default=TARGET_H)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[1/7] Loading NuScenes …  (device: {DEVICE})")
    nusc = NuScenes(version=args.version, dataroot=args.nuscenes, verbose=False)

    if args.sample is None:
        best = max(nusc.sample,
                   key=lambda s: len(s['anns']) if args.cam in s['data'] else 0)
        args.sample = best['token']
        print(f"  Auto-selected: {args.sample[:24]}…  ({len(best['anns'])} anns)")

    image, K, R_cam, t_cam = load_data(nusc, args.sample, args.cam)
    cam_h   = float(t_cam[2])
    delta_h = args.target_h - cam_h
    H, W    = image.shape[:2]
    print(f"  Image: {W}×{H}   Camera height: {cam_h:.3f}m → {args.target_h:.1f}m  "
          f"(Δh={delta_h:+.3f}m)")

    print("[2/7] MiDaS depth estimation …")
    model, tf  = load_midas()
    inv_depth  = midas_relative_depth(model, tf, image)
    print(f"  Relative depth range: {inv_depth.min():.2f}–{inv_depth.max():.2f}")

    print("[3/7] Scaling to metric depth …")
    depth_metric, horizon_row = scale_to_metric(inv_depth, K, cam_h)
    valid_mask = (depth_metric > 0.5) & (depth_metric < 100.0)
    print(f"  Metric depth:  median={np.median(depth_metric[valid_mask]):.2f}m  "
          f"horizon_row={horizon_row}")

    print("[4/7] Building dense point cloud …")
    pts_cam, pts_rgb = build_point_cloud(image, depth_metric, K, stride=1)
    print(f"  Points: {len(pts_cam):,}")

    print("[5/7] Reprojecting to 3 m …")
    rendered, pc_canvas, hole_mask, hole_pct = reproject_pc_to_height(
        pts_cam, pts_rgb, K, R_cam, t_cam, args.target_h, H, W)
    print(f"  Hole coverage: {hole_pct:.1f}%")

    print("[6/7] Detecting features …")
    tl_orig = detect_traffic_lights(image)
    tl_3m   = detect_traffic_lights(rendered)
    rs_orig = detect_road_symbols(image)
    rs_3m   = detect_road_symbols(rendered)
    print(f"  Traffic lights  @ {cam_h:.2f}m: {len(tl_orig)}  "
          f"@ {args.target_h:.1f}m: {len(tl_3m)}")
    print(f"  Lane lines      @ {cam_h:.2f}m: {len(rs_orig['lanes'])}  "
          f"@ {args.target_h:.1f}m: {len(rs_3m['lanes'])}")
    print(f"  Road symbols    @ {cam_h:.2f}m: {len(rs_orig['symbols'])}  "
          f"@ {args.target_h:.1f}m: {len(rs_3m['symbols'])}")
    marking_orig = int((rs_orig["markings_mask"] > 0).sum())
    marking_3m   = int((rs_3m["markings_mask"]   > 0).sum())
    print(f"  Marking pixels  @ {cam_h:.2f}m: {marking_orig:,}  "
          f"@ {args.target_h:.1f}m: {marking_3m:,}  "
          f"({(marking_3m-marking_orig)/max(marking_orig,1)*100:+.1f}%)")

    print("[7/7] Saving figures …")
    fig_pipeline(image, depth_metric, pc_canvas, rendered,
                 horizon_row, hole_pct, cam_h, args.output_dir)
    fig_traffic_lights(image, rendered, tl_orig, tl_3m, cam_h, args.output_dir)
    fig_road_symbols(image, rendered, rs_orig, rs_3m, cam_h, args.output_dir)
    fig_side_by_side(image, rendered, cam_h, args.output_dir)
    fig_pointcloud(image, depth_metric, cam_h, args.output_dir)

    print(f"\nOutputs → {os.path.abspath(args.output_dir)}/")
    print("  mono_01_pipeline.png      — original | depth | point cloud | rendered")
    print("  mono_02_traffic_lights.png — TL detection @ 1.5m vs 3m")
    print("  mono_03_road_symbols.png  — lane + marking detection @ 1.5m vs 3m")
    print("  mono_04_comparison.png    — clean 3-panel + heatmap")
    print("  mono_05_pointcloud.png    — depth map + dense point cloud overlay")


if __name__ == "__main__":
    main()
