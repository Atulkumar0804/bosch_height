"""
3-Panel Height Comparison  —  matching comparison_300cm style
=============================================================

Panel 1: Original image at camera height h
Panel 2: Raw point-cloud warp (holes = black displacement bands)
Panel 3: Rendered view from 3.0 m  (inpainted)

Usage:
  python run_comparison_300cm.py --nuscenes nuscenes_data
  python run_comparison_300cm.py --nuscenes nuscenes_data --sample <token> --cam CAM_FRONT
"""

import argparse, os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

from src.nuscenes_reprojector import create_lidar_depth_map
from src.real_image_transformer import point_cloud_reproject, inpaint_holes

TARGET_H   = 3.0
OUTPUT_DIR = "outputs/comparison_300cm"
DARK_BG    = "#0d0d1a"


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
    H, W    = image.shape[:2]

    lid_sd  = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    cs_lid  = nusc.get('calibrated_sensor', lid_sd['calibrated_sensor_token'])
    R_lid   = Quaternion(cs_lid['rotation']).rotation_matrix
    t_lid   = np.array(cs_lid['translation'])
    pts_raw = np.fromfile(f"nuscenes_data/{lid_sd['filename']}", dtype=np.float32).reshape(-1,5)
    pts_l   = pts_raw[:,:3]

    delta_h = TARGET_H - float(t_cam[2])

    return dict(
        image=image, K=K, R_cam=R_cam, t_cam=t_cam,
        R_lid=R_lid, t_lid=t_lid, pts_lidar=pts_l,
        cam_height=float(t_cam[2]), delta_h=delta_h,
        H=H, W=W,
        sample_token=sample_token, cam_name=cam_name,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_3m(data):
    """Return (raw_warp_black_holes, inpainted, hole_pct, depth_map)."""
    image   = data["image"]
    K       = data["K"]
    R_cam   = data["R_cam"]
    t_cam   = data["t_cam"]
    pts_l   = data["pts_lidar"]
    R_lid   = data["R_lid"]
    t_lid   = data["t_lid"]
    H, W    = data["H"], data["W"]
    delta_h = data["delta_h"]

    # Depth map from ORIGINAL camera (so point_cloud_reproject gets correct depths)
    depth_map, _ = create_lidar_depth_map(
        pts_l, R_lid, t_lid, R_cam, t_cam, K, H, W, dilation_ksize=11)

    # Warp original image to new height using actual R_cam calibration
    h_orig   = float(t_cam[2])
    h_new    = h_orig + delta_h
    warped, hole_mask = point_cloud_reproject(
        image, depth_map, h_orig, h_new, K, R_cam=R_cam, t_cam=t_cam)

    hole_pct = 100.0 * hole_mask.sum() / (H * W)

    # Raw warp: holes shown as black
    raw_holes = warped.copy()
    raw_holes[hole_mask] = 0

    # Inpainted render
    rendered = inpaint_holes(warped, hole_mask) if hole_mask.any() else warped.copy()

    return raw_holes, rendered, hole_pct, depth_map, hole_mask


# ─────────────────────────────────────────────────────────────────────────────
# LiDAR depth overlay (jet-coloured dots)
# ─────────────────────────────────────────────────────────────────────────────

def lidar_color_render(data, delta_h_override=None):
    """Project LiDAR from new height, colour each point from original image."""
    image  = data["image"]
    K      = data["K"]
    R_cam  = data["R_cam"]
    t_cam  = data["t_cam"]
    pts_l  = data["pts_lidar"]
    R_lid  = data["R_lid"]
    t_lid  = data["t_lid"]
    H, W   = data["H"], data["W"]
    delta_h = delta_h_override if delta_h_override is not None else data["delta_h"]

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    t_cam_new = t_cam + np.array([0., 0., delta_h])

    # Colours from original camera
    pts_ego   = (R_lid @ pts_l.T).T + t_lid
    pts_co    = (R_cam.T @ (pts_ego - t_cam).T).T
    fwd_o     = pts_co[:,2] > 0.5
    d_o       = pts_co[fwd_o,2]
    u_o       = (fx * pts_co[fwd_o,0]/d_o + cx).round().astype(int)
    v_o       = (fy * pts_co[fwd_o,1]/d_o + cy).round().astype(int)
    in_o      = (u_o>=0)&(u_o<W)&(v_o>=0)&(v_o<H)
    pts_3d    = pts_ego[fwd_o][in_o]
    colors    = image[v_o[in_o], u_o[in_o]]

    # Re-project from new height
    pts_cn    = (R_cam.T @ (pts_3d - t_cam_new).T).T
    fwd_n     = pts_cn[:,2] > 0.5
    d_n       = pts_cn[fwd_n,2]
    u_n       = (fx * pts_cn[fwd_n,0]/d_n + cx).round().astype(int)
    v_n       = (fy * pts_cn[fwd_n,1]/d_n + cy).round().astype(int)
    c_n       = colors[fwd_n]
    in_n      = (u_n>=0)&(u_n<W)&(v_n>=0)&(v_n<H)

    # Depth-coloured background using JET
    depth_map, _ = create_lidar_depth_map(
        pts_l, R_lid, t_lid, R_cam, t_cam_new, K, H, W, dilation_ksize=5)
    d_norm = np.clip(depth_map / 60.0, 0, 1)
    jet    = (plt.cm.jet(d_norm)[:,:,:3] * 255).astype(np.uint8)
    jet_bgr = cv2.cvtColor(jet, cv2.COLOR_RGB2BGR)

    # Blend original image with jet depth for background
    canvas = cv2.addWeighted(image, 0.4, jet_bgr, 0.6, 0)
    canvas[depth_map == 0] = image[depth_map == 0]   # keep undepth areas original

    # Paint LiDAR colour dots on top
    for i in np.where(in_n)[0]:
        cv2.circle(canvas, (u_n[i], v_n[i]), 3,
                   tuple(int(c) for c in c_n[i]), -1, cv2.LINE_AA)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Per-object detail strip (zoom-in pairs: original ROI | 3m ROI)
# ─────────────────────────────────────────────────────────────────────────────

def _box2d(ann, R_cam, t_cam, K, H, W):
    """Return (u_min,v_min,u_max,v_max) of 3D box in camera image, or None."""
    from pyquaternion import Quaternion as Q
    from nuscenes.utils.data_classes import Box

    center  = np.array(ann['translation'])
    size    = np.array(ann['size'])
    orient  = Q(ann['rotation'])
    box     = Box(center, size, orient)

    corners_global = box.corners().T          # (8,3)
    return corners_global


def project_corners(corners_global, ego_R, ego_t, R_cam, t_cam, K, H, W):
    corners_ego = (ego_R.T @ (corners_global - ego_t).T).T
    pts_cam = (R_cam.T @ (corners_ego - t_cam).T).T
    in_front = pts_cam[:,2] > 0.2
    if in_front.sum() < 4:
        return None
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    d   = pts_cam[in_front,2]
    u   = fx * pts_cam[in_front,0] / d + cx
    v   = fy * pts_cam[in_front,1] / d + cy
    u1, v1 = int(u.min()), int(v.min())
    u2, v2 = int(u.max()), int(v.max())
    if u2<=u1 or v2<=v1: return None
    u1=max(u1,0); v1=max(v1,0); u2=min(u2,W); v2=min(v2,H)
    if u2-u1<8 or v2-v1<8: return None
    return (u1,v1,u2,v2)


# ─────────────────────────────────────────────────────────────────────────────
# Main figure: 3 panels matching comparison_300cm style
# ─────────────────────name──────────────────────────────────────────────────────

def fig_3panel(data, raw_holes, rendered, hole_pct, out_dir):
    """
    Recreate the comparison_300cm.png style:
      [Original at h_orig]  [Raw point-cloud warp]  [Rendered at 3m]
    """
    img_orig  = data["image"]
    h_orig    = data["cam_height"]
    delta_h   = data["delta_h"]
    h_new     = h_orig + delta_h

    # Build the 3 panels (all BGR → convert to RGB for matplotlib)
    p1 = cv2.cvtColor(img_orig,  cv2.COLOR_BGR2RGB)
    p2 = cv2.cvtColor(raw_holes, cv2.COLOR_BGR2RGB)
    p3 = cv2.cvtColor(rendered,  cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(24, 8), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.03, hspace=0)

    titles = [
        f"ORIGINAL  ·  h = {h_orig:.2f} m",
        f"POINT CLOUD WARP  ·  black = displaced pixels",
        f"RENDERED FROM {h_new:.1f} m  ·  hole {hole_pct:.1f}% inpainted",
    ]
    border_cols = ["#44dd44", "#ff6633", "#4499ff"]
    panels      = [p1, p2, p3]

    for col, (panel, title, bc) in enumerate(zip(panels, titles, border_cols)):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(panel)
        ax.set_title(title, color="white", fontsize=11, pad=8,
                     fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(bc); sp.set_linewidth(4)

    fig.suptitle(
        f"Camera Height Shift: {h_orig:.2f} m  →  {h_new:.1f} m  (Δh = {delta_h:+.2f} m)  ·  "
        f"NuScenes dataset  ·  {data['cam_name']}",
        color="white", fontsize=13, y=1.02, fontweight="bold")

    path = os.path.join(out_dir, "comparison_300cm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: 4-panel with change heatmap
# ─────────────────────────────────────────────────────────────────────────────

def fig_4panel_diff(data, raw_holes, rendered, hole_mask, out_dir):
    """Add a 4th panel: pixel-change heatmap (|original - rendered| × 4)."""
    img_orig = data["image"]
    h_orig   = data["cam_height"]
    h_new    = h_orig + data["delta_h"]

    diff_g = cv2.absdiff(img_orig, rendered)
    diff_g = cv2.cvtColor(diff_g, cv2.COLOR_BGR2GRAY)
    diff_a = np.clip(diff_g.astype(np.float32) * 4, 0, 255).astype(np.uint8)
    heat   = cv2.applyColorMap(diff_a, cv2.COLORMAP_JET)

    panels = [img_orig, raw_holes, rendered, heat]
    titles = [
        f"ORIGINAL  h = {h_orig:.2f} m",
        "RAW WARP  (black = displaced pixels)",
        f"RENDERED  h = {h_new:.1f} m",
        "CHANGE HEATMAP  (×4 amplified)\nblue=no change  red=max shift",
    ]
    bcolors = ["#44dd44", "#ff6633", "#4499ff", "#ffaa22"]

    fig, axes = plt.subplots(1, 4, figsize=(32, 7), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    for ax, img, title, bc in zip(axes, panels, titles, bcolors):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, color="white", fontsize=10, pad=6, linespacing=1.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(bc); sp.set_linewidth(3)

    fig.suptitle(
        f"ADASAdapt — Camera Height: {h_orig:.2f} m → {h_new:.1f} m  "
        f"(Δh={data['delta_h']:+.2f}m)  ·  {data['cam_name']}",
        color="white", fontsize=13, y=1.02, fontweight="bold")
    plt.tight_layout(pad=0.5)

    path = os.path.join(out_dir, "comparison_4panel.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Object zoom-in pairs (original ROI | 3m ROI)
# ─────────────────────────────────────────────────────────────────────────────

def fig_object_zoom(data, rendered, out_dir):
    """6 closest annotated objects: side-by-side crop at 1m and 3m."""
    from pyquaternion import Quaternion as Q
    from nuscenes.utils.data_classes import Box

    nusc_ref = data.get("_nusc_")
    if nusc_ref is None:
        return

    nusc       = nusc_ref
    sample_tok = data["sample_token"]
    sample     = nusc.get('sample', sample_tok)
    ep         = nusc.get('ego_pose', nusc.get('sample_data', sample['data'][data['cam_name']])['ego_pose_token'])
    ego_R      = Q(ep['rotation']).rotation_matrix
    ego_t      = np.array(ep['translation'])

    R_cam = data["R_cam"]
    t_cam = data["t_cam"]
    K     = data["K"]
    H, W  = data["H"], data["W"]
    delta_h = data["delta_h"]
    t_cam_new = t_cam + np.array([0., 0., delta_h])

    img_1m  = data["image"]
    img_3m  = rendered

    # Collect objects visible from this camera
    objs = []
    for ann_tok in sample['anns']:
        ann = nusc.get('sample_annotation', ann_tok)
        c   = ann['category_name'].split('.')[0]
        center_global = np.array(ann['translation'])
        center_ego    = ego_R.T @ (center_global - ego_t)
        dist = float(np.linalg.norm(center_ego[:2]))

        corners_g = _box2d(ann, R_cam, t_cam, K, H, W)
        box2d = project_corners(corners_g, ego_R, ego_t, R_cam, t_cam, K, H, W)
        if box2d is None:
            continue
        objs.append((dist, c, box2d, ann))

    objs.sort()   # closest first
    objs = objs[:8]  # top 8 closest visible

    if not objs:
        return

    n = len(objs)
    CROP_H = 160
    fig, axes = plt.subplots(n, 2, figsize=(10, n * 2.2), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    if n == 1:
        axes = [axes]

    for row, (dist, cat, (u1,v1,u2,v2), ann) in enumerate(objs):
        pad = max(10, int((v2-v1)*0.15))
        u1p = max(0, u1-pad); v1p = max(0, v1-pad)
        u2p = min(W, u2+pad); v2p = min(H, v2+pad)

        crop_1m = img_1m[v1p:v2p, u1p:u2p]
        crop_3m = img_3m[v1p:v2p, u1p:u2p]

        if crop_1m.size == 0 or crop_3m.size == 0:
            continue

        ax1, ax2 = axes[row]
        ax1.imshow(cv2.cvtColor(crop_1m, cv2.COLOR_BGR2RGB))
        ax2.imshow(cv2.cvtColor(crop_3m, cv2.COLOR_BGR2RGB))

        ax1.set_title(f"{cat}  @{dist:.1f}m  ·  h={data['cam_height']:.2f}m",
                      color="#88ff88", fontsize=8, pad=3)
        ax2.set_title(f"{cat}  @{dist:.1f}m  ·  h=3.0m",
                      color="#ffaa44", fontsize=8, pad=3)
        for ax in (ax1, ax2):
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Per-Object Zoom: Original (left) vs Rendered 3m (right)",
                 color="white", fontsize=12, y=1.01, fontweight="bold")
    plt.tight_layout(pad=0.4)

    path = os.path.join(out_dir, "comparison_objects.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: LiDAR point cloud view at both heights (side-by-side)
# ─────────────────────────────────────────────────────────────────────────────

def fig_lidar_panels(data, out_dir):
    orig_render = lidar_color_render(data, delta_h_override=0.0)
    new_render  = lidar_color_render(data, delta_h_override=data["delta_h"])

    h_orig = data["cam_height"]
    h_new  = h_orig + data["delta_h"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)

    ax1.imshow(cv2.cvtColor(orig_render, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"LiDAR Point Cloud  ·  h = {h_orig:.2f} m (original)",
                  color="white", fontsize=11, pad=6)
    ax2.imshow(cv2.cvtColor(new_render,  cv2.COLOR_BGR2RGB))
    ax2.set_title(f"LiDAR Point Cloud  ·  h = {h_new:.1f} m (raised)",
                  color="white", fontsize=11, pad=6)

    for ax in (ax1, ax2):
        ax.set_xticks([]); ax.set_yticks([])

    for sp in ax1.spines.values(): sp.set_edgecolor("#44dd44"); sp.set_linewidth(3)
    for sp in ax2.spines.values(): sp.set_edgecolor("#ff6633"); sp.set_linewidth(3)

    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors  import Normalize
    cax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    cb  = ColorbarBase(cax, cmap=plt.cm.jet,
                       norm=Normalize(0, 60), orientation="vertical")
    cb.set_label("Depth (m)", color="white", fontsize=10)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    fig.suptitle(
        "LiDAR Point Cloud: Same scan re-projected from original vs raised camera",
        color="white", fontsize=13, y=1.02, fontweight="bold")
    plt.tight_layout(rect=[0,0,0.91,1.0])

    path = os.path.join(out_dir, "comparison_lidar.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


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

    print("[1/5] Loading NuScenes …")
    nusc = NuScenes(version=args.version, dataroot=args.nuscenes, verbose=False)
    print(f"  {len(nusc.sample)} samples")

    if args.sample is None:
        # Auto-pick: most annotated forward sample
        best = max(nusc.sample, key=lambda s: len(s['anns'])
                   if args.cam in s['data'] else 0)
        args.sample = best['token']
        print(f"  Auto-selected sample: {args.sample[:16]}…  "
              f"({len(best['anns'])} annotations)")

    print("[2/5] Loading sample data …")
    data = load_data(nusc, args.sample, args.cam)
    data["_nusc_"] = nusc
    data["delta_h"] = args.target_h - data["cam_height"]
    print(f"  Camera: {args.cam}  h = {data['cam_height']:.3f}m  "
          f"→  {TARGET_H:.1f}m  (Δh = {data['delta_h']:+.3f}m)")
    print(f"  LiDAR pts: {len(data['pts_lidar']):,}   Image: {data['W']}×{data['H']}")

    print("[3/5] Rendering from 3m …")
    raw_holes, rendered, hole_pct, depth_map, hole_mask = render_3m(data)
    print(f"  Hole coverage: {hole_pct:.1f}%  "
          f"  Pixels that moved: {(cv2.absdiff(data['image'],rendered).mean()):.1f} avg diff")

    print("[4/5] Generating figures …")
    fig_3panel(data, raw_holes, rendered, hole_pct, args.output_dir)
    fig_4panel_diff(data, raw_holes, rendered, hole_mask, args.output_dir)
    fig_object_zoom(data, rendered, args.output_dir)
    fig_lidar_panels(data, args.output_dir)

    print("[5/5] Done.")
    print(f"\nOutputs in {os.path.abspath(args.output_dir)}/")
    print("  comparison_300cm.png    ← 3-panel main figure")
    print("  comparison_4panel.png   ← + change heatmap")
    print("  comparison_objects.png  ← per-object zoom pairs")
    print("  comparison_lidar.png    ← LiDAR point cloud both heights")


if __name__ == "__main__":
    main()
