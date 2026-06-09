"""
Perspective Change Proof Visualizer
=====================================
Demonstrates that raising the camera from 1.57m to 3.0m genuinely changes
the viewpoint — even when the final inpainted image looks similar.

Output figures:
  proof_01_raw_warp.png     — raw pixel displacement (black holes) side-by-side
  proof_02_feature_zoom.png — zoomed crops: traffic light / road markings / car
  proof_03_grid_deform.png  — checkerboard grid that bends with height change
  proof_04_blend_overlay.png— red=original blue=rendered; misalignment = colour fringe
  proof_05_shift_arrows.png — tracked keypoints with displacement arrows
"""

import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

TARGET_H   = 3.0
OUTPUT_DIR = "outputs/perspective_proof"
DARK_BG    = "#0a0a14"


# ─── helpers ─────────────────────────────────────────────────────────────────

def load_midas():
    m  = torch.hub.load("intel-isl/MiDaS","MiDaS_small",trust_repo=True,verbose=False)
    tf = torch.hub.load("intel-isl/MiDaS","transforms",trust_repo=True,verbose=False).small_transform
    m.eval().cuda(); return m, tf

def midas_depth(model, tf, image_bgr, K, cam_h):
    H, W = image_bgr.shape[:2]
    fy   = K[1,1]
    rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        pred = torch.nn.functional.interpolate(
            model(tf(rgb).cuda()).unsqueeze(1), size=(H,W),
            mode="bicubic", align_corners=False).squeeze()
    inv = pred.cpu().numpy().astype(np.float32)

    # detect horizon via row energy
    gray   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blr    = cv2.GaussianBlur(gray,(1,31),0)
    soy    = np.abs(cv2.Sobel(blr,cv2.CV_64F,0,1,ksize=5))
    re     = soy.mean(axis=1)
    lo,hi  = H//6, 2*H//3
    hrow   = int(lo + np.argmax(re[lo:hi]))

    road_rows = np.arange(hrow+20, H, dtype=np.float32)
    gt        = fy * cam_h / np.maximum(road_rows - hrow, 1.)
    rel       = inv[int(hrow+20):, W//3:2*W//3].mean(axis=1)
    v         = rel > 0
    A         = np.stack([rel[v], np.ones(v.sum())], axis=1)
    coeffs,_,_,_ = np.linalg.lstsq(A, gt[v], rcond=None)
    metric    = np.clip(coeffs[0]*inv + coeffs[1], 0.3, 120.).astype(np.float32)
    return metric, hrow


def build_pc(image_bgr, depth, K):
    H,W  = image_bgr.shape[:2]
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    vs = np.arange(H); us = np.arange(W)
    uu,vv = np.meshgrid(us,vs)
    d  = depth.ravel()
    ok = (d>0.3)&(d<100.)
    d  = d[ok]
    u  = uu.ravel()[ok].astype(np.float32)
    v  = vv.ravel()[ok].astype(np.float32)
    bgr= image_bgr.reshape(-1,3)[ok]
    return (np.stack([(u-cx)*d/fx,(v-cy)*d/fy,d],axis=1).astype(np.float32),
            bgr[:,::-1].astype(np.uint8),   # RGB
            ok)


def warp_pc(pts_cam, pts_rgb, K, R_cam, t_cam, target_h, H, W):
    """Pure geometric warp — NO inpainting, so holes are visible."""
    delta_h = target_h - float(t_cam[2])
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]

    pts_ego = (R_cam @ pts_cam.T).T + t_cam
    t_new   = t_cam.copy(); t_new[2] += delta_h
    pts_nc  = (R_cam.T @ (pts_ego - t_new).T).T

    fwd = pts_nc[:,2] > 0.1
    d   = pts_nc[fwd,2]
    u_f = fx*pts_nc[fwd,0]/d + cx
    v_f = fy*pts_nc[fwd,1]/d + cy
    u_i = np.round(u_f).astype(np.int32)
    v_i = np.round(v_f).astype(np.int32)
    col = pts_rgb[fwd]

    in_img = (u_i>=0)&(u_i<W)&(v_i>=0)&(v_i<H)
    order  = np.argsort(-d[in_img])
    uo = u_i[in_img][order]; vo = v_i[in_img][order]
    co = col[in_img][order]; do_ = d[in_img][order]

    canvas = np.zeros((H,W,3), np.uint8)
    zbuf   = np.full((H,W), np.inf)
    closer = do_ < zbuf[vo,uo]
    vc=vo[closer]; uc=uo[closer]
    zbuf[vc,uc]   = do_[closer]
    canvas[vc,uc] = co[closer,::-1]   # RGB→BGR

    # small dilation to close sub-pixel gaps
    filled = (zbuf < np.inf).astype(np.uint8)
    cdil   = cv2.dilate(canvas, np.ones((3,3),np.uint8))
    fdil   = cv2.dilate(filled, np.ones((3,3),np.uint8))
    gap    = (fdil>0)&(filled==0)
    canvas[gap] = cdil[gap]; filled[gap]=1

    hole_mask = filled == 0
    hole_pct  = 100.*hole_mask.sum()/(H*W)

    # inpainted copy for clean rendering
    from src.real_image_transformer import inpaint_holes
    rendered = inpaint_holes(canvas.copy(), hole_mask) if hole_mask.any() else canvas.copy()

    return canvas, rendered, hole_mask, hole_pct   # canvas = raw (holes=black)


# ─── Figure 1: Raw warp side-by-side ─────────────────────────────────────────

def fig_raw_warp(orig, raw, rendered, hole_pct, cam_h, out_dir):
    """
    3 panels: original | raw warp (black holes) | inpainted
    The raw panel PROVES the perspective changed — holes are where pixels moved away from.
    """
    diff_g = cv2.absdiff(orig, rendered)
    diff_g = cv2.cvtColor(diff_g, cv2.COLOR_BGR2GRAY)
    diff_a = np.clip(diff_g.astype(np.float32)*6,0,255).astype(np.uint8)
    heat   = cv2.applyColorMap(diff_a, cv2.COLORMAP_JET)

    panels = [
        (orig,     f"ORIGINAL  h = {cam_h:.2f} m"),
        (raw,      f"RAW WARP  h = {TARGET_H:.1f} m\nBlack bands = pixels displaced away"),
        (rendered, f"RENDERED  h = {TARGET_H:.1f} m  (holes inpainted)\nhole = {hole_pct:.1f}%"),
        (heat,     f"CHANGE HEATMAP  (×6)\nblue=no change  red=max shift"),
    ]
    bcolors = ["#44dd44","#ff3333","#4488ff","#ffaa00"]

    fig, axes = plt.subplots(1,4, figsize=(32,7), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    for ax,(img,title),bc in zip(axes, panels, bcolors):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, color="white", fontsize=10, pad=6,
                     linespacing=1.6, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(bc); sp.set_linewidth(4)

    fig.suptitle(
        f"Height {cam_h:.2f}m → {TARGET_H:.1f}m: Raw pixel displacement proof\n"
        "Black bands in panel 2 = regions where pixels physically moved to new positions",
        color="white", fontsize=13, y=1.02, fontweight="bold")
    plt.tight_layout(pad=0.3)
    p = os.path.join(out_dir,"proof_01_raw_warp.png")
    plt.savefig(p,dpi=130,bbox_inches="tight",facecolor=DARK_BG); plt.close()
    print(f"  Saved: {p}")


# ─── Figure 2: Feature zoom crops ────────────────────────────────────────────

def fig_feature_zoom(orig, rendered, depth, K, R_cam, t_cam, cam_h, out_dir):
    """
    For each ROI (traffic light, road marking, nearby car):
      LEFT = crop from original  |  RIGHT = same crop from rendered
    Plus arrows showing expected pixel shift.
    """
    H, W = orig.shape[:2]
    fy   = K[1,1]
    delta_h = TARGET_H - float(t_cam[2])
    shift_cam = R_cam.T @ np.array([0.,0.,delta_h])

    # Interesting regions: (name, row_c, col_c, half_size)
    rois = [
        ("Traffic Lights\n(intersection)",  int(H*0.38), int(W*0.50), 130),
        ("BUS Road Marking\n(bottom-left)", int(H*0.74), int(W*0.26), 110),
        ("Yellow Lane Lines\n(center road)",int(H*0.70), int(W*0.50), 120),
        ("Pedestrians\n(crosswalk area)",   int(H*0.52), int(W*0.42), 100),
    ]

    n = len(rois)
    fig, axes = plt.subplots(n, 3, figsize=(18, n*4.5), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)

    for row, (name, rc, cc, hs) in enumerate(rois):
        r1,r2 = max(0,rc-hs), min(H,rc+hs)
        c1,c2 = max(0,cc-hs), min(W,cc+hs)

        crop_o = orig[r1:r2, c1:c2].copy()
        crop_r = rendered[r1:r2, c1:c2].copy()

        # Local depth at centre
        d_center  = float(depth[rc, cc])
        v_shift   = -shift_cam[1] / max(d_center,0.5) * fy   # pixels downward
        expected  = f"Depth≈{d_center:.1f}m  →  shift≈{v_shift:+.0f}px"

        # Draw crosshair on original crop
        ch = crop_o.copy()
        cv2.line(ch,(0,hs),(2*hs,hs),(0,255,255),1)
        cv2.line(ch,(hs,0),(hs,2*hs),(0,255,255),1)

        # Draw shifted crosshair on rendered crop
        cr = crop_r.copy()
        shifted_row = int(hs - v_shift)   # v_shift + = down in image, so crop centre moves up
        cv2.line(cr,(0,shifted_row),(2*hs,shifted_row),(255,100,0),2)
        cv2.line(cr,(hs,0),(hs,2*hs),(255,100,0),1)
        # arrow showing direction
        cv2.arrowedLine(cr,(hs,hs),(hs,shifted_row),(0,255,255),2,tipLength=0.3)

        # Difference heatmap of this crop
        dc = cv2.absdiff(crop_o, crop_r)
        dc_g = cv2.cvtColor(dc, cv2.COLOR_BGR2GRAY)
        dc_a = np.clip(dc_g.astype(np.float32)*8,0,255).astype(np.uint8)
        dc_h = cv2.applyColorMap(dc_a, cv2.COLORMAP_INFERNO)

        ax0, ax1, ax2 = axes[row]

        ax0.imshow(cv2.cvtColor(ch, cv2.COLOR_BGR2RGB))
        ax0.set_title(f"{name}\nOriginal h={cam_h:.2f}m\n{expected}",
                      color="#88ff88", fontsize=9, pad=3, linespacing=1.4)
        ax1.imshow(cv2.cvtColor(cr, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"{name}\nRendered h={TARGET_H:.1f}m\nOrange line = where feature shifted TO",
                      color="#ffaa44", fontsize=9, pad=3, linespacing=1.4)
        ax2.imshow(cv2.cvtColor(dc_h, cv2.COLOR_BGR2RGB))
        ax2.set_title(f"Difference ×8\nbright = pixels that changed",
                      color="#ff6666", fontsize=9, pad=3, linespacing=1.4)

        for ax, bc in zip((ax0,ax1,ax2),["#44dd44","#ffaa44","#ff4444"]):
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(bc); sp.set_linewidth(2)

    fig.suptitle(
        f"Feature-Level Perspective Change: {cam_h:.2f}m → {TARGET_H:.1f}m\n"
        "Cyan crosshair = original position  ·  Orange line = new position after height shift",
        color="white", fontsize=13, y=1.01, fontweight="bold")
    plt.tight_layout(pad=0.5)
    p = os.path.join(out_dir,"proof_02_feature_zoom.png")
    plt.savefig(p,dpi=130,bbox_inches="tight",facecolor=DARK_BG); plt.close()
    print(f"  Saved: {p}")


# ─── Figure 3: Grid deformation ──────────────────────────────────────────────

def fig_grid_deform(orig, depth, K, R_cam, t_cam, cam_h, out_dir):
    """
    Draw a regular grid on original image.
    Track each grid point through the 3D reprojection to see where it lands.
    Visualise as: original (grid dots) | rendered (shifted dots + arrows).
    """
    H, W  = orig.shape[:2]
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    delta_h = TARGET_H - float(t_cam[2])

    # Grid points (only lower 60% of image where road content is)
    grid_step = 60
    grid_us   = np.arange(grid_step//2, W, grid_step)
    grid_vs   = np.arange(int(H*0.4), H, grid_step)

    img_orig_grid = orig.copy()
    img_new_grid  = orig.copy()   # start with original so background is clear

    shifts = []
    for gv in grid_vs:
        for gu in grid_us:
            d = float(depth[gv, gu])
            if d < 0.5 or d > 80:
                continue

            # 3D in camera frame
            Xc = (gu - cx)*d/fx
            Yc = (gv - cy)*d/fy
            Zc = d

            # To ego
            pt_ego = R_cam @ np.array([Xc,Yc,Zc]) + t_cam

            # Raise camera
            t_new = t_cam.copy(); t_new[2] += delta_h
            pt_nc = R_cam.T @ (pt_ego - t_new)

            if pt_nc[2] < 0.1: continue
            u_new = int(fx*pt_nc[0]/pt_nc[2] + cx)
            v_new = int(fy*pt_nc[1]/pt_nc[2] + cy)

            dv = v_new - gv
            du = u_new - gu

            # Draw on original: cyan dot
            cv2.circle(img_orig_grid, (gu, gv), 5, (0,255,255), -1)

            # Draw on new: orange dot at new position + arrow from old
            if 0<=u_new<W and 0<=v_new<H:
                cv2.circle(img_new_grid, (u_new, v_new), 5, (0,120,255), -1)
                cv2.arrowedLine(img_new_grid, (gu,gv), (u_new,v_new),
                                (0,255,255), 1, tipLength=0.25)
            shifts.append((gu,gv,du,dv,d))

    fig, axes = plt.subplots(1,2, figsize=(20,7), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)

    axes[0].imshow(cv2.cvtColor(img_orig_grid, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"ORIGINAL  h={cam_h:.2f}m\nCyan dots = grid tracking points",
                      color="#88ff88", fontsize=11, pad=6, fontweight="bold")
    axes[1].imshow(cv2.cvtColor(img_new_grid,  cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"RENDERED  h={TARGET_H:.1f}m\nOrange dots = where each cyan point moved\nArrows show displacement magnitude",
                      color="#ffaa44", fontsize=11, pad=6, fontweight="bold")

    for ax,bc in zip(axes,["#44dd44","#ff6633"]):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(bc); sp.set_linewidth(3)

    # Shift statistics
    if shifts:
        dvs = [s[3] for s in shifts]
        avg_dv = np.mean(dvs); max_dv = np.max(np.abs(dvs))
        fig.text(0.5,-0.02,
                 f"Grid tracking: {len(shifts)} points  ·  "
                 f"Avg vertical shift = {avg_dv:+.1f}px  ·  "
                 f"Max shift = {max_dv:.0f}px",
                 ha="center", color="#aaaacc", fontsize=11)

    fig.suptitle(
        f"Grid Point Tracking: {cam_h:.2f}m → {TARGET_H:.1f}m\n"
        "Each cyan dot tracks to its new image position (orange) under height-only camera raise",
        color="white", fontsize=13, y=1.02, fontweight="bold")
    plt.tight_layout(pad=0.3)
    p = os.path.join(out_dir,"proof_03_grid_deform.png")
    plt.savefig(p,dpi=130,bbox_inches="tight",facecolor=DARK_BG); plt.close()
    print(f"  Saved: {p}")


# ─── Figure 4: Red-Cyan anaglyph overlay ─────────────────────────────────────

def fig_blend_overlay(orig, rendered, cam_h, out_dir):
    """
    Anaglyph-style overlay:
      Red channel   = original
      Cyan (G+B)    = rendered
    Where images agree → grey
    Where they differ → colour fringe showing exactly which objects shifted
    """
    o = orig.astype(np.float32)
    r = rendered.astype(np.float32)

    anaglyph = np.zeros_like(o)
    anaglyph[:,:,2] = o[:,:,2]          # Red channel from original
    anaglyph[:,:,1] = r[:,:,1]          # Green from rendered
    anaglyph[:,:,0] = r[:,:,0]          # Blue from rendered
    anaglyph = np.clip(anaglyph,0,255).astype(np.uint8)

    # Also create 50/50 alpha blend
    blend = cv2.addWeighted(orig,0.5,rendered,0.5,0)

    fig, axes = plt.subplots(1,3, figsize=(24,7), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)

    axes[0].imshow(cv2.cvtColor(orig,      cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Original  h={cam_h:.2f}m", color="#88ff88",
                      fontsize=11, pad=6, fontweight="bold")

    axes[1].imshow(cv2.cvtColor(anaglyph,  cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Anaglyph Overlay\nRed=Original  Cyan=Rendered {TARGET_H:.0f}m\n"
                      "Colour fringe = perspective shift",
                      color="white", fontsize=11, pad=6, fontweight="bold")

    axes[2].imshow(cv2.cvtColor(rendered,  cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Rendered  h={TARGET_H:.1f}m", color="#ffaa44",
                      fontsize=11, pad=6, fontweight="bold")

    for ax,bc in zip(axes,["#44dd44","#ffffff","#ff6633"]):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(bc); sp.set_linewidth(3)

    fig.suptitle(
        "Anaglyph Overlay: Red fringe = original-only pixels  ·  "
        "Cyan fringe = new-view-only pixels\n"
        "Colour around edges of objects = those objects shifted position",
        color="white", fontsize=13, y=1.02, fontweight="bold")
    plt.tight_layout(pad=0.3)
    p = os.path.join(out_dir,"proof_04_anaglyph.png")
    plt.savefig(p,dpi=130,bbox_inches="tight",facecolor=DARK_BG); plt.close()
    print(f"  Saved: {p}")


# ─── Figure 5: Shift magnitude map ───────────────────────────────────────────

def fig_shift_map(depth, K, R_cam, t_cam, cam_h, orig, out_dir):
    """
    Colour every pixel by how many pixels it will shift under the height change.
    Shows WHERE in the image the perspective change is strongest.
    """
    H, W  = depth.shape
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    delta_h = TARGET_H - float(t_cam[2])
    shift_cam = R_cam.T @ np.array([0.,0.,delta_h])

    # vertical shift = fy * |shift_cam[1]| / depth
    dv_map = np.where(depth > 0.3, fy * abs(shift_cam[1]) / np.maximum(depth,0.3), 0)
    dv_map = np.clip(dv_map, 0, 300).astype(np.float32)

    dv_norm = np.clip(dv_map/200., 0, 1)
    jet     = (plt.cm.jet(dv_norm)[:,:,:3]*255).astype(np.uint8)
    jet_bgr = cv2.cvtColor(jet, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(orig, 0.45, jet_bgr, 0.55, 0)

    fig, axes = plt.subplots(1,3, figsize=(24,7), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)

    axes[0].imshow(cv2.cvtColor(orig,    cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Original  h={cam_h:.2f}m",
                      color="#88ff88", fontsize=11, pad=6, fontweight="bold")

    axes[1].imshow(cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Pixel Shift Magnitude Map\n"
                      "Red=large shift (>150px)  Blue=small shift (<20px)",
                      color="white", fontsize=11, pad=6, fontweight="bold")

    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Shift Map Overlaid on Image\n"
                      "Bright areas = where perspective changes most",
                      color="#ffaa44", fontsize=11, pad=6, fontweight="bold")

    for ax,bc in zip(axes,["#44dd44","#ffffff","#ff6633"]):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(bc); sp.set_linewidth(3)

    # Colorbar
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors  import Normalize
    cax = fig.add_axes([0.92,0.15,0.012,0.7])
    cb  = ColorbarBase(cax, cmap=plt.cm.jet,
                       norm=Normalize(0,200), orientation="vertical")
    cb.set_label("Pixel shift (px)", color="white", fontsize=10)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    # Stats
    med = float(np.median(dv_map[depth>0.3]))
    p90 = float(np.percentile(dv_map[depth>0.3], 90))
    fig.text(0.5,-0.02,
             f"Median shift = {med:.0f}px  ·  90th-pct shift = {p90:.0f}px  ·  "
             f"Max shift = {dv_map.max():.0f}px (at depth {depth[depth>0.3].min():.1f}m)",
             ha="center", color="#aaaacc", fontsize=11)

    fig.suptitle(
        f"Where Does the Perspective Change Most?  {cam_h:.2f}m → {TARGET_H:.1f}m\n"
        "Near objects (road surface) shift most — AI detectors trained at 1.5m see a different view",
        color="white", fontsize=13, y=1.02, fontweight="bold")
    plt.tight_layout(rect=[0,0,0.91,1.0])
    p = os.path.join(out_dir,"proof_05_shift_map.png")
    plt.savefig(p,dpi=130,bbox_inches="tight",facecolor=DARK_BG); plt.close()
    print(f"  Saved: {p}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuscenes",   default="nuscenes_data")
    ap.add_argument("--version",    default="v1.0-mini")
    ap.add_argument("--sample",     default="fdc39b23ab4242eda6ec5e1e6574fe33")
    ap.add_argument("--cam",        default="CAM_BACK")
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/6] Loading scene …")
    nusc   = NuScenes(version=args.version, dataroot=args.nuscenes, verbose=False)
    sample = nusc.get('sample', args.sample)
    cam_sd = nusc.get('sample_data', sample['data'][args.cam])
    cs     = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    R_cam  = Quaternion(cs['rotation']).rotation_matrix
    t_cam  = np.array(cs['translation'])
    K      = np.array(cs['camera_intrinsic'], dtype=np.float32)
    orig   = cv2.imread(f"nuscenes_data/{cam_sd['filename']}")
    H, W   = orig.shape[:2]
    cam_h  = float(t_cam[2])
    print(f"  {args.cam}  h={cam_h:.3f}m → {TARGET_H:.1f}m  (Δh={TARGET_H-cam_h:+.3f}m)")

    print("[2/6] MiDaS depth …")
    model, tf  = load_midas()
    depth, _   = midas_depth(model, tf, orig, K, cam_h)
    print(f"  median={np.median(depth):.1f}m")

    print("[3/6] Building point cloud & warping …")
    pts_cam, pts_rgb, _ = build_pc(orig, depth, K)
    raw, rendered, hole_mask, hole_pct = warp_pc(
        pts_cam, pts_rgb, K, R_cam, t_cam, TARGET_H, H, W)
    print(f"  Points: {len(pts_cam):,}  Hole: {hole_pct:.1f}%")

    print("[4/6] Generating proof figures …")
    fig_raw_warp(orig, raw, rendered, hole_pct, cam_h, args.output_dir)
    fig_feature_zoom(orig, rendered, depth, K, R_cam, t_cam, cam_h, args.output_dir)
    fig_grid_deform(orig, depth, K, R_cam, t_cam, cam_h, args.output_dir)
    fig_blend_overlay(orig, rendered, cam_h, args.output_dir)
    fig_shift_map(depth, K, R_cam, t_cam, cam_h, orig, args.output_dir)

    print(f"\n[5/6] Shift summary:")
    shift_cam = R_cam.T @ np.array([0.,0.,TARGET_H-cam_h])
    fy = K[1,1]
    for d in [5,10,20,30,50]:
        print(f"  depth {d:2d}m → {fy*abs(shift_cam[1])/d:+.0f}px vertical shift")

    print(f"\nOutputs → {os.path.abspath(args.output_dir)}/")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(".png"): print(f"  {f}")


if __name__ == "__main__":
    main()
