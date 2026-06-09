"""
Feature Extraction Accuracy: 1.5 m vs 3.0 m Camera Height
===========================================================
Shows HOW detection quality changes, not just WHERE boxes move.

For each visible NuScenes object this script computes:
  • ORB keypoints   – distinctive feature points an AI descriptor can latch on to
  • Canny edges     – structural outlines (backbone of most object detectors)
  • Texture entropy – information density (flat roof vs. rich side-view)
  • Gradient RMS    – HOG-proxy; how "textured" the region is
  • Appearance diff – pixel-level change showing texture starvation

The core insight being visualised
  At 1.5 m the camera sees the FRONT / SIDE of vehicles → rich texture,
  many keypoints, strong edges → detectors work well.
  At 3.0 m the camera sees the ROOF (never in training data) → flat,
  featureless, very few keypoints → detector confidence collapses.

Outputs  (outputs/feature_accuracy/)
  acc_01_keypoints.png      ORB keypoint maps on both full images
  acc_02_per_object.png     Per-object 5-column analysis strip
  acc_03_feature_maps.png   Edge density / gradient / entropy heatmaps
  acc_04_scores.png         Detection quality score bars per category

Usage
-----
  python run_feature_accuracy.py --nuscenes nuscenes_data
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

sys.path.insert(0, os.path.dirname(__file__))
from src.nuscenes_reprojector import (
    create_lidar_depth_map, reproject_frame, project_3d_box_to_camera,
)
from src.real_image_transformer import point_cloud_reproject, inpaint_holes

OUTPUT_DIR = "outputs/feature_accuracy"
TARGET_H   = 3.0
DARK_BG    = "#1a1a2e"

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

def load_sample(nusc, token, cam="CAM_FRONT"):
    from pyquaternion import Quaternion
    from nuscenes.utils.data_classes import LidarPointCloud

    sample  = nusc.get("sample", token)
    cam_sd  = nusc.get("sample_data", sample["data"][cam])
    image   = cv2.imread(os.path.join(nusc.dataroot, cam_sd["filename"]))

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

    anns = []
    for box in nusc.get_boxes(sample["data"][cam]):
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

    return dict(image=image, K=K, R_cam=R_cam, t_cam=t_cam,
                pts_lidar=pts_l, R_lidar=R_lid, t_lidar=t_lid,
                annotations=anns, cam_height=float(t_cam[2]))


def build_3m_render(data):
    """Return dense inpainted render at 3.0 m using LiDAR depth."""
    delta_h = TARGET_H - data["cam_height"]
    t_cam_new = data["t_cam"] + np.array([0., 0., delta_h])
    H, W = data["image"].shape[:2]
    depth_map, _ = create_lidar_depth_map(
        data["pts_lidar"], data["R_lidar"], data["t_lidar"],
        data["R_cam"], data["t_cam"], data["K"], H, W, dilation_ksize=11)
    warped, hole_mask = point_cloud_reproject(
        data["image"], depth_map,
        data["cam_height"], TARGET_H, data["K"],
        R_cam=data["R_cam"], t_cam=data["t_cam"])
    rendered = inpaint_holes(warped, hole_mask) if hole_mask.any() else warped
    return rendered, hole_mask, depth_map, delta_h


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction metrics
# ─────────────────────────────────────────────────────────────────────────────

_ORB = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)

def roi_metrics(roi_bgr: np.ndarray) -> dict:
    """
    Compute feature-extraction quality metrics for a single ROI.

    Returns
    -------
    keypoints     int    ORB keypoints found (higher = more detectable)
    edge_density  float  Canny edge pixels / total pixels  (0-1)
    entropy       float  Texture entropy in bits  (0-8)
    gradient_rms  float  RMS Sobel magnitude  (HOG proxy)
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return dict(keypoints=0, edge_density=0.0, entropy=0.0, gradient_rms=0.0)

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < 10 or gray.shape[1] < 10:
        return dict(keypoints=0, edge_density=0.0, entropy=0.0, gradient_rms=0.0)

    # ── ORB keypoints ────────────────────────────────────────────────────────
    kps, _ = _ORB.detectAndCompute(gray, None)
    n_kp   = len(kps)

    # ── Canny edge density ───────────────────────────────────────────────────
    edges     = cv2.Canny(gray, 50, 150)
    edge_dens = float(edges.sum() // 255) / max(gray.size, 1)

    # ── Texture entropy ──────────────────────────────────────────────────────
    hist      = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist_norm = hist / hist.sum()
    h_pos     = hist_norm[hist_norm > 0]
    entropy   = float(-np.sum(h_pos * np.log2(h_pos)))

    # ── Gradient RMS (HOG proxy) ─────────────────────────────────────────────
    gx    = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy    = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gmag  = np.sqrt(gx**2 + gy**2)
    grad_rms = float(np.sqrt(np.mean(gmag**2)))

    return dict(keypoints=n_kp, edge_density=edge_dens,
                entropy=entropy, gradient_rms=grad_rms)


def detection_score(m: dict) -> float:
    """
    Proxy for 'how well a learned detector can identify this object'.
    Weighted combination of normalised metrics (0–1 scale).
    """
    kp_n   = min(m["keypoints"] / 80.0, 1.0)          # ≥80 kp = full score
    ed_n   = min(m["edge_density"] / 0.12, 1.0)        # ≥12% edge = full
    ent_n  = min(m["entropy"] / 6.5, 1.0)              # ≥6.5 bits = full
    grad_n = min(m["gradient_rms"] / 40.0, 1.0)        # ≥40 = full
    return 0.30*kp_n + 0.25*ed_n + 0.25*ent_n + 0.20*grad_n


# ─────────────────────────────────────────────────────────────────────────────
# ROI extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_roi(image, ann, R_cam, t_cam, K, delta_h, pad_frac=0.25):
    """Extract padded ROI for one annotation at given height offset."""
    H, W = image.shape[:2]
    pix, front = project_3d_box_to_camera(
        ann["corners_ego"], R_cam, t_cam, K, delta_h=delta_h)
    vis = pix[front]
    if len(vis) < 2:
        return None, None
    x1 = max(0, int(vis[:,0].min()));  y1 = max(0, int(vis[:,1].min()))
    x2 = min(W, int(vis[:,0].max())); y2 = min(H, int(vis[:,1].max()))
    if x2 <= x1 or y2 <= y1:
        return None, None
    bw, bh = x2-x1, y2-y1
    pad = max(10, int(max(bw,bh)*pad_frac))
    rx0 = max(0,x1-pad); ry0 = max(0,y1-pad)
    rx1 = min(W,x2+pad); ry1 = min(H,y2+pad)
    roi = image[ry0:ry1, rx0:rx1]
    return roi if roi.size else None, (x1,y1,bw,bh)


# ─────────────────────────────────────────────────────────────────────────────
# Figure helpers
# ─────────────────────────────────────────────────────────────────────────────

def _edge_img(bgr, lo=50, hi=150):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    e = cv2.Canny(g, lo, hi)
    return cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)


def _gradient_heat(bgr):
    g  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag = np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8) if mag.max() > 0 else mag.astype(np.uint8)
    return cv2.applyColorMap(mag, cv2.COLORMAP_JET)


def _entropy_heat(bgr, block=32):
    g  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    out = np.zeros((H, W), dtype=np.float32)
    for y in range(0, H, block):
        for x in range(0, W, block):
            patch = g[y:y+block, x:x+block]
            h = cv2.calcHist([patch],[0],None,[64],[0,256]).ravel()
            h = h / h.sum() if h.sum() > 0 else h
            hp = h[h > 0]
            ent = float(-np.sum(hp * np.log2(hp))) if len(hp) else 0.0
            out[y:y+block, x:x+block] = ent
    out = np.clip(out / 7.0 * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(out, cv2.COLORMAP_INFERNO)


def _kp_density_heat(bgr, block=32):
    g   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    kps, _ = _ORB.detectAndCompute(g, None)
    density = np.zeros((H, W), dtype=np.float32)
    for kp in kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 0 <= x < W and 0 <= y < H:
            # Gaussian blob around each keypoint
            r = max(1, int(kp.size))
            cv2.circle(density, (x,y), r, 1.0, -1)
    # Blur to make it a smooth heatmap
    density = cv2.GaussianBlur(density, (31,31), 10)
    density = np.clip(density / max(density.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(density, cv2.COLORMAP_HOT)


def _draw_kp(bgr, max_kp=1000):
    g   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kps, _ = _ORB.detectAndCompute(g, None)
    kps = sorted(kps, key=lambda k: -k.response)[:max_kp]
    out = bgr.copy()
    for kp in kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = max(3, int(kp.size // 2))
        cv2.circle(out, (x,y), r, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(out, (x,y), 1, (0, 255, 0), -1)
    return out, len(kps)


def _score_bar(ax, m, height_label, color):
    """Draw a small horizontal score bar on a given Axes."""
    labels  = ["keypoints\n(÷80)", "edge\ndensity", "texture\nentropy", "gradient\nRMS"]
    norms   = [
        min(m["keypoints"]/80.0, 1.0),
        min(m["edge_density"]/0.12, 1.0),
        min(m["entropy"]/6.5, 1.0),
        min(m["gradient_rms"]/40.0, 1.0),
    ]
    score = detection_score(m)
    ax.set_facecolor("#0d0d22")
    bars = ax.barh(labels, norms, color=color, alpha=0.85,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.axvline(score, color="yellow", lw=1.5, linestyle="--")
    ax.set_xlabel(f"Det. score: {score:.2f}", color="yellow",
                  fontsize=7, labelpad=2)
    ax.set_title(height_label, color="white", fontsize=7, pad=2)
    ax.tick_params(colors="white", labelsize=6)
    for sp in ax.spines.values(): sp.set_color("#444466")
    for bar, v in zip(bars, norms):
        ax.text(min(v + 0.03, 0.95), bar.get_y() + bar.get_height()/2,
                f"{v:.2f}", va="center", color="white", fontsize=6)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: ORB keypoint maps on full images
# ─────────────────────────────────────────────────────────────────────────────

def fig_keypoints(img_1m, img_3m, h_orig, out_dir):
    print("  Building keypoint maps …")
    kp_1m, n1 = _draw_kp(img_1m)
    kp_3m, n3 = _draw_kp(img_3m)

    H, W = img_1m.shape[:2]

    # Gradient heat maps
    gh_1m = _gradient_heat(img_1m)
    gh_3m = _gradient_heat(img_3m)

    fig, axes = plt.subplots(2, 2, figsize=(22, 11))
    fig.patch.set_facecolor(DARK_BG)

    panels = [
        (axes[0,0], kp_1m,
         f"ORB Keypoints @ {h_orig:.2f} m  — {n1} features detected"),
        (axes[0,1], kp_3m,
         f"ORB Keypoints @ {TARGET_H:.1f} m  — {n3} features detected  "
         f"(Δ {n3-n1:+d},  {100*(n3-n1)/max(n1,1):+.1f}%)"),
        (axes[1,0], gh_1m,
         f"Gradient Magnitude @ {h_orig:.2f} m  (HOG proxy — red=strong edges)"),
        (axes[1,1], gh_3m,
         f"Gradient Magnitude @ {TARGET_H:.1f} m  (flat roof areas = blue/cold)"),
    ]

    border = ["#44dd44","#ffaa44","#44dd44","#ffaa44"]
    for ax, img, title, bc in zip(
            [ax for row in axes for ax in row],
            [p[1] for p in panels],
            [p[2] for p in panels],
            border):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, color="white", fontsize=9, pad=5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(bc); sp.set_linewidth(2)

    # Annotation: how keypoints relate to detection
    note = (f"ORB keypoints = distinctive image features (corners, blobs).\n"
            f"Object detectors use similar features internally.\n"
            f"Fewer keypoints on car roofs (flat) → harder to identify at 3 m.\n"
            f"1m: {n1} kp  →  3m: {n3} kp  ({100*(n3-n1)/max(n1,1):+.1f}%)")
    fig.text(0.5, 0.01, note, ha="center", color="#aaccff", fontsize=9,
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#12122a", alpha=0.85))

    fig.suptitle("Feature Detection Quality: ORB Keypoints & Gradient Strength",
                 color="white", fontsize=13, y=0.99)
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    path = os.path.join(out_dir, "acc_01_keypoints.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")
    return n1, n3


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Per-object 5-column analysis strip
# ─────────────────────────────────────────────────────────────────────────────

def fig_per_object(img_1m, img_3m, annotations, R_cam, t_cam, K,
                   delta_h, h_orig, hole_mask_3m, out_dir):
    print("  Building per-object analysis …")

    # Collect at most 2 instances per category, max 50m
    cat_counts = defaultdict(int)
    rows = []
    for ann in sorted(annotations, key=lambda a: a["distance"]):
        cat  = ann["category"]
        dist = ann["distance"]
        if cat_counts[cat] >= 2 or dist > 50 or cat not in CAT_BGR:
            continue

        roi_1m, bb1 = get_roi(img_1m, ann, R_cam, t_cam, K, 0.0)
        roi_3m, bb3 = get_roi(img_3m, ann, R_cam, t_cam, K, delta_h)
        if roi_1m is None: continue

        if roi_1m.shape[0] > 8 and roi_1m.shape[1] > 8:
            roi_1m_r = cv2.resize(roi_1m, (200, 200))
        else:
            continue
        roi_3m_r = (cv2.resize(roi_3m, (200, 200))
                    if roi_3m is not None and roi_3m.size > 0
                    else np.zeros((200,200,3), dtype=np.uint8))

        m1  = roi_metrics(roi_1m_r)
        m3  = roi_metrics(roi_3m_r)
        s1  = detection_score(m1)
        s3  = detection_score(m3)

        # Difference image (shows what changed — texture starvation)
        diff = cv2.absdiff(roi_1m_r, roi_3m_r)
        diff_heat = cv2.applyColorMap(
            np.clip(diff.mean(axis=2).astype(np.uint8)*3, 0, 255).astype(np.uint8),
            cv2.COLORMAP_HOT)

        rows.append(dict(
            cat=cat, dist=dist,
            roi_1m=roi_1m_r, roi_3m=roi_3m_r, diff=diff_heat,
            m1=m1, m3=m3, s1=s1, s3=s3, bb1=bb1, bb3=bb3,
        ))
        cat_counts[cat] += 1

    if not rows:
        print("  No valid ROIs found.")
        return {}

    # ── Layout: N rows × 7 columns ───────────────────────────────────────────
    # Col 0: object crop @1m | Col 1: edges @1m | Col 2: score bar @1m
    # Col 3: diff (appearance change)
    # Col 4: object crop @3m | Col 5: edges @3m | Col 6: score bar @3m

    n = len(rows)
    col_w  = [2, 1.5, 1.5, 1.5, 2, 1.5, 1.5]
    fig_w  = sum(col_w) + 1
    fig_h  = n * 2.4 + 1.2

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(DARK_BG)
    gs  = GridSpec(n, 7, figure=fig, hspace=0.06, wspace=0.05,
                   width_ratios=col_w)

    # Column headers
    col_labels = [
        f"Object @ {h_orig:.2f}m",
        "Canny edges",
        f"Det. score\n@ {h_orig:.2f}m",
        "Appearance\nchange",
        f"Object @ {TARGET_H:.1f}m",
        "Canny edges",
        f"Det. score\n@ {TARGET_H:.1f}m",
    ]

    for ri, row in enumerate(rows):
        cat   = row["cat"]
        color = CAT_BGR.get(cat, (200,200,200))
        rgb   = CAT_RGB.get(cat, (0.7,0.7,0.7))

        e1 = _edge_img(row["roi_1m"])
        e3 = _edge_img(row["roi_3m"])

        img_cols  = [row["roi_1m"], e1, None, row["diff"], row["roi_3m"], e3, None]
        score_col = [False, False, True, False, False, False, True]
        metrics   = [None, None, row["m1"], None, None, None, row["m3"]]
        hlabels   = [None, None, f"{h_orig:.2f}m", None,
                     None, None, f"{TARGET_H:.1f}m"]
        bcolors   = ["#44dd44","#44dd44","#44dd44",
                     "#888888",
                     "#ffaa44","#ffaa44","#ffaa44"]

        for ci in range(7):
            ax = fig.add_subplot(gs[ri, ci])

            if score_col[ci]:
                # Score bar subplot
                m   = metrics[ci]
                lbl = hlabels[ci]
                clr = ("#44dd44" if "1m" in lbl or h_orig<2 else "#ffaa44")
                _score_bar(ax, m, lbl, clr)
            else:
                img = img_cols[ci]
                if img is not None and img.size > 0:
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    ax.set_facecolor("#0d0d22")
                    ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                            ha="center", color="#555577", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])

            # Border colour = health
            bc = bcolors[ci]
            for sp in ax.spines.values():
                sp.set_edgecolor(bc); sp.set_linewidth(1.5)

            # Column header on first row
            if ri == 0:
                ax.set_title(col_labels[ci], color="white",
                             fontsize=7.5, pad=3)

        # Row label
        s1 = row["s1"]; s3 = row["s3"]
        pct = (s3-s1)/max(s1,0.01)*100
        col_r = "#44dd44" if pct > -10 else "#ffaa00" if pct > -30 else "#ff4444"
        fig.add_subplot(gs[ri, 0]).set_ylabel(
            f"{cat}\n{row['dist']:.0f}m",
            color=rgb, fontsize=8, fontweight="bold")

        # Score delta annotation between score bars
        ax_mid = fig.add_subplot(gs[ri, 3])
        ax_mid.set_facecolor("#0d0d22")
        ax_mid.text(0.5, 0.55,
                    f"Score:\n{s1:.2f}→{s3:.2f}\n({pct:+.0f}%)",
                    transform=ax_mid.transAxes,
                    ha="center", va="center",
                    color=col_r, fontsize=8, fontweight="bold")
        ax_mid.set_xticks([]); ax_mid.set_yticks([])

    fig.suptitle(
        f"Per-Object Feature Extraction Accuracy: {h_orig:.2f}m → {TARGET_H:.1f}m\n"
        f"Cols: [crop] [edges] [score]  ‖  diff  ‖  [crop@3m] [edges@3m] [score@3m]",
        color="white", fontsize=11, y=0.999)

    path = os.path.join(out_dir, "acc_02_per_object.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")
    return {r["cat"]: r for r in rows}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Full-image feature maps (edge / gradient / entropy / keypoint)
# ─────────────────────────────────────────────────────────────────────────────

def fig_feature_maps(img_1m, img_3m, h_orig, out_dir):
    print("  Building feature maps …")

    maps = {
        "Canny Edges":        (_edge_img(img_1m),    _edge_img(img_3m)),
        "Gradient Magnitude": (_gradient_heat(img_1m), _gradient_heat(img_3m)),
        "Texture Entropy":    (_entropy_heat(img_1m),  _entropy_heat(img_3m)),
        "ORB Keypoint Density":(_kp_density_heat(img_1m), _kp_density_heat(img_3m)),
    }

    n_maps = len(maps)
    fig, axes = plt.subplots(n_maps, 2, figsize=(18, 4.5 * n_maps))
    fig.patch.set_facecolor(DARK_BG)
    if n_maps == 1:
        axes = [axes]

    for ri, (label, (m1, m3)) in enumerate(maps.items()):
        for ci, (ax, img, h_label) in enumerate(zip(
                axes[ri], [m1, m3],
                [f"{h_orig:.2f} m  (original)",
                 f"{TARGET_H:.1f} m  (rendered)"])):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{label}  @  {h_label}", color="white",
                         fontsize=9, pad=4)
            ax.set_xticks([]); ax.set_yticks([])
            bc = "#44dd44" if ci == 0 else "#ffaa44"
            for sp in ax.spines.values():
                sp.set_edgecolor(bc); sp.set_linewidth(2)

        # Row label
        axes[ri][0].set_ylabel(label, color="#aaccff", fontsize=9,
                                fontweight="bold")

    explainer = (
        "Canny edges: outlines used by CNN detectors.   "
        "Gradient map: texture richness (cold=flat roof at 3m).   "
        "Entropy: information density per block.   "
        "ORB density: where distinctive features cluster."
    )
    fig.text(0.5, 0.005, explainer, ha="center", color="#aaccff",
             fontsize=8.5, fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#12122a", alpha=0.85))

    fig.suptitle("Feature Map Comparison: What the Detector 'Sees' at Each Height",
                 color="white", fontsize=12, y=0.998)
    plt.tight_layout(rect=[0, 0.03, 1, 0.995])
    path = os.path.join(out_dir, "acc_03_feature_maps.png")
    plt.savefig(path, dpi=110, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Detection quality score bars per category
# ─────────────────────────────────────────────────────────────────────────────

def fig_scores(row_data: dict, img_1m, img_3m,
               annotations, R_cam, t_cam, K, delta_h, h_orig, out_dir):
    print("  Building detection score chart …")

    # Compute full-image metrics
    m_full_1m = roi_metrics(img_1m)
    m_full_3m = roi_metrics(img_3m)
    s_full_1m = detection_score(m_full_1m)
    s_full_3m = detection_score(m_full_3m)

    cats      = sorted(row_data.keys())
    s1_list   = [row_data[c]["s1"] for c in cats]
    s3_list   = [row_data[c]["s3"] for c in cats]

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(DARK_BG)
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    def styled_ax(ax):
        ax.set_facecolor("#12122a")
        ax.tick_params(colors="white", labelsize=9)
        for sp in ax.spines.values(): sp.set_color("#555577")
        ax.yaxis.grid(True, color="#333355", lw=0.5)
        ax.set_axisbelow(True)

    # ── A: Detection score per category ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    styled_ax(ax)
    x  = np.arange(len(cats))
    bw = 0.32
    b1 = ax.bar(x-bw/2, s1_list, bw, label=f"{h_orig:.2f}m (original)",
                color="#44dd88", edgecolor="white", lw=0.5)
    b3 = ax.bar(x+bw/2, s3_list, bw, label=f"{TARGET_H:.1f}m (rendered)",
                color="#ff8844", edgecolor="white", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([c.upper() for c in cats],
                                          color="white", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Detection Quality Score  (0–1)", color="white", fontsize=10)
    ax.set_title("Overall Feature Extraction Quality per Object Category",
                 color="white", fontsize=11, pad=8)
    ax.legend(facecolor="#22224a", labelcolor="white", fontsize=9)
    ax.axhline(0.5, color="#ffaaaa", lw=1, ls=":", alpha=0.7)
    ax.text(len(cats)-0.1, 0.51, "detection threshold", color="#ffaaaa",
            fontsize=7, ha="right")

    for bar_list, scores in [(b1, s1_list), (b3, s3_list)]:
        for bar, s in zip(bar_list, scores):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{s:.2f}", ha="center", color="white", fontsize=8)

    # Score drop annotation
    for xi, (cat, s1, s3) in enumerate(zip(cats, s1_list, s3_list)):
        drop = (s3-s1)/max(s1,0.01)*100
        col  = "#44dd44" if drop > -10 else "#ffaa00" if drop > -30 else "#ff4444"
        ax.text(xi, max(s1,s3)+0.06, f"{drop:+.0f}%",
                ha="center", color=col, fontsize=8, fontweight="bold")

    # ── B: Metric breakdown at 1m ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    styled_ax(ax)
    metric_names = ["keypoints\n(÷80)", "edges", "entropy", "gradient"]
    for i, cat in enumerate(cats):
        m = row_data[cat]["m1"]
        vals = [min(m["keypoints"]/80,1), min(m["edge_density"]/0.12,1),
                min(m["entropy"]/6.5,1), min(m["gradient_rms"]/40,1)]
        ax.plot(metric_names, vals, "o-",
                color=CAT_RGB.get(cat,(0.7,0.7,0.7)),
                label=cat, lw=1.5, ms=6)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Feature Profile @ {h_orig:.2f}m", color="white", fontsize=10)
    ax.legend(fontsize=7, facecolor="#22224a", labelcolor="white",
              loc="lower right")
    ax.tick_params(axis="x", colors="white", labelsize=8)

    # ── C: Metric breakdown at 3m ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    styled_ax(ax)
    for i, cat in enumerate(cats):
        m = row_data[cat]["m3"]
        vals = [min(m["keypoints"]/80,1), min(m["edge_density"]/0.12,1),
                min(m["entropy"]/6.5,1), min(m["gradient_rms"]/40,1)]
        ax.plot(metric_names, vals, "o-",
                color=CAT_RGB.get(cat,(0.7,0.7,0.7)),
                label=cat, lw=1.5, ms=6)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Feature Profile @ {TARGET_H:.1f}m", color="white", fontsize=10)
    ax.legend(fontsize=7, facecolor="#22224a", labelcolor="white",
              loc="lower right")
    ax.tick_params(axis="x", colors="white", labelsize=8)

    # ── D: Full-image metrics comparison ─────────────────────────────────────
    ax = fig.add_subplot(gs[2, :])
    styled_ax(ax)
    metric_labels = ["Keypoints\n(total)", "Edge density\n(×100)",
                     "Entropy\n(bits)", "Gradient RMS"]
    v1 = [m_full_1m["keypoints"], m_full_1m["edge_density"]*100,
          m_full_1m["entropy"],    m_full_1m["gradient_rms"]]
    v3 = [m_full_3m["keypoints"], m_full_3m["edge_density"]*100,
          m_full_3m["entropy"],    m_full_3m["gradient_rms"]]
    x2 = np.arange(4); bw2 = 0.35
    b1f = ax.bar(x2-bw2/2, v1, bw2, label=f"{h_orig:.2f}m",
                 color="#44dd88", edgecolor="white", lw=0.5)
    b3f = ax.bar(x2+bw2/2, v3, bw2, label=f"{TARGET_H:.1f}m",
                 color="#ff8844", edgecolor="white", lw=0.5)
    ax.set_xticks(x2); ax.set_xticklabels(metric_labels, color="white", fontsize=9)
    ax.set_title(
        f"Full-Image Feature Metrics  —  "
        f"Overall detection score: {h_orig:.2f}m={s_full_1m:.3f}  vs  "
        f"{TARGET_H:.1f}m={s_full_3m:.3f}  ({100*(s_full_3m-s_full_1m)/max(s_full_1m,0.01):+.1f}%)",
        color="white", fontsize=10, pad=6)
    ax.legend(facecolor="#22224a", labelcolor="white", fontsize=9)
    for bars, vals in [(b1f,v1),(b3f,v3)]:
        for b,v in zip(bars,vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.01,
                    f"{v:.2f}" if v<10 else f"{v:.0f}",
                    ha="center", color="white", fontsize=8)

    fig.suptitle(
        "Detection Accuracy Analysis: Feature Quality Scores at 1.5m vs 3.0m",
        color="white", fontsize=13, y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    path = os.path.join(out_dir, "acc_04_scores.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_accuracy_summary(row_data, n_kp_1m, n_kp_3m, h_orig):
    print("\n" + "═"*70)
    print(f"  FEATURE EXTRACTION ACCURACY: {h_orig:.2f}m → {TARGET_H:.1f}m")
    print("═"*70)
    print(f"  Full-image ORB keypoints  : {n_kp_1m} → {n_kp_3m}"
          f"  ({100*(n_kp_3m-n_kp_1m)/max(n_kp_1m,1):+.1f}%)")
    print()
    print(f"  {'Category':<16}  {'Dist':>5}  "
          f"{'KP@1m':>6}{'KP@3m':>7}  "
          f"{'Edge@1m':>8}{'Edge@3m':>8}  "
          f"{'Score@1m':>9}{'Score@3m':>9}  {'Δscore':>7}")
    print("  " + "-"*80)
    for cat, r in sorted(row_data.items(), key=lambda x: -x[1]["s1"]):
        m1, m3 = r["m1"], r["m3"]
        s1, s3 = r["s1"], r["s3"]
        dscore = (s3-s1)/max(s1,0.01)*100
        flag   = "✓" if dscore > -10 else "!" if dscore > -30 else "✗"
        print(f"  {cat:<16}  {r['dist']:>5.0f}  "
              f"{m1['keypoints']:>6}{m3['keypoints']:>7}  "
              f"{m1['edge_density']:>8.4f}{m3['edge_density']:>8.4f}  "
              f"{s1:>9.3f}{s3:>9.3f}  {dscore:>+6.1f}%  {flag}")
    print("═"*70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuscenes", default="nuscenes_data")
    ap.add_argument("--version",  default="v1.0-mini")
    ap.add_argument("--sample",   default=None)
    ap.add_argument("--cam",      default="CAM_FRONT")
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[1/6] Loading NuScenes …")
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.nuscenes, verbose=False)

    token = args.sample
    if not token:
        best_t, best_n = None, 0
        for s in nusc.sample[:40]:
            n = sum(1 for b in nusc.get_boxes(s["data"][args.cam])
                    if CAT_MAP.get(b.name,"") in CAT_BGR)
            if n > best_n: best_n, best_t = n, s["token"]
        token = best_t or nusc.sample[0]["token"]

    print("\n[2/6] Loading sample …")
    data = load_sample(nusc, token, args.cam)
    h_orig = data["cam_height"]
    print(f"  Camera height: {h_orig:.3f}m,  target: {TARGET_H}m,  "
          f"delta_h: {TARGET_H-h_orig:+.3f}m")
    print(f"  Annotations: {len(data['annotations'])}")

    # ── Build 3m render ───────────────────────────────────────────────────────
    print(f"\n[3/6] Rendering scene at {TARGET_H}m …")
    img_3m, hole_mask, depth_map, delta_h = build_3m_render(data)
    img_1m = data["image"]
    print(f"  Hole coverage: {hole_mask.mean()*100:.1f}%")

    # ── Figure 1: Keypoint maps ───────────────────────────────────────────────
    print("\n[4/6] Figure 1: ORB keypoints + gradient maps …")
    n1, n3 = fig_keypoints(img_1m, img_3m, h_orig, args.output_dir)

    # ── Figure 2: Per-object analysis ─────────────────────────────────────────
    print("\n[5/6] Figures 2–4: Per-object, feature maps, scores …")
    row_data = fig_per_object(
        img_1m, img_3m, data["annotations"],
        data["R_cam"], data["t_cam"], data["K"],
        delta_h, h_orig, hole_mask, args.output_dir)

    # ── Figures 3 & 4 ─────────────────────────────────────────────────────────
    fig_feature_maps(img_1m, img_3m, h_orig, args.output_dir)

    if row_data:
        fig_scores(row_data, img_1m, img_3m,
                   data["annotations"], data["R_cam"], data["t_cam"],
                   data["K"], delta_h, h_orig, args.output_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n[6/6] Summary …")
    if row_data:
        print_accuracy_summary(row_data, n1, n3, h_orig)

    print(f"Outputs → {os.path.abspath(args.output_dir)}/\n")


if __name__ == "__main__":
    main()
