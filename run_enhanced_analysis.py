#!/usr/bin/env python3
"""
Enhanced height-change analysis:
  • Per-class stats   (car vs person separately, every height)
  • Per-distance stats (near 0-15m, mid 15-30m, far 30-60m)
  • Confidence distributions per height (box-plot ready)
  • Per-image galleries for 8 selected samples (incl. user-requested sample-17)
  • Sample × height miss-rate heatmap
"""
import os, sys, json, warnings
import numpy as np
import cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from PIL import Image as PILImage

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from ultralytics import YOLO
from transformers import pipeline as hf_pipeline
from src.real_image_transformer import inpaint_holes

# ── Config ────────────────────────────────────────────────────────────────────
NUSC_ROOT  = "nuscenes_data"
SCENE_NAME = "scene-0103"
CAM_NAME   = "CAM_BACK"
H_NATIVE   = 1.568

HEIGHTS_FULL = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
HEIGHTS_VIZ  = [1.568, 3.0, 3.5, 5.0]
SELECTED     = [0, 6, 10, 17, 25, 32, 36, 39]    # user-requested sample 17 included
DIST_BINS    = [(0, 15, 'near'), (15, 30, 'mid'), (30, 60, 'far')]

OUT_DIR   = Path('outputs/enhanced')
PROOF_DIR = Path('outputs/research_proof')
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROOF_DIR.mkdir(parents=True, exist_ok=True)

BG, GRID = '#0a0a1a', '#1e2030'
COL_CAR     = '#00ff88'
COL_PERSON  = '#ffaa44'
COL_NATIVE  = '#00ff88'
COL_3M      = '#ffcc00'
COL_3p5M    = '#ff9933'
COL_5M      = '#ff4444'

print("Loading models...")
dav2 = hf_pipeline('depth-estimation',
                   model='depth-anything/Depth-Anything-V2-Small-hf',
                   device=0)
yolo = YOLO('yolov8n.pt')
print("  loaded")


# ── DAV2 metric depth ─────────────────────────────────────────────────────────
def dav2_metric_depth(image_bgr, pts_lidar, R_lidar, t_lidar, R_cam, t_cam, K):
    H, W = image_bgr.shape[:2]
    rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    raw  = dav2(PILImage.fromarray(rgb))['predicted_depth'].squeeze().cpu().numpy()
    raw  = cv2.resize(raw.astype(np.float32), (W, H))

    pts_ego = (R_lidar @ pts_lidar.T).T + t_lidar
    pts_c   = (R_cam.T @ (pts_ego - t_cam).T).T
    valid   = pts_c[:, 2] > 0.5
    pv      = pts_c[valid]
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    d_l = pv[:, 2]
    ui  = np.round(fx * pv[:, 0] / d_l + cx).astype(int)
    vi  = np.round(fy * pv[:, 1] / d_l + cy).astype(int)
    in_img = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (d_l < 80)

    d_lidar = d_l[in_img].astype(np.float32)
    d_dav2  = raw[vi[in_img], ui[in_img]]
    inv_l   = 1.0 / (d_lidar + 1e-6)
    A       = np.column_stack([d_dav2, np.ones_like(d_dav2)])
    a, b    = np.linalg.lstsq(A, inv_l, rcond=None)[0]
    inv_m   = np.clip(a * raw + b, 1.0/200., 1.0/0.1)
    return (1.0 / inv_m).astype(np.float32)


# ── Warp ──────────────────────────────────────────────────────────────────────
def warp_h(image, depth, R_cam, t_cam, K, h_new):
    H, W = image.shape[:2]
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    Z = depth; X = (uu - cx) * Z / fx; Y = (vv - cy) * Z / fy
    pts_cam = np.stack([X.ravel(), Y.ravel(), Z.ravel()])
    pts_ego = R_cam @ pts_cam + t_cam[:, None]
    t_new = t_cam.copy(); t_new[2] = h_new
    pts_new = R_cam.T @ (pts_ego - t_new[:, None])
    Xn, Yn, Zn = pts_new[0].reshape(H,W), pts_new[1].reshape(H,W), pts_new[2].reshape(H,W)
    valid = (depth > 0.1) & (depth < 200)
    in_f = (Zn > 0.1) & valid
    with np.errstate(divide='ignore', invalid='ignore'):
        un = np.where(in_f, fx * Xn / Zn + cx, -1.0)
        vn = np.where(in_f, fy * Yn / Zn + cy, -1.0)
    ui = np.round(un).astype(np.int32); vi = np.round(vn).astype(np.int32)
    ok = (in_f & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H))
    warped = np.zeros_like(image); filled = np.zeros((H, W), dtype=bool)
    zbuf = np.full((H, W), np.inf)
    idx = np.where(ok.ravel())[0]
    order = np.argsort(-Zn.ravel()[idx]); idx = idx[order]
    sv, su = idx // W, idx % W
    dv, du = vi.ravel()[idx], ui.ravel()[idx]
    dz = Zn.ravel()[idx]
    closer = dz < zbuf[dv, du]
    zbuf[dv[closer], du[closer]] = dz[closer]
    warped[dv[closer], du[closer]] = image[sv[closer], su[closer]]
    filled[dv[closer], du[closer]] = True
    sky = ~valid; warped[sky] = image[sky]; filled[sky] = True
    hole = (~filled) & valid
    if hole.any(): warped = inpaint_holes(warped, hole, inpaint_radius=8)
    return warped


# ── GT projection with class + distance ──────────────────────────────────────
def gt_boxes(nusc, sample_token, cam_name, K, R_cam, t_cam, h_target):
    sample = nusc.get('sample', sample_token)
    cam_sd = nusc.get('sample_data', sample['data'][cam_name])
    boxes  = nusc.get_boxes(cam_sd['token'])
    ego    = nusc.get('ego_pose', cam_sd['ego_pose_token'])
    R_ego  = Quaternion(ego['rotation']).rotation_matrix
    t_ego  = np.array(ego['translation'])
    t_new  = t_cam.copy(); t_new[2] = h_target
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    H, W = 900, 1600
    out = []
    for box in boxes:
        nm = box.name.split('.')[0]
        if nm == 'vehicle' and 'pedestrian' not in box.name:
            cat = 'car'
        elif nm == 'human':
            cat = 'person'
        else:
            continue
        cg = box.corners().T
        ce = (R_ego.T @ (cg - t_ego).T).T
        pc = (R_cam.T @ (ce - t_new).T).T
        inf = pc[:, 2] > 0.1
        if inf.sum() < 4: continue
        Z = np.where(inf, pc[:, 2], np.nan)
        pu = np.where(inf, fx * pc[:, 0] / (Z + 1e-9) + cx, np.nan)
        pv = np.where(inf, fy * pc[:, 1] / (Z + 1e-9) + cy, np.nan)
        x1, y1 = np.nanmin(pu), np.nanmin(pv)
        x2, y2 = np.nanmax(pu), np.nanmax(pv)
        if x2 < 0 or y2 < 0 or x1 >= W or y1 >= H: continue
        x1 = max(0., x1); y1 = max(0., y1)
        x2 = min(float(W), x2); y2 = min(float(H), y2)
        if (x2 - x1) < 5 or (y2 - y1) < 5: continue
        dist = float(np.nanmean(Z))
        out.append({'class': cat, 'box': [x1, y1, x2, y2], 'dist': dist})
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., ix2 - ix1) * max(0., iy2 - iy1)
    if inter == 0: return 0.
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)


_REMAP = {'car':'car','truck':'car','bus':'car','motorcycle':'car','bicycle':'car','person':'person'}

def parse_yolo(yolo_res):
    out = []
    if not yolo_res or len(yolo_res[0].boxes) == 0: return out
    bxs = yolo_res[0].boxes
    names = yolo_res[0].names
    for i in range(len(bxs)):
        cn = names[int(bxs.cls[i])]
        cat = _REMAP.get(cn)
        if cat is None: continue
        out.append({'class': cat,
                    'box': list(bxs.xyxy[i].cpu().numpy()),
                    'conf': float(bxs.conf[i])})
    return out


def match_detections(gt, dets, iou_thr=0.25):
    """Return matched_gt_idx, matched_det_idx, and per-GT result list."""
    matched_g, matched_d = set(), set()
    per_gt = []
    for gi, g in enumerate(gt):
        best_iou = 0; best_di = -1; best_conf = 0
        for di, d in enumerate(dets):
            if di in matched_d: continue
            if d['class'] != g['class']: continue
            i = iou(g['box'], d['box'])
            if i > best_iou and i >= iou_thr:
                best_iou = i; best_di = di; best_conf = d['conf']
        if best_di >= 0:
            matched_g.add(gi); matched_d.add(best_di)
            per_gt.append({'matched': True, 'iou': best_iou, 'conf': best_conf,
                           'gt': g})
        else:
            per_gt.append({'matched': False, 'iou': 0, 'conf': 0, 'gt': g})
    return matched_g, matched_d, per_gt


# ── Load NuScenes ─────────────────────────────────────────────────────────────
print("Loading NuScenes…")
nusc = NuScenes('v1.0-mini', NUSC_ROOT, verbose=False)
scene_obj = [s for s in nusc.scene if s['name'] == SCENE_NAME][0]
tokens = []
tok = scene_obj['first_sample_token']
while tok:
    s = nusc.get('sample', tok)
    tokens.append(tok); tok = s['next']
print(f"  {len(tokens)} samples")


def load_sample(tok):
    s = nusc.get('sample', tok)
    cam_sd  = nusc.get('sample_data', s['data'][CAM_NAME])
    img     = cv2.imread(os.path.join(NUSC_ROOT, cam_sd['filename']))
    cal     = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    K       = np.array(cal['camera_intrinsic'], dtype=np.float64)
    R_cam   = Quaternion(cal['rotation']).rotation_matrix
    t_cam   = np.array(cal['translation'], dtype=np.float64)
    lid_sd  = nusc.get('sample_data', s['data']['LIDAR_TOP'])
    pc      = LidarPointCloud.from_file(os.path.join(NUSC_ROOT, lid_sd['filename']))
    pts     = pc.points[:3].T
    lcal    = nusc.get('calibrated_sensor', lid_sd['calibrated_sensor_token'])
    R_lid   = Quaternion(lcal['rotation']).rotation_matrix
    t_lid   = np.array(lcal['translation'], dtype=np.float64)
    fname   = cam_sd['filename']
    return img, K, R_cam, t_cam, pts, R_lid, t_lid, fname


# ── PHASE 1: Cache depth + warped images for every (sample, height) ──────────
# Store detailed records to compute per-class, per-distance, confidence stats
records = []  # one record per GT object
det_records = []  # one record per YOLO detection
heatmap = np.zeros((len(tokens), len(HEIGHTS_FULL)))  # miss rate

# Save renders for selected samples
viz_renders = {}   # {sample_idx: {height: rendered_image}}
viz_gt = {}
viz_yolo = {}
viz_fnames = {}

print(f"\nProcessing {len(tokens)} samples × {len(HEIGHTS_FULL)} heights…")
for idx, token in enumerate(tokens):
    print(f"  [{idx+1:02d}/{len(tokens)}]", end='', flush=True)
    img, K, Rc, tc, pts, Rl, tl, fname = load_sample(token)
    if img is None:
        print(' SKIP'); continue

    depth = dav2_metric_depth(img, pts, Rl, tl, Rc, tc, K)

    if idx in SELECTED:
        viz_renders[idx] = {}
        viz_gt[idx] = {}
        viz_yolo[idx] = {}
        viz_fnames[idx] = os.path.basename(fname)

    for hi, h in enumerate(HEIGHTS_FULL):
        if abs(h - H_NATIVE) < 0.05:
            rendered = img.copy()
        else:
            rendered = warp_h(img, depth, Rc, tc, K, h)

        yres = yolo(rendered, verbose=False, conf=0.25)
        dets = parse_yolo(yres)
        gt   = gt_boxes(nusc, token, CAM_NAME, K, Rc, tc, h)

        mg, md, per_gt = match_detections(gt, dets, iou_thr=0.25)
        for r in per_gt:
            records.append({
                'sample_idx': idx, 'height': h, 'class': r['gt']['class'],
                'dist': r['gt']['dist'], 'matched': bool(r['matched']),
                'conf': r['conf'], 'iou': r['iou'],
            })
        for di, d in enumerate(dets):
            det_records.append({
                'sample_idx': idx, 'height': h, 'class': d['class'],
                'conf': d['conf'], 'matched': bool(di in md),
            })

        n_gt = len(gt)
        n_miss = n_gt - len(mg)
        heatmap[idx, hi] = n_miss / max(1, n_gt)

        # Cache viz height + render
        if idx in SELECTED and any(abs(h - hv) < 0.05 for hv in HEIGHTS_VIZ):
            viz_renders[idx][h] = rendered
            viz_gt[idx][h] = gt
            viz_yolo[idx][h] = dets

    print(' done')

# ── Save raw records ──────────────────────────────────────────────────────────
with open(OUT_DIR / 'gt_records.json', 'w') as f:
    json.dump(records, f)
with open(OUT_DIR / 'det_records.json', 'w') as f:
    json.dump(det_records, f)
np.save(OUT_DIR / 'heatmap.npy', heatmap)
print(f"\nSaved {len(records)} GT records, {len(det_records)} detection records")


# ── Aggregation helpers ───────────────────────────────────────────────────────
def per_class_miss(records, cls):
    out = {}
    for h in HEIGHTS_FULL:
        rels = [r for r in records if r['class'] == cls and abs(r['height'] - h) < 0.01]
        if not rels: out[h] = (0.0, 0)
        else:
            mr = 1.0 - sum(r['matched'] for r in rels) / len(rels)
            out[h] = (mr, len(rels))
    return out

def dist_strat_miss(records):
    out = {h: {} for h in HEIGHTS_FULL}
    for h in HEIGHTS_FULL:
        for lo, hi, name in DIST_BINS:
            rels = [r for r in records if abs(r['height'] - h) < 0.01 and lo <= r['dist'] < hi]
            if not rels: out[h][name] = (0.0, 0)
            else:
                mr = 1.0 - sum(r['matched'] for r in rels) / len(rels)
                out[h][name] = (mr, len(rels))
    return out

def conf_distribution(det_records):
    out = {}
    for h in HEIGHTS_FULL:
        rels = [d['conf'] for d in det_records if abs(d['height'] - h) < 0.01]
        out[h] = np.array(rels) if rels else np.array([])
    return out


# ── Fig 14: Per-class miss rate vs height ────────────────────────────────────
print("\nFigure 14: per-class miss rate…")
car_stats    = per_class_miss(records, 'car')
person_stats = per_class_miss(records, 'person')

fig, ax = plt.subplots(figsize=(11, 6.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
for sp in ax.spines.values(): sp.set_color(GRID)
ax.tick_params(colors='#cccccc')

car_mrs = [car_stats[h][0] for h in HEIGHTS_FULL]
car_ns  = [car_stats[h][1] for h in HEIGHTS_FULL]
per_mrs = [person_stats[h][0] for h in HEIGHTS_FULL]
per_ns  = [person_stats[h][1] for h in HEIGHTS_FULL]

ax.plot(HEIGHTS_FULL, car_mrs, 'o-', color=COL_CAR, lw=3, ms=10,
        label=f'Cars (n={sum(car_ns)} obs)', alpha=0.95)
ax.plot(HEIGHTS_FULL, per_mrs, 's-', color=COL_PERSON, lw=3, ms=10,
        label=f'Pedestrians (n={sum(per_ns)} obs)', alpha=0.95)

# Annotate point counts
for h, m, n in zip(HEIGHTS_FULL, car_mrs, car_ns):
    ax.annotate(f'{n}', (h, m), xytext=(0,-18), textcoords='offset points',
                ha='center', color=COL_CAR, fontsize=8, alpha=0.7)

ax.axvline(H_NATIVE, color='#888', ls='--', alpha=0.6, label=f'Default {H_NATIVE}m')
ax.set_xlabel('Camera Height (m)', fontsize=12, color='#cccccc')
ax.set_ylabel('Miss Rate  (1 - matched/GT)', fontsize=12, color='#cccccc')
ax.set_title('Per-Class Detection Miss Rate vs Camera Height\n(scene-0103 CAM_BACK, IoU≥0.25, 40 samples × 9 heights)',
             color='white', fontsize=13, fontweight='bold')
ax.legend(facecolor='#111', edgecolor=GRID, labelcolor='#cccccc', fontsize=10)
ax.grid(True, color=GRID, alpha=0.6)
ax.set_ylim(0, 1.05)
ax.xaxis.label.set_color('#cccccc'); ax.yaxis.label.set_color('#cccccc')
plt.tight_layout()
fig14 = PROOF_DIR / 'fig14_per_class_miss.png'
plt.savefig(fig14, dpi=150, bbox_inches='tight', facecolor=BG); plt.close()
print(f"  → {fig14}")


# ── Fig 15: Distance-stratified miss rate ────────────────────────────────────
print("Figure 15: distance-stratified miss…")
dist_stats = dist_strat_miss(records)
fig, ax = plt.subplots(figsize=(11, 6.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
for sp in ax.spines.values(): sp.set_color(GRID)
ax.tick_params(colors='#cccccc')

dist_colors = {'near': '#00ff88', 'mid': '#ffcc00', 'far': '#ff4444'}
for lo, hi, name in DIST_BINS:
    mrs = [dist_stats[h][name][0] for h in HEIGHTS_FULL]
    ns  = [dist_stats[h][name][1] for h in HEIGHTS_FULL]
    ax.plot(HEIGHTS_FULL, mrs, 'o-', color=dist_colors[name], lw=3, ms=9,
            label=f'{name.upper()} ({lo}-{hi}m, n={sum(ns)})', alpha=0.95)

ax.axvline(H_NATIVE, color='#888', ls='--', alpha=0.6, label=f'Default {H_NATIVE}m')
ax.set_xlabel('Camera Height (m)', fontsize=12, color='#cccccc')
ax.set_ylabel('Miss Rate', fontsize=12, color='#cccccc')
ax.set_title('Distance-Stratified Detection Miss Rate vs Camera Height\nFar objects degrade fastest as camera rises',
             color='white', fontsize=13, fontweight='bold')
ax.legend(facecolor='#111', edgecolor=GRID, labelcolor='#cccccc', fontsize=10)
ax.grid(True, color=GRID, alpha=0.6); ax.set_ylim(0, 1.05)
ax.xaxis.label.set_color('#cccccc'); ax.yaxis.label.set_color('#cccccc')
plt.tight_layout()
fig15 = PROOF_DIR / 'fig15_distance_strat.png'
plt.savefig(fig15, dpi=150, bbox_inches='tight', facecolor=BG); plt.close()
print(f"  → {fig15}")


# ── Fig 16: Confidence distribution box plot ─────────────────────────────────
print("Figure 16: confidence distribution boxplot…")
conf_dist = conf_distribution(det_records)
fig, ax = plt.subplots(figsize=(12, 6.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
for sp in ax.spines.values(): sp.set_color(GRID)
ax.tick_params(colors='#cccccc')

box_data = [conf_dist[h] for h in HEIGHTS_FULL if len(conf_dist[h]) > 0]
positions = [h for h in HEIGHTS_FULL if len(conf_dist[h]) > 0]
bp = ax.boxplot(box_data, positions=positions, widths=0.28,
                patch_artist=True, showfliers=True,
                medianprops=dict(color='#00ff88', linewidth=2),
                whiskerprops=dict(color='#888'),
                capprops=dict(color='#888'),
                flierprops=dict(marker='.', markersize=3, markerfacecolor='#666', markeredgecolor='#444'))

cmap = LinearSegmentedColormap.from_list('grad', ['#00ff88', '#ffcc00', '#ff4444'])
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(cmap(i / max(1, len(bp['boxes'])-1)))
    patch.set_alpha(0.65)
    patch.set_edgecolor('#cccccc')

# Mean line
means = [np.mean(d) for d in box_data]
ax.plot(positions, means, 'D-', color='#4499ff', ms=8, lw=2,
        label='Mean confidence', zorder=10)

ax.axvline(H_NATIVE, color='#888', ls='--', alpha=0.6, label=f'Default {H_NATIVE}m')
ax.set_xlabel('Camera Height (m)', fontsize=12, color='#cccccc')
ax.set_ylabel('YOLOv8n Detection Confidence', fontsize=12, color='#cccccc')
ax.set_title('Confidence Distribution per Camera Height\n(Box = 25/50/75 percentile, whiskers = 1.5×IQR)',
             color='white', fontsize=13, fontweight='bold')
ax.legend(facecolor='#111', edgecolor=GRID, labelcolor='#cccccc', fontsize=10)
ax.grid(True, color=GRID, alpha=0.6); ax.set_ylim(0.15, 1.0)
ax.xaxis.label.set_color('#cccccc'); ax.yaxis.label.set_color('#cccccc')
ax.set_xticks(HEIGHTS_FULL); ax.set_xticklabels([f'{h}' for h in HEIGHTS_FULL])
plt.tight_layout()
fig16 = PROOF_DIR / 'fig16_conf_boxplot.png'
plt.savefig(fig16, dpi=150, bbox_inches='tight', facecolor=BG); plt.close()
print(f"  → {fig16}")


# ── Fig 17: Sample × height heatmap ──────────────────────────────────────────
print("Figure 17: sample × height heatmap…")
fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
for sp in ax.spines.values(): sp.set_color(GRID)

cmap = LinearSegmentedColormap.from_list('mr', ['#003300', '#00ff88', '#ffcc00', '#ff4444', '#770000'])
im = ax.imshow(heatmap, aspect='auto', cmap=cmap, vmin=0, vmax=1,
               interpolation='nearest')

# Mark target sample 17
ax.axhline(17, color='#4499ff', lw=2, ls='--', alpha=0.8)
ax.text(len(HEIGHTS_FULL) - 0.4, 17, '← sample 17 (target)', color='#4499ff',
        fontsize=9, va='center', fontweight='bold')

ax.set_xticks(range(len(HEIGHTS_FULL)))
ax.set_xticklabels([f'{h}m' for h in HEIGHTS_FULL], color='#cccccc')
ax.set_yticks(range(0, len(tokens), 4))
ax.set_yticklabels([f'#{i}' for i in range(0, len(tokens), 4)], color='#cccccc')
ax.set_xlabel('Camera Height', color='#cccccc', fontsize=12)
ax.set_ylabel('Sample Index (40 total)', color='#cccccc', fontsize=12)
ax.set_title('Per-Sample Miss-Rate Heatmap\n(rows = 40 NuScenes samples, columns = 9 heights)',
             color='white', fontsize=13, fontweight='bold')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Miss Rate', color='#cccccc')
cbar.ax.yaxis.set_tick_params(color='#cccccc')
for tk in cbar.ax.get_yticklabels(): tk.set_color('#cccccc')
plt.tight_layout()
fig17 = PROOF_DIR / 'fig17_sample_heatmap.png'
plt.savefig(fig17, dpi=150, bbox_inches='tight', facecolor=BG); plt.close()
print(f"  → {fig17}")


# ── Fig 18: Detection count + miss + FP bars per height ──────────────────────
print("Figure 18: detection count stacked bars…")
n_dets = {h: 0 for h in HEIGHTS_FULL}
n_gts  = {h: 0 for h in HEIGHTS_FULL}
n_miss = {h: 0 for h in HEIGHTS_FULL}
n_fp   = {h: 0 for h in HEIGHTS_FULL}
for r in records:
    n_gts[r['height']] += 1
    if not r['matched']: n_miss[r['height']] += 1
for d in det_records:
    n_dets[d['height']] += 1
    if not d['matched']: n_fp[d['height']] += 1

fig, ax = plt.subplots(figsize=(12, 6.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
for sp in ax.spines.values(): sp.set_color(GRID)
ax.tick_params(colors='#cccccc')

x = np.arange(len(HEIGHTS_FULL))
w = 0.32
matched_counts = [n_gts[h] - n_miss[h] for h in HEIGHTS_FULL]
missed_counts  = [n_miss[h] for h in HEIGHTS_FULL]
fp_counts      = [n_fp[h] for h in HEIGHTS_FULL]

ax.bar(x - w, matched_counts, w, label='Matched GT (TP)', color='#00ff88', edgecolor=GRID)
ax.bar(x,    missed_counts,  w, label='Missed GT (FN)',  color='#ff4444', edgecolor=GRID)
ax.bar(x + w, fp_counts,     w, label='False positives (FP)', color='#ffcc00', edgecolor=GRID)

ax.set_xticks(x); ax.set_xticklabels([f'{h}m' for h in HEIGHTS_FULL])
ax.set_xlabel('Camera Height', color='#cccccc', fontsize=12)
ax.set_ylabel('Count (across 40 samples)', color='#cccccc', fontsize=12)
ax.set_title('Detection Outcomes per Height: TP / FN / FP',
             color='white', fontsize=13, fontweight='bold')
ax.legend(facecolor='#111', edgecolor=GRID, labelcolor='#cccccc', fontsize=10)
ax.grid(True, axis='y', color=GRID, alpha=0.6)
ax.xaxis.label.set_color('#cccccc'); ax.yaxis.label.set_color('#cccccc')
plt.tight_layout()
fig18 = PROOF_DIR / 'fig18_tp_fn_fp_bars.png'
plt.savefig(fig18, dpi=150, bbox_inches='tight', facecolor=BG); plt.close()
print(f"  → {fig18}")


# ── Per-image galleries (figs 19-26) for selected samples ────────────────────
print("\nGenerating per-image galleries…")

def render_gallery(idx, save_path):
    fig, axes = plt.subplots(2, 4, figsize=(28, 14))
    fig.patch.set_facecolor(BG)
    col_list = [COL_NATIVE, COL_3M, COL_3p5M, COL_5M]

    for col, (h, c) in enumerate(zip(HEIGHTS_VIZ, col_list)):
        img_bgr = viz_renders[idx][h]
        gt_h = viz_gt[idx][h]
        det_h = viz_yolo[idx][h]

        # TOP: image with GT
        ax = axes[0][col]
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax.axis('off'); ax.set_facecolor(BG)
        h_label = f'{h}m' + (' (native)' if abs(h - H_NATIVE) < 0.05 else '')
        ax.set_title(f'{h_label}\nGround Truth', color=c, fontsize=12, fontweight='bold')

        n_car = sum(1 for g in gt_h if g['class']=='car')
        n_per = sum(1 for g in gt_h if g['class']=='person')
        for g in gt_h:
            x1,y1,x2,y2 = g['box']
            box_col = COL_CAR if g['class']=='car' else COL_PERSON
            ax.add_patch(mpatches.Rectangle((x1,y1), x2-x1, y2-y1,
                lw=2, edgecolor=box_col, facecolor='none', alpha=0.9))
            ax.text(x1+2, y1+12, f"{g['class']} {g['dist']:.0f}m",
                    color=box_col, fontsize=7, fontweight='bold',
                    bbox=dict(facecolor='#000a', pad=1, edgecolor='none'))
        ax.text(10, 870, f'{n_car}c + {n_per}p  =  {n_car+n_per} GT',
                color=c, fontsize=10, fontweight='bold',
                bbox=dict(facecolor='#0009', pad=4, edgecolor=c, boxstyle='round'))

        # BOTTOM: YOLO detections
        ax = axes[1][col]
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax.axis('off'); ax.set_facecolor(BG)
        ax.set_title('YOLOv8n Detections', color=c, fontsize=11)

        # Match-aware coloring
        _, md, _ = match_detections(gt_h, det_h, iou_thr=0.25)
        for di, d in enumerate(det_h):
            x1,y1,x2,y2 = d['box']
            col_d = '#4499ff' if di in md else '#ff8800'   # blue=matched, orange=FP
            ax.add_patch(mpatches.Rectangle((x1,y1), x2-x1, y2-y1,
                lw=2, edgecolor=col_d, facecolor='none', alpha=0.9))
            ax.text(x1+2, y1+12, f"{d['class']} {d['conf']:.2f}",
                    color=col_d, fontsize=7,
                    bbox=dict(facecolor='#000a', pad=1, edgecolor='none'))

        mean_conf = np.mean([d['conf'] for d in det_h]) if det_h else 0
        n_tp = len(md); n_fp = len(det_h) - n_tp; n_fn = len(gt_h) - n_tp
        miss_rate = n_fn / max(1, len(gt_h)) * 100
        ax.text(10, 870,
                f'{len(det_h)} dets | TP={n_tp} FP={n_fp} FN={n_fn} | miss={miss_rate:.0f}% | μ-conf={mean_conf:.2f}',
                color=c, fontsize=10, fontweight='bold',
                bbox=dict(facecolor='#0009', pad=4, edgecolor=c, boxstyle='round'))

    fig.suptitle(f'Sample #{idx} — Detection Degradation Across 4 Heights\n'
                 f'{viz_fnames.get(idx, "")}',
                 fontsize=13, color='white', fontweight='bold', y=1.01)
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor=BG)
    plt.close()

# Fig indices: 19, 20, 21, 22, 23, 24, 25, 26
gallery_paths = {}
for i, idx in enumerate(SELECTED):
    fig_num = 19 + i
    save_path = PROOF_DIR / f'fig{fig_num}_gallery_sample{idx:02d}.png'
    render_gallery(idx, save_path)
    gallery_paths[idx] = save_path
    print(f"  fig{fig_num} (sample {idx}) → {save_path}")


# ── Summary CSV/JSON ──────────────────────────────────────────────────────────
summary = {
    'meta': {
        'scene': SCENE_NAME, 'cam': CAM_NAME, 'n_samples': len(tokens),
        'heights_full': HEIGHTS_FULL, 'heights_viz': HEIGHTS_VIZ,
        'selected_samples': SELECTED,
        'target_image': 'n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151611887558.jpg (sample 17)',
    },
    'per_height': {
        str(h): {
            'n_gt':       int(sum(1 for r in records if abs(r['height']-h)<0.01)),
            'n_matched':  int(sum(1 for r in records if abs(r['height']-h)<0.01 and r['matched'])),
            'n_det':      int(sum(1 for d in det_records if abs(d['height']-h)<0.01)),
            'n_fp':       int(sum(1 for d in det_records if abs(d['height']-h)<0.01 and not d['matched'])),
            'miss_rate':  1.0 - sum(r['matched'] for r in records if abs(r['height']-h)<0.01) /
                          max(1, sum(1 for r in records if abs(r['height']-h)<0.01)),
            'mean_conf':  float(np.mean([d['conf'] for d in det_records if abs(d['height']-h)<0.01])) if any(abs(d['height']-h)<0.01 for d in det_records) else 0,
            'median_conf':float(np.median([d['conf'] for d in det_records if abs(d['height']-h)<0.01])) if any(abs(d['height']-h)<0.01 for d in det_records) else 0,
        } for h in HEIGHTS_FULL
    },
    'per_class': {
        'car':    {str(h): {'miss_rate': float(car_stats[h][0]),    'n': int(car_stats[h][1])} for h in HEIGHTS_FULL},
        'person': {str(h): {'miss_rate': float(person_stats[h][0]), 'n': int(person_stats[h][1])} for h in HEIGHTS_FULL},
    },
    'per_distance': {
        str(h): {name: {'miss_rate': float(dist_stats[h][name][0]),
                        'n': int(dist_stats[h][name][1])}
                 for _, _, name in DIST_BINS}
        for h in HEIGHTS_FULL
    },
    'selected_samples_info': {
        str(idx): {
            'filename': viz_fnames.get(idx, ''),
            'per_height': {
                str(h): {
                    'n_gt':    len(viz_gt[idx][h]) if h in viz_gt.get(idx, {}) else 0,
                    'n_det':   len(viz_yolo[idx][h]) if h in viz_yolo.get(idx, {}) else 0,
                    'mean_conf': float(np.mean([d['conf'] for d in viz_yolo[idx][h]])) if viz_yolo.get(idx,{}).get(h) else 0,
                } for h in HEIGHTS_VIZ
            }
        } for idx in SELECTED
    },
}
with open(PROOF_DIR / 'enhanced_stats.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved enhanced_stats.json")
print("\n=== Done ===")
for h in HEIGHTS_FULL:
    s = summary['per_height'][str(h)]
    print(f"  h={h}m  miss={s['miss_rate']*100:.1f}%  conf={s['mean_conf']:.3f}  GT={s['n_gt']}  TP={s['n_matched']}  FP={s['n_fp']}")
