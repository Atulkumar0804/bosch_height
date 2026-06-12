#!/usr/bin/env python3
"""Generate per-image galleries for 8 selected samples (uses DAV2 + YOLO)."""
import os, sys
sys.path.insert(0, '.')
import numpy as np, cv2, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image as PILImage

warnings.filterwarnings('ignore')

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from ultralytics import YOLO
from transformers import pipeline as hf_pipeline
from src.real_image_transformer import inpaint_holes

NUSC_ROOT  = "nuscenes_data"
SCENE_NAME = "scene-0103"
CAM_NAME   = "CAM_BACK"
H_NATIVE   = 1.568
HEIGHTS_VIZ = [1.568, 3.0, 3.5, 5.0]
SELECTED   = [0, 6, 10, 17, 25, 32, 36, 39]
PROOF_DIR  = Path('outputs/research_proof')
BG, GRID   = '#0a0a1a', '#1e2030'
COL_CAR, COL_PERSON = '#00ff88', '#ffaa44'
COL_LIST = ['#00ff88', '#ffcc00', '#ff9933', '#ff4444']

print("Loading models…")
dav2 = hf_pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf', device=0)
yolo = YOLO('yolov8n.pt')

def dav2_metric(image_bgr, pts_lidar, R_lidar, t_lidar, R_cam, t_cam, K):
    H, W = image_bgr.shape[:2]
    rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    raw  = dav2(PILImage.fromarray(rgb))['predicted_depth'].squeeze().cpu().numpy()
    raw  = cv2.resize(raw.astype(np.float32), (W, H))
    pts_ego = (R_lidar @ pts_lidar.T).T + t_lidar
    pts_c   = (R_cam.T @ (pts_ego - t_cam).T).T
    valid   = pts_c[:, 2] > 0.5; pv = pts_c[valid]
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    d_l = pv[:, 2]
    ui  = np.round(fx * pv[:, 0] / d_l + cx).astype(int)
    vi  = np.round(fy * pv[:, 1] / d_l + cy).astype(int)
    in_img = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (d_l < 80)
    d_lidar = d_l[in_img].astype(np.float32)
    d_dav2  = raw[vi[in_img], ui[in_img]]
    inv_l   = 1.0 / (d_lidar + 1e-6)
    A = np.column_stack([d_dav2, np.ones_like(d_dav2)])
    a, b = np.linalg.lstsq(A, inv_l, rcond=None)[0]
    return (1.0 / np.clip(a * raw + b, 1/200., 1/0.1)).astype(np.float32)

def warp_h(image, depth, R_cam, t_cam, K, h_new):
    H, W = image.shape[:2]
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
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
    idx = np.where(ok.ravel())[0]; order = np.argsort(-Zn.ravel()[idx]); idx = idx[order]
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

def gt_boxes(nusc, sample_token, K, R_cam, t_cam, h_target):
    sample = nusc.get('sample', sample_token)
    cam_sd = nusc.get('sample_data', sample['data'][CAM_NAME])
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
        if nm == 'vehicle' and 'pedestrian' not in box.name: cat = 'car'
        elif nm == 'human': cat = 'person'
        else: continue
        cg = box.corners().T
        ce = (R_ego.T @ (cg - t_ego).T).T
        pc = (R_cam.T @ (ce - t_new).T).T
        inf = pc[:, 2] > 0.1
        if inf.sum() < 4: continue
        Z = np.where(inf, pc[:, 2], np.nan)
        pu = np.where(inf, fx * pc[:, 0] / (Z+1e-9) + cx, np.nan)
        pv = np.where(inf, fy * pc[:, 1] / (Z+1e-9) + cy, np.nan)
        x1, y1 = np.nanmin(pu), np.nanmin(pv)
        x2, y2 = np.nanmax(pu), np.nanmax(pv)
        if x2 < 0 or y2 < 0 or x1 >= W or y1 >= H: continue
        x1, y1 = max(0., x1), max(0., y1)
        x2, y2 = min(float(W), x2), min(float(H), y2)
        if (x2-x1) < 5 or (y2-y1) < 5: continue
        out.append({'class': cat, 'box': [x1, y1, x2, y2], 'dist': float(np.nanmean(Z))})
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
    bxs = yolo_res[0].boxes; names = yolo_res[0].names
    for i in range(len(bxs)):
        cn = names[int(bxs.cls[i])]
        cat = _REMAP.get(cn)
        if cat is None: continue
        out.append({'class': cat,
                    'box': list(bxs.xyxy[i].cpu().numpy()),
                    'conf': float(bxs.conf[i])})
    return out

def match(gt, dets, iou_thr=0.25):
    matched_d = set()
    matched_g = set()
    for gi, g in enumerate(gt):
        best_iou = 0; best_di = -1
        for di, d in enumerate(dets):
            if di in matched_d: continue
            if d['class'] != g['class']: continue
            i = iou(g['box'], d['box'])
            if i > best_iou and i >= iou_thr:
                best_iou = i; best_di = di
        if best_di >= 0:
            matched_g.add(gi); matched_d.add(best_di)
    return matched_g, matched_d


print("Loading NuScenes…")
nusc = NuScenes('v1.0-mini', NUSC_ROOT, verbose=False)
scene_obj = [s for s in nusc.scene if s['name'] == SCENE_NAME][0]
tokens = []
tok = scene_obj['first_sample_token']
while tok:
    s = nusc.get('sample', tok)
    tokens.append(tok); tok = s['next']

for fig_idx, sample_idx in enumerate(SELECTED):
    print(f"\nGallery {fig_idx+1}/{len(SELECTED)}: sample #{sample_idx}")
    token = tokens[sample_idx]
    sample = nusc.get('sample', token)
    cam_sd = nusc.get('sample_data', sample['data'][CAM_NAME])
    img_path = os.path.join(NUSC_ROOT, cam_sd['filename'])
    fname = os.path.basename(cam_sd['filename'])
    img = cv2.imread(img_path)
    cal = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    K = np.array(cal['camera_intrinsic'], dtype=np.float64)
    R_cam = Quaternion(cal['rotation']).rotation_matrix
    t_cam = np.array(cal['translation'], dtype=np.float64)
    lid_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pc = LidarPointCloud.from_file(os.path.join(NUSC_ROOT, lid_sd['filename']))
    pts = pc.points[:3].T
    lcal = nusc.get('calibrated_sensor', lid_sd['calibrated_sensor_token'])
    R_lid = Quaternion(lcal['rotation']).rotation_matrix
    t_lid = np.array(lcal['translation'], dtype=np.float64)

    depth = dav2_metric(img, pts, R_lid, t_lid, R_cam, t_cam, K)

    fig, axes = plt.subplots(2, 4, figsize=(28, 14))
    fig.patch.set_facecolor(BG)

    for col, (h, c) in enumerate(zip(HEIGHTS_VIZ, COL_LIST)):
        if abs(h - H_NATIVE) < 0.05:
            img_h = img.copy()
        else:
            img_h = warp_h(img, depth, R_cam, t_cam, K, h)

        gt_h = gt_boxes(nusc, token, K, R_cam, t_cam, h)
        det_h = parse_yolo(yolo(img_h, verbose=False, conf=0.25))
        mg, md = match(gt_h, det_h, iou_thr=0.25)

        # TOP: GT
        ax = axes[0][col]
        ax.imshow(cv2.cvtColor(img_h, cv2.COLOR_BGR2RGB))
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

        # BOTTOM: YOLO
        ax = axes[1][col]
        ax.imshow(cv2.cvtColor(img_h, cv2.COLOR_BGR2RGB))
        ax.axis('off'); ax.set_facecolor(BG)
        ax.set_title('YOLOv8n Detections', color=c, fontsize=11)

        for di, d in enumerate(det_h):
            x1,y1,x2,y2 = d['box']
            col_d = '#4499ff' if di in md else '#ff8800'
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

    is_target = ' [USER-REQUESTED TARGET]' if sample_idx == 17 else ''
    fig.suptitle(f'Sample #{sample_idx}{is_target} — Detection Degradation Across 4 Heights\n{fname}',
                 fontsize=13, color='white', fontweight='bold', y=1.005)
    plt.tight_layout(pad=0.5)
    save = PROOF_DIR / f'fig{19+fig_idx}_gallery_sample{sample_idx:02d}.png'
    plt.savefig(save, dpi=120, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  → {save}")

print("\nAll galleries done.")
