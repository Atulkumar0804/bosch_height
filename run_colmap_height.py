"""
COLMAP-based Height Change Pipeline
=====================================
Pipeline:
  1. Export NuScenes scene (6 cams × N frames) → COLMAP format
     - cameras.txt  : per-camera PINHOLE intrinsics
     - images.txt   : world-to-camera poses (from NuScenes ego_pose + calib)
     - points3D.txt : empty (filled by triangulator)
  2. COLMAP: feature extract → sequential match → point_triangulator
  3. Color each 3D point from its source image pixel
  4. Augment with LiDAR coloured points for density
  5. Open3D: render from h=1.5m  AND  h=3.0m  (height-only shift, no tilt)
  6. Compare: car front/side (1.5m) vs car roof (3m) + traffic light perspective

Usage:
  python run_colmap_height.py --nuscenes nuscenes_data --scene scene-0103
"""

import argparse, os, subprocess, csv, shutil, time
import numpy as np
import cv2
import open3d as o3d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

TARGET_H   = 3.0
OUTPUT_DIR = "outputs/colmap_height"
DARK_BG    = "#0a0a14"
CAMS       = ["CAM_FRONT","CAM_FRONT_RIGHT","CAM_FRONT_LEFT",
              "CAM_BACK", "CAM_BACK_RIGHT", "CAM_BACK_LEFT"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Export NuScenes → COLMAP workspace
# ─────────────────────────────────────────────────────────────────────────────

def export_to_colmap(nusc, scene_name, workspace, max_frames=10,
                     cams=None):
    """
    Write COLMAP workspace:
      workspace/images/        - image files (symlinked)
      workspace/sparse/0/      - cameras.txt, images.txt, points3D.txt
    Returns list of (cam_id, image_name, cam_token, ego_pose_token) for each image.
    """
    if cams is None:
        cams = CAMS

    img_dir   = os.path.join(workspace, "images")
    sparse_dir = os.path.join(workspace, "sparse", "0")
    os.makedirs(img_dir,    exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    # Find scene
    scene = next(s for s in nusc.scene if s['name'] == scene_name)
    print(f"  Scene: {scene['name']} ({scene['nbr_samples']} samples)")

    # Collect frames
    tok = scene['first_sample_token']
    samples = []
    while tok and len(samples) < max_frames:
        samples.append(nusc.get('sample', tok))
        tok = samples[-1]['next']

    # ── cameras.txt ───────────────────────────────────────────────────────────
    # Map calibrated_sensor_token → COLMAP camera_id
    cs_to_cid = {}
    cam_lines  = []
    cid        = 1
    for cam_name in cams:
        if cam_name not in samples[0]['data']:
            continue
        sd = nusc.get('sample_data', samples[0]['data'][cam_name])
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        K  = np.array(cs['camera_intrinsic'])
        fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
        W  = sd['width']; H = sd['height']
        tok_cs = sd['calibrated_sensor_token']
        if tok_cs not in cs_to_cid:
            cs_to_cid[tok_cs] = cid
            cam_lines.append(f"{cid} PINHOLE {W} {H} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}")
            cid += 1

    with open(os.path.join(sparse_dir, "cameras.txt"), "w") as f:
        f.write("# camera_id model width height params\n")
        f.write("\n".join(cam_lines) + "\n")

    # ── images.txt ────────────────────────────────────────────────────────────
    img_records = []
    iid = 1
    for s_idx, sample in enumerate(samples):
        ep = nusc.get('ego_pose',
                      nusc.get('sample_data',
                               sample['data'][cams[0]])['ego_pose_token'])
        R_ew = Quaternion(ep['rotation']).rotation_matrix    # ego → world
        t_ew = np.array(ep['translation'])                   # ego origin in world

        for cam_name in cams:
            if cam_name not in sample['data']:
                continue
            sd  = nusc.get('sample_data', sample['data'][cam_name])
            cs  = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            R_ce = Quaternion(cs['rotation']).rotation_matrix  # cam → ego
            t_ce = np.array(cs['translation'])                  # cam origin in ego

            # Camera centre in world
            t_cw = t_ew + R_ew @ t_ce          # (3,)

            # World-to-camera rotation
            R_cw = R_ew @ R_ce                  # cam → world
            R_wc = R_cw.T                       # world → cam

            q = Quaternion(matrix=R_wc)
            qw,qx,qy,qz = q.w, q.x, q.y, q.z

            # COLMAP translation: t = -R_wc @ t_cw
            t_col = -R_wc @ t_cw
            tx,ty,tz = t_col

            # Image filename
            fname = os.path.basename(sd['filename'])
            dst   = os.path.join(img_dir, fname)
            src   = os.path.join("nuscenes_data", sd['filename'])
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)

            cam_id = cs_to_cid[sd['calibrated_sensor_token']]
            img_records.append({
                "iid": iid, "qw":qw,"qx":qx,"qy":qy,"qz":qz,
                "tx":tx,"ty":ty,"tz":tz, "cam_id":cam_id,
                "fname": fname, "cam_name": cam_name,
                "R_cw": R_cw, "t_cw": t_cw,
                "R_ce": R_ce, "t_ce": t_ce,
                "K": np.array(cs['camera_intrinsic']),
                "cam_h": float(t_ce[2]),
            })
            iid += 1

    with open(os.path.join(sparse_dir, "images.txt"), "w") as f:
        f.write("# iid qw qx qy qz tx ty tz cam_id fname\n")
        for r in img_records:
            f.write(f"{r['iid']} {r['qw']:.9f} {r['qx']:.9f} {r['qy']:.9f} "
                    f"{r['qz']:.9f} {r['tx']:.9f} {r['ty']:.9f} {r['tz']:.9f} "
                    f"{r['cam_id']} {r['fname']}\n\n")   # blank line after each image (no points)

    # Empty points file
    open(os.path.join(sparse_dir, "points3D.txt"), "w").close()

    print(f"  Exported {len(img_records)} images ({len(samples)} frames × {len(cams)} cams)")
    return img_records, sparse_dir


# ─────────────────────────────────────────────────────────────────────────────
# 2. Run COLMAP
# ─────────────────────────────────────────────────────────────────────────────

def _fix_db_camera_ids(workspace):
    """
    After feature_extractor runs, align the database camera/image IDs to match
    our cameras.txt / images.txt (NuScenes intrinsics + per-camera grouping).
    COLMAP assigns its own camera IDs; this corrects them to ours.
    """
    import sqlite3
    db     = os.path.join(workspace, "database.db")
    sparse = os.path.join(workspace, "sparse", "0")

    cams_txt = {}
    with open(os.path.join(sparse, "cameras.txt")) as f:
        for line in f:
            if line.startswith('#'): continue
            p = line.split()
            cid, W, H = int(p[0]), int(p[2]), int(p[3])
            fx, fy, cx, cy = float(p[4]), float(p[5]), float(p[6]), float(p[7])
            cams_txt[cid] = (fx, fy, cx, cy, W, H)

    fname_to_cam = {}
    with open(os.path.join(sparse, "images.txt")) as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            p = line.split()
            if len(p) >= 10:
                fname_to_cam[p[9]] = int(p[8])

    conn = sqlite3.connect(db)
    cur  = conn.cursor()

    # Replace DB cameras with our NuScenes calibration
    cur.execute('DELETE FROM cameras')
    for cid, (fx, fy, cx, cy, W, H) in cams_txt.items():
        params = np.array([fx, fy, cx, cy], dtype=np.float64).tobytes()
        cur.execute(
            'INSERT INTO cameras (camera_id,model,width,height,params,prior_focal_length) '
            'VALUES (?,1,?,?,?,1)', (cid, W, H, params))

    # Remap DB image → camera IDs by filename
    cur.execute('SELECT image_id, name FROM images ORDER BY image_id')
    db_imgs = cur.fetchall()
    new_images_txt_rows = []
    for iid, name in db_imgs:
        our_cam = fname_to_cam.get(name)
        if our_cam is not None:
            cur.execute('UPDATE images SET camera_id=? WHERE image_id=?', (our_cam, iid))

    conn.commit(); conn.close()

    # Rewrite images.txt with DB image IDs (so image_id matches database)
    fname_to_pose = {}
    with open(os.path.join(sparse, "images.txt")) as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            p = line.split()
            if len(p) >= 10:
                fname_to_pose[p[9]] = (' '.join(p[1:8]), int(p[8]))

    with open(os.path.join(sparse, "images.txt"), 'w') as f:
        f.write('# iid qw qx qy qz tx ty tz cam_id fname\n')
        for iid, name in db_imgs:
            if name not in fname_to_pose: continue
            pose_str, cam_id = fname_to_pose[name]
            f.write(f"{iid} {pose_str} {cam_id} {name}\n\n")


def run_colmap(workspace):
    """
    Known-pose sparse reconstruction:
      feature_extractor → fix_db_ids → sequential_matcher → point_triangulator
    Returns path to sparse model with points3D.txt (text format).
    """
    db      = os.path.join(workspace, "database.db")
    img_dir = os.path.join(workspace, "images")
    sparse  = os.path.join(workspace, "sparse", "0")
    tri_bin = os.path.join(workspace, "sparse", "triangulated_bin")
    tri_txt = os.path.join(workspace, "sparse", "triangulated")
    os.makedirs(tri_bin, exist_ok=True)
    os.makedirs(tri_txt, exist_ok=True)

    def run(cmd, desc):
        print(f"  → {desc}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    STDERR: {r.stderr[-500:]}")
        return r.returncode == 0

    # 1. Feature extraction (skip if database already has features)
    if not os.path.exists(db) or os.path.getsize(db) < 1024:
        ok = run([
            "colmap", "feature_extractor",
            "--database_path", db,
            "--image_path",    img_dir,
            "--ImageReader.camera_model", "PINHOLE",
            "--ImageReader.single_camera_per_folder", "0",
            "--SiftExtraction.max_image_size", "1600",
            "--SiftExtraction.max_num_features", "4096",
            "--SiftExtraction.use_gpu", "0",
        ], "Feature extraction (SIFT)")
        if not ok:
            return None
    else:
        print("  → Feature extraction (skipped — DB exists)")

    # Always align database camera/image IDs to our NuScenes calibration
    # (export_to_colmap re-writes cameras.txt/images.txt each run)
    _fix_db_camera_ids(workspace)

    # 2. Sequential matching
    ok = run([
        "colmap", "sequential_matcher",
        "--database_path", db,
        "--SiftMatching.use_gpu", "0",
        "--SequentialMatching.overlap", "5",
    ], "Sequential feature matching")

    # 3. Triangulate with known poses
    ok = run([
        "colmap", "point_triangulator",
        "--database_path",  db,
        "--image_path",     img_dir,
        "--input_path",     sparse,
        "--output_path",    tri_bin,
        "--Mapper.tri_min_angle", "1.5",
    ], "Point triangulation")

    if not ok:
        return sparse  # fallback: no points

    # 4. Convert binary → text
    run([
        "colmap", "model_converter",
        "--input_path",  tri_bin,
        "--output_path", tri_txt,
        "--output_type", "TXT",
    ], "Convert model to TXT")

    return tri_txt


# ─────────────────────────────────────────────────────────────────────────────
# 3. Color 3D points from images
# ─────────────────────────────────────────────────────────────────────────────

def load_colmap_points(model_dir):
    """
    Parse COLMAP points3D.txt → Nx3 positions, Nx3 RGB colours.
    """
    pts_path = os.path.join(model_dir, "points3D.txt")
    if not os.path.exists(pts_path):
        return np.zeros((0,3)), np.zeros((0,3))

    pts_xyz = []; pts_rgb = []
    with open(pts_path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            # point3D_id X Y Z R G B error track...
            if len(parts) < 7:
                continue
            try:
                x,y,z = float(parts[1]),float(parts[2]),float(parts[3])
                r,g,b = int(parts[4]),int(parts[5]),int(parts[6])
                pts_xyz.append([x,y,z])
                pts_rgb.append([r/255.,g/255.,b/255.])
            except (ValueError, IndexError):
                continue

    return (np.array(pts_xyz, dtype=np.float32) if pts_xyz else np.zeros((0,3)),
            np.array(pts_rgb, dtype=np.float32) if pts_rgb else np.zeros((0,3)))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Build colored LiDAR point cloud (dense backup / augmentation)
# ─────────────────────────────────────────────────────────────────────────────

def build_lidar_pc(nusc, sample_token, cam_name="CAM_FRONT"):
    """
    Build colored point cloud from LiDAR + camera for one sample.
    Returns (pts_world Nx3, rgb Nx3 in [0,1]).
    """
    sample  = nusc.get('sample', sample_token)
    cam_sd  = nusc.get('sample_data', sample['data'][cam_name])
    cs_cam  = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    ep      = nusc.get('ego_pose', cam_sd['ego_pose_token'])

    R_ew    = Quaternion(ep['rotation']).rotation_matrix
    t_ew    = np.array(ep['translation'])
    R_ce    = Quaternion(cs_cam['rotation']).rotation_matrix
    t_ce    = np.array(cs_cam['translation'])
    K       = np.array(cs_cam['camera_intrinsic'], dtype=np.float32)
    image   = cv2.imread(f"nuscenes_data/{cam_sd['filename']}")
    H, W    = image.shape[:2]
    fx,fy   = K[0,0],K[1,1]
    cx,cy   = K[0,2],K[1,2]

    lid_sd  = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    cs_lid  = nusc.get('calibrated_sensor', lid_sd['calibrated_sensor_token'])
    R_le    = Quaternion(cs_lid['rotation']).rotation_matrix
    t_le    = np.array(cs_lid['translation'])
    pts_raw = np.fromfile(f"nuscenes_data/{lid_sd['filename']}",
                          dtype=np.float32).reshape(-1,5)
    pts_l   = pts_raw[:,:3]

    # LiDAR → ego → world
    pts_ego   = (R_le @ pts_l.T).T + t_le
    pts_world = (R_ew @ pts_ego.T).T + t_ew        # world frame

    # Project to camera for colours
    pts_cam   = (R_ce.T @ (pts_ego - t_ce).T).T
    fwd       = pts_cam[:,2] > 0.3
    d         = pts_cam[fwd,2]
    u_f       = fx*pts_cam[fwd,0]/d + cx
    v_f       = fy*pts_cam[fwd,1]/d + cy
    u_i       = np.round(u_f).astype(int)
    v_i       = np.round(v_f).astype(int)
    in_img    = (u_i>=0)&(u_i<W)&(v_i>=0)&(v_i<H)

    pts_w_col = pts_world[fwd][in_img]
    bgr_col   = image[v_i[in_img], u_i[in_img]]
    rgb_col   = bgr_col[:,::-1] / 255.

    return pts_w_col.astype(np.float32), rgb_col.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Open3D: Render from a given camera pose
# ─────────────────────────────────────────────────────────────────────────────

def render_open3d(pts_world, rgb, K, R_cw, t_cw, H, W, pt_size=3.0):
    """
    Render coloured point cloud from camera with pose (R_cw, t_cw).
    Uses Open3D offscreen renderer.
    Returns (H, W, 3) uint8 BGR image.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb,0,1).astype(np.float64))

    # Open3D intrinsics
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    # Camera extrinsics for Open3D (4×4 world-to-camera)
    # R_wc = R_cw.T,  t_col = -R_wc @ t_cw
    R_wc   = R_cw.T
    t_col  = -R_wc @ t_cw
    extr   = np.eye(4)
    extr[:3,:3] = R_wc
    extr[:3, 3] = t_col

    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background([0.05,0.05,0.1,1.0])

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = pt_size
    renderer.scene.add_geometry("pcd", pcd, mat)

    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = intr
    cam.extrinsic = extr
    renderer.setup_camera(cam.intrinsic, cam.extrinsic)

    img_o3d = renderer.render_to_image()
    img_np  = np.asarray(img_o3d)          # RGB uint8
    renderer.scene.clear_geometry()
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)


def raise_camera_height(R_cw, t_cw, delta_h):
    """
    Raise camera by delta_h in world Z.
    Only Z component of camera world-position changes.
    Camera orientation (R_cw) is UNCHANGED.
    """
    t_new = t_cw.copy()
    t_new[2] += delta_h
    return R_cw, t_new


# ─────────────────────────────────────────────────────────────────────────────
# 6. Feature detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def detect_traffic_lights(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H,W = img_bgr.shape[:2]
    boxes = []
    for label, ranges in {
        "RED":   [(np.array([0,120,100]),np.array([10,255,255])),
                  (np.array([160,120,100]),np.array([180,255,255]))],
        "GREEN": [(np.array([35,60,60]),np.array([90,255,255]))],
    }.items():
        mask = np.zeros(hsv.shape[:2],np.uint8)
        for lo,hi in ranges: mask |= cv2.inRange(hsv,lo,hi)
        mask[int(H*0.65):] = 0
        mask = cv2.dilate(mask,np.ones((5,5),np.uint8),iterations=2)
        for c in cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(c) < 50: continue
            x,y,w,h = cv2.boundingRect(c)
            if w>W*0.25: continue
            boxes.append((x,y,w,h,label))
    return boxes


def draw_tl(img, boxes):
    out = img.copy()
    cols = {"RED":(0,0,220),"GREEN":(0,200,0),"AMBER":(0,165,255)}
    for x,y,w,h,lbl in boxes:
        c = cols.get(lbl,(200,200,200))
        cv2.rectangle(out,(x,y),(x+w,y+h),c,2)
        cv2.putText(out,lbl,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    return out


def detect_road_markings(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H,W = img_bgr.shape[:2]
    white  = cv2.inRange(hsv,np.array([0,0,160]),np.array([180,55,255]))
    yellow = cv2.inRange(hsv,np.array([15,70,70]),np.array([40,255,255]))
    combo  = cv2.bitwise_or(white,yellow)
    combo[:H//2] = 0
    lines  = cv2.HoughLinesP(cv2.Canny(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY),50,150),
                              1,np.pi/180,60,minLineLength=40,maxLineGap=30)
    return combo, lines if lines is not None else []


# ─────────────────────────────────────────────────────────────────────────────
# 7. Figures
# ─────────────────────────────────────────────────────────────────────────────

def fig_main(img_15, img_3m, cam_h, n_pts, out_dir):
    """3-panel: original 1.5m | rendered 3m | diff heatmap."""
    diff  = cv2.absdiff(img_15, img_3m)
    diff_g = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    heat  = cv2.applyColorMap(np.clip(diff_g.astype(np.float32)*5,0,255).astype(np.uint8),
                               cv2.COLORMAP_JET)

    panels = [
        (img_15, f"ORIGINAL  h = {cam_h:.2f} m\nPoint cloud rendered from {cam_h:.2f}m"),
        (img_3m, f"RAISED  h = {TARGET_H:.1f} m\nSame point cloud, camera Z only"),
        (heat,   f"CHANGE HEATMAP (×5)\nblue=no shift  red=max shift"),
    ]
    fig, axes = plt.subplots(1,3,figsize=(24,7),facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    for ax,(img,title),bc in zip(axes,panels,["#44dd44","#ff5533","#ffaa00"]):
        ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        ax.set_title(title,color="white",fontsize=11,pad=6,
                     fontweight="bold",linespacing=1.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(bc); sp.set_linewidth(4)

    fig.suptitle(
        f"COLMAP + LiDAR Point Cloud ({n_pts:,} pts)  ·  "
        f"Height {cam_h:.2f}m → {TARGET_H:.1f}m  (Δh={TARGET_H-cam_h:+.2f}m)\n"
        "Camera Z raised only — no tilt, no pitch change",
        color="white",fontsize=13,y=1.02,fontweight="bold")
    plt.tight_layout(pad=0.3)
    p = os.path.join(out_dir,"col_01_main.png")
    plt.savefig(p,dpi=130,bbox_inches="tight",facecolor=DARK_BG); plt.close()
    print(f"  Saved: {p}")


def fig_car_view(img_15, img_3m, nusc, sample_token, img_rec, out_dir):
    """
    Zoom into detected vehicles to show roof vs front-face change.
    """
    from pyquaternion import Quaternion as Q

    sample = nusc.get('sample', sample_token)
    ep     = nusc.get('ego_pose',
                      nusc.get('sample_data', sample['data']['CAM_FRONT'])['ego_pose_token'])
    R_ew   = Q(ep['rotation']).rotation_matrix
    t_ew   = np.array(ep['translation'])

    # Use camera pose from img_rec (first CAM_FRONT record)
    rec = next((r for r in img_rec if r['cam_name']=='CAM_FRONT'), img_rec[0])
    K   = rec['K']
    R_cw, t_cw = rec['R_cw'], rec['t_cw']
    H, W = img_15.shape[:2]
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]

    # Get NuScenes 3D boxes for vehicles
    cars = []
    for ann_tok in sample['anns']:
        ann = nusc.get('sample_annotation', ann_tok)
        if 'vehicle' not in ann['category_name']:
            continue
        center_g = np.array(ann['translation'])
        # Ego frame centre
        center_e = R_ew.T @ (center_g - t_ew)
        dist = float(np.linalg.norm(center_e[:2]))

        # Project 8 corners to image at h=1.5m
        from nuscenes.utils.data_classes import Box
        box = Box(ann['translation'], ann['size'], Q(ann['rotation']))
        corners_g = box.corners().T          # (8,3)
        corners_e = (R_ew.T @ (corners_g - t_ew).T).T
        pts_cam   = (R_cw.T @ (corners_e.T)).T + (R_cw.T @ (-t_cw[:,None])).T

        fwd = pts_cam[:,2] > 0.1
        if fwd.sum() < 4: continue
        d   = pts_cam[fwd,2]
        u   = fx*pts_cam[fwd,0]/d + cx
        v   = fy*pts_cam[fwd,1]/d + cy
        u1,v1,u2,v2 = int(u.min()),int(v.min()),int(u.max()),int(v.max())
        if u2<=u1 or v2<=v1: continue
        u1=max(0,u1); v1=max(0,v1); u2=min(W,u2); v2=min(H,v2)
        if u2-u1<20 or v2-v1<20: continue
        cars.append((dist, u1,v1,u2,v2, ann['size']))

    cars.sort()
    cars = cars[:6]
    if not cars:
        print("  No close cars found for zoom figure")
        return

    n   = len(cars)
    fig = plt.figure(figsize=(18, n*3.5+1), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(n, 3, figure=fig, hspace=0.45, wspace=0.08)

    for row,(dist,u1,v1,u2,v2,size) in enumerate(cars):
        pad  = max(15,int(max(u2-u1,v2-v1)*0.2))
        r1=max(0,v1-pad); r2=min(H,v2+pad)
        c1=max(0,u1-pad); c2=min(W,u2+pad)

        crop_15 = img_15[r1:r2,c1:c2]
        crop_3m = img_3m[r1:r2,c1:c2]
        diff_c  = cv2.absdiff(crop_15,crop_3m)
        diff_g  = cv2.cvtColor(diff_c,cv2.COLOR_BGR2GRAY)
        heat_c  = cv2.applyColorMap(
            np.clip(diff_g.astype(np.float32)*8,0,255).astype(np.uint8),
            cv2.COLORMAP_INFERNO)

        l,w,h_sz = size   # NuScenes: length,width,height
        roof_h   = h_sz   # approximate roof height above ground ≈ car height
        angle_15 = np.degrees(np.arctan2(1.568-roof_h/2, max(dist,1)))
        angle_3m = np.degrees(np.arctan2(3.0  -roof_h/2, max(dist,1)))

        ax0=fig.add_subplot(gs[row,0])
        ax1=fig.add_subplot(gs[row,1])
        ax2=fig.add_subplot(gs[row,2])

        ax0.imshow(cv2.cvtColor(crop_15,cv2.COLOR_BGR2RGB))
        ax0.set_title(f"Vehicle @{dist:.1f}m  ·  h=1.57m\n"
                      f"View angle to roof: {angle_15:+.1f}°  "
                      f"{'(front/side visible)' if angle_15>=-5 else '(top-down)'}",
                      color="#88ff88",fontsize=9,pad=3)

        ax1.imshow(cv2.cvtColor(crop_3m,cv2.COLOR_BGR2RGB))
        ax1.set_title(f"Vehicle @{dist:.1f}m  ·  h=3.0m\n"
                      f"View angle to roof: {angle_3m:+.1f}°  "
                      f"{'(roof visible!)' if angle_3m>5 else '(side still visible)'}",
                      color="#ffaa44",fontsize=9,pad=3)

        ax2.imshow(cv2.cvtColor(heat_c,cv2.COLOR_BGR2RGB))
        ax2.set_title(f"Diff ×8 — Car size: {l:.1f}×{w:.1f}×{h_sz:.1f}m\n"
                      f"Angle change: {angle_3m-angle_15:+.1f}°",
                      color="#ff6666",fontsize=9,pad=3)

        for ax,bc in zip((ax0,ax1,ax2),["#44dd44","#ffaa44","#ff4444"]):
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor(bc); sp.set_linewidth(2)

    fig.suptitle(
        "Car View Change: 1.57m → 3.0m\n"
        "At 1.57m: see car FRONT/SIDE  ·  At 3.0m: camera above roof → ROOF visible\n"
        "AI detectors trained at 1.5m FAIL at 3m because they never saw car tops",
        color="white",fontsize=12,y=1.01,fontweight="bold")
    p = os.path.join(out_dir,"col_02_car_view.png")
    plt.savefig(p,dpi=130,bbox_inches="tight",facecolor=DARK_BG); plt.close()
    print(f"  Saved: {p}")


def fig_traffic_light(img_15, img_3m, cam_h, out_dir):
    """Traffic light detection + zoom at both heights."""
    tl_15 = detect_traffic_lights(img_15)
    tl_3m = detect_traffic_lights(img_3m)
    ann_15 = draw_tl(img_15, tl_15)
    ann_3m = draw_tl(img_3m, tl_3m)

    fig, axes = plt.subplots(1,2, figsize=(18,7), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    axes[0].imshow(cv2.cvtColor(ann_15,cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Traffic Lights  h={cam_h:.2f}m  ·  {len(tl_15)} detected",
                      color="#88ff88",fontsize=11,pad=6,fontweight="bold")
    axes[1].imshow(cv2.cvtColor(ann_3m,cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Traffic Lights  h={TARGET_H:.1f}m  ·  {len(tl_3m)} detected\n"
                      "Signals appear lower in frame — viewing angle to signal head changes",
                      color="#ffaa44",fontsize=11,pad=6,fontweight="bold")
    for ax,bc in zip(axes,["#44dd44","#ff6633"]):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(bc); sp.set_linewidth(3)

    fig.suptitle(
        f"Traffic Light Perspective Change: {cam_h:.2f}m → {TARGET_H:.1f}m\n"
        "Higher mount → look UP less steeply at signal heads → different housing angle",
        color="white",fontsize=13,y=1.02,fontweight="bold")
    plt.tight_layout(pad=0.3)
    p = os.path.join(out_dir,"col_03_traffic_lights.png")
    plt.savefig(p,dpi=130,bbox_inches="tight",facecolor=DARK_BG); plt.close()
    print(f"  Saved: {p}")


def fig_road_markings(img_15, img_3m, cam_h, out_dir):
    """Lane markings at both heights."""
    mk_15, ln_15 = detect_road_markings(img_15)
    mk_3m, ln_3m = detect_road_markings(img_3m)

    def overlay(img, mask, lines):
        out = img.copy()
        ch  = np.zeros_like(out); ch[mask>0] = (0,255,180)
        out = cv2.addWeighted(out,0.8,ch,0.4,0)
        for l in lines:
            x1,y1,x2,y2 = l[0]
            cv2.line(out,(x1,y1),(x2,y2),(0,220,255),2)
        return out

    fig, axes = plt.subplots(1,2,figsize=(18,7),facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    axes[0].imshow(cv2.cvtColor(overlay(img_15,mk_15,ln_15),cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Road Markings  h={cam_h:.2f}m\n"
                      f"Marking px={int((mk_15>0).sum()):,}  Lines={len(ln_15)}",
                      color="#88ff88",fontsize=11,pad=6,fontweight="bold")
    axes[1].imshow(cv2.cvtColor(overlay(img_3m,mk_3m,ln_3m),cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Road Markings  h={TARGET_H:.1f}m\n"
                      f"Marking px={int((mk_3m>0).sum()):,}  Lines={len(ln_3m)}",
                      color="#ffaa44",fontsize=11,pad=6,fontweight="bold")
    for ax,bc in zip(axes,["#44dd44","#ff6633"]):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(bc); sp.set_linewidth(3)

    mk_d = (int((mk_3m>0).sum()) - int((mk_15>0).sum())) / max(int((mk_15>0).sum()),1) * 100
    fig.suptitle(
        f"Road Marking Detection: {cam_h:.2f}m → {TARGET_H:.1f}m  ·  "
        f"Marking pixels {mk_d:+.1f}%\n"
        "At 3m road surface is seen from steeper angle → markings appear more foreshortened",
        color="white",fontsize=13,y=1.02,fontweight="bold")
    plt.tight_layout(pad=0.3)
    p = os.path.join(out_dir,"col_04_road_markings.png")
    plt.savefig(p,dpi=130,bbox_inches="tight",facecolor=DARK_BG); plt.close()
    print(f"  Saved: {p}")


def fig_point_cloud(pts_w, rgb, out_dir):
    """Top-down view of the full colored point cloud."""
    fig, ax = plt.subplots(1,1,figsize=(14,10),facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    # Top-down: X=right, Y=up in plot = world X-Y plane
    ax.scatter(pts_w[:,0], pts_w[:,1],
               c=np.clip(rgb,0,1), s=0.3, alpha=0.6)
    ax.set_facecolor(DARK_BG)
    ax.set_xlabel("World X (m)", color="white")
    ax.set_ylabel("World Y (m)", color="white")
    ax.tick_params(colors="white")
    ax.set_aspect('equal')
    ax.set_title("COLMAP + LiDAR Point Cloud — Top-Down View",
                 color="white",fontsize=13,pad=6,fontweight="bold")
    plt.tight_layout()
    p = os.path.join(out_dir,"col_00_pointcloud_topdown.png")
    plt.savefig(p,dpi=130,bbox_inches="tight",facecolor=DARK_BG); plt.close()
    print(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuscenes",   default="nuscenes_data")
    ap.add_argument("--version",    default="v1.0-mini")
    ap.add_argument("--scene",      default="scene-0103")
    ap.add_argument("--sample",     default="fdc39b23ab4242eda6ec5e1e6574fe33")
    ap.add_argument("--cam",        default="CAM_FRONT")
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--skip-colmap",action="store_true",
                    help="Skip COLMAP (use LiDAR-only point cloud)")
    args = ap.parse_args()

    workspace = os.path.join(args.output_dir, "colmap_ws")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(workspace,       exist_ok=True)

    print("[1/6] Loading NuScenes …")
    nusc = NuScenes(version=args.version, dataroot=args.nuscenes, verbose=False)

    # ── Export to COLMAP format ──────────────────────────────────────────────
    print("[2/6] Exporting to COLMAP format …")
    img_records, sparse_dir = export_to_colmap(
        nusc, args.scene, workspace,
        max_frames=args.max_frames, cams=CAMS)

    # Pick the reference camera record (cam matching args.cam, first frame)
    ref_rec = next((r for r in img_records if r['cam_name']==args.cam), img_records[0])
    cam_h   = ref_rec['cam_h']
    K       = ref_rec['K']
    R_cw    = ref_rec['R_cw']
    t_cw    = ref_rec['t_cw']

    # Get image size
    sample  = nusc.get('sample', args.sample)
    cam_sd  = nusc.get('sample_data', sample['data'][args.cam])
    img_ref = cv2.imread(f"nuscenes_data/{cam_sd['filename']}")
    H, W    = img_ref.shape[:2]
    print(f"  Camera: {args.cam}  h={cam_h:.3f}m  →  {TARGET_H:.1f}m")

    # ── Run COLMAP ─────────────────────────────────────────────────────────
    colmap_pts = np.zeros((0,3)); colmap_rgb = np.zeros((0,3))
    if not args.skip_colmap:
        print("[3/6] Running COLMAP (feature extract → match → triangulate) …")
        tri_dir = run_colmap(workspace)
        if tri_dir:
            colmap_pts, colmap_rgb = load_colmap_points(tri_dir)
            print(f"  COLMAP sparse points: {len(colmap_pts):,}")

    # ── LiDAR coloured point cloud ──────────────────────────────────────────
    print("[4/6] Building LiDAR colored point cloud …")
    all_pts = [colmap_pts] if len(colmap_pts) > 0 else []
    all_rgb = [colmap_rgb] if len(colmap_rgb) > 0 else []

    # Accumulate LiDAR over all frames in scene for density
    tok = nusc.get('scene', next(s['token'] for s in nusc.scene
                                  if s['name']==args.scene))['first_sample_token']
    frame_count = 0
    while tok and frame_count < args.max_frames:
        s = nusc.get('sample', tok)
        for cn in CAMS:
            if cn in s['data']:
                pts_w, rgb_l = build_lidar_pc(nusc, s['token'], cn)
                all_pts.append(pts_w); all_rgb.append(rgb_l)
        tok = s['next']
        frame_count += 1

    pts_world = np.vstack(all_pts).astype(np.float32)
    rgb_world = np.vstack(all_rgb).astype(np.float32)
    print(f"  Total points: {len(pts_world):,}  "
          f"(COLMAP={len(colmap_pts):,}  LiDAR={len(pts_world)-len(colmap_pts):,})")

    # ── Render from 1.5m and 3m ────────────────────────────────────────────
    print("[5/6] Rendering point cloud from 1.5m and 3.0m …")
    img_15 = render_open3d(pts_world, rgb_world, K, R_cw, t_cw, H, W, pt_size=2.5)
    R_cw_3m, t_cw_3m = raise_camera_height(R_cw, t_cw, TARGET_H - cam_h)
    img_3m = render_open3d(pts_world, rgb_world, K, R_cw_3m, t_cw_3m, H, W, pt_size=2.5)

    cv2.imwrite(os.path.join(args.output_dir,"col_render_15m.png"), img_15)
    cv2.imwrite(os.path.join(args.output_dir,"col_render_3m.png"),  img_3m)
    print(f"  Renders saved.")

    # ── Figures ────────────────────────────────────────────────────────────
    print("[6/6] Generating comparison figures …")
    fig_point_cloud(pts_world, rgb_world, args.output_dir)
    fig_main(img_15, img_3m, cam_h, len(pts_world), args.output_dir)
    fig_car_view(img_15, img_3m, nusc, args.sample, img_records, args.output_dir)
    fig_traffic_light(img_15, img_3m, cam_h, args.output_dir)
    fig_road_markings(img_15, img_3m, cam_h, args.output_dir)

    print(f"\nAll outputs → {os.path.abspath(args.output_dir)}/")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(".png"): print(f"  {f}")


if __name__ == "__main__":
    main()
