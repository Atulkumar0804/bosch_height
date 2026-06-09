"""
All visualisation utilities.

Outputs:
  1. Rendered camera image with LiDAR overlay  (per height, coloured by feature / depth)
  2. Bird's Eye View (BEV) with feature-coloured point cloud
  3. Bar-chart: point count per feature per height
  4. Line-chart: detection range vs height per feature
  5. Summary comparison figure (grid of all 3 heights)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless – saves PNG files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

from src.config import (
    CAM_WIDTH, CAM_HEIGHT, CAM_INTRINSIC, CAM_R_FROM_WORLD,
    FEATURE_LABELS, FEATURE_COLORS, OUTPUT_DIR,
)
from src.projector import (
    project_points, depth_to_color, label_to_color, overlay_points_on_image
)


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic background renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_background(sensor_height: float) -> np.ndarray:
    """
    Perspective-correct sky + ground background rendered from sensor_height.
    Returns (H, W, 3) uint8 BGR image.
    """
    H, W = CAM_HEIGHT, CAM_WIDTH
    f  = CAM_INTRINSIC[1, 1]
    cy = CAM_INTRINSIC[1, 2]

    img = np.zeros((H, W, 3), dtype=np.uint8)

    # Pre-compute the world-Z for every row in the image
    v_coords = np.arange(H, dtype=float)
    # cam_y = (v - cy) / f  →  world_z_angle = cam_y (when looking forward)
    # A pixel row v corresponds to world elevation: z_world = sensor_height − (v−cy)/f * depth
    # For the background (infinite depth), the horizon is at v = cy.
    horizon_v = int(cy + f * sensor_height / 30.0)   # where z_world=0 would project at 30m

    for v in range(H):
        t = v / H
        if v < horizon_v:
            # Sky – gradient light blue
            sky_t = v / max(horizon_v, 1)
            r = int(100 + sky_t * 35)
            g = int(149 + sky_t * 31)
            b = int(237 - sky_t * 30)
        else:
            # Ground – gradient grey-brown
            gnd_t = (v - horizon_v) / max(H - horizon_v, 1)
            r = int(80  + gnd_t * 40)
            g = int(80  + gnd_t * 30)
            b = int(70  + gnd_t * 20)
        img[v, :] = (b, g, r)   # BGR

    return img


def _project_box_edges(
    center, dims, sensor_height, K=CAM_INTRINSIC, R=CAM_R_FROM_WORLD
):
    """Project 8 corners of a box onto the image. Returns (8,2) or None."""
    cx, cy, cz = center
    dx, dy, dz = dims[0]/2, dims[1]/2, dims[2]/2
    corners = np.array([
        [cx+dx, cy+dy, cz+dz], [cx+dx, cy-dy, cz+dz],
        [cx-dx, cy+dy, cz+dz], [cx-dx, cy-dy, cz+dz],
        [cx+dx, cy+dy, cz-dz], [cx+dx, cy-dy, cz-dz],
        [cx-dx, cy+dy, cz-dz], [cx-dx, cy-dy, cz-dz],
    ], dtype=float)
    pixels, depths, valid = project_points(corners, sensor_height, K, R)
    return pixels, valid


def render_scene_objects(img: np.ndarray, sensor_height: float) -> np.ndarray:
    """
    Draw simplified object shapes on the background image.
    Returns modified (H, W, 3) uint8 BGR image.
    """
    from src.synthetic_scene import UrbanScene
    from src.config import FEATURE_COLORS
    scene = UrbanScene()
    canvas = img.copy()

    BOX_EDGES = [
        (0,1),(2,3),(4,5),(6,7),   # along y
        (0,2),(1,3),(4,6),(5,7),   # along x
        (0,4),(1,5),(2,6),(3,7),   # vertical
    ]
    from src.synthetic_scene import Box, Cylinder
    for obj in scene.objects:
        if isinstance(obj, Box):
            color_rgb = FEATURE_COLORS.get(obj.label_id, (200,200,200))
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            pixels, valid = _project_box_edges(
                obj.center, obj.dims, sensor_height)
            if valid.sum() >= 2:
                for i, j in BOX_EDGES:
                    if valid[i] and valid[j]:
                        p1 = tuple(pixels[i].astype(int))
                        p2 = tuple(pixels[j].astype(int))
                        cv2.line(canvas, p1, p2, color_bgr, 1, cv2.LINE_AA)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Camera image with LiDAR overlay
# ─────────────────────────────────────────────────────────────────────────────

def camera_with_lidar(
    scan: dict,
    color_mode: str = "feature",   # "feature" or "depth"
    point_radius: int = 3,
) -> np.ndarray:
    """
    Render the scene background and overlay projected LiDAR points.
    Returns (H, W, 3) uint8 BGR image.
    """
    h = scan["sensor_height"]
    bg = render_background(h)
    bg = render_scene_objects(bg, h)

    world_pts = scan["world_pts"]    # (N, 3)
    pixels, depths, valid = project_points(world_pts, h)

    if color_mode == "feature":
        colors = label_to_color(scan["labels"])
    else:
        colors = depth_to_color(depths)

    img = overlay_points_on_image(bg, pixels, colors, valid, radius=point_radius)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Bird's Eye View
# ─────────────────────────────────────────────────────────────────────────────

def bird_eye_view(
    scan: dict,
    x_range=(0, 75),
    y_range=(-12, 12),
    ppm: int = 10,          # pixels per metre
) -> np.ndarray:
    """
    Return a BEV image coloured by feature label.
    Sensor origin = left edge, centre row.
    """
    W = int((x_range[1] - x_range[0]) * ppm)
    H = int((y_range[1] - y_range[0]) * ppm)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)   # dark background

    pts = scan["world_pts"]
    labels = scan["labels"]

    for i, pt in enumerate(pts):
        xi = int((pt[0] - x_range[0]) * ppm)
        yi = H - int((pt[1] - y_range[0]) * ppm) - 1  # flip Y
        if 0 <= xi < W and 0 <= yi < H:
            r, g, b = FEATURE_COLORS.get(int(labels[i]), (128,128,128))
            img[yi, xi] = (b, g, r)   # BGR

    # Sensor position
    sx = int((0 - x_range[0]) * ppm)
    sy = H - int((0 - y_range[0]) * ppm) - 1
    cv2.circle(img, (sx, sy), 5, (0, 255, 255), -1)
    cv2.putText(img, f"h={scan['sensor_height']:.1f}m",
                (sx+8, sy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Comparison grid
# ─────────────────────────────────────────────────────────────────────────────

def save_camera_comparison(height_scans: dict, filename: str = "camera_comparison.png"):
    """3-column figure: camera view for each height, both feature- and depth-coloured."""
    n = len(height_scans)
    fig, axes = plt.subplots(2, n, figsize=(7*n, 9))

    for col, (name, scan) in enumerate(height_scans.items()):
        for row, mode in enumerate(("feature", "depth")):
            img_bgr = camera_with_lidar(scan, color_mode=mode, point_radius=3)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            axes[row, col].imshow(img_rgb)
            axes[row, col].axis("off")
            title = (f"{name}\nh={scan['sensor_height']:.1f}m\n"
                     f"{'Feature labels' if mode=='feature' else 'Depth map'}")
            axes[row, col].set_title(title, fontsize=10)

    # Legend for feature colours
    patches = [
        mpatches.Patch(color=np.array(c)/255, label=lbl)
        for lid, lbl in FEATURE_LABELS.items()
        if lid in FEATURE_COLORS and lid != 0
        for c in [FEATURE_COLORS[lid]]
    ]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))

    plt.suptitle("LiDAR Projection on Camera – Sensor Height Comparison", fontsize=13)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def save_bev_comparison(height_scans: dict, filename: str = "bev_comparison.png"):
    """Side-by-side BEV for all heights."""
    n = len(height_scans)
    bevs = [bird_eye_view(scan) for scan in height_scans.values()]
    combined = np.hstack(bevs)
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, combined)
    print(f"  Saved: {path}")
    return path


def save_feature_bar_chart(
    comparison_df,
    filename: str = "feature_point_counts.png",
):
    """Bar chart: point counts per feature per height."""
    import pandas as pd
    features = list(comparison_df["feature"].unique())
    variants = list(comparison_df["variant"].unique())

    x = np.arange(len(features))
    width = 0.8 / len(variants)

    fig, ax = plt.subplots(figsize=(14, 6))
    palette = plt.cm.tab10(np.linspace(0, 1, len(variants)))

    for i, var in enumerate(variants):
        sub = comparison_df[comparison_df["variant"] == var].set_index("feature")
        counts = [int(sub.loc[f, "point_count"]) if f in sub.index else 0
                  for f in features]
        bars = ax.bar(x + i*width, counts, width, label=var, color=palette[i])
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(cnt), ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x + width*(len(variants)-1)/2)
    ax.set_xticklabels(features, rotation=20, ha='right')
    ax.set_ylabel("LiDAR Point Count")
    ax.set_title("Feature Detection: Point Count vs. Sensor Mounting Height")
    ax.legend()
    ax.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"  Saved: {path}")
    return path


def save_range_comparison(
    comparison_df,
    filename: str = "detection_range.png",
):
    """Line plot: mean detection range vs height for each feature."""
    features = [f for f in comparison_df["feature"].unique()
                if comparison_df[comparison_df["feature"]==f]["point_count"].max() > 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    heights_sorted = sorted(comparison_df["sensor_height_m"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, len(features)))

    for feat, color in zip(features, palette):
        sub = comparison_df[comparison_df["feature"] == feat].copy()
        sub = sub.sort_values("sensor_height_m")
        h_vals  = sub["sensor_height_m"].tolist()
        rng_vals = [v if v is not None else 0 for v in sub["mean_range_m"]]
        ax.plot(h_vals, rng_vals, marker='o', label=feat, color=color, linewidth=2)

    ax.set_xlabel("Sensor Mounting Height (m)")
    ax.set_ylabel("Mean Detection Range (m)")
    ax.set_title("Mean Detection Range per Feature vs. Sensor Height")
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    ax.grid(alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def save_blind_spot_diagram(
    comparison_df,
    filename: str = "blind_spot.png",
):
    """Illustrate blind-spot radius and vertical coverage for each height."""
    from src.config import LIDAR_V_FOV_DEG
    import matplotlib.patches as patches

    heights = sorted(comparison_df["sensor_height_m"].unique())
    colors  = ["#3498db", "#2ecc71", "#e74c3c"][:len(heights)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: blind-spot radii ───────────────────────────────────────────────
    ax = axes[0]
    max_r = 0
    for h, c in zip(heights, colors):
        bs = h / np.tan(np.radians(abs(min(LIDAR_V_FOV_DEG))))
        circle = plt.Circle((0, 0), bs, fill=False, color=c, linewidth=2,
                             label=f"h={h:.1f}m  (blind ≥ {bs:.1f}m)")
        ax.add_patch(circle)
        max_r = max(max_r, bs)

    ax.set_xlim(-max_r*1.1, max_r*1.1)
    ax.set_ylim(-max_r*1.1, max_r*1.1)
    ax.set_aspect('equal')
    ax.set_title("Ground Blind-Spot Radius (Top View)")
    ax.set_xlabel("Y (m)"); ax.set_ylabel("X forward (m)")
    ax.scatter([0],[0], c='black', s=50, zorder=5, label="Sensor")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Right: vertical coverage to traffic light at 10 m ahead ─────────────
    ax2 = axes[1]
    tl_height = 5.0
    x_dist    = 10.0
    for h, c in zip(heights, colors):
        # Elevation angle to traffic-light top
        dz     = tl_height - h
        ang    = np.degrees(np.arctan2(dz, x_dist))
        within = abs(ang) <= max(abs(v) for v in LIDAR_V_FOV_DEG)
        label = (f"h={h:.1f}m  θ={ang:+.1f}°"
                 + ("  ✓" if within else "  ✗ outside FOV"))
        ax2.barh(f"h={h:.1f}m", ang, color=c, label=label, height=0.5)

    fov_min, fov_max = LIDAR_V_FOV_DEG
    ax2.axvline(fov_min, color='grey', linestyle='--', alpha=0.6, label=f"FOV min {fov_min}°")
    ax2.axvline(fov_max, color='grey', linestyle=':',  alpha=0.6, label=f"FOV max {fov_max}°")
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.set_xlabel("Elevation angle to traffic-light head (degrees)")
    ax2.set_title(f"Traffic Light (z={tl_height}m) at {x_dist}m ahead")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"  Saved: {path}")
    return path


def save_detection_heatmap(
    comparison_df,
    filename: str = "detection_heatmap.png",
):
    """Heatmap: detected (green) / not-detected (red) per feature × height."""
    import pandas as pd
    pivot = comparison_df.pivot_table(
        index="feature", columns="variant", values="point_count", aggfunc="first"
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    data  = pivot.values.astype(float)
    im    = ax.imshow(data, aspect='auto', cmap='YlOrRd')
    plt.colorbar(im, ax=ax, label="Point Count")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=15, ha='right')
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Feature Detection Heatmap (point count)")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = int(data[i, j])
            ax.text(j, i, str(val), ha='center', va='center',
                    color='black' if data[i,j] < data.max()*0.6 else 'white',
                    fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"  Saved: {path}")
    return path
