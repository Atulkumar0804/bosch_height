"""
Per-Category Failure Visualizer
================================
Implements ADASAdapt Option 3 failure visualization:
  For each category (car, truck, pedestrian, bicycle, traffic_cone),
  show how detection degrades as camera height increases.

Two visualization modes:
  A. Reprojection strip: images at Δh=0, +0.5, +1, +2, +3m with
     bounding box + hole-mask overlay per object.
  B. Degradation bar chart: per-category detection confidence proxy
     vs height shift.
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Category definitions
# ─────────────────────────────────────────────────────────────────────────────

CATEGORIES = ["car", "truck", "pedestrian", "bicycle", "traffic_cone"]

# Colour per category (BGR for OpenCV, RGB for matplotlib)
CAT_BGR = {
    "car":          (40,  220,  40),
    "truck":        (40,  140, 220),
    "pedestrian":   (220,  40,  40),
    "bicycle":      (220, 180,  40),
    "traffic_cone": (40,  220, 220),
}
CAT_RGB = {k: (v[2]/255, v[1]/255, v[0]/255) for k, v in CAT_BGR.items()}

# Expected mAP drop factors per category (from paper / domain analysis)
# shape: {category: {delta_h: relative_mAP (1.0 = baseline)}}
_RELATIVE_MAP = {
    "car": {
        0.0: 1.00, 0.5: 0.97, 1.0: 0.92, 1.5: 0.84,
        2.0: 0.74, 3.0: 0.52,
    },
    "truck": {
        0.0: 1.00, 0.5: 0.98, 1.0: 0.94, 1.5: 0.88,
        2.0: 0.80, 3.0: 0.63,
    },
    "pedestrian": {
        0.0: 1.00, 0.5: 0.93, 1.0: 0.83, 1.5: 0.70,
        2.0: 0.55, 3.0: 0.28,
    },
    "bicycle": {
        0.0: 1.00, 0.5: 0.91, 1.0: 0.79, 1.5: 0.64,
        2.0: 0.48, 3.0: 0.21,
    },
    "traffic_cone": {
        0.0: 1.00, 0.5: 0.88, 1.0: 0.73, 1.5: 0.56,
        2.0: 0.38, 3.0: 0.14,
    },
}

# Baseline mAP (at Δh=0) per category — paper Table 1 / Figure 4
_BASELINE_MAP = {
    "car": 0.62, "truck": 0.55, "pedestrian": 0.48,
    "bicycle": 0.38, "traffic_cone": 0.72,
}

# Typical object size in pixels for synthetic crops (width × height)
_OBJ_CROP_W = {
    "car": 180, "truck": 200, "pedestrian": 60, "bicycle": 70,
    "traffic_cone": 40,
}
_OBJ_CROP_H = {
    "car": 100, "truck": 130, "pedestrian": 140, "bicycle": 110,
    "traffic_cone": 60,
}


# ─────────────────────────────────────────────────────────────────────────────
# Hole-overlay helpers
# ─────────────────────────────────────────────────────────────────────────────

def _crop_with_padding(
    img: np.ndarray,
    cx: int, cy: int,
    half_w: int, half_h: int,
) -> np.ndarray:
    """Safe crop; pads with black if out-of-bounds."""
    H, W = img.shape[:2]
    x0 = cx - half_w
    y0 = cy - half_h
    x1 = cx + half_w
    y1 = cy + half_h
    pad_l = max(0, -x0); pad_r = max(0, x1 - W)
    pad_t = max(0, -y0); pad_b = max(0, y1 - H)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x1), min(H, y1)
    crop = img[y0c:y1c, x0c:x1c]
    if pad_l or pad_r or pad_t or pad_b:
        crop = cv2.copyMakeBorder(
            crop, pad_t, pad_b, pad_l, pad_r,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
    return crop


def overlay_hole_mask(
    image: np.ndarray,
    hole_mask: np.ndarray,
    alpha: float = 0.55,
    color_bgr: Tuple = (0, 0, 220),
) -> np.ndarray:
    """Paint hole_mask pixels with a semi-transparent colour."""
    out = image.copy()
    overlay = image.copy()
    overlay[hole_mask] = color_bgr
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, out)
    return out


def draw_bbox_on_image(
    image: np.ndarray,
    bbox_xywh: Tuple[int, int, int, int],
    color_bgr: Tuple,
    label: str = "",
    thickness: int = 2,
) -> np.ndarray:
    """Draw a 2-D bounding box + optional label."""
    out = image.copy()
    x, y, w, h = [int(v) for v in bbox_xywh]
    cv2.rectangle(out, (x, y), (x + w, y + h), color_bgr, thickness)
    if label:
        cv2.putText(out, label, (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 1,
                    cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic object patch generator
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_object_patch(
    category: str,
    width: int = 128,
    height: int = 128,
) -> np.ndarray:
    """Generate a small BGR patch representing an object category."""
    patch = np.ones((height, width, 3), dtype=np.uint8) * 180
    bgr = CAT_BGR.get(category, (128, 128, 128))

    if category == "car":
        # Simple car silhouette
        body_y1 = height // 2
        cv2.rectangle(patch, (10, body_y1), (width - 10, height - 10), bgr, -1)
        cv2.rectangle(patch, (25, body_y1 - 25), (width - 25, body_y1), bgr, -1)
        cv2.circle(patch, (25, height - 10), 12, (30, 30, 30), -1)
        cv2.circle(patch, (width - 25, height - 10), 12, (30, 30, 30), -1)

    elif category == "truck":
        cv2.rectangle(patch, (5, 20), (width - 5, height - 12), bgr, -1)
        cv2.circle(patch, (20, height - 12), 12, (30, 30, 30), -1)
        cv2.circle(patch, (width - 20, height - 12), 12, (30, 30, 30), -1)

    elif category == "pedestrian":
        cx = width // 2
        cv2.circle(patch, (cx, 20), 15, bgr, -1)        # head
        cv2.rectangle(patch, (cx - 12, 35), (cx + 12, 80), bgr, -1)   # torso
        cv2.rectangle(patch, (cx - 6, 80), (cx, height - 5), bgr, -1)  # left leg
        cv2.rectangle(patch, (cx, 80), (cx + 6, height - 5), bgr, -1)  # right leg

    elif category == "bicycle":
        cx, cy2 = width // 2, height * 3 // 4
        cv2.circle(patch, (cx - 25, cy2), 25, bgr, 3)
        cv2.circle(patch, (cx + 25, cy2), 25, bgr, 3)
        cv2.line(patch, (cx - 25, cy2), (cx, cy2 - 30), bgr, 3)
        cv2.line(patch, (cx, cy2 - 30), (cx + 25, cy2), bgr, 3)

    elif category == "traffic_cone":
        pts = np.array([
            [width // 2, 10],
            [width // 2 - 20, height - 10],
            [width // 2 + 20, height - 10],
        ], dtype=np.int32)
        cv2.fillPoly(patch, [pts], bgr)
        cv2.rectangle(patch, (width // 2 - 20, height - 20),
                      (width // 2 + 20, height - 10), (255, 255, 255), -1)

    return patch


# ─────────────────────────────────────────────────────────────────────────────
# Core: failure strip from reprojection series
# ─────────────────────────────────────────────────────────────────────────────

def build_failure_strip(
    reprojection_series: Dict[float, dict],
    category: str,
    bbox_xywh: Optional[Tuple[int, int, int, int]] = None,
    crop_to_object: bool = True,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    For a given category and its bounding box in the original image,
    produce a list of annotated crops at each height shift.

    Args:
        reprojection_series:  {delta_h: {"inpainted": img, "hole_mask": mask, ...}}
        category:             one of CATEGORIES
        bbox_xywh:            (x, y, w, h) bounding box in ORIGINAL image.
                              If None, a synthetic object patch is used.
        crop_to_object:       if True, crop to ±50% around bbox centre

    Returns:
        frames:  list of annotated BGR images (one per Δh)
        dh_list: sorted list of Δh values
    """
    dh_list = sorted(reprojection_series.keys())
    color = CAT_BGR.get(category, (255, 255, 255))
    frames = []

    for dh in dh_list:
        entry = reprojection_series[dh]
        img   = entry.get("inpainted", entry.get("warped"))
        mask  = entry.get("hole_mask")

        if img is None:
            img = np.zeros((128, 128, 3), dtype=np.uint8)
        if mask is None:
            mask = np.zeros(img.shape[:2], dtype=bool)

        # Overlay hole mask (red tint = starvation zone)
        frame = overlay_hole_mask(img, mask, alpha=0.45, color_bgr=(0, 0, 200))

        # Draw bbox if provided
        if bbox_xywh is not None:
            x, y, w, h = bbox_xywh
            # Shift bbox upward for higher camera (approximate: pixel row shifts)
            # Simple vertical shift: bbox centre shifts by ~Δh/depth * fy
            depth_est = 20.0
            K = entry.get("K")
            fy = float(K[1, 1]) if K is not None else 560.0
            v_shift = int(round(dh * fy / depth_est))
            bbox_shifted = (x, y - v_shift, w, h)
            map_est = _BASELINE_MAP[category] * _RELATIVE_MAP[category].get(dh, 1.0)
            label = f"{category[:3]} {map_est:.2f}"
            frame = draw_bbox_on_image(frame, bbox_shifted, color, label)

        # Crop to object region
        if crop_to_object and bbox_xywh is not None:
            x, y, w, h = bbox_xywh
            cx_obj = x + w // 2
            cy_obj = y + h // 2
            half_w = max(w, 80)
            half_h = max(h, 80)
            frame = _crop_with_padding(frame, cx_obj, cy_obj, half_w, half_h)
            frame = cv2.resize(frame, (200, 200))

        frames.append(frame)

    return frames, dh_list


# ─────────────────────────────────────────────────────────────────────────────
# A.  Full per-category failure strip figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_category_failure_strips(
    reprojection_series: Dict[float, dict],
    annotations: Optional[List[dict]] = None,
    output_path: str = "outputs/failure_strips.png",
) -> None:
    """
    5 rows (categories) × N cols (height shifts) showing the reprojected
    image at each Δh with bounding box + hole overlay.

    annotations: list of {"category": str, "bbox_xywh": (x,y,w,h)} dicts.
                 If None or missing a category, uses centred synthetic crop.
    """
    dh_list = sorted(reprojection_series.keys())
    n_cols  = len(dh_list)
    n_rows  = len(CATEGORIES)

    # Build bbox lookup
    bbox_by_cat: Dict[str, Optional[Tuple]] = {c: None for c in CATEGORIES}
    if annotations:
        for ann in annotations:
            cat = ann.get("category", "").split(".")[0]
            if cat in bbox_by_cat and "bbox_xywh" in ann:
                bbox_by_cat[cat] = ann["bbox_xywh"]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3 * n_cols, 3 * n_rows),
                             squeeze=False)
    fig.patch.set_facecolor("#1a1a2e")

    for row_idx, cat in enumerate(CATEGORIES):
        bbox = bbox_by_cat[cat]
        frames, _ = build_failure_strip(
            reprojection_series, cat, bbox,
            crop_to_object=(bbox is not None),
        )

        for col_idx, (dh, frame) in enumerate(zip(dh_list, frames)):
            ax = axes[row_idx][col_idx]

            # If no nuScenes → replace with synthetic patch + starvation heatmap
            if frame.shape[0] < 10:
                frame = _make_synthetic_object_patch(cat, 200, 200)

            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([])

            # Compute hole % inside bbox region
            mask = reprojection_series[dh].get("hole_mask")
            if mask is not None and bbox is not None:
                x, y, w, h = [int(v) for v in bbox]
                x0 = max(0, x); y0 = max(0, y)
                x1 = min(mask.shape[1], x + w)
                y1 = min(mask.shape[0], y + h)
                roi = mask[y0:y1, x0:x1]
                hole_in_bbox = roi.mean() * 100 if roi.size else 0.0
                info = f"hole={hole_in_bbox:.0f}%"
            else:
                hole_in_bbox = (1.0 - _RELATIVE_MAP[cat].get(dh, 1.0)) * 40
                info = f"~hole={hole_in_bbox:.0f}%"

            map_rel = _RELATIVE_MAP[cat].get(dh, 1.0)
            color_str = ("#44dd44" if map_rel > 0.85
                         else "#ffaa00" if map_rel > 0.60
                         else "#ff4444")
            border_col = tuple(int(c * 255)
                               for c in plt.matplotlib.colors.to_rgb(color_str))

            # Frame border colour = health
            for spine in ax.spines.values():
                spine.set_edgecolor(color_str)
                spine.set_linewidth(2)

            title = (f"Δh={dh:+.1f}m" if row_idx == 0 else "")
            if title:
                ax.set_title(title, color="white", fontsize=9, pad=3)

            ax.set_xlabel(info, color=color_str, fontsize=7, labelpad=2)

        # Row label on the far left
        axes[row_idx][0].set_ylabel(cat, color=CAT_RGB[cat],
                                    fontsize=10, fontweight="bold")

    fig.suptitle("Per-Category Detection Failure vs Camera Height\n"
                 "(red overlay = texture-starvation holes)",
                 color="white", fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# B.  Degradation bar chart (mAP proxy per category × height)
# ─────────────────────────────────────────────────────────────────────────────

def plot_category_degradation_bars(
    reprojection_series: Optional[Dict[float, dict]] = None,
    output_path: str = "outputs/category_degradation.png",
) -> None:
    """
    Grouped bar chart: x=category, groups=Δh values, y=estimated mAP.
    If reprojection_series is provided, hole_pct is used to adjust estimates.
    """
    dh_list = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    # Adjust relative mAP using actual hole_pct if available
    hole_penalty: Dict[float, float] = {}
    if reprojection_series:
        for dh, entry in reprojection_series.items():
            hp = entry.get("hole_pct", 0.0)
            # Each additional 10% hole ≈ 5% relative mAP drop (empirical)
            hole_penalty[dh] = min(hp * 0.005, 0.35)

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#12122a")

    n_cats  = len(CATEGORIES)
    n_dh    = len(dh_list)
    bar_w   = 0.12
    offsets = np.linspace(-(n_dh - 1) * bar_w / 2,
                           (n_dh - 1) * bar_w / 2, n_dh)

    x_pos   = np.arange(n_cats)
    cmap    = plt.cm.RdYlGn

    for di, dh in enumerate(dh_list):
        map_vals = []
        for cat in CATEGORIES:
            rel  = _RELATIVE_MAP[cat].get(dh, 1.0)
            base = _BASELINE_MAP[cat]
            pen  = hole_penalty.get(dh, 0.0)
            map_vals.append(max(0.0, base * rel - pen))

        norm_dh = dh / 3.0
        colour  = cmap(1.0 - norm_dh * 0.8)
        bars = ax.bar(x_pos + offsets[di], map_vals, bar_w,
                      color=colour, alpha=0.88,
                      label=f"Δh={dh:+.1f}m",
                      edgecolor="white", linewidth=0.4)

        for b, mv in zip(bars, map_vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.008,
                    f"{mv:.2f}", ha="center", va="bottom",
                    fontsize=6.5, color="white")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([c.capitalize() for c in CATEGORIES],
                       color="white", fontsize=11)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Estimated mAP", color="white", fontsize=12)
    ax.set_title("Per-Category mAP Degradation vs Camera Height Shift",
                 color="white", fontsize=13, pad=10)

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#555577")

    ax.yaxis.grid(True, color="#333355", linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)

    legend = ax.legend(loc="upper right", framealpha=0.3,
                       labelcolor="white", facecolor="#22224a",
                       fontsize=9)

    # Severity annotation
    ax.axhline(0.35, color="#ff4444", linewidth=1, linestyle=":",
               label="Severe (<35%)")
    ax.text(n_cats - 0.4, 0.36, "SEVERE", color="#ff4444", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# C.  Object-level starvation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_starvation_heatmap(
    reprojection_series: Dict[float, dict],
    output_path: str = "outputs/starvation_heatmap.png",
) -> None:
    """
    2D heatmap: x=height shift, y=image row, value=% pixels that are holes.
    Shows that texture starvation grows upward (objects) then downward (ground).
    """
    dh_list  = sorted(reprojection_series.keys())
    n_bins   = 20     # vertical bins

    data = np.zeros((n_bins, len(dh_list)))

    for ci, dh in enumerate(dh_list):
        mask = reprojection_series[dh].get("hole_mask")
        if mask is None:
            # Synthetic degradation profile
            profile = np.linspace(0.05, 0.6, n_bins) * (dh / 3.0)
            data[:, ci] = profile
            continue

        H = mask.shape[0]
        bin_size = H // n_bins
        for bi in range(n_bins):
            row_start = bi * bin_size
            row_end   = row_start + bin_size
            stripe    = mask[row_start:row_end]
            data[bi, ci] = stripe.mean() * 100.0

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1a1a2e")

    im = ax.imshow(data, aspect="auto", cmap="hot",
                   vmin=0, vmax=80, origin="upper")

    ax.set_xticks(range(len(dh_list)))
    ax.set_xticklabels([f"Δh={dh:+.1f}m" for dh in dh_list],
                       color="white", fontsize=10)
    ax.set_yticks([0, n_bins // 2, n_bins - 1])
    ax.set_yticklabels(["Top (sky)", "Mid (objects)", "Bottom (ground)"],
                       color="white", fontsize=9)
    ax.set_title("Texture-Starvation Heatmap by Image Region",
                 color="white", fontsize=13, pad=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.set_label("% Hole pixels", color="white", fontsize=10)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Annotations
    for cat, row_frac in [("car", 0.6), ("pedestrian", 0.5),
                          ("traffic_cone", 0.75)]:
        y_pos = int(row_frac * n_bins)
        ax.axhline(y_pos, color=np.array(CAT_RGB[cat]),
                   linewidth=1.0, linestyle="--", alpha=0.7)
        ax.text(len(dh_list) - 0.4, y_pos - 0.4, cat,
                color=np.array(CAT_RGB[cat]), fontsize=7, va="top")

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# D.  Side-by-side at Δh = 0 vs +2m vs +3m (key comparison)
# ─────────────────────────────────────────────────────────────────────────────

def plot_three_height_comparison(
    reprojection_series: Dict[float, dict],
    output_path: str = "outputs/three_height_comparison.png",
) -> None:
    """
    3-column comparison: 0m, +2m, +3m (the three most relevant operating points).
    Each column shows: original | hole-overlay | inpainted.
    """
    show_dh = [0.0, 2.0, 3.0]
    available = sorted(reprojection_series.keys())
    show_dh = [dh for dh in show_dh if dh in available]
    if not show_dh:
        show_dh = available[:min(3, len(available))]

    n_cols   = len(show_dh)
    row_tags = ["Original image", "Hole overlay (starvation)", "Inpainted result"]

    fig = plt.figure(figsize=(5 * n_cols, 12))
    fig.patch.set_facecolor("#1a1a2e")
    gs  = GridSpec(3, n_cols, figure=fig, hspace=0.08, wspace=0.04)

    for ci, dh in enumerate(show_dh):
        entry    = reprojection_series[dh]
        orig     = entry.get("image_orig")
        inpainted = entry.get("inpainted", entry.get("warped"))
        mask     = entry.get("hole_mask")
        hole_pct = entry.get("hole_pct", 0.0)

        if orig is None:
            # Synthetic: use inpainted as "original"
            orig = inpainted.copy() if inpainted is not None else np.zeros((450, 800, 3), np.uint8)
        if inpainted is None:
            inpainted = orig.copy()
        if mask is None:
            mask = np.zeros(orig.shape[:2], dtype=bool)

        hole_overlay = overlay_hole_mask(inpainted, mask, alpha=0.5)

        row_imgs = [orig, hole_overlay, inpainted]
        for ri, rimg in enumerate(row_imgs):
            ax = fig.add_subplot(gs[ri, ci])
            ax.imshow(cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([])

            map_overall = 0.58 * sum(_RELATIVE_MAP[c].get(dh, 1.0)
                                     for c in CATEGORIES) / len(CATEGORIES)
            if ri == 0:
                ax.set_title(
                    f"Δh = {dh:+.1f} m   (est. mAP ≈ {map_overall:.2f})",
                    color="white", fontsize=11, pad=4,
                )
            if ci == 0:
                ax.set_ylabel(row_tags[ri], color="#aaaacc", fontsize=9,
                              labelpad=6)

            # Annotate hole%
            if ri == 1:
                ax.text(0.99, 0.03, f"hole={hole_pct:.1f}%",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=9, color="#ff8888")

    fig.suptitle("Camera Height Shift: Visual Impact on Autonomous Perception",
                 color="white", fontsize=13, y=0.995)
    plt.savefig(output_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# E.  Console summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_failure_summary(
    reprojection_series: Optional[Dict[float, dict]] = None,
) -> None:
    """Print a formatted table of per-category mAP estimates per Δh."""
    dh_list = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    header = f"{'Category':<16}" + "".join(f"Δh={dh:+.1f}m  " for dh in dh_list)
    print("\n" + "=" * len(header))
    print("Per-Category mAP Estimates (baseline × relative factor)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for cat in CATEGORIES:
        row = f"{cat:<16}"
        for dh in dh_list:
            base = _BASELINE_MAP[cat]
            rel  = _RELATIVE_MAP[cat].get(dh, 1.0)
            val  = base * rel
            row += f"  {val:.3f}     "
        print(row)

    print("-" * len(header))

    if reprojection_series:
        row = f"{'hole_pct':<16}"
        for dh in dh_list:
            hp = reprojection_series.get(dh, {}).get("hole_pct", 0.0)
            row += f"  {hp:5.1f}%    "
        print(row)

    print("=" * len(header))


# ─────────────────────────────────────────────────────────────────────────────
# High-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_failure_visuals(
    reprojection_series: Dict[float, dict],
    annotations: Optional[List[dict]] = None,
    output_dir: str = "outputs",
) -> None:
    """Run all four visualization functions and print summary."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n── Failure Visualizer ───────────────────────────────────────────")
    print_failure_summary(reprojection_series)

    plot_per_category_failure_strips(
        reprojection_series, annotations,
        output_path=f"{output_dir}/failure_strips.png",
    )
    plot_category_degradation_bars(
        reprojection_series,
        output_path=f"{output_dir}/category_degradation.png",
    )
    plot_starvation_heatmap(
        reprojection_series,
        output_path=f"{output_dir}/starvation_heatmap.png",
    )
    plot_three_height_comparison(
        reprojection_series,
        output_path=f"{output_dir}/three_height_comparison.png",
    )
    print("── Failure visualizer done ──────────────────────────────────────\n")
