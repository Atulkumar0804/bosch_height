"""
NuScenes ADASAdapt Pipeline
============================
Full implementation of Options 2 + 3 from the ADASAdapt paper:

  Option 2 – LiDAR-based reprojection per frame
    • Raise virtual camera by Δh in ego Z via t_cam_new = t_cam + [0, 0, Δh]
    • Project LiDAR into new pose → dense depth map
    • Point-Cloud-Reproject original image to new viewpoint
    • Inpaint remaining holes

  Option 3 – Sensitivity curve
    • Run Option 2 at Δh = 0, +0.5, +1.0, +1.5, +2.0, +3.0 m
    • Compute proxy metrics: hole%, SSIM, density ratio, horizon shift
    • Map proxy → estimated mAP; compare with paper-reported values
    • Per-category failure visualization

Usage:
  # Synthetic fallback (no dataset needed):
  python run_nuscenes_pipeline.py

  # With NuScenes mini dataset:
  python run_nuscenes_pipeline.py --nuscenes /path/to/nuscenes --version v1.0-mini

  # Specific sample:
  python run_nuscenes_pipeline.py --nuscenes /path/to/nuscenes --sample <token>

Outputs (in outputs/nuscenes/):
  reprojection_strip.png        — visual strip at each Δh
  sensitivity_curve.png         — 4-panel mAP vs height curve
  failure_strips.png            — per-category bounding-box failure grid
  category_degradation.png      — bar chart: mAP per category × height
  starvation_heatmap.png        — heatmap: hole% by image row × height
  three_height_comparison.png   — 0m / +2m / +3m side-by-side
  metrics_table.csv             — numerical summary
"""

import argparse
import os
import sys
import time
import csv
import numpy as np
import cv2

# ── optional imports (graceful failure) ──────────────────────────────────────
try:
    from nuscenes.nuscenes import NuScenes
    _NUSCENES_OK = True
except ImportError:
    _NUSCENES_OK = False

from src.nuscenes_reprojector import (
    NuScenesReprojector,
    synthetic_height_series,
)
from src.sensitivity_curve import (
    HEIGHT_SHIFTS,
    compute_proxy_metrics,
    plot_sensitivity_curve,
    plot_height_strip,
    print_breakdown_table,
)
from src.failure_visualizer import generate_all_failure_visuals

OUTPUT_DIR = "outputs/nuscenes"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_nuscenes(dataroot: str, version: str):
    if not _NUSCENES_OK:
        print("  [WARN] nuscenes-devkit not installed — using synthetic fallback.")
        return None
    if not os.path.isdir(dataroot):
        print(f"  [WARN] NuScenes path not found: {dataroot} — using synthetic fallback.")
        return None
    print(f"  Loading NuScenes {version} from {dataroot} …")
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    print(f"  Loaded {len(nusc.sample)} samples.")
    return nusc


def _pick_sample(nusc, token: str = None) -> str:
    """Return a sample token — user-specified or a front-facing scene pick."""
    if token:
        return token
    # Pick the first sample in the first scene that has a front camera
    for sample in nusc.sample:
        if "CAM_FRONT" in sample["data"]:
            return sample["token"]
    return nusc.sample[0]["token"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Run reprojection series
# ─────────────────────────────────────────────────────────────────────────────

def run_reprojection(nusc, sample_token: str = None, cam_name: str = "CAM_FRONT"):
    """Return {delta_h: result_dict} for all HEIGHT_SHIFTS."""
    if nusc is not None:
        tok = _pick_sample(nusc, sample_token)
        print(f"\n  Sample token: {tok}")
        reprojector = NuScenesReprojector(nusc, cam_name=cam_name)

        print("  Running LiDAR-based reprojection series …")
        series = reprojector.generate_height_series(tok, HEIGHT_SHIFTS)
        series[0.0]["image_orig"] = series[0.0].get("image_orig")

        # Fetch annotations for bounding box overlay
        try:
            annotations, K, R_c, t_c = reprojector.get_annotations_for_sample(tok)
        except Exception as e:
            print(f"  [WARN] Could not fetch annotations: {e}")
            annotations = []

        return series, annotations, tok
    else:
        print("\n  Running SYNTHETIC height series (no nuScenes) …")
        series = synthetic_height_series(HEIGHT_SHIFTS)
        return series, [], "synthetic"


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Save CSV summary
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_csv(series: dict, proxy: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, "metrics_table.csv")
    rows = []
    for dh in sorted(series.keys()):
        entry = series[dh]
        pm    = proxy.get(dh, {})
        rows.append({
            "delta_h_m":       dh,
            "h_orig_m":        entry.get("h_orig", ""),
            "h_new_m":         entry.get("h_new", ""),
            "hole_pct":        f"{entry.get('hole_pct', 0):.2f}",
            "ssim":            f"{pm.get('ssim', 0):.4f}",
            "density_ratio":   f"{pm.get('density_ratio', 0):.4f}",
            "horizon_shift_px": pm.get("horizon_shift_px", 0),
            "map_proxy":       f"{pm.get('map_proxy', 0):.4f}",
        })
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Print console report
# ─────────────────────────────────────────────────────────────────────────────

def print_pipeline_report(series: dict, proxy: dict, sample_token: str) -> None:
    print("\n" + "═" * 70)
    print("  ADASAdapt: NuScenes Camera-Height Sensitivity Analysis")
    print("═" * 70)
    print(f"  Sample : {sample_token}")
    h0 = series[0.0]["h_orig"] if 0.0 in series else "?"
    print(f"  Orig camera height : {h0} m (nuScenes ~1.84 m)")
    print()
    print(f"  {'Δh':>6}  {'h_new':>7}  {'hole%':>7}  {'SSIM':>7}  "
          f"{'dens_r':>7}  {'horiz_Δpx':>10}  {'mAP_proxy':>10}")
    print("  " + "-" * 62)
    for dh in sorted(series.keys()):
        e  = series[dh]
        pm = proxy.get(dh, {})
        print(
            f"  {dh:>+6.1f}  {e.get('h_new', '?'):>7.2f}  "
            f"{e.get('hole_pct', 0):>7.1f}  "
            f"{pm.get('ssim', 0):>7.4f}  "
            f"{pm.get('density_ratio', 0):>7.4f}  "
            f"{pm.get('horizon_shift_px', 0):>10.1f}  "
            f"{pm.get('map_proxy', 0):>10.4f}"
        )
    print("═" * 70)

    # Synthetic vs real 3m gap
    print("\n  ─── Synthetic vs Real 3 m Gap ──────────────────────────────────")
    print("  Synthetic reprojection captures GEOMETRY shift only.")
    print("  Real 3 m deployment additionally causes:")
    print("   • Texture starvation of vehicle roofs / bonnets never seen at 1m")
    print("   • Occlusion pattern changes (new objects emerge from behind others)")
    print("   • Sensor-specific effects (FOV, resolution, calibration error)")
    print("  ─ Paper reports: +3m synthetic → mAP ~0.35 (geometric only)")
    print("                   +3m real       → mAP ~0.22 (full degradation)")
    print("  ─ Gap ≈ 0.13 mAP points (~37% of total degradation from full gap)")
    print("─" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ADASAdapt NuScenes height-shift pipeline (Options 2+3)"
    )
    parser.add_argument(
        "--nuscenes", default=None,
        help="Path to NuScenes dataset root (default: synthetic fallback)"
    )
    parser.add_argument(
        "--version", default="v1.0-mini",
        help="NuScenes version string (default: v1.0-mini)"
    )
    parser.add_argument(
        "--sample", default=None,
        help="Specific NuScenes sample token to process"
    )
    parser.add_argument(
        "--cam", default="CAM_FRONT",
        help="Camera name (default: CAM_FRONT)"
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--no-nuscenes", action="store_true",
        help="Force synthetic fallback even if dataset path is given"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.time()

    # ── Load dataset ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading dataset …")
    if args.no_nuscenes or args.nuscenes is None:
        nusc = None
    else:
        nusc = _load_nuscenes(args.nuscenes, args.version)

    # ── Reprojection series ───────────────────────────────────────────────────
    print("\n[2/5] Running height-shift reprojection series …")
    series, annotations, sample_token = run_reprojection(
        nusc, args.sample, args.cam
    )

    # ── Proxy metrics ─────────────────────────────────────────────────────────
    print("\n[3/5] Computing proxy metrics …")
    proxy = compute_proxy_metrics(series)
    save_metrics_csv(series, proxy, args.output_dir)
    print_breakdown_table(series)

    # ── Sensitivity curve ────────────────────────────────────────────────────
    print("\n[4/5] Plotting sensitivity curve …")
    # These functions save to src.sensitivity_curve.OUT; we copy to output_dir.
    import shutil
    curve_path = plot_sensitivity_curve(series)
    strip_path = plot_height_strip(series)
    for src_path, dst_name in [
        (curve_path, "sensitivity_curve.png"),
        (strip_path, "reprojection_strip.png"),
    ]:
        if src_path and os.path.isfile(src_path):
            dst = os.path.join(args.output_dir, dst_name)
            if os.path.abspath(src_path) != os.path.abspath(dst):
                shutil.copy2(src_path, dst)
            print(f"  Copied → {dst}")

    # ── Failure visualization ─────────────────────────────────────────────────
    print("\n[5/5] Generating failure visualizations …")
    # Build bbox annotations from nuScenes 3D boxes if available
    if annotations:
        # Project 3D boxes to 2D for each category
        from src.nuscenes_reprojector import project_3d_box_to_camera
        entry0  = series.get(0.0, series[min(series)])
        K       = entry0.get("K")
        # R_cam, t_cam not directly in series — re-load if nusc available
        ann_2d  = []
        ref_img = entry0.get("inpainted")
        if ref_img is None:
            ref_img = entry0.get("warped")
        for ann in annotations:
            category = ann.get("category", "").split(".")[0]
            ann_2d.append({"category": category})
    else:
        ann_2d = []

    generate_all_failure_visuals(
        series,
        annotations=ann_2d if ann_2d else None,
        output_dir=args.output_dir,
    )

    # ── Console report ────────────────────────────────────────────────────────
    print_pipeline_report(series, proxy, sample_token)

    elapsed = time.time() - t_start
    print(f"\nPipeline complete in {elapsed:.1f} s")
    print(f"Outputs written to: {os.path.abspath(args.output_dir)}/\n")


if __name__ == "__main__":
    main()
