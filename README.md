# Camera Height Impact on AI Object Detection

Research pipeline demonstrating how raising a camera from **1.57 m** (standard car mount) to **3.0 m** (elevated mount) fundamentally changes what an AI detector sees — causing missed detections, false positives, and confidence degradation.

## Key Findings

| Metric | Value |
|--------|-------|
| Cars that flip front→roof view | **65.8 %** |
| Pedestrians that change view | **70.8 %** |
| Median viewing angle shift | **+3.2°** |
| YOLOv8 confidence drop (matched objects) | **−0.176** |
| Poles misclassified as persons at 3m | **3 false positives** (0 at 1.57m) |

## Pipeline Overview

```
NuScenes LiDAR + Camera
        │
        ├── Geometric Analysis (GT 3D annotations)
        │     └── Project to 1.57m and 3m virtual cameras
        │
        ├── Image Rendering (MiDaS dense depth + LiDAR calibration)
        │     └── Warp real image to 3m viewpoint
        │
        ├── YOLOv8 Detection Comparison
        │     └── Show missed/false-positive detections at 3m
        │
        └── 3D LiDAR Detection (DBSCAN + box fitting)
              └── Height-invariant: same 3D boxes, any camera height
```

## Scripts

| Script | Description |
|--------|-------------|
| `run_colmap_height.py` | COLMAP sparse reconstruction + Open3D render at 1.5m and 3m |
| `run_monocular_3m.py` | MiDaS monocular depth → 3m render → feature detection |
| `run_perspective_proof.py` | 5-figure proof that perspective changes (grid tracking, heatmap) |
| `run_comparison_300cm.py` | 3-panel comparison: original | point cloud warp | rendered 3m |
| `run_height_render_compare.py` | 4-panel: original | raw warp | inpainted | diff heatmap |
| `run_feature_accuracy.py` | Feature detection accuracy at both heights |

## Research Figures (outputs/research_proof/)

| Figure | Content |
|--------|---------|
| `fig1_gt_box_comparison.png` | GT 3D boxes at 1.57m vs 3m with shift arrows |
| `fig2_viewing_angle_dist.png` | Viewing angle distribution (4849 cars, 4578 pedestrians) |
| `fig3_angle_shift_and_flip.png` | 65.8% car flip rate, distance-dependent shift |
| `fig4_real_vs_rendered_gt.png` | Real photo + rendered 3m photo with GT annotations |
| `fig5_summary_stats.png` | Key numbers summary |
| `fig6_yolo_comparison.png` | YOLOv8 missed/new detections + confidence scatter |
| `fig7_false_positive_pole.png` | Pole misclassified as person at 3m |
| `fig8_3d_lidar_solution.png` | 3D LiDAR detection: height-invariant solution |
| `camera_height_impact_report.pdf` | Full PDF report |

## Solution: 3D LiDAR Detection

The root fix is to detect in 3D world space (LiDAR), not 2D image space:
- Detect once in 3D → project to **any** camera height without re-detection
- No domain shift: physical size classification is height-invariant
- No false positives from perspective-altered appearances

## Setup

```bash
pip install -r requirements.txt
# Download NuScenes mini dataset → nuscenes_data/
python setup_nuscenes.py
```

## Data

Uses [NuScenes mini dataset](https://www.nuscenes.org/nuscenes#download) — scene-0103 (Boston, 40 samples, 6 cameras).
Camera heights: CAM_FRONT=1.495m, CAM_BACK=1.568m (NuScenes default).
