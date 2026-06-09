"""
Synthetic urban intersection scene.

Geometry primitives used by the LiDAR ray caster:
  GroundPlane, Box, Cylinder, ThinRect (for markings/signs).

All coordinates are in the world frame:
  X = forward, Y = left, Z = up.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from src.config import FEATURE_LABELS


# ── Reverse lookup: name → int label ─────────────────────────────────────────
NAME_TO_ID = {v: k for k, v in FEATURE_LABELS.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Geometry primitives
# Each primitive implements:
#   ray_intersect(origin: (3,), dirs: (N,3)) → (t: (N,), valid: (N,bool))
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GroundPlane:
    z: float = 0.0
    label_id: int = field(default_factory=lambda: NAME_TO_ID["ground"])
    intensity: float = 0.2

    def ray_intersect(self, origin: np.ndarray, dirs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dz = dirs[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (self.z - origin[2]) / dz
        valid = (np.abs(dz) > 1e-9) & (t > 0.05)
        t = np.where(valid, t, np.inf)
        return t, valid


@dataclass
class Box:
    """Axis-aligned box. center=(x,y,z), dims=(dx,dy,dz)."""
    center: np.ndarray
    dims: np.ndarray
    label_id: int = 0
    intensity: float = 0.5

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.dims   = np.asarray(self.dims,   dtype=float)
        self.bmin = self.center - self.dims / 2
        self.bmax = self.center + self.dims / 2

    def ray_intersect(self, origin: np.ndarray, dirs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        inv = np.where(np.abs(dirs) > 1e-12, 1.0 / dirs, np.sign(dirs) * 1e12)
        t1 = (self.bmin - origin) * inv   # (N, 3)
        t2 = (self.bmax - origin) * inv
        tmin_slab = np.minimum(t1, t2)
        tmax_slab = np.maximum(t1, t2)
        t_enter = np.max(tmin_slab, axis=1)
        t_exit  = np.min(tmax_slab, axis=1)
        valid = (t_exit >= t_enter) & (t_enter > 0.05)
        t = np.where(valid, t_enter, np.inf)
        return t, valid


@dataclass
class Cylinder:
    """Vertical cylinder. cx, cy = horizontal centre; z_min, z_max = height range."""
    cx: float
    cy: float
    radius: float
    z_min: float
    z_max: float
    label_id: int = 0
    intensity: float = 0.4

    def ray_intersect(self, origin: np.ndarray, dirs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ox = origin[0] - self.cx
        oy = origin[1] - self.cy
        dx = dirs[:, 0]
        dy = dirs[:, 1]
        dz = dirs[:, 2]

        a = dx ** 2 + dy ** 2
        b = 2 * (ox * dx + oy * dy)
        c = ox ** 2 + oy ** 2 - self.radius ** 2

        disc = b ** 2 - 4 * a * c
        valid_disc = (disc >= 0) & (a > 1e-12)
        sqrt_disc = np.sqrt(np.maximum(disc, 0))

        t1 = np.where(valid_disc, (-b - sqrt_disc) / (2 * a), np.inf)
        t2 = np.where(valid_disc, (-b + sqrt_disc) / (2 * a), np.inf)

        # Use nearer positive root
        t = np.where(t1 > 0.05, t1, t2)

        z_hit = origin[2] + t * dz
        in_height = (z_hit >= self.z_min) & (z_hit <= self.z_max)
        valid = valid_disc & (t > 0.05) & in_height
        t = np.where(valid, t, np.inf)
        return t, valid


@dataclass
class ThinRect:
    """Flat rectangle on z=const plane (used for road markings / signs)."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z: float = 0.01           # slightly above ground to avoid z-fighting
    label_id: int = 0
    intensity: float = 0.9

    def ray_intersect(self, origin: np.ndarray, dirs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dz = dirs[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (self.z - origin[2]) / dz
        valid = (np.abs(dz) > 1e-9) & (t > 0.05)
        hit_x = origin[0] + t * dirs[:, 0]
        hit_y = origin[1] + t * dirs[:, 1]
        in_bounds = ((hit_x >= self.x_min) & (hit_x <= self.x_max) &
                     (hit_y >= self.y_min) & (hit_y <= self.y_max))
        valid = valid & in_bounds
        t = np.where(valid, t, np.inf)
        return t, valid


# ─────────────────────────────────────────────────────────────────────────────
# Scene builder
# ─────────────────────────────────────────────────────────────────────────────

class UrbanScene:
    """
    Urban intersection with:
      - ground plane
      - 4 lane markings (dashed white lines)
      - 3 road symbols (stop line + direction arrows)
      - 2 traffic lights (pole + head)
      - 1 traffic sign (speed-limit board on pole)
      - 3 cars, 1 truck
      - 2 pedestrians
    """

    def __init__(self):
        self.objects: List = []
        self._build()

    def _build(self):
        G = NAME_TO_ID
        # Ground
        self.objects.append(GroundPlane(z=0.0, label_id=G["ground"], intensity=0.2))

        # ── Lane markings ─────────────────────────────────────────────────────
        # Two-lane road, lanes at y ∈ [-3.5, 0] and [0, 3.5]
        # Centre dashes (yellow) at y=0
        for x0 in range(2, 70, 6):
            self.objects.append(ThinRect(x0, x0+3, -0.075, 0.075,
                                         z=0.01, label_id=G["lane_marking"], intensity=0.95))
        # Right edge (white) at y=-3.5
        for x0 in range(0, 70, 5):
            self.objects.append(ThinRect(x0, x0+2.5, -3.575, -3.425,
                                         z=0.01, label_id=G["lane_marking"], intensity=0.9))
        # Left edge (white) at y=3.5
        for x0 in range(0, 70, 5):
            self.objects.append(ThinRect(x0, x0+2.5, 3.425, 3.575,
                                         z=0.01, label_id=G["lane_marking"], intensity=0.9))
        # Inner lane dividers
        for x0 in range(0, 70, 5):
            self.objects.append(ThinRect(x0, x0+2.5, -0.65, -0.575,
                                         z=0.01, label_id=G["lane_marking"], intensity=0.88))
            self.objects.append(ThinRect(x0, x0+2.5,  0.575,  0.65,
                                         z=0.01, label_id=G["lane_marking"], intensity=0.88))

        # ── Road symbols ──────────────────────────────────────────────────────
        # Stop line across both lanes at x=58 m
        self.objects.append(ThinRect(57.5, 58.5, -3.5, 3.5,
                                     z=0.015, label_id=G["road_symbol"], intensity=0.95))
        # Forward-arrow markings at x=20 and x=40
        for ax in (20.0, 40.0):
            self.objects.append(ThinRect(ax-0.6, ax+2.0, -2.5, -1.0,
                                         z=0.015, label_id=G["road_symbol"], intensity=0.92))
            self.objects.append(ThinRect(ax-0.6, ax+2.0,  1.0,  2.5,
                                         z=0.015, label_id=G["road_symbol"], intensity=0.92))

        # ── Traffic lights ────────────────────────────────────────────────────
        # Two poles at x=60, y=±5
        for sign_y in (-5.0, 5.0):
            # pole (thin cylinder)
            self.objects.append(Cylinder(60.0, sign_y, 0.08, 0.0, 5.0,
                                          label_id=G["pole"], intensity=0.3))
            # light head (box at top)
            self.objects.append(Box(center=[60.0, sign_y, 5.3],
                                    dims=[0.5, 0.3, 0.6],
                                    label_id=G["traffic_light"], intensity=0.8))

        # ── Traffic sign (speed limit 50) ─────────────────────────────────────
        # Pole at x=15, y=4.5
        self.objects.append(Cylinder(15.0, 4.5, 0.05, 0.0, 2.5,
                                      label_id=G["pole"], intensity=0.3))
        self.objects.append(Box(center=[15.0, 4.5, 3.0],
                                dims=[0.6, 0.08, 0.6],
                                label_id=G["traffic_sign"], intensity=0.7))

        # ── Vehicles ──────────────────────────────────────────────────────────
        # Car 1 – ahead in right lane
        self.objects.append(Box([18.0,  -1.75, 0.75], [4.2, 1.9, 1.5],
                                label_id=G["car"], intensity=0.55))
        # Car 2 – ahead in left lane
        self.objects.append(Box([28.0,   1.75, 0.75], [4.2, 1.9, 1.5],
                                label_id=G["car"], intensity=0.55))
        # Car 3 – further ahead
        self.objects.append(Box([48.0,  -1.75, 0.75], [4.2, 1.9, 1.5],
                                label_id=G["car"], intensity=0.60))
        # Truck – far ahead in right lane
        self.objects.append(Box([65.0,  -1.75, 1.75], [8.5, 2.4, 3.5],
                                label_id=G["truck"], intensity=0.50))

        # ── Pedestrians ───────────────────────────────────────────────────────
        self.objects.append(Cylinder(58.5, -4.5, 0.30, 0.0, 1.75,
                                      label_id=G["pedestrian"], intensity=0.45))
        self.objects.append(Cylinder(60.5,  5.2, 0.30, 0.0, 1.75,
                                      label_id=G["pedestrian"], intensity=0.45))

    def ray_intersect_all(
        self, origin: np.ndarray, dirs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return closest hit for all rays.
        Returns:
          best_t      (N,)  – intersection distances (inf = no hit)
          best_label  (N,)  – feature label of closest hit
          best_intens (N,)  – intensity of closest hit
        """
        N = len(dirs)
        best_t      = np.full(N, np.inf)
        best_label  = np.full(N, -1, dtype=np.int32)
        best_intens = np.zeros(N)

        for obj in self.objects:
            t, valid = obj.ray_intersect(origin, dirs)
            closer = valid & (t < best_t)
            best_t[closer]      = t[closer]
            best_label[closer]  = obj.label_id
            best_intens[closer] = obj.intensity

        return best_t, best_label, best_intens


def build_scene() -> UrbanScene:
    return UrbanScene()
