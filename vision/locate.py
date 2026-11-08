"""Finds the green foam block and the purple peg in one camera frame and places them in the arm's frame.

Colour first, no learning: the block is a saturated lime, the peg a saturated magenta, the desk pale.
Each blob's pixels become rays from the calibrated camera, met with the plane the object's surface is
on (the block's top, 34.5 mm up; a lying peg's centreline, 11.3 mm up; a standing peg's top, 100.5 mm
up), so the answer is a position in the right arm's base frame (x forward, y left, z up) and a heading.

The camera model is the pinhole MuJoCo renders with (and a calibrated real camera reduces to):
`Camera(width, height, fovy_deg, pos, R)` with R's columns the camera's x (right), y (up), z (backwards)
axes in the arm's frame. Real cameras: calibrate pos and R first (and undistort).
"""

from __future__ import annotations

import dataclasses
import math

import cv2
import numpy as np

BLOCK = dict(x=0.0507, y=0.0499, h=0.0345)
PEG = dict(d=0.02265, length=0.1005)

# HSV ranges (OpenCV: H 0..180); widen for the venue's light once real frames are in
GREEN = ((25, 90, 60), (55, 255, 255))
MAGENTA = ((135, 60, 50), (175, 255, 255))


@dataclasses.dataclass
class Camera:
    width: int
    height: int
    fovy_deg: float
    pos: np.ndarray  # camera centre, arm frame, m
    R: np.ndarray  # columns: camera x right, y up, z backwards, in the arm frame
    k1: float = 0.0  # radial distortion of a real camera (calib/cam_mid.json); 0 for sim renders

    @classmethod
    def from_calib(cls, c: dict) -> "Camera":
        """A real camera from a calib/fit_overhead.py file (pinhole centred on the frame, plus k1)."""
        return cls(c["width"], c["height"], c["fovy_deg"], np.array(c["pos"]), np.array(c["R_cols_xyz"]), c.get("k1", 0.0))

    @property
    def f(self) -> float:
        return self.height / 2 / math.tan(math.radians(self.fovy_deg) / 2)

    def rays(self, uv: np.ndarray) -> np.ndarray:
        """Unit rays (arm frame) through pixel centres uv (N x 2, column, row)."""
        if self.k1:  # the fit's convention: principal point at (W/2, H/2), OpenCV distortion
            K = np.array([[self.f, 0, self.width / 2], [0, self.f, self.height / 2], [0, 0, 1.0]])
            n = cv2.undistortPoints(np.asarray(uv, float).reshape(-1, 1, 2), K, np.array([self.k1, 0, 0, 0, 0.0])).reshape(-1, 2)
            x, y = n[:, 0], -n[:, 1]
        else:
            x = (uv[:, 0] + 0.5 - self.width / 2) / self.f
            y = -(uv[:, 1] + 0.5 - self.height / 2) / self.f
        d = np.stack([x, y, -np.ones_like(x)], 1) @ self.R.T
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def project(self, xyz: np.ndarray) -> np.ndarray:
        """Pixels (column, row) of arm-frame points."""
        rel = (np.asarray(xyz, dtype=float).reshape(-1, 3) - self.pos) @ self.R
        if self.k1:
            K = np.array([[self.f, 0, self.width / 2], [0, self.f, self.height / 2], [0, 0, 1.0]])
            pts = np.c_[rel[:, 0], -rel[:, 1], -rel[:, 2]]
            uv, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, np.array([self.k1, 0, 0, 0, 0.0]))
            return uv.reshape(-1, 2)
        return np.c_[self.width / 2 + self.f * rel[:, 0] / -rel[:, 2] - 0.5, self.height / 2 - self.f * rel[:, 1] / -rel[:, 2] - 0.5]

    def on_plane(self, uv: np.ndarray, z: float) -> np.ndarray:
        """Where each pixel's ray meets the horizontal plane at height z (arm frame)."""
        d = self.rays(np.asarray(uv, dtype=float).reshape(-1, 2))
        t = (z - self.pos[2]) / d[:, 2]
        return self.pos + d * t[:, None]


def mask(rgb: np.ndarray, rng) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, np.array(rng[0]), np.array(rng[1]))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def biggest(m: np.ndarray, min_px: int = 30):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < min_px:
        return None
    v, u = np.nonzero(lab == k)
    return np.stack([u, v], 1)


def _rect(xy: np.ndarray):
    """Centre, heading (rad, mod 90 deg for a square-ish block, mod 180 for a peg) and size of points on a plane."""
    (cx, cy), (w, h), ang = cv2.minAreaRect(xy.astype(np.float32))
    return np.array([cx, cy]), math.radians(ang), (w, h)


def _far_centre(cam: Camera, pts: np.ndarray, half: float) -> np.ndarray:
    """Centre of an object from pixels met with the plane of its top: side faces land nearer the camera
    than they are, so only the far edge is true. Centre = far edge (98th percentile along the view)
    minus the object's half size along the view; across the view, the median."""
    v = pts.mean(0) - cam.pos[:2]
    v /= np.linalg.norm(v)
    n = np.array([-v[1], v[0]])
    along, across = pts @ v, pts @ n
    return v * (np.percentile(along, 98) - half) + n * np.median(across)


def _axis(pts: np.ndarray) -> float:
    """Heading of the long axis of points on a plane, rad mod pi."""
    c = pts - pts.mean(0)
    w, vecs = np.linalg.eigh(c.T @ c)
    a = vecs[:, -1]
    return math.atan2(a[1], a[0]) % math.pi


def locate(rgb: np.ndarray, cam: Camera, desk_z: float = 0.0) -> dict:
    """The block's and the peg's poses in the arm frame, from one frame. None for what is not seen."""
    out = {"block": None, "peg": None}
    g = biggest(mask(rgb, GREEN))
    block_xy = None
    if g is not None:
        # the block's top face dominates a camera looking down; its pixels met with the top's plane.
        # Its sides are in the blob too, which pulls the centre toward the camera: estimate from the
        # far half of the blob along the view (the top), then centre the block's footprint there.
        top = cam.on_plane(g, desk_z + BLOCK["h"])[:, :2]
        _, yaw, _ = _rect(top)
        c = _far_centre(cam, top, (BLOCK["x"] + BLOCK["y"]) / 4)
        block_xy = c
        out["block"] = {"xyz": [float(c[0]), float(c[1]), desk_z + BLOCK["h"] / 2], "yaw": yaw % (math.pi / 2), "pixels": int(len(g))}
    p = biggest(mask(rgb, MAGENTA))
    if p is not None:
        # plugged in: the peg's top, met with the top's plane, sits over the block (the far edge of its pixels
        # there is the cap's far rim); lying: its pixels met with its centreline's plane, a long thin shape
        top_xy = _far_centre(cam, cam.on_plane(p, desk_z + PEG["length"])[:, :2], PEG["d"] / 2)
        standing = block_xy is not None and np.linalg.norm(top_xy - block_xy) < 0.025
        if standing:
            out["peg"] = {"xyz": [float(top_xy[0]), float(top_xy[1]), desk_z + PEG["length"] / 2], "standing": True, "pixels": int(len(p))}
        else:
            flat = cam.on_plane(p, desk_z + PEG["d"] / 2)[:, :2]
            c = np.median(flat, 0)  # its side pulls this about 1 cm toward the camera (sim_test): next to fix
            out["peg"] = {"xyz": [float(c[0]), float(c[1]), desk_z + PEG["d"] / 2], "yaw": _axis(flat), "standing": False, "pixels": int(len(p))}
    return out


# Real station pieces (NUCK-1 overhead camera, venue light): OpenCV hue bins of saturated table pixels.
# Measured on runs/station/calib/poses-1/pose03.jpg: green 66, blue 101, purple 165 (a block and a stick), red 0.
HUES = {"red": [(0, 10), (176, 181)], "green": [(40, 88)], "blue": [(88, 125)], "purple": [(135, 176)], "orange": [(10, 24)]}
SAT_MIN, VAL_MIN = 55, 60
# the station's five pieces, plus golf balls (42.7 mm: lime, orange, white; white has no colour to find it by)
KINDS = {"green": {"block", "ball"}, "blue": {"block", "ball"}, "purple": {"block", "peg", "ball"}, "red": {"peg", "ball"}, "orange": {"ball"}}
BALL_D = 0.0427
SIZE = {"block": ((0.03, 0.085), (0.03, 0.085)), "peg": ((0.06, 0.16), (0.012, 0.05))}  # long, short side on the table, m


def _hue_name(h: float) -> str | None:
    for name, bins in HUES.items():
        if any(lo <= h < hi for lo, hi in bins):
            return name
    return None


def _far_centre_across(cam: Camera, pts: np.ndarray, axis: float, half: float) -> np.ndarray:
    """Centre of a lying peg: along its axis, the median; across it, the edge away from the camera minus
    its half width (the near side face lands short of where it is, as in _far_centre)."""
    a = np.array([math.cos(axis), math.sin(axis)])
    n = np.array([-a[1], a[0]])
    if n @ (pts.mean(0) - cam.pos[:2]) < 0:
        n = -n  # n points away from the camera
    along, across = pts @ a, pts @ n
    return a * np.median(along) + n * (np.percentile(across, 98) - half)


def track(rgb: np.ndarray, cam: Camera, table_z: float, min_px: int = 120, ignore=None, table_row: int = 0,
          area=((0.05, 0.85), (-0.95, 0.3))) -> list[dict]:
    """Every coloured block, peg and golf ball on the table: colour, kind, position in the arm frame, heading.

    Saturated blobs (the table, the arms and the cables are grey or thin), named by hue, then sized on
    the table plane: a round blob of a 42.7 mm ball's size is a ball (never a block), two sides of 3-8.5 cm is a block, a 6-16 cm by 1-5 cm one is a lying peg. A peg
    coloured blob sitting on a block's top is that block's plugged peg. `ignore`: boxes (x0, y0, x1, y1)
    of pixels to skip (hands, the other team's things); `table_row`: the table's far edge in the frame,
    above it is the room; `area`: (x range, y range) in the arm frame where pieces can be, on the table."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 1] >= SAT_MIN) & (hsv[..., 2] >= VAL_MIN)).astype(np.uint8)
    for x0, y0, x1, y1 in ignore or []:
        m[y0:y1, x0:x1] = 0
    m[:table_row] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    hue = hsv[..., 0].astype(float)
    hue[hue >= 180] -= 180
    # split touching pieces of different colours: label each hue bin on its own
    found = []
    for name, bins in HUES.items():
        mb = np.zeros_like(m)
        for lo, hi in bins:
            mb |= ((hue >= lo) & (hue < hi)).astype(np.uint8)
        mb &= m
        mb = cv2.morphologyEx(mb, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mb)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < min_px:
                continue
            v, u = np.nonzero(lab == k)
            uv = np.stack([u, v], 1)
            found.append((name, uv, stats[k]))
    blocks, pegs, balls, rest = [], [], [], []
    for name, uv, st in found:
        # a ball first: a round blob (it fills its enclosing circle, a box's corners leave about a third empty)
        # that is a ball's size on the plane through a ball's centre
        mask = np.zeros((st[3] + 2, st[2] + 2), np.uint8)
        mask[uv[:, 1] - st[1] + 1, uv[:, 0] - st[0] + 1] = 1
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c_ = max(cs, key=cv2.contourArea)
        (_, _), rr = cv2.minEnclosingCircle(c_)
        fill = cv2.contourArea(c_) / max(math.pi * rr * rr, 1)
        solid = cv2.contourArea(c_) / max(cv2.contourArea(cv2.convexHull(c_)), 1)
        mid = cam.on_plane(uv, table_z + BALL_D / 2)[:, :2]
        _, _, (bw, bh) = _rect(mid)
        # on a horizontal plane a ball's outline stretches along the line of sight (by 1 / sin of the camera's
        # elevation), so only its short side is a ball's width
        if "ball" in KINDS[name] and fill > 0.8 and solid > 0.93 and 0.7 * BALL_D <= min(bw, bh) <= 1.45 * BALL_D and max(bw, bh) <= 2.6 * BALL_D:
            c = _far_centre(cam, mid, BALL_D / 2)
            balls.append({"kind": "ball", "colour": {"green": "lime", "red": "orange"}.get(name, name), "xyz": [float(c[0]), float(c[1]), table_z + BALL_D / 2],
                          "yaw": 0.0, "size_m": [float(max(bw, bh)), float(min(bw, bh))], "round": [round(fill, 2), round(solid, 2)],
                          "px": int(len(uv)), "box": [int(st[0]), int(st[1]), int(st[0] + st[2]), int(st[1] + st[3])]})
            continue
        top = cam.on_plane(uv, table_z + BLOCK["h"])[:, :2]
        _, yaw, (w, h) = _rect(top)
        lng, sht = max(w, h), min(w, h)
        flat = cam.on_plane(uv, table_z + PEG["d"])[:, :2]  # a lying peg's top line
        _, _, (fw, fh) = _rect(flat)
        flng, fsht = max(fw, fh), min(fw, fh)
        box = [int(st[0]), int(st[1]), int(st[0] + st[2]), int(st[1] + st[3])]
        if "peg" in KINDS[name] and SIZE["peg"][0][0] <= flng <= SIZE["peg"][0][1] and SIZE["peg"][1][0] <= fsht <= SIZE["peg"][1][1] and flng > 1.8 * fsht:
            axis = _axis(flat)
            c = _far_centre_across(cam, flat, axis, PEG["d"] / 2)
            pegs.append({"kind": "stick", "colour": name, "standing": False, "xyz": [float(c[0]), float(c[1]), table_z + PEG["d"] / 2],
                         "yaw": axis, "size_m": [float(flng), float(fsht)], "px": int(len(uv)), "box": box})
        elif "block" in KINDS[name] and SIZE["block"][0][0] <= sht and lng <= SIZE["block"][0][1] * 1.6:
            c = _far_centre(cam, top, (BLOCK["x"] + BLOCK["y"]) / 4)
            blocks.append({"kind": "block", "colour": name, "xyz": [float(c[0]), float(c[1]), table_z + BLOCK["h"] / 2],
                           "yaw": yaw % (math.pi / 2), "size_m": [float(lng), float(sht)], "px": int(len(uv)), "box": box})
        else:
            rest.append((name, uv, box, float(lng), float(sht)))
    # small peg-coloured blobs over a block: a plugged peg (its cap and upper side)
    for name, uv, box, lng, sht in rest:
        if "peg" not in KINDS[name]:
            continue
        top_xy = _far_centre(cam, cam.on_plane(uv, table_z + PEG["length"])[:, :2], PEG["d"] / 2)
        near = [b for b in blocks if np.linalg.norm(top_xy - np.array(b["xyz"][:2])) < 0.03]
        if near:
            pegs.append({"kind": "stick", "colour": name, "standing": True, "xyz": [float(top_xy[0]), float(top_xy[1]), table_z + PEG["length"] / 2],
                         "in_block": near[0]["colour"], "px": int(len(uv)), "box": box})
    (x0, x1), (y0, y1) = area
    return [p for p in blocks + pegs + balls if x0 <= p["xyz"][0] <= x1 and y0 <= p["xyz"][1] <= y1]
