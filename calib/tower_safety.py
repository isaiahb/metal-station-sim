"""Hard limits for every scripted tower motion, sim and real: calib/safety.py's gate (wrist roll and flex limits, the
camera box clear of the arm's links and 4 cm over the table) with the table's edges and a fingers-down rule added.

Rules, at every sampled point of every path (the joint-space line, every 2 deg of the largest joint change):
  1. wrist_roll within 1.6 rad of camera-up, wrist_flex 0.3 rad inside its limits (calib/safety.py).
  2. The wrist camera: when the fingers point down (approach . down >= 0.6) it may face down, but its image-up must
     not point back toward the arm base (horizontal part . outward from the base >= -0.05) and never be rolled under
     (image-up . world-up >= -0.5); otherwise it is never upside down (image-up . world-up >= 0, calib/safety.py).
  3. The camera box (its real size plus 2 cm) clear of the arm's other links, and 4 cm over the table.
  4. Every point of the arm from the forearm out (links 3 to 6, both jaws) and the camera box: over the table top by
     at least 1 cm, and inside the table's top: x past the base plate's front edge (0.05 m) and y inside the table's
     side edges, for every point lower than 10 cm over the table. The one exception: the moves out of and back to
     the arm's folded home (it rests lying back over its base), where only the height rules hold.
The sim's table matches the real one's top (TABLE below): set by SimRobot so nothing passes through its edges.

Positions are the arm's model frame (x forward, y left, z up, the base plate's top at 0; the table top 8 mm under it).
"""
from __future__ import annotations

import math

import numpy as np

from calib.safety import BOX_HI, BOX_LO, FLEX_IN, MARGIN, NEAR, ROLL_MAX, ROLL_UP, STEP, Safety

TABLE_Z = -0.008  # the table top in the arm frame (the 8 mm base plate on it)
FRONT_X = 0.05  # low points stay in front of this (the base plate's front edge; the table's near edge is behind it)
SIDE_Y = (-0.30, 0.42)  # low points stay between these (the table's left edge; the right arm's side)
LOW = 0.10  # "low": under this over the table
TABLE_CLEAR_LINKS = 0.010
TABLE_CLEAR_CAM = 0.040
DOWN_MIN = 0.6  # fingers count as pointing down at approach . down >= this
OUT_MIN = -0.05
UNDER_MIN = -0.5
UP_MIN = 0.0  # as calib/safety.py: never upside down, sideways allowed
CHECKED = ("right_link3", "right_link4", "right_link5", "right_link6", "right_jaw0", "right_jaw1")
# the real table, for the sim: top at TABLE_Z, 25 mm thick, its near edge 6 cm behind the base's centre (the base
# plate, 112 mm deep, sits flush with it), 0.75 m deep, from the right arm's base (y -0.69) less 0.4 m to 0.45 m left
TABLE = dict(x=(-0.06, 0.69), y=(-1.09, 0.45), thick=0.025)


class TowerSafety(Safety):
    def __init__(self):
        super().__init__()
        m = self.s.model
        rng = np.random.default_rng(1)
        self.points = []  # (geom, local points) of the links checked against the table
        import mujoco
        for g in range(m.ngeom):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            if b in CHECKED:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                self.points.append((g, v[rng.choice(len(v), min(400, len(v)), replace=False)], b))
        self.jaw_q = self.s.jaw_qadr["right"]

    def check_pose(self, q, opening=0.07, home=False):
        """home: a move to or from the arm's folded home (it rests lying back over its base, behind the plate's front
        edge): the front and side edge rules are off, the rest stay."""
        q = np.asarray(q, float)
        if abs(q[5] - ROLL_UP) > ROLL_MAX:
            return False, f"wrist_roll {q[5]:+.2f} rad is more than {ROLL_MAX} from camera-up"
        if q[3] < self.lo[3] + FLEX_IN or q[3] > self.hi[3] - FLEX_IN:
            return False, f"wrist_flex {q[3]:+.2f} rad within {FLEX_IN} of its limits"
        Rc, pc = self.camera(q)
        d = self.s.data
        for a in self.jaw_q:
            d.qpos[a] = opening / 2
        import mujoco
        mujoco.mj_kinematics(self.s.model, d)
        Rt = d.site_xmat[self.sid].reshape(3, 3)
        up = -Rc[:, 1]  # the image's up, in the world
        if Rt[:, 2] @ np.array([0, 0, -1.0]) >= DOWN_MIN:  # fingers down
            out = pc[:2] / max(np.linalg.norm(pc[:2]), 1e-6)
            h = up[:2] / max(np.linalg.norm(up[:2]), 1e-6)
            if h @ out < OUT_MIN:
                return False, f"fingers down with the camera's image-up turned back toward the base ({h @ out:+.2f})"
            if up[2] < UNDER_MIN:
                return False, "camera rolled under"
        elif up[2] < UP_MIN:
            return False, f"camera tilted {math.degrees(math.acos(np.clip(up[2], -1, 1))):.0f} deg from upright"
        corners = np.array([[x, y, z] for x in (BOX_LO[0], BOX_HI[0]) for y in (BOX_LO[1], BOX_HI[1]) for z in (BOX_LO[2], BOX_HI[2])])
        cw = corners @ Rc.T + pc
        if cw[:, 2].min() < TABLE_Z + TABLE_CLEAR_CAM:
            return False, f"camera {(cw[:, 2].min() - TABLE_Z) * 1000:.0f} mm over the table (min {TABLE_CLEAR_CAM * 1000:.0f})"
        pts = [cw]
        for g, v, b in self.points:
            pts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        P = np.concatenate(pts)
        if P[:, 2].min() < TABLE_Z + TABLE_CLEAR_LINKS:
            return False, f"arm {(P[:, 2].min() - TABLE_Z) * 1000:.0f} mm over the table (min {TABLE_CLEAR_LINKS * 1000:.0f})"
        low = P[P[:, 2] < TABLE_Z + LOW] if not home else P[:0]
        if len(low) and low[:, 0].min() < FRONT_X:
            return False, f"arm low ({(low[:, 2].min() - TABLE_Z) * 1000:.0f} mm) behind the plate's front edge (x {low[:, 0].min() * 1000:.0f} mm)"
        if len(low) and (low[:, 1].min() < SIDE_Y[0] or low[:, 1].max() > SIDE_Y[1]):
            return False, "arm low past the table's side"
        for g, v, b in self.geoms:  # the camera box against the other links (calib/safety.py)
            pw = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            pcam = (pw - pc) @ Rc
            mg = NEAR.get(b, MARGIN)
            if np.all((pcam >= BOX_LO - mg) & (pcam <= BOX_HI + mg), axis=1).any():
                return False, f"camera box ({mg * 1000:.0f} mm margin) hits {b.replace('right_', '')}"
        return True, "ok"

    def check_path(self, q0, q1, opening=0.07, recover=False, home=False):
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        n = max(2, int(math.ceil(np.max(np.abs(q1 - q0)) / STEP)) + 1)
        for t in np.linspace(0, 1, n):
            ok, why = self.check_pose(q0 + (q1 - q0) * t, opening, home)
            if not ok:
                return False, f"at {t * 100:.0f}% of the path: {why}"
        return True, "ok"


def outward_ok(R: np.ndarray, xy) -> bool:
    """A fingers-down tool frame passes rule 2 at xy: the camera (on the across axis side) faces outward."""
    from calib.tower_plan import ROOT
    import json
    Rm = np.array(json.loads((ROOT / "calib" / "cam_wrist_l.json").read_text())["R_in_tcp"])
    up = -(R @ Rm)[:, 1]
    out = np.asarray(xy, float) / max(np.linalg.norm(xy), 1e-6)
    h = up[:2] / max(np.linalg.norm(up[:2]), 1e-6)
    return bool(h @ out >= OUT_MIN + 0.02)
