"""One gate for every scripted move of the left Metal arm: check_path(q0, q1) -> (ok, reason).

Isaiah's rules (Nov 7, after the wrist camera nearly hit the arm's own body during a cube-view run):
  1. The wrist camera never goes upside down: its image-up axis keeps a component along world up of at least 0
     (sideways is allowed; it was 0.5 until Isaiah's Nov 7 evening change), at every sampled point of the path.
  2. The wrist camera, as a box on top of the gripper (its real size plus a 2 cm margin), clears every other arm link,
     the base and plate, the cubes, and stays 4 cm over the table.
  3. wrist_roll within 1.6 rad of the camera-up roll (0: the camera on top with the fingers straight), wrist_flex at
     least 0.3 rad inside its limits.
The path is the server's straight line in joint space, sampled every 2 deg of the largest joint change.

The camera's mount comes from calib/cam_wrist_l.json (fitted by calib/fit_tagcubes.py) when it has one, otherwise
the fitted values below. The sim's Metal arm has no camera, which is why this exists.
"""

import json
import math
import pathlib

import mujoco
import numpy as np

from sim.physics import Scene

ROOT = pathlib.Path(__file__).resolve().parent.parent
TABLE_Z = -0.008
MARGIN = 0.020
TABLE_CLEAR = 0.040
UP_MIN = 0.0  # Isaiah, Nov 7 evening: never upside down; sideways is fine (was 0.5, which also refused sideways)
ROLL_UP, ROLL_MAX = 0.0, 1.6
FLEX_IN = 0.3
STEP = math.radians(2.0)
# the camera body in its own frame (x right, y down, z forward), metres: lens at the origin, the housing behind it
BOX_LO, BOX_HI = np.array([-0.022, -0.018, -0.035]), np.array([0.022, 0.018, 0.005])
# links the camera rides on: not checked. The wrist links next to it (link4, link5) sit 11 and 30 mm from the box at
# the arm's rest pose, where nothing touches, so they get a smaller margin
OWN = {"right_link6", "right_jaw0", "right_jaw1"}
NEAR = {"right_link4": 0.005, "right_link5": 0.005}
# cubes on the table (centre xy, half size): both estimates (overhead and wrist chain) until they agree
CUBES = [((0.199, 0.272), 0.035), ((0.255, 0.395), 0.035), ((0.044, 0.175), 0.0196), ((0.100, 0.293), 0.0196)]


def _mount():
    f = ROOT / "calib" / "cam_wrist_l.json"
    try:
        d = json.load(open(f))
        if "R_in_tcp" in d and d.get("fitted_with", "").startswith("calib/fit_tagcubes"):
            return np.array(d["R_in_tcp"]), np.array(d["pos_in_tcp"])
    except (OSError, ValueError):
        pass
    R = np.array([[-0.99, -0.1, -0.07], [0.11, -0.94, -0.33], [-0.04, -0.34, 0.94]])
    return R, np.array([0.011, 0.074, -0.085])


class Safety:
    def __init__(self):
        self.s = Scene("metal", "right", props=[], draw=True)
        m = self.s.model
        self.sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_tcp")
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"right_{n}") for n in self.s.joint_names]
        self.lo, self.hi = m.jnt_range[jids, 0].copy(), m.jnt_range[jids, 1].copy()
        self.Rm, self.tm = _mount()
        rng = np.random.default_rng(0)
        self.geoms = []  # (geom id, local vertices, body name)
        for g in range(m.ngeom):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            if not b.startswith("right_") or b in OWN:
                continue
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            self.geoms.append((g, v[rng.choice(len(v), min(600, len(v)), replace=False)], b))

    def camera(self, q):
        """The camera's axes in the world (columns x right, y down, z forward) and its lens position."""
        m, d = self.s.model, self.s.data
        self.s.set_arm("right", q, 0.0, snap=True)
        mujoco.mj_forward(m, d)
        Rt = d.site_xmat[self.sid].reshape(3, 3)
        return Rt @ self.Rm, d.site_xpos[self.sid] + Rt @ self.tm

    def check_pose(self, q):
        q = np.asarray(q, float)
        if abs(q[5] - ROLL_UP) > ROLL_MAX:
            return False, f"wrist_roll {q[5]:+.2f} rad is more than {ROLL_MAX} from camera-up"
        if q[3] < self.lo[3] + FLEX_IN or q[3] > self.hi[3] - FLEX_IN:
            return False, f"wrist_flex {q[3]:+.2f} rad within {FLEX_IN} of its limits [{self.lo[3]:.2f}, {self.hi[3]:.2f}]"
        Rc, pc = self.camera(q)
        up = -Rc[:, 1] @ np.array([0, 0, 1.0])
        if up < UP_MIN:
            return False, f"camera tilted {math.degrees(math.acos(np.clip(up, -1, 1))):.0f} deg from upright (image-up . world-up {up:.2f})"
        corners = np.array([[x, y, z] for x in (BOX_LO[0], BOX_HI[0]) for y in (BOX_LO[1], BOX_HI[1]) for z in (BOX_LO[2], BOX_HI[2])])
        cw = corners @ Rc.T + pc
        if cw[:, 2].min() < TABLE_Z + TABLE_CLEAR:
            return False, f"camera {(cw[:, 2].min() - TABLE_Z) * 1000:.0f} mm over the table (min {TABLE_CLEAR * 1000:.0f})"
        for (cx, cy), h in CUBES:
            top = TABLE_Z + 2 * h
            # nearest distance from the cube (a box) to the camera's corners, coarse but conservative with the margin
            dx = np.maximum(np.abs(cw[:, 0] - cx) - h, 0)
            dy = np.maximum(np.abs(cw[:, 1] - cy) - h, 0)
            dz = np.maximum(cw[:, 2] - top, 0)
            if np.sqrt(dx ** 2 + dy ** 2 + dz ** 2).min() < MARGIN:
                return False, f"camera within {MARGIN * 1000:.0f} mm of the cube at {cx * 1000:.0f}, {cy * 1000:.0f} mm"
        d = self.s.data
        for g, v, b in self.geoms:  # every other link's points, in the camera's frame, against the padded box
            pw = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            pcam = (pw - pc) @ Rc
            mg = NEAR.get(b, MARGIN)
            inside = np.all((pcam >= BOX_LO - mg) & (pcam <= BOX_HI + mg), axis=1)
            if inside.any():
                return False, f"camera box ({mg * 1000:.0f} mm margin) hits {b.replace('right_', '')}"
        return True, "ok"

    def upness(self, q):
        Rc, _ = self.camera(q)
        return float(-Rc[:, 1][2])

    def check_path(self, q0, q1, recover=False):
        """recover: the arm already sits a little outside the upright limit (a stop leaves the command at the sagging
        reading); the move may start there if only uprightness fails, by at most 0.1, never gets worse along the
        way, and the rest of the path and the end pass everything."""
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        n = max(2, int(math.ceil(np.max(np.abs(q1 - q0)) / STEP)) + 1)
        u_prev = None
        for t in np.linspace(0, 1, n):
            q = q0 + (q1 - q0) * t
            ok, why = self.check_pose(q)
            if not ok:
                u = self.upness(q)
                tilted = why.startswith("camera tilted")
                if not (recover and tilted and u >= UP_MIN - 0.1 and (u_prev is None or u >= u_prev - 1e-3) and t < 1):
                    return False, f"at {t * 100:.0f}% of the path: {why}"
                u_prev = u
            else:
                u_prev = None if not recover else self.upness(q)
                recover = False  # once inside the limits, the rest of the path must stay inside
        return True, "ok"


_S = None


def check_path(q0, q1, recover=False):
    """(ok, reason) for the left arm moving in a straight joint-space line from q0 to q1."""
    global _S
    if _S is None:
        _S = Safety()
    return _S.check_path(q0, q1, recover)


def check_pose(q):
    global _S
    if _S is None:
        _S = Safety()
    return _S.check_pose(q)
