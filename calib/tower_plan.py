"""The tallest-tower plan for the left Metal arm: two foam blocks flat, the third standing on its side on top
(34.5 + 34.5 + about 50 mm: about 119 mm).

Shared by the sim runner (sim/tower.py) and the real-arm runner (calib/tower_real.py); they differ only in the robot
object (moves, joints, wrist frames). Kinematics are the sim's Metal arm (Scene "metal", arm "right" = our left arm).

How the third block ends up standing (worked out in sim, inside calib/tower_safety.py's limits): with the camera
kept upright (or, fingers down, facing outward) the Metal wrist cannot turn the fingers sideways near the table, so the
block is tipped over instead (flip_poses): set flat at a flip spot about 44 cm out, gripped from above across the faces
along the radial line's normal, tipped toward the arm about its inner bottom edge; the wrist turns with it to 57 deg
(its flex limit less the margin), the pads drag the top on round, and once past 55 deg (its weight over the edge) the
jaws open and it lies down on its side. Then it is picked again from above, fingers down, and set on the tower.

Every grasp squeezes twice: the jaws close gently, open, the hand turns 90 deg and closes again. Each squeeze slides the
block to the hand's centre along the jaws and squares it, so after two its centre and yaw are the hand's.

The real arm (Nov 7 calibration, STATUS.md): measured joints sag under gravity (lift -0.06..-0.10 rad, elbow -0.04)
and fingertips landed centimetres off the plan. So:
  - sag: the driver has no gravity feedforward; the sag is the gravity torque over kp (sim model, servo_fit kp), fitted
    per joint on the touch run's loaded poses (SagModel). Every command is q_target - sag(q_target).
  - the blocks are seen only through the wrist camera (calib/cam_wrist_l.json mount and lens; no overhead camera).
    Look first: the hand high over three fixed spots, each block found by colour. Then before each grasp: hover over
    the block's estimate with the jaws squared to its estimated yaw, fit x, y, yaw of its box to the colour blob
    (jaws masked out by their CAD), move there, again until the fit stops moving (centred between the jaws, square to
    its faces), then descend and squeeze.
  - placing: the next block goes where the hand set (and squeezed) the one below, in the arm's commanded frame; the
    wrist look on the tower's top, hand empty, checks it, and before the third block is set the top is looked at
    again and the target moved by the change between the two looks. Same pose, same camera: its small mount and
    joint errors cancel in the difference (in sim, fits off by up to 10 mm absolute still stack within 3 mm).
Positions are all in the arm's own (model) frame: model joints = measured + calib/arm_left.json dq_rad.

Frame: the left arm's base at the origin, x forward, y left, z up; the sim's desk_z is the base bottom.

    python -m calib.tower_plan --fit-sag      # fit the sag model on the touch run, writes calib/sag_left.json
"""
from __future__ import annotations

import glob
import json
import math
import pathlib
import sys

import cv2
import mujoco
import numpy as np

from sim.physics import ARMS, BLOCK, Scene

ROOT = pathlib.Path(__file__).resolve().parent.parent
A = ARMS["metal"]
REST = np.array(A["rest"], dtype=float)
H = BLOCK["h"]  # 34.5 mm, along the hole
FACE = (BLOCK["x"] + BLOCK["y"]) / 2  # 50.3 mm: the tracker cannot tell the 50.7 and 49.9 faces apart
STAND_H = FACE  # the standing block's height (49.9 or 50.7)
PICK_TIP = 0.013  # fingertips this far over the surface under the block when gripping (the jaws stay 1 cm over the table)
CLEAR = 0.03  # a carried block's bottom this far over anything it passes
SQUEEZE_IN = 0.006  # the squeeze aims this much under the block's width: holds, gently (foam)
OPEN = 0.068  # open jaws (m): 9 mm a side on a 50 mm block, so the pads keep off blocks close by
HOVER = 0.025  # fingertips this far over a block's top for the wrist look
DROP = 0.003  # a placed block's bottom this far over what it lands on

BASE_ZONE = ((0.30, 0.38), (-0.06, 0.08))  # where block 1 may stay
BASE_SPOTS = [(x, y) for x in (0.32, 0.34, 0.30, 0.36) for y in (0.02, 0.0, 0.04, -0.02, 0.06, -0.04, 0.08, -0.06)]


def frame_down(yaw: float) -> np.ndarray:
    """Tool axes (columns: x jaw travel, y across the pads, z approach) for fingers straight down, jaws along yaw."""
    x = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([x, np.cross(z, x), z])


def tilt(R0: np.ndarray, sign: int, frac: float) -> np.ndarray:
    """R0 turned frac x 90 deg about its own jaw axis; sign +1 turns the fingers toward the jaw axis + 90 deg (the
    across axis then points down), -1 the other way (across axis up)."""
    x = R0[:, 0]
    K = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    a = sign * frac * math.pi / 2
    return (np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K) @ R0


def frame_side(jaw_yaw: float, sign: int) -> np.ndarray:
    """Fingers horizontal, jaws along jaw_yaw: frame_down(jaw_yaw) turned 90 deg about the jaw axis."""
    return tilt(frame_down(jaw_yaw), sign, 1.0)


def squares(yaw: float) -> list[np.ndarray]:
    return [frame_down(yaw + k * math.pi / 2) for k in range(4)]


def fold(yaw: float) -> float:
    """A square block's yaw folded to (-45, 45] deg."""
    return (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4


def yaw_of(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class Kin:
    """The sim's Metal arm as kinematics: IK on the tool point with many seeds, gravity torques, the wrist camera's
    pose, and the lowest points of the hand along a joint-space move."""

    def __init__(self, scene: Scene | None = None, side: str = "right"):
        self.s = scene or Scene("metal", "right", props=[], draw=True)  # drawn: the jaws' meshes for the wrist view
        self.m, self.side = self.s.model, side
        self.d = mujoco.MjData(self.m)
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")
        self.cam = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, f"{side}_wrist")
        jids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{n}") for n in A["joints"]]
        self.lo, self.hi = self.m.jnt_range[jids, 0] + 0.02, self.m.jnt_range[jids, 1] - 0.02
        # calib/safety.py's wrist rules as hard limits: wrist_roll within 1.6 rad of camera-up, wrist_flex 0.3 inside
        self.lo[5], self.hi[5] = max(self.lo[5], -1.58), min(self.hi[5], 1.58)
        self.lo[3], self.hi[3] = self.m.jnt_range[jids[3], 0] + 0.31, self.m.jnt_range[jids[3], 1] - 0.31
        self.dofs, self.qadr = self.s.arm_dadr[side], self.s.arm_qadr[side]
        rng = np.random.default_rng(7)
        self.seeds = [REST, REST + [0, 0, 0, 0, 0, math.pi / 2], REST - [0, 0, 0, 0, 0, math.pi / 2], np.zeros(6),
                      np.array([0, -1.5, 2.0, -1.0, 0, 0]), np.array([0, -1.0, 1.4, 0.0, 1.4, 0]), np.array([0, -1.0, 1.4, 0.0, -1.4, 0])]
        self.seeds += [rng.uniform(self.lo, self.hi) for _ in range(6)]
        mujoco.mj_forward(self.m, self.d)
        pads = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_pad{i}") for i in range(2)]
        R = self.d.site_xmat[self.site].reshape(3, 3)
        self.tip = max(((self.d.geom_xpos[g] - self.d.site_xpos[self.site]) @ R)[2] + self.m.geom_size[g][2] for g in pads)
        name = lambda g: mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""  # noqa: E731
        self.boxes = [g for g in range(self.m.ngeom) if name(g).startswith(f"{side}_") and name(g).endswith(("pad0", "pad1", "palm"))]

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def cam_pose(self, q):
        """The wrist camera's position and axes (columns x right, y up, looking along -z) at joints q."""
        self.fk(q)
        return self.d.cam_xpos[self.cam].copy(), self.d.cam_xmat[self.cam].reshape(3, 3).copy()

    def gravity(self, q):
        """Joint torques gravity puts on the arm at q (what the motors hold without feedforward)."""
        self.d.qpos[self.qadr] = q
        self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)
        return self.d.qfrc_bias[self.dofs].copy()

    def ik_one(self, pos, Rw, q0, iters=300):
        q = np.array(q0, dtype=float)
        for _ in range(iters):
            self.d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            R = self.d.site_xmat[self.site].reshape(3, 3)
            e_pos = pos - self.d.site_xpos[self.site]
            e_rot = 0.5 * sum(np.cross(R[:, k], Rw[:, k]) for k in range(3))
            if np.linalg.norm(e_pos) < 3e-4 and np.linalg.norm(e_rot) < 3e-3:
                # the cross-product error also vanishes at a half turn: only a near match counts
                return q, bool(np.trace(R.T @ Rw) > 2.99)
            jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
            J = np.vstack([jp[:, self.dofs], 0.3 * jr[:, self.dofs]])
            e = np.concatenate([e_pos, 0.3 * e_rot])
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e), self.lo, self.hi)
        return q, False

    def ik(self, pos, frames, q_ref, margin=0.05, extra=()):
        """The converged solution nearest q_ref over the frames given; solutions under margin rad from a joint limit
        cost extra. Returns (q, frame) or (None, None)."""
        frames = frames if isinstance(frames, list) else [frames]
        q_ref = np.asarray(q_ref, float)
        best = None
        for Rw in frames:
            for i, q0 in enumerate([q_ref] + list(extra) + self.seeds):
                q, ok = self.ik_one(np.asarray(pos, float), Rw, q0)
                if not ok:
                    continue
                inside = min(np.min(q - self.lo), np.min(self.hi - q))
                cost = np.abs(q - q_ref).sum() + (2.0 if inside < margin else 0.0)
                if best is None or cost < best[0]:
                    best = (cost, q, Rw)
                if i == 0 and inside >= margin:
                    break  # the present posture reaches it: stay on that branch
        return (best[1], best[2]) if best else (None, None)

    def margin(self, q):
        return float(min(np.min(q - self.lo), np.min(self.hi - q)))

    def lowest(self, q0, q1, n=12):
        """The lowest z of the hand's boxes (pads, palm) along the joint-space line q0 -> q1."""
        lo = 9.0
        for t in np.linspace(0, 1, n):
            self.d.qpos[self.qadr] = np.asarray(q0) + (np.asarray(q1) - np.asarray(q0)) * t
            mujoco.mj_kinematics(self.m, self.d)
            for g in self.boxes:
                R, c, h = self.d.geom_xmat[g].reshape(3, 3), self.d.geom_xpos[g], self.m.geom_size[g]
                lo = min(lo, c[2] - np.abs(R[2]) @ h)
        return lo


class SagModel:
    """Measured minus commanded joints under gravity: -alpha * gravity torque / kp per joint (kp from
    sim/servo_fit.json, alpha fitted on loaded poses). The pan and the wrist roll and yaw are left at zero."""

    def __init__(self, kin: Kin, alpha=None, path=ROOT / "calib" / "sag_left.json"):
        self.kin = kin
        self.kp = np.array(json.loads((ROOT / "sim" / "servo_fit.json").read_text())["params"]["kp"])
        if alpha is None:
            alpha = json.loads(pathlib.Path(path).read_text())["alpha"] if pathlib.Path(path).exists() else [0, 1, 1, 1, 0, 0]
        self.alpha = np.asarray(alpha, float)

    def __call__(self, q):
        return -self.alpha * self.kin.gravity(q) / self.kp

    def command(self, q_target):
        """What to command so the joints settle at q_target (the sag barely changes over the few hundredths of a
        radian it moves the arm)."""
        q = np.asarray(q_target, float)
        return q - self(q)


def fit_sag(kin: Kin) -> dict:
    """alpha per joint by least squares on the Nov 7 touch run's loaded poses (state q measured, goal commanded)."""
    kp = np.array(json.loads((ROOT / "sim" / "servo_fit.json").read_text())["params"]["kp"])
    rows = []
    for f in sorted(glob.glob(str(ROOT / "runs/station/calib2/run-*/*.json"))):
        j = json.loads(pathlib.Path(f).read_text())
        if not isinstance(j, dict) or "state" not in j or not j["state"].get("goal"):
            continue
        q, g = np.array(j["state"]["q"]), np.array(j["state"]["goal"])
        if np.abs(q - g).max() < 0.02:  # the folded rest pose leans on its stops: no sag there
            continue
        rows.append((pathlib.Path(f).name, q - g, -kin.gravity(q) / kp))
    meas, pred = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    alpha = np.zeros(6)
    for j in (1, 2, 3):
        alpha[j] = float(np.clip((meas[:, j] @ pred[:, j]) / max(pred[:, j] @ pred[:, j], 1e-9), 0, 2))
    resid = meas - alpha * pred
    return {"what": "measured minus commanded joints = -alpha * gravity torque / kp (sim Metal arm, kp from sim/servo_fit.json); "
                    "fitted by calib/tower_plan.py --fit-sag on the touch run's loaded poses",
            "alpha": alpha.round(3).tolist(), "poses": [r[0] for r in rows],
            "rms_rad_before": np.sqrt((meas ** 2).mean(0)).round(4).tolist(), "rms_rad_after": np.sqrt((resid ** 2).mean(0)).round(4).tolist(),
            "max_rad_after": np.abs(resid).max(0).round(4).tolist()}


class Shape:
    """A block as a box: extent along its yaw, across it, and up; its bottom's height. A flat block is
    (FACE, FACE, H); the standing third block (FACE, H, STAND_H), its yaw the normal of its 50 mm faces."""

    def __init__(self, dims, z0, period=math.pi / 2):
        self.dims, self.z0, self.period = tuple(dims), float(z0), period

    def corners(self, x, y, yaw):
        L, Wd, Hh = self.dims
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[x + a * c - b * s, y + a * s + b * c, z] for a in (-L / 2, L / 2) for b in (-Wd / 2, Wd / 2)
                         for z in (self.z0, self.z0 + Hh)])

    def fold(self, yaw):
        p = self.period
        return (yaw + p / 2) % p - p / 2


class WristView:
    """The left wrist camera as calibrated (calib/cam_wrist_l.json: mount in the tool frame, f, k1, k2 at 1280x720),
    at a working size of 640x360. The camera's pose comes from the arm's (model) joints, so everything it finds is
    relative to the hand. Finds a block of a given colour: colour mask (vision/locate.py's hue bins) minus the jaws
    (their CAD drawn through the camera), then x, y, yaw of the block's box whose silhouette best matches the blob
    (calib/wrist_overlay.py's fit, here with a starting guess)."""

    def __init__(self, kin: Kin, width=640, height=360, path=ROOT / "calib" / "cam_wrist_l.json"):
        c = json.loads(pathlib.Path(path).read_text())
        self.kin, self.W, self.H = kin, width, height
        self.Rm, self.tm = np.array(c["R_in_tcp"]), np.array(c["pos_in_tcp"])
        self.f = c["f"] * width / c["width"]
        self.k1, self.k2 = c["k1"], c["k2"]
        m = kin.m
        self.jaw_q = kin.s.jaw_qadr[kin.side]
        jaws = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{kin.side}_jaw{i}") for i in range(2)}
        self.meshes = []
        for g in range(m.ngeom):
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_bodyid[g] in jaws:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].copy()
                f = m.mesh_face[m.mesh_faceadr[mid]: m.mesh_faceadr[mid] + m.mesh_facenum[mid]].copy()
                self.meshes.append((g, v, f))

    def pose(self, q, opening=OPEN):
        d = self.kin.d
        d.qpos[self.kin.qadr] = q
        for a in self.jaw_q:
            d.qpos[a] = opening / 2
        mujoco.mj_kinematics(self.kin.m, d)
        Rt, pt = d.site_xmat[self.kin.site].reshape(3, 3), d.site_xpos[self.kin.site]
        return Rt @ self.Rm, pt + Rt @ self.tm

    def project(self, cam, P):
        Rc, pc = cam
        Pc = (np.asarray(P, float).reshape(-1, 3) - pc) @ Rc
        z = np.maximum(Pc[:, 2], 1e-6)
        x, y = Pc[:, 0] / z, Pc[:, 1] / z
        r2 = x * x + y * y
        dd = 1 + self.k1 * r2 + self.k2 * r2 * r2
        return np.stack([self.f * x * dd + self.W / 2, self.f * y * dd + self.H / 2], 1), Pc[:, 2] > 0.01

    def ray_on_plane(self, cam, uv, z):
        Rc, pc = cam
        uv = np.atleast_2d(np.asarray(uv, float))
        xd, yd = (uv[:, 0] - self.W / 2) / self.f, (uv[:, 1] - self.H / 2) / self.f
        x, y = xd.copy(), yd.copy()
        for _ in range(10):
            r2 = x * x + y * y
            dd = 1 + self.k1 * r2 + self.k2 * r2 * r2
            x, y = xd / dd, yd / dd
        r = np.stack([x, y, np.ones_like(x)], 1) @ Rc.T
        t = (z - pc[2]) / r[:, 2]
        return pc + t[:, None] * r

    def jaws_mask(self, q, opening):
        """The jaws' CAD (both fingers at this opening) drawn through the camera."""
        cam = self.pose(q, opening)
        d = self.kin.d
        mask = np.zeros((self.H, self.W), np.uint8)
        for g, v, f in self.meshes:
            pw = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            uv, ok = self.project(cam, pw)
            tri = f[ok[f].all(1)]
            pts = np.round(uv[tri]).astype(np.int32)
            pts = pts[(np.abs(pts) < 20000).all((1, 2))]
            for t in pts:
                cv2.fillConvexPoly(mask, t, 255)
        return mask

    def silhouette(self, cam, shape: Shape, x, y, yaw):
        uv, ok = self.project(cam, shape.corners(x, y, yaw))
        if not ok.all():
            return None
        return cv2.convexHull(np.round(uv).astype(np.int32))

    @staticmethod
    def colour_mask(bgr, colour, jaws):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        from vision.locate import HUES, SAT_MIN, VAL_MIN
        m = np.zeros(hsv.shape[:2], bool)
        for lo, hi in HUES[colour]:
            m |= (hsv[..., 0] >= lo) & (hsv[..., 0] < hi)
        m = (m & (hsv[..., 1] >= SAT_MIN) & (hsv[..., 2] >= VAL_MIN) & (jaws == 0)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    def find(self, bgr, colour, q, opening, shape: Shape, guess=None, min_px=150):
        """The block of this colour in this wrist frame: dict(x, y, yaw, iou, edge, px) in the arm frame, or None.
        guess (x, y, yaw) picks the blob nearest it and seeds the fit."""
        cam = self.pose(q, opening)
        jaws = self.jaws_mask(q, opening)
        m = self.colour_mask(bgr, colour, jaws)
        n, lab, stats, cen = cv2.connectedComponentsWithStats(m)
        comps = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px]
        if not comps:
            return None
        if guess is not None:
            g_uv, ok = self.project(cam, [[guess[0], guess[1], shape.z0 + shape.dims[2] / 2]])
            pick = min(comps, key=lambda i: np.linalg.norm(cen[i] - g_uv[0]) - 0.02 * math.sqrt(stats[i, cv2.CC_STAT_AREA]))
            return self._fit(cam, jaws, lab, stats, pick, shape, guess)
        # no guess: the three biggest blobs, the best fit inside the arm's workspace
        best = None
        for i in sorted(comps, key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:3]:
            f = self._fit(cam, jaws, lab, stats, i, shape, None)
            if f is None or not (WORKSPACE[0][0] < f["x"] < WORKSPACE[0][1] and WORKSPACE[1][0] < f["y"] < WORKSPACE[1][1]):
                continue
            if best is None or f["iou"] > best["iou"]:
                best = f
        return best

    def _fit(self, cam, jaws, lab, stats, pick, shape, guess):
        blob = (lab == pick).astype(np.uint8)
        cs, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(blob, cs, -1, 1, -1)  # the hole filled: the block's silhouette
        blob = blob > 0
        x0, y0, w0, h0 = stats[pick, :4]
        edge = x0 <= 1 or y0 <= 1 or x0 + w0 >= self.W - 2 or y0 + h0 >= self.H - 2
        ys, xs = np.nonzero(blob)
        P = self.ray_on_plane(cam, np.stack([xs, ys], 1)[:: max(1, len(xs) // 300)], shape.z0 + shape.dims[2] / 2)
        c = P[:, :2].mean(0)
        yaw0 = guess[2] if guess is not None else math.atan2(*np.linalg.svd(P[:, :2] - c, full_matrices=False)[2][0][::-1])
        pad = 40
        X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(self.W, x0 + w0 + pad), min(self.H, y0 + h0 + pad)
        target = blob[Y0:Y1, X0:X1]
        valid = jaws[Y0:Y1, X0:X1] == 0

        def cost(p):
            hull = self.silhouette(cam, shape, *p)
            if hull is None:
                return 1.0
            mm = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
            cv2.fillConvexPoly(mm, hull - [X0, Y0], 1)
            mm = (mm > 0) & valid
            return 1 - np.sum(mm & target) / max(np.sum((mm | target) & valid), 1)

        from scipy.optimize import minimize
        best = None
        for yaw in (yaw0, yaw0 + shape.period / 2):
            x_init = np.array([c[0], c[1], yaw])
            simplex = np.array([x_init, x_init + [0.008, 0, 0], x_init + [0, 0.008, 0], x_init + [0, 0, 0.25]])
            r = minimize(cost, x_init, method="Nelder-Mead", options={"xatol": 2e-4, "fatol": 1e-4, "initial_simplex": simplex, "maxiter": 300})
            if best is None or r.fun < best.fun:
                best = r
        x, y, yaw = best.x
        return dict(x=float(x), y=float(y), yaw=float(shape.fold(yaw)), iou=float(1 - best.fun), edge=bool(edge), px=int(blob.sum()))

    def overlay(self, bgr, q, opening, shape: Shape, fit=None, truth=None, label=""):
        """The wrist frame with the jaw CAD (cyan), the fitted block's outline (amber) and, in sim, the true one (white)."""
        img = bgr.copy()
        cam = self.pose(q, opening)
        cs, _ = cv2.findContours(self.jaws_mask(q, opening), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (255, 224, 63), 1, cv2.LINE_AA)
        if truth is not None:
            h = self.silhouette(cam, truth[0], *truth[1])
            if h is not None:
                cv2.polylines(img, [h], True, (255, 255, 255), 1, cv2.LINE_AA)
        if fit is not None:
            h = self.silhouette(cam, shape, fit["x"], fit["y"], fit["yaw"])
            if h is not None:
                cv2.polylines(img, [h], True, (40, 170, 245), 2, cv2.LINE_AA)
            label += f" iou {fit['iou']:.2f}"
        cv2.putText(img, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 170, 245), 1, cv2.LINE_AA)
        return img


def side_reach(kin: Kin, xy, tool_z, q_ref=REST, step_deg=10):
    """(margin, jaw yaw, sign, q) for each fingers-sideways frame that reaches the tool point at (xy, tool_z)."""
    out = []
    for k in range(0, 360, step_deg):
        yaw = math.radians(k - 180)
        for sign in (1, -1):
            q, _ = kin.ik([xy[0], xy[1], tool_z], frame_side(yaw, sign), q_ref)
            if q is not None:
                out.append((kin.margin(q), yaw, sign, q))
    return sorted(out, key=lambda r: -r[0])


def plan_tower(blocks, kin: Kin, desk_z: float, others=()) -> dict:
    """blocks: three dicts with "xy" (m) and "yaw" (rad), from the tracker or the sim. others: xy of anything else
    on the table to keep clear of. Returns the base spot and yaw and the block order (the flip spot: plan_flip)."""
    xs = [np.asarray(b["xy"], float) for b in blocks]
    clear = lambda p, skip=(): all(np.linalg.norm(xs[i] - p) > 0.10 for i in range(len(xs)) if i not in skip) and \
        all(np.linalg.norm(np.asarray(o) - p) > 0.10 for o in others)  # noqa: E731
    (x0, x1), (y0, y1) = BASE_ZONE
    stay = [i for i in range(3) if x0 <= xs[i][0] <= x1 and y0 <= xs[i][1] <= y1 and clear(xs[i], (i,))]
    if stay:
        b1 = min(stay, key=lambda i: np.linalg.norm(xs[i] - [0.33, 0.02]))
        spot, move = xs[b1], False
    else:
        spot = None
        for s in BASE_SPOTS:
            s = np.array(s)
            b1 = int(np.argmin([np.linalg.norm(x - s) for x in xs]))
            if clear(s, (b1,)):
                spot, move = s, True
                break
        if spot is None:
            raise RuntimeError("no clear base spot")
    rest = sorted([i for i in range(3) if i != b1], key=lambda i: np.linalg.norm(xs[i] - spot))
    b2, b3 = rest
    tower_yaw = fold(blocks[b1]["yaw"]) if not move else 0.0
    return dict(order=[b1, b2, b3], base=spot.tolist(), move_base=bool(move), tower_yaw=float(tower_yaw))


# The flip (third block onto its side), inside every limit (calib/tower_safety.py): the block lies flat at a flip spot
# far out (r about 0.44 m), a face square to the arm's radial line. The hand grips it from above, fingers down, the
# jaws across the two faces along that line's normal, the pads on their upper part (fingertips FLIP_TIP over the
# table), the grasp FLIP_G further out than the block's centre, the camera on the outer side. Then the block is tipped
# toward the arm about its inner bottom edge: the hand turns with it about the jaw axis to FLIP_HAND deg (the wrist's
# flex limit less the 0.3 rad margin stops it near there), then keeps that angle while the pads drag the top on round
# to FLIP_TO deg (the gentle hold lets the block turn in the pads); past 55 deg its weight is over the edge, so when
# the jaws open it lies down on its side. In sim: tipped 59 to 62 deg in the pads, it lands standing every time.
FLIP_R = (0.44, 0.43, 0.42)
FLIP_ANG = (0, -10, 10, -20, 20, -30, 30)  # deg, the flip spot's direction from the base
FLIP_TIP, FLIP_G, FLIP_HAND, FLIP_TO = 0.019, 0.0, 57.0, 84.0
FLIP_MUST = 66.0  # the drag must reach this far (its weight is over the edge by about 59)
FLIP_PRESS = 0.006
FLIP_SHIFT = FACE / 2 + H / 2  # the standing block's centre ends this far in from the flat block's centre


def _rot(a, phi):
    a = np.asarray(a, float) / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(phi) * K + (1 - math.cos(phi)) * K @ K


def flip_poses(kin: Kin, desk_z, centre_xy, d, tip=None, press=None):
    """The hand's poses (position, frame, tag) for the flip of a flat block centred at centre_xy, tipped toward -d.
    tip, press: the fingertips' height over the table and how far under the table the turn aims (FLIP_TIP, FLIP_PRESS
    unless given: a retry grips lower and presses harder)."""
    tip = FLIP_TIP if tip is None else tip
    press = FLIP_PRESS if press is None else press
    d = np.array([d[0], d[1], 0.0]) / np.linalg.norm(d[:2])
    R0 = frame_down(math.atan2(d[1], d[0]) + math.pi / 2)  # the across axis (camera side) along +d
    tool0 = np.array([centre_xy[0], centre_xy[1], desk_z + tip + kin.tip]) + FLIP_G * d
    E = np.array([centre_xy[0], centre_xy[1], desk_z - press]) - d * FACE / 2  # the inner bottom edge, aimed a
    # little under the table so the turn presses the edge down and it pivots rather than slides
    ax = np.cross([0, 0, 1.0], -d)
    out = [(tool0 + [0, 0, 0.05], R0, "flip over"), (tool0, R0, "flip down")]
    for deg in np.linspace(0, FLIP_TO, 16)[1:]:
        Rf, Rh = _rot(ax, math.radians(deg)), _rot(ax, math.radians(min(deg, FLIP_HAND)))
        out.append((E + Rf @ (tool0 - E), Rh @ R0, f"flip {deg:.0f}"))
    return out


def plan_flip(kin: Kin, desk_z, tower_xy, others=(), safety=None, exclude=()):
    """A flip spot: clear of the tower and anything else (where the block lies, where it ends, the hand between), every
    flip pose reachable and inside the limits. Returns dict(flip=xy, flip_d=d)."""
    tower_xy = np.asarray(tower_xy, float)
    for r in FLIP_R:
        for ang in FLIP_ANG:
            d = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
            F = r * d
            if any(np.linalg.norm(F - np.asarray(e)) < 0.03 for e in exclude):
                continue  # a spot where a flip already failed
            end = F - FLIP_SHIFT * d
            # the tower clear of where the block lies and ends, and of the hand's sweep in toward the arm
            seg = [F - t * d for t in np.linspace(0, 0.17, 9)]
            if min(np.linalg.norm(p - tower_xy) for p in seg) < 0.10:
                continue
            if any(min(np.linalg.norm(F - o), np.linalg.norm(end - o)) < 0.09 for o in others):
                continue
            q, ok = REST.copy(), True
            for pos, R, tag in flip_poses(kin, desk_z, F, d):
                q1, _ = kin.ik(pos, [R], q)
                if q1 is None or (safety is not None and not safety.check_pose(q1)[0]):
                    deg = float(tag.split()[-1]) if tag.split()[-1].replace(".", "").isdigit() else 0.0
                    ok = deg > FLIP_MUST  # the end of the drag may be cut short (it lets go from there)
                    break
                q = q1
            if ok:
                return dict(flip=F.tolist(), flip_d=d.tolist())
    raise RuntimeError("no flip spot inside the limits")


class Stop(Exception):
    """A stage found something not as planned: stop rather than go on blindly."""


WORKSPACE = ((0.15, 0.50), (-0.25, 0.30))  # where blocks can be (x, y), m
SCAN_VIEWS = [(0.27, 0.02), (0.27, 0.14), (0.27, -0.10)]  # where the hand looks down from, high, to find the blocks
SCAN_TIP_Z = 0.20  # fingertips this high over the table for the scan


class Builder:
    """Runs the tower on a robot, seeing the blocks only through the wrist camera. The robot gives:
    move(q_cmd, opening, sec), q() (joints, the arm's own model), wrist_image() (a fresh 640x360 BGR wrist frame),
    snap(tag). Every command goes through the sag model.

    Look first (the hand high over a few fixed spots, the blocks found by colour), then for each grasp or place:
    hover over the block's estimate with the jaws squared to its estimated yaw, fit x, y, yaw from the wrist frame,
    move there, again until the fix stops moving (centred between the jaws, square to its faces); then descend."""

    def __init__(self, robot, kin: Kin, sag: SagModel | None, view: WristView | None, desk_z: float, log=print, safety=None):
        self.r, self.kin, self.sag, self.view, self.desk, self.log = robot, kin, sag, view, desk_z, log
        self.safety = safety  # calib/tower_safety.TowerSafety: every pose and every path checked before it is sent
        self.unsafe = []
        self.q = np.asarray(robot.q(), float)
        self.opening = OPEN
        self.looks = []  # every look: tag, fit, the frame with the overlay (kept for the last look of each)
        self.sheet = []  # (tag, overlay image) of each look's final frame
        self.recent = []
        self.others = []  # xy of anything on the table besides the tower and the third block
        self.approach_error = None  # demos: (rng, probability, metres) of arriving off target at a look

    # motion
    def go(self, q, opening=None, sec=1.0, settle=0.3, home=False):
        opening = self.opening if opening is None else opening
        if self.safety is not None and np.abs(np.asarray(q) - self.q).max() > 1e-6:
            ok, why = self.safety.check_path(self.q, q, min(opening, self.opening), home=home)
            if not ok:
                self.unsafe.append(why)
                raise Stop(f"unsafe path: {why}")
        cmd = self.sag.command(q) if self.sag is not None else np.asarray(q)
        self.r.move(cmd, opening, sec, settle)
        self.q, self.opening = np.asarray(q, float), opening

    def allowed(self, frames, xy):
        """Fingers-down frames whose camera faces outward (calib/tower_safety.py rule 2); others as given."""
        if self.safety is None:
            return frames
        from calib.tower_safety import DOWN_MIN, outward_ok
        return [R for R in frames if R[:, 2] @ np.array([0, 0, -1.0]) < DOWN_MIN or outward_ok(R, xy)]

    def solve(self, pos, frames, what):
        frames = self.allowed(frames if isinstance(frames, list) else [frames], pos[:2])
        cands = []
        for R in frames:
            q, _ = self.kin.ik(pos, [R], self.q, extra=self.recent[-6:])
            if q is not None:
                cands.append((float(np.abs(q - self.q).sum()), q, R))
        why = "no IK"
        for _, q, R in sorted(cands, key=lambda c: c[0]):
            if self.safety is not None:
                ok, why = self.safety.check_pose(q, self.opening)
                if not ok:
                    continue
            self.recent.append(q)
            return q, R
        raise Stop(f"{what} at {np.round(np.asarray(pos) * 1000).tolist()} mm: {why}")

    def to(self, pos, frames, what, opening=None, sec=1.0, settle=0.3):
        q, R = self.solve(pos, frames, what)
        self.go(q, opening, sec, settle)
        return R

    def hand_xy(self):
        return self.kin.fk(self.q)[0][:2]

    def tool_z(self, tip_z):
        return tip_z + self.kin.tip

    def see(self, colour, shape, guess, tag):
        """One wrist frame at the present pose, the block fitted; the overlay kept."""
        q = np.asarray(self.r.q(), float)
        img = self.r.wrist_image()
        fit = self.view.find(img, colour, q, self.opening, shape, guess)
        truth = self.r.truth(colour) if hasattr(self.r, "truth") else None
        return fit, (tag, self.view.overlay(img, q, self.opening, shape, fit, truth, tag))

    def hole_fill(self, colour, fit, top_z, r=0.007):
        """The share of the block's colour in a 7 mm disc at the middle of a flat fit's top face (a fresh frame at
        the present pose): about 0 over the hole of a block lying flat, about 1 on a block standing on its side."""
        if self.view is None:
            return None
        q = np.asarray(self.r.q(), float)
        img = self.r.wrist_image()
        cam = self.view.pose(q, self.opening)
        ring = [[fit["x"] + r * math.cos(a), fit["y"] + r * math.sin(a), top_z] for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
        uv, ok = self.view.project(cam, ring)
        if not ok.all():
            return None
        m = np.zeros((self.view.H, self.view.W), np.uint8)
        cv2.fillConvexPoly(m, np.round(uv).astype(np.int32), 1)
        if m.sum() < 20:
            return None
        col = self.view.colour_mask(img, colour, self.view.jaws_mask(q, self.opening))
        return float((col[m > 0] > 0).mean())

    # the wrist look
    def look(self, colour, shape: Shape, guess, tip_z, tag, tries=5, tol=0.0015, tol_yaw=math.radians(2)):
        """Hover over guess (x, y, yaw) with the jaws squared to it, fit the block in the wrist frame, move over the
        fit, again until it stops moving. Returns (x, y, yaw) in the arm frame."""
        x, y, yaw = guess
        lifted = False
        last = None
        for i in range(tries + (1 if self.approach_error else 0)):
            hover = np.array([x, y])
            if i == 0 and self.approach_error is not None:  # demos: arrive off target now and then, so they show the fix
                rng, prob, mag = self.approach_error
                if rng.random() < prob:
                    a = rng.uniform(0, 2 * math.pi)
                    hover = hover + rng.uniform(0.5, 1.0) * mag * np.array([math.cos(a), math.sin(a)])
            frames = [frame_down(yaw + k * shape.period) for k in range(int(round(2 * math.pi / shape.period)))]
            for pull in range(4):  # out of reach: look from a little closer in (the camera still sees past the hand)
                try:
                    self.to([hover[0], hover[1], self.tool_z(tip_z)], frames, f"look {tag}", sec=0.8, settle=0.4)
                    break
                except Stop:
                    if pull == 3:
                        raise
                    hover = hover - 0.02 * hover / np.linalg.norm(hover)
            if self.view is None:
                return x, y, yaw
            fit, ov = self.see(colour, shape, (x, y, yaw), f"{tag} {i}")
            last = ov
            if fit is None or fit["iou"] < 0.5:
                if not lifted:  # not seen: look again from 6 cm higher (a wider view)
                    lifted, tip_z = True, tip_z + 0.06
                    continue
                self.sheet.append(ov)
                raise Stop(f"{tag}: the {colour} block is not seen by the wrist camera")
            step = np.array([fit["x"], fit["y"]]) - hover
            dyaw = shape.fold(fit["yaw"] - yaw)
            if np.linalg.norm(step) > 0.06:  # a far fix from one view: go part way and look again
                step *= 0.04 / np.linalg.norm(step)
            rec = dict(tag=tag, look=i, step_mm=round(float(np.linalg.norm(step)) * 1000, 1),
                       dyaw_deg=round(math.degrees(dyaw), 1), iou=round(fit["iou"], 2), edge=fit["edge"])
            if hasattr(self.r, "truth"):  # sim only: how far the fit is from the truth
                _, (tx, ty, tyaw) = self.r.truth(colour)
                rec["err_mm"] = round(math.hypot(fit["x"] - tx, fit["y"] - ty) * 1000, 1)
                rec["err_yaw_deg"] = round(math.degrees(shape.fold(fit["yaw"] - tyaw)), 1)
            self.looks.append(rec)
            x, y, yaw = hover[0] + step[0], hover[1] + step[1], yaw + dyaw
            if np.linalg.norm(step) < tol and abs(dyaw) < tol_yaw and not fit["edge"]:
                break
        self.sheet.append(last)
        return x, y, yaw

    def scan(self, colours):
        """Look first: the hand high over a few fixed spots, fingers down; each colour's block fitted where it is
        seen best (the most pixels, not cut by the frame's edge). A rough start for the looks."""
        flat = Shape((FACE, FACE, H), self.desk)
        best = {}
        for vx, vy in SCAN_VIEWS:
            try:
                self.to([vx, vy, self.tool_z(self.desk + SCAN_TIP_Z)], squares(0.0), "scan", sec=1.2, settle=0.4)
            except Stop:
                continue
            q = np.asarray(self.r.q(), float)
            img = self.r.wrist_image()
            for c in colours:
                fit = self.view.find(img, c, q, self.opening, flat)
                if fit is None or fit["iou"] < 0.5:
                    continue
                score = fit["px"] * (0.3 if fit["edge"] else 1.0)
                if c not in best or score > best[c][0]:
                    best[c] = (score, fit)
            if all(c in best and not best[c][1]["edge"] for c in colours):
                break
        missing = [c for c in colours if c not in best]
        if missing:
            raise Stop(f"scan: no {', '.join(missing)} block seen")
        return [{"xy": [best[c][1]["x"], best[c][1]["y"]], "yaw": best[c][1]["yaw"], "colour": c} for c in colours]

    # grasps
    def grasp(self, colour, shape: Shape, guess, widths, offset=0.0, tag="", hold_period=None):
        """Look, then the double squeeze and hold. widths: the block's width along its yaw and across it (the first
        squeeze is across, the hold along). hold_period: the hold's jaw directions repeat every this (default the
        shape's). Returns the hold frame, the block centre in the tool frame, and the block's (x, y, yaw)."""
        surface, top = shape.z0, shape.z0 + shape.dims[2]
        if hasattr(self.r, "before_grasp"):
            self.r.before_grasp(colour)  # demos: the sim may nudge the block now; the look finds it again
        x, y, yaw = self.look(colour, shape, guess, top + HOVER, f"{tag} look")
        grip_z = self.tool_z(surface + PICK_TIP)
        period = hold_period or shape.period
        hold = [frame_down(yaw + k * period) for k in range(int(round(2 * math.pi / period)))]
        q_hold, R_hold = self.solve([x, y, grip_z], hold, f"{tag} hold")
        first = [frame_down(yaw_of(R_hold) + s * math.pi / 2) for s in (1, -1)]
        R1 = self.to([x, y, self.tool_z(top + HOVER)], first, f"{tag} over", sec=0.8)
        self.to([x, y, grip_z], [R1], f"{tag} down", opening=OPEN, sec=0.8)
        self.go(self.q, widths[1] - SQUEEZE_IN, 0.6, 0.4)
        self.go(self.q, OPEN, 0.4, 0.2)
        self.to([x, y, self.tool_z(top + 0.01)], [R1], f"{tag} up", sec=0.5)
        p = np.array([x, y, 0.0]) - offset * R_hold[:, 1] * np.array([1, 1, 0])  # the hold, offset along the pads
        self.to([p[0], p[1], self.tool_z(top + 0.01)], [R_hold], f"{tag} over 2", sec=0.6)
        self.to([p[0], p[1], grip_z], [R_hold], f"{tag} down 2", sec=0.6)
        self.go(self.q, widths[0] - SQUEEZE_IN, 0.6, 0.4)
        self.r.snap(f"{tag}-held")
        centre_in_tool = R_hold.T @ (np.array([x, y, surface + shape.dims[2] / 2]) - np.array([p[0], p[1], grip_z]))
        return R_hold, centre_in_tool, (x, y, yaw)

    def lift_to(self, xy, bottom_z, frames, tag, sec=1.0):
        """Carry: the held block's bottom at bottom_z (the hold's fingertips PICK_TIP over it)."""
        return self.to([xy[0], xy[1], self.tool_z(bottom_z + PICK_TIP)], frames, tag, sec=sec)

    def place(self, target, under_top, height, tag):
        """Set the held block (fingers down) centred on target (x, y, yaw), squared to it; let go; up."""
        x, y, yaw = target
        R = self.lift_to((x, y), self.desk + 2 * H + CLEAR, squares(yaw), f"{tag} over", sec=1.4)
        self.lift_to((x, y), under_top + DROP, [R], f"{tag} set", sec=0.9)
        self.go(self.q, OPEN, 0.4, 0.4)
        self.to([x, y, self.tool_z(under_top + height + 0.04)], [R], f"{tag} up", sec=0.6)
        self.r.snap(f"{tag}-placed")

    # the tower
    def run(self, colours, plan=None, stages=("base", "b2", "b3side", "b3top"), blocks=None):
        """colours: the three blocks' colours. Scans (unless blocks, rough (xy, yaw) guesses, are given), plans
        (plan_tower) unless a plan is given, builds. Returns what it found, and the plan it used under "plan"."""
        d = self.desk
        flat0, flat1 = Shape((FACE, FACE, H), d), Shape((FACE, FACE, H), d + H)
        self.go(REST, OPEN, 1.0, home=True)
        out = {}
        blocks = blocks or self.scan(colours)
        out["scanned"] = blocks
        if plan is None:
            plan = plan_tower(blocks, self.kin, d)
        out["plan"] = self.last_plan = plan
        b1, b2, b3 = plan["order"]
        c1, c2, c3 = colours[b1], colours[b2], colours[b3]
        g = lambda i: (*blocks[i]["xy"], blocks[i]["yaw"])  # noqa: E731
        B = np.asarray(plan["base"], float)
        Y = plan["tower_yaw"]
        if "base" in stages:
            if plan["move_base"]:
                self.grasp(c1, flat0, g(b1), (FACE, FACE), tag="b1")
                self.lift_to(self.hand_xy(), d + 2 * H + CLEAR, [self.kin.fk(self.q)[1]], "b1 lift", sec=0.8)
                self.place((B[0], B[1], Y), d, H, "b1")
                guess = (B[0], B[1], Y)
            else:  # squeeze it square where it is
                Rh, _, guess = self.grasp(c1, flat0, g(b1), (FACE, FACE), tag="b1")
                self.go(self.q, OPEN, 0.4, 0.3)
                self.to([guess[0], guess[1], self.tool_z(d + H + 0.04)], [Rh], "b1 up", sec=0.6)
            # where the hand set it is where the next block goes (the squeeze centred it under the jaws there); the
            # wrist look checks it, and its reading there is the reference later looks are compared with (the camera's
            # small mount and joint errors are then the same in both and cancel)
            out["base_cmd"] = guess
            T1 = self.look(c1, flat0, guess, d + H + HOVER + 0.02, "b1 check")
            out["base"] = T1
            if np.linalg.norm(np.subtract(T1[:2], guess[:2])) > 0.012:
                raise Stop(f"block 1 reads {np.linalg.norm(np.subtract(T1[:2], guess[:2])) * 1000:.0f} mm from where it was set")
        base_cmd = out.get("base_cmd", (B[0], B[1], Y))
        T1 = out.get("base", base_cmd)
        if "b2" in stages:
            self.grasp(c2, flat0, g(b2), (FACE, FACE), tag="b2")
            self.lift_to(self.hand_xy(), d + 2 * H + CLEAR, [self.kin.fk(self.q)[1]], "b2 lift", sec=0.6)
            self.place(base_cmd, d + H, H, "b2")
            T2 = self.look(c2, flat1, T1, d + 2 * H + HOVER + 0.02, "b2 check")
            out["b2"] = T2
            if np.linalg.norm(np.subtract(T2[:2], T1[:2])) > 0.008:
                raise Stop(f"block 2 is {np.linalg.norm(np.subtract(T2[:2], T1[:2])) * 1000:.0f} mm off block 1")
        T2 = out.get("b2", T1)
        if "b3side" in stages:
            fl = plan_flip(self.kin, d, T2[:2], others=self.others, safety=self.safety)
            plan.update(fl)
            out["flip"] = fl["flip"]
            F, dv = np.asarray(fl["flip"]), np.asarray(fl["flip_d"])
            ang = math.atan2(dv[1], dv[0])
            self.grasp(c3, flat0, g(b3), (FACE, FACE), tag="b3")
            self.lift_to(self.hand_xy(), d + 2 * H + CLEAR, [self.kin.fk(self.q)[1]], "b3 lift", sec=0.8)
            self.place((F[0], F[1], ang), d, H, "b3 flat")  # flat at the flip spot, a face square to the radial line
            # where the hand set it is where it is (the squeeze centred and squared it there); the look is the reference
            # a later look is compared with, as for the tower
            ref = self.look(c3, flat0, (F[0], F[1], ang), d + H + HOVER + 0.02, "b3 flip look")
            if np.linalg.norm(np.subtract(ref[:2], F)) > 0.012:
                raise Stop(f"the third block reads {np.linalg.norm(np.subtract(ref[:2], F)) * 1000:.0f} mm from where it was set")
            x, y, yaw = F[0], F[1], ang
            stand = Shape((FACE, H, STAND_H), d, period=math.pi)
            tried = []
            for attempt in range(3):
                yd = yaw + round((ang - yaw) / (math.pi / 2)) * math.pi / 2  # its face normal nearest the radial line
                dv = np.array([math.cos(yd), math.sin(yd)])
                fp = flip_poses(self.kin, d, (x, y), dv, tip=FLIP_TIP - 0.003 * attempt, press=FLIP_PRESS + 0.006 * attempt)
                for i, (pos, R, tag) in enumerate(fp):
                    try:
                        self.to(pos, [R], tag, sec=0.8 if i < 2 else 0.35, settle=0.3 if i < 2 else 0.05)
                    except Stop:
                        if float(tag.split()[-1]) <= 66 if tag.startswith("flip ") and tag[5:].replace(".", "").isdigit() else True:
                            raise
                        break  # the last of the drag out of reach here: let go from where it got to (the check below)
                    if i == 1:
                        self.go(self.q, FACE - SQUEEZE_IN, 0.6, 0.4)  # the gentle hold
                self.go(self.q, OPEN, 0.4, 0.6)  # let go: past its edge, it lies down on its side
                self.r.snap("b3-flipped")
                pos, R = self.kin.fk(self.q)
                self.to(pos - 0.04 * R[:, 2], [R], "flip back", sec=0.5)
                self.to(pos - 0.04 * R[:, 2] + [0, 0, 0.06], [R], "flip up", sec=0.5)
                endc = np.array([x, y]) - FLIP_SHIFT * dv
                if attempt == 0:
                    self.stand_target = endc.copy()
                tang = yd + math.pi / 2  # the standing block's 50 mm faces face along the jaw axis of the flip
                self.to([endc[0], endc[1], self.tool_z(d + STAND_H + HOVER + 0.02)], [frame_down(tang + k * math.pi / 2) for k in range(4)],
                        "b3 above", sec=1.0)
                if self.view is None:
                    break
                # did it go over? fitted as standing and as lying flat, hovering over the better fit until one fits
                # clearly better (a block half out of view fits both badly)
                over = None
                gx, gy = endc
                for k in range(3):
                    if k:
                        self.to([gx, gy, self.tool_z(d + STAND_H + HOVER + 0.02)], [frame_down(tang + j * math.pi / 2) for j in range(4)],
                                "flip check hover", sec=0.6, settle=0.3)
                    fs, ov = self.see(c3, stand, (gx, gy, tang), f"flip check {attempt}.{k}")
                    ff, ovf = self.see(c3, flat0, (gx, gy, yaw), f"flat check {attempt}.{k}")
                    i_s, i_f = (fs or {}).get("iou", 0.0), (ff or {}).get("iou", 0.0)
                    best = fs if i_s >= i_f else ff
                    if best is None:
                        break
                    gx, gy = best["x"], best["y"]
                    if max(i_s, i_f) > 0.9 and abs(i_s - i_f) > 0.02:
                        over = i_s > i_f
                        self.sheet.append(ov if over else ovf)
                        break
                hole = None
                if over is None and ff is not None:
                    # the silhouettes alone are unsure: a block lying flat shows its hole (dark, not its colour) in the
                    # middle of its top; one on its side shows solid foam there
                    hole = self.hole_fill(c3, ff, d + H)
                    over = None if hole is None or 0.35 <= hole <= 0.85 else hole > 0.85
                self.looks.append(dict(tag=f"flip check {attempt}", over=over, iou_stand=round(i_s, 2), iou_flat=round(i_f, 2),
                                       hole_fill=None if hole is None else round(hole, 2)))
                if over:
                    endc, tang = np.array([fs["x"], fs["y"]]), fs["yaw"]
                    break
                if over is None:
                    raise Stop("after the flip the wrist camera cannot tell whether the third block went over")
                now = self.look(c3, flat0, (ff["x"], ff["y"], ff["yaw"]), d + H + HOVER + 0.02, f"b3 flip again {attempt}")
                if np.linalg.norm(np.subtract(now[:2], ref[:2])) > 0.008 or abs(flat0.fold(now[2] - ref[2])) > math.radians(5):
                    # it slid or turned: put it back on the flip spot (planned inside the limits) rather than tip it
                    # from wherever it ended
                    self.grasp(c3, flat0, now, (FACE, FACE), tag=f"b3 reset {attempt}")
                    self.lift_to(self.hand_xy(), d + H + CLEAR, [self.kin.fk(self.q)[1]], "b3 reset lift", sec=0.6)
                    tried.append(F.copy())
                    try:  # another flip spot: the same one tends to fail the same way
                        fl = plan_flip(self.kin, d, T2[:2], others=self.others, safety=self.safety, exclude=tried)
                        F, dv = np.asarray(fl["flip"]), np.asarray(fl["flip_d"])
                        ang = math.atan2(dv[1], dv[0])
                    except RuntimeError:
                        pass
                    self.place((F[0], F[1], ang), d, H, f"b3 reset {attempt}")
                    ref = self.look(c3, flat0, (F[0], F[1], ang), d + H + HOVER + 0.02, f"b3 reset look {attempt}")
                    x, y, yaw = F[0], F[1], ang
                else:
                    x, y, yaw = F[0] + now[0] - ref[0], F[1] + now[1] - ref[1], ang + flat0.fold(now[2] - ref[2])
            else:
                raise Stop("the third block would not go over onto its side")
            out["flip_attempts"] = attempt + 1
            out["stand"] = (float(endc[0]), float(endc[1]), float(tang))
        if "b3top" in stages:
            stand = Shape((FACE, H, STAND_H), d, period=math.pi)
            # the double squeeze: first across the hole faces (34.5 mm), the hold across the 50 mm faces
            # the tower's top again, hand empty, against its reading when block 2 was set: where it is now
            T2now = self.look(c2, flat1, T2, d + 2 * H + HOVER + 0.02, "b2 again")
            out["b2_again"] = T2now
            target = (base_cmd[0] + T2now[0] - T2[0], base_cmd[1] + T2now[1] - T2[1], base_cmd[2] + flat1.fold(T2now[2] - T2[2]))
            R_hold, _, (x, y, yaw) = self.grasp(c3, stand, out["stand"], (FACE, H), tag="b3s", hold_period=math.pi)
            self.lift_to((x, y), d + STAND_H + CLEAR, [R_hold], "b3s lift", sec=0.6)
            self.lift_to(target[:2], d + 2 * H + CLEAR + 0.01, squares(target[2]), "b3s carry", sec=1.4)
            self.place(target, d + 2 * H, STAND_H, "b3s")
            top = Shape((FACE, H, STAND_H), d + 2 * H, period=math.pi)
            out["b3"] = self.look(c3, top, (T2now[0], T2now[1], yaw_of(self.kin.fk(self.q)[1])), d + 2 * H + STAND_H + HOVER, "b3 check")
        self.go(REST, OPEN, 1.2, 0.5)
        return out


if __name__ == "__main__":
    if "--fit-sag" in sys.argv:
        res = fit_sag(Kin())
        (ROOT / "calib" / "sag_left.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
