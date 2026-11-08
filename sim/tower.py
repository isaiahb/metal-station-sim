"""The tallest tower in the CPU MuJoCo sim: calib/tower_plan.py's Builder on the sim's Metal arm (our left arm).

Three foam blocks (green, blue and purple, as at the station) at random places and yaws. Block 1 stays (squeezed square)
or is moved to the base spot, block 2 goes flat on it, block 3 is tipped onto its side at a flip spot and then set on
top (see calib/tower_plan.py for how). Every pose and path passes calib/tower_safety.py's limits (a stop otherwise);
the sim's table is the real one's (its top 8 mm under the base, its near edge just behind the base). Success: each block's centre within 5 mm (xy) of the one below, resting on it,
block 3 standing (its hole axis within 5 deg of horizontal), after the arm lets go and goes back to rest.

The Builder sees the blocks only through the wrist camera: the sim renders the wrist view through the calibrated
mount and lens (calib/cam_wrist_l.json: the camera placed on the flange where the fit puts it, its field of view from
f, the k1/k2 distortion applied to the render), at 640x360, and the Builder finds blocks by colour in that image,
exactly as on the real arm. Nothing reads the blocks' true poses but the score and the white outlines on the sheets.
Every mode: the joints sag under gravity as the fitted model says (the real driver has no gravity feedforward; the
Builder commands through the model). Modes:
  true    a clean render, the arm and camera exactly as modelled.
  noise   pixel noise (sigma 6 of 255), brightness and colour jitter per frame, and the camera mount off by up to
          2 mm and 1 deg from the calibration.
  real    noise, plus the sag 0.8 to 1.2 times the model's and joint zero errors up to 0.75 deg on pan, lift, elbow and
          wrist flex (the calibration's residual is about 0.4 deg).
  nosag   real, without the sag compensation.

    python -m sim.tower [mode] [trials] [sheet prefix]    # sheets: <prefix>-scene.jpg (3D) and <prefix>-wrist.jpg
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import cv2
import mujoco
import numpy as np

from calib.tower_plan import FACE, H, OPEN, STAND_H, Builder, Kin, SagModel, Shape, Stop, WristView
from calib.tower_safety import TowerSafety
from sim.physics import ARMS, Scene

ROOT = pathlib.Path(__file__).resolve().parent.parent
A = ARMS["metal"]
SIDE = "right"
COLOURS = ["green", "blue", "purple"]
RGBA = {"green": (0.36, 0.72, 0.38, 1), "blue": (0.22, 0.52, 0.82, 1), "purple": (0.80, 0.25, 0.60, 1)}  # hues in locate.HUES' bins
W, Hpx = 640, 360


def quat_of(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


class SimRobot:
    """The sim arm as the Builder's robot. The plant: servo target = command + true sag(command) + joint zero error;
    the encoders read the joints minus the zero error (the arm's belief). The wrist camera: the calibrated mount, moved
    by mount_err (metres, radians) if given."""

    def __init__(self, props, sag_true=None, dq=None, record=False, noise=None, mount_err=None, rng=None):
        self.s = Scene("metal", "right", props=props, draw=True)
        self.m, self.d = self.s.model, self.s.data
        self.sag_true, self.dq = sag_true, np.zeros(6) if dq is None else np.asarray(dq, float)
        self.rng = rng or np.random.default_rng(0)
        self.noise = noise
        for k, p in enumerate(self.s.props):
            for g in range(self.m.ngeom):  # the drawn mesh only (the contact boxes stay hidden)
                if self.m.geom_bodyid[g] == p["body"] and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    self.m.geom_rgba[g] = RGBA[COLOURS[k]]
        # a room's diffuse light: the blocks' side faces show their colour, as in the real wrist frames (with the
        # scene's lights alone the far faces render near black and the colour mask loses them)
        self.m.vis.headlight.ambient[:] = [0.45] * 3
        self.m.vis.headlight.diffuse[:] = [0.35] * 3
        self.m.vis.headlight.specular[:] = [0.05] * 3
        self.cam_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, f"{SIDE}_wrist")
        self._place_camera(mount_err)
        self._real_table()
        self.s.step(250)
        self.ctrl = np.array(A["start"], float)
        self.grip = OPEN / A["stroke"]
        self.wr = mujoco.Renderer(self.m, Hpx, W)
        self.record = record
        self.frames = []
        if record:
            self.vr = mujoco.Renderer(self.m, 360, 480)
            self.vc = mujoco.MjvCamera()
            self.vc.lookat[:] = [0.30, 0.0, self.s.desk_z + 0.05]
            self.vc.distance, self.vc.azimuth, self.vc.elevation = 0.75, 205, -28

    def _place_camera(self, err):
        """The sim's wrist camera where calib/cam_wrist_l.json puts it on the hand, its lens as a pinhole with the
        fitted f (the k1/k2 distortion is applied to each render)."""
        c = json.loads((ROOT / "calib" / "cam_wrist_l.json").read_text())
        R_cv, t = np.array(c["R_in_tcp"]), np.array(c["pos_in_tcp"])
        if err is not None:
            dt, dr = err
            t = t + dt
            rv = np.asarray(dr, float)
            R_cv = cv2.Rodrigues(rv)[0] @ R_cv
        sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, f"{SIDE}_tcp")
        assert self.m.site_bodyid[sid] == self.m.cam_bodyid[self.cam_id]
        Rs = np.zeros(9)
        mujoco.mju_quat2Mat(Rs, self.m.site_quat[sid])
        Rs = Rs.reshape(3, 3)
        R_mj = Rs @ R_cv @ np.diag([1.0, -1.0, -1.0])  # OpenCV axes (x right, y down, z ahead) to MuJoCo's (y up, z back)
        self.m.cam_pos[self.cam_id] = self.m.site_pos[sid] + Rs @ t
        self.m.cam_quat[self.cam_id] = quat_of(R_mj)
        f = c["f"] * W / c["width"]
        self.m.cam_fovy[self.cam_id] = math.degrees(2 * math.atan(Hpx / 2 / f))
        # the distortion as a remap: each output pixel's source in the pinhole render
        k1, k2 = c["k1"], c["k2"]
        u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(Hpx, dtype=np.float32))
        xd, yd = (u - W / 2) / f, (v - Hpx / 2) / f
        x, y = xd.copy(), yd.copy()
        for _ in range(10):
            r2 = x * x + y * y
            dd = 1 + k1 * r2 + k2 * r2 * r2
            x, y = xd / dd, yd / dd
        self.map = (x * f + W / 2).astype(np.float32), (y * f + Hpx / 2).astype(np.float32)

    def _real_table(self):
        """The real table (calib/tower_safety.TABLE): its top 8 mm under the base (the base plate on it), its near edge
        just behind the base, 25 mm thick; the blocks settle onto it."""
        from calib.tower_safety import TABLE, TABLE_Z
        self.table_z = self.s.desk_z + TABLE_Z
        (x0, x1), (y0, y1), t = TABLE["x"], TABLE["y"], TABLE["thick"]
        for name, z_top, half_t in (("desk", self.table_z, t / 2), ("paper", self.table_z + 0.001, 0.0005)):
            g = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, name)
            if name == "desk":
                self.m.geom_size[g] = [(x1 - x0) / 2, (y1 - y0) / 2, half_t]
                self.m.geom_pos[g] = [(x0 + x1) / 2, (y0 + y1) / 2, z_top - half_t]
            else:
                self.m.geom_pos[g][2] = z_top - half_t
        for p in self.s.props:  # down onto it
            adr = self.m.jnt_qposadr[self.m.body_jntadr[p["body"]]]
            self.d.qpos[adr + 2] += TABLE_Z
        mujoco.mj_forward(self.m, self.d)

    def plant(self, cmd):
        tgt = np.asarray(cmd, float)
        if self.sag_true is not None:
            tgt = tgt + self.sag_true(tgt)
        return tgt + self.dq

    def move(self, q_cmd, opening, sec, settle=0.3):
        q0, g0 = self.ctrl.copy(), self.grip
        q1, g1 = np.asarray(q_cmd, float), opening / A["stroke"]
        n = max(1, int(sec / self.m.opt.timestep))
        for i in range(n + int(settle / self.m.opt.timestep)):
            a = min(1.0, (i + 1) / n)
            self.s.set_arm(SIDE, self.plant(q0 + (q1 - q0) * a), g0 + (g1 - g0) * a)
            self.s.step()
        self.ctrl, self.grip = q1, g1

    def q(self):
        return np.array([self.d.qpos[a] for a in self.s.arm_qadr[SIDE]]) - self.dq

    def render_wrist(self):
        """The clean wrist view, 640x360 RGB, through the lens."""
        self.wr.update_scene(self.d, camera=self.cam_id)
        return cv2.remap(self.wr.render(), *self.map, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def wrist_image(self):
        bgr = self.render_wrist()[..., ::-1].astype(np.float32)
        if self.noise:
            bgr *= self.rng.uniform(0.85, 1.15) * self.rng.uniform(0.95, 1.05, 3)
            bgr += self.rng.normal(0, self.noise, bgr.shape)
        return np.clip(bgr, 0, 255).astype(np.uint8)

    def truth(self, colour):
        """For the overlay only: the block's true box (the Builder never reads it)."""
        k = COLOURS.index(colour)
        c, R = self.pose(k)
        hole = R[:, 2]
        if abs(hole[2]) > 0.7:  # lying flat
            return Shape((BLOCK_X, BLOCK_Y, H), c[2] - H / 2), (c[0], c[1], math.atan2(R[1, 0], R[0, 0]))
        yaw = math.atan2(hole[1], hole[0]) + math.pi / 2
        return Shape((FACE, H, STAND_H), c[2] - STAND_H / 2, math.pi), (c[0], c[1], yaw)

    def snap(self, tag):
        if tag in ("b3-stood", "b3-flipped"):
            self.stood = [self.pose(k) for k in range(len(self.s.props))]
        if not self.record:
            return
        self.vr.update_scene(self.d, self.vc)
        view = self.vr.render().copy()
        inset = cv2.resize(self.render_wrist(), (192, 108))
        view[4:4 + inset.shape[0], -4 - inset.shape[1]:-4] = inset
        cv2.putText(view, tag, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
        self.frames.append((tag, view))

    def pose(self, k):
        b = self.s.props[k]["body"]
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()


from sim.physics import BLOCK  # noqa: E402

BLOCK_X, BLOCK_Y = BLOCK["x"], BLOCK["y"]


def random_blocks(rng, n=3, min_gap=0.095):
    pts = []
    while len(pts) < n:
        p = np.array([rng.uniform(0.22, 0.40), rng.uniform(-0.12, 0.14)])
        if all(np.linalg.norm(p - q) > min_gap for q in pts):
            pts.append(p)
    return pts


class TrueSag:
    def __init__(self, sag, scale):
        self.sag, self.scale = sag, scale

    def __call__(self, q):
        return self.scale * self.sag(q)


def setup(seed, mode, record=False, base_seed=1000):
    rng = np.random.default_rng(base_seed + seed)
    pts = random_blocks(rng)
    yaws = rng.uniform(-math.pi / 4, math.pi / 4, 3)
    props = [{"kind": "block", "at": [float(p[0]), float(p[1])], "yaw": float(y)} for p, y in zip(pts, yaws)]
    sag_true = TrueSag(SagModel(Kin()), 1.0)
    dq = np.zeros(6)
    noise, mount = None, None
    if mode != "true":
        noise = 6.0
        mount = (rng.uniform(-1, 1, 3) * 0.002, rng.uniform(-1, 1, 3) * math.radians(1.0) / math.sqrt(3))
    if mode in ("real", "nosag"):
        dq[:4] = rng.uniform(-1, 1, 4) * math.radians(0.75)
        sag_true.scale = rng.uniform(0.8, 1.2)
    return props, dict(sag_true=sag_true, dq=dq, noise=noise, mount_err=mount, rng=rng, record=record)


def score(robot, order):
    P = [robot.pose(k) for k in order]
    off = [float(np.linalg.norm(P[i][0][:2] - P[i - 1][0][:2])) for i in (1, 2)]
    dz1 = P[1][0][2] - P[0][0][2] - H
    hole_tilt = math.degrees(math.asin(min(1.0, abs(P[2][1][2, 2]))))
    stand_h = 2 * (P[2][0][2] - (P[1][0][2] + H / 2))
    two = off[0] < 0.005 and abs(dz1) < 0.003
    three = two and off[1] < 0.005 and hole_tilt < 5 and abs(stand_h - 0.0503) < 0.004
    return dict(xy_off_mm=[round(v * 1000, 1) for v in off], dz2_mm=round(float(dz1) * 1000, 1), b3_hole_from_horizontal_deg=round(hole_tilt, 1),
                top_mm=round(float(P[2][0][2] + stand_h / 2 - robot.table_z) * 1000, 1), two_ok=bool(two), success=bool(three))


def trial(seed: int, mode: str = "true", record=False) -> dict:
    props, kw = setup(seed, mode, record)
    robot = SimRobot(props, **kw)
    kin = Kin()
    desk = robot.s.desk_z
    res = dict(seed=seed, mode=mode)
    t0 = time.time()
    b = Builder(robot, kin, None if mode == "nosag" else SagModel(kin), WristView(kin), robot.table_z, log=lambda *a: None,
                safety=TowerSafety())
    try:
        b.run(COLOURS)
    except (Stop, RuntimeError) as e:
        res["stopped"] = str(e)
    plan = getattr(b, "last_plan", None)
    res["unsafe"] = b.unsafe[:3]
    res["looks"] = len(b.looks)
    res["max_look_step_mm"] = max((l["step_mm"] for l in b.looks if "step_mm" in l), default=0)
    res["min_iou"] = min((l["iou"] for l in b.looks if "iou" in l), default=0)
    res["flip_checks"] = [l for l in b.looks if l["tag"].startswith("flip check")]
    errs = [l["err_mm"] for l in b.looks if "err_mm" in l]
    res["fit_err_mm"] = dict(median=round(float(np.median(errs)), 1), max=round(float(np.max(errs)), 1)) if errs else None
    res["fit_err_yaw_deg"] = round(float(np.max([abs(l["err_yaw_deg"]) for l in b.looks if "err_yaw_deg" in l] or [0])), 1)
    robot.move(robot.ctrl, OPEN, 0.1, 1.0)
    res["sec"] = round(time.time() - t0)
    if plan is not None and getattr(robot, "stood", None) is not None and getattr(b, "stand_target", None) is not None:
        c3 = robot.stood[plan["order"][2]][0]  # where the third block lay when let go, against where it was meant to
        res["slide_mm"] = round(float(np.linalg.norm(c3[:2] - b.stand_target)) * 1000, 1)
    if plan is None:
        res.update(success=False, two_ok=False)
    else:
        res.update(score(robot, plan["order"]))
    return res | {"frames": robot.frames, "wrist": b.sheet}


def sheet(rows_of_images, path, cols=8, size=None):
    rows = []
    for imgs in rows_of_images:
        if not imgs:
            continue
        imgs = [cv2.resize(i, size) if size else i for i in imgs[:cols]]
        while len(imgs) < cols:
            imgs.append(np.full_like(imgs[0], 255))
        rows.append(np.concatenate(imgs, axis=1))
    if rows:
        cv2.imwrite(str(path), np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "true"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    out = sys.argv[3] if len(sys.argv) > 3 else None
    results = []
    for i in range(n):
        r = trial(i, mode, record=bool(out) and i < 3)
        results.append(r)
        print(i, json.dumps({k: v for k, v in r.items() if k not in ("frames", "wrist")}), flush=True)
    two = sum(r.get("two_ok", False) for r in results)
    ok = sum(r["success"] for r in results)
    print(f"tower {mode}: two flat {two}/{n}, full tower {ok}/{n}")
    log = ROOT / "runs" / "sim" / "tower" / f"{mode}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps([{k: v for k, v in r.items() if k not in ("frames", "wrist")} for r in results], indent=1))
    if out:
        sheet([[f[1][..., ::-1] for f in r["frames"]] for r in results[:3]], f"{out}-scene.jpg")
        # the wrist looks: each look's last frame, jaw CAD (cyan), fit (amber), truth (white)
        for i, r in enumerate(results[:3]):
            ims = [im for _, im in r["wrist"] if im is not None]
            sheet([ims[k:k + 6] for k in range(0, len(ims), 6)], f"{out}-wrist{i}.jpg", cols=6, size=(426, 240))
        print("wrote", out + "-*.jpg")
