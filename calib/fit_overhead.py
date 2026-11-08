"""Fits the overhead camera to the left arm: the CAD arm at measured joints laid onto the dark arm pixels.

Input: a station/calib_poses.py run (frames + measured joints of the LEFT arm). The arm is black on a
pale table, so in each frame the dark pixels on the table (minus the other dark things: power strip,
clip, phone, the right arm's gripper, the laptop) are the arm. The camera (pose, focal length, one
radial distortion term) that best overlays the Metal CAD (sim/physics.py) at each pose's measured
joints, both ways (CAD points onto dark pixels, dark pixels onto CAD points), is the answer, in the
left arm's base frame (x forward, y left, z up).

    python -m calib.fit_overhead runs/station/calib/poses-1 [--out calib/cam_mid.json] [--free]

By default the camera starts at, and is held near (MOUNT_SIGMA), where the Metal Arm Station's mount
puts it (MOUNT, from MakerMods' station drawing); --free searches without that prior.
"""

import argparse
import json
import math
import pathlib

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from sim.physics import Scene

W, H = 1280, 720
# not the arm: dark things on the table in this station's view (x0, y0, x1, y1), full-resolution pixels
EXCLUDE = [(690, 300, 830, 495), (900, 410, 1000, 480), (30, 600, 230, 720), (930, 520, 1280, 720), (1020, 200, 1280, 520)]
TABLE_TOP_ROW = 305  # the table's far edge in this view; above it is the room
# MakerMods' Metal Arm Station (makermods.ai/scripts/station-fixture.js): bases 690 mm apart on one plate,
# the camera mast midway between them 35 mm behind the base line, lens centre 600 mm over the plate top,
# tilted 30 deg down, looking forward. In the left arm's frame (origin on the plate top) the lens sits at:
SPACING, CAM_HEIGHT, CAM_PITCH_DEG = 0.690, 0.600, 30.0
MOUNT = [-0.0043, -SPACING / 2, CAM_HEIGHT]
MOUNT_SIGMA = 0.01  # m: one residual unit (10 px) per cm off the drawing


def arm_points(scene: Scene, q, per_mesh: int = 250) -> np.ndarray:
    m, d = scene.model, scene.data
    scene.set_arm("right", q, 0.0, snap=True)
    mujoco.mj_forward(m, d)
    rng = np.random.default_rng(0)
    pts = []
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.body_weldid[m.geom_bodyid[g]] == 0:
            continue  # moving parts only (the base is off the frame's edge)
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        v = v[rng.choice(len(v), min(per_mesh, len(v)), replace=False)]
        pts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    return np.concatenate(pts)


def dark_pixels(img) -> np.ndarray:
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    m = (g < 60).astype(np.uint8)
    m[:TABLE_TOP_ROW] = 0
    for x0, y0, x1, y1 in EXCLUDE:
        m[y0:y1, x0:x1] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    v, u = np.nonzero(m)
    return np.stack([u, v], 1).astype(float)


def project(params, pts):
    rvec, tvec, f, k1 = params[:3], params[3:6], params[6], params[7]
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    uv, _ = cv2.projectPoints(pts, rvec, tvec, K, np.array([k1, 0, 0, 0, 0.0]))
    cam = (cv2.Rodrigues(rvec)[0] @ pts.T).T + tvec
    return uv.reshape(-1, 2), cam[:, 2] > 0.05


def look_at(pos, target):
    """OpenCV extrinsics (rvec, tvec) of a camera at pos looking at target, z up in the world."""
    f = np.subtract(target, pos); f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
    d = np.cross(f, r)  # image down
    R = np.stack([r, d, f])  # rows: camera x (right), y (down), z (forward) in the world
    return cv2.Rodrigues(R)[0].ravel(), -R @ np.asarray(pos, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default="calib/cam_mid.json")
    ap.add_argument("--free", action="store_true", help="no mount prior: grid of starts")
    a = ap.parse_args()
    run = pathlib.Path(a.run)
    rows = json.load(open(run / "poses.json"))
    s = Scene("metal", "right", props=[], draw=True)
    frames = []
    for r in rows:
        img = cv2.imread(str(run / r["frame"]))
        dark = dark_pixels(img)
        dark = dark[np.random.default_rng(1).choice(len(dark), min(1500, len(dark)), replace=False)] if len(dark) else dark
        anydark = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0) < 60  # the arm's own body above the table too
        frames.append({"pts": arm_points(s, r["measured"]), "dark": dark,
                       "dist": cv2.distanceTransform((~anydark).astype(np.uint8), cv2.DIST_L2, 5)})

    def residuals(p):
        res = []
        for fr in frames:
            uv, front = project(p, fr["pts"])
            inimg = front & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
            # every CAD point gives one residual: in the frame, its distance to the nearest dark pixel; off the
            # frame, a flat 1.5 (15 px) so leaving the frame does not pay; behind the camera, 4
            r1 = np.where(front, 1.5, 4.0)
            u = np.clip(uv, 0, [W - 1, H - 1]).astype(int)
            d = np.minimum(fr["dist"][u[:, 1], u[:, 0]], 40) / 10
            r1 = np.where(inimg, d, r1)
            res.append(r1)
            # dark arm pixels -> the nearest projected CAD point
            if len(fr["dark"]):
                tree = cKDTree(np.where(front[:, None], uv, 1e6))
                dd, _ = tree.query(fr["dark"])
                res.append(np.minimum(dd, 60) / 10)
        if not a.free:  # the mount: the lens stays near where the drawing puts it
            R = cv2.Rodrigues(p[:3])[0]
            res.append((-R.T @ p[3:6] - MOUNT) / MOUNT_SIGMA)
        return np.concatenate(res)

    best = None
    if a.free:
        starts = [([back, -sp, h], [0.5, -sp, 0.0]) for sp in (0.25, 0.3, 0.35) for h in (0.35, 0.45, 0.55) for back in (-0.15, 0.0)]
    else:  # the drawing's aim, and a few degrees either way in pitch and yaw
        starts = [(MOUNT, [MOUNT[0] + math.cos(math.radians(pitch)) * math.cos(math.radians(yaw)), MOUNT[1] + math.cos(math.radians(pitch)) * math.sin(math.radians(yaw)),
                           MOUNT[2] - math.sin(math.radians(pitch))]) for pitch in (CAM_PITCH_DEG - 8, CAM_PITCH_DEG, CAM_PITCH_DEG + 8) for yaw in (-6, 0, 6)]
    for pos0, look0 in starts:
        rv, tv = look_at(pos0, look0)
        for f in (550.0, 700.0):
            p0 = np.r_[rv, tv, f, -0.2]
            r = least_squares(residuals, p0, loss="huber", f_scale=1.0, max_nfev=200, x_scale=np.r_[[0.05] * 3, [0.05] * 3, 50, 0.05])
            cost = float(np.mean(r.fun ** 2))
            if best is None or cost < best[0]:
                best = (cost, r.x)
                print(f"start at {np.round(pos0, 3).tolist()} looking at {np.round(look0, 2).tolist()} f {f}: cost {cost:.3f}", flush=True)
    p = best[1]
    R = cv2.Rodrigues(p[:3])[0]
    pos = -R.T @ p[3:6]
    fovy = math.degrees(2 * math.atan(H / 2 / p[6]))
    out = {"arm": "left", "width": W, "height": H, "rvec": p[:3].tolist(), "tvec": p[3:6].tolist(), "f": float(p[6]), "k1": float(p[7]),
           "K": [[p[6], 0, W / 2], [0, p[6], H / 2], [0, 0, 1]], "pos": pos.tolist(), "fovy_deg": fovy,
           "R_cols_xyz": (R.T @ np.diag([1.0, -1.0, -1.0])).tolist(), "fit_cost": best[0], "err_px": [math.sqrt(best[0]) * 10],
           "mount_prior": None if a.free else {"lens": MOUNT, "sigma_m": MOUNT_SIGMA, "source": "makermods.ai/scripts/station-fixture.js"},
           "pitch_down_deg": math.degrees(math.asin(-R[2, 2])), "yaw_deg": math.degrees(math.atan2(R[2, 1], R[2, 0]))}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"camera at {np.round(pos, 3).tolist()} m in the left arm's frame, f {p[6]:.0f} px (fovy {fovy:.1f} deg), k1 {p[7]:.3f}, cost {best[0]:.3f} -> {a.out}")
    # overlay check images
    for i, (r, fr) in enumerate(zip(rows, frames)):
        img = cv2.imread(str(run / r["frame"]))
        uv, front = project(p, fr["pts"])
        for u, v in uv[front].astype(int):
            if 0 <= u < W and 0 <= v < H:
                cv2.circle(img, (u, v), 1, (255, 255, 0), -1)
        cv2.imwrite(str(run / f"overlay{i:02d}.jpg"), img)


if __name__ == "__main__":
    main()
