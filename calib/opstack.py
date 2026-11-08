"""Operator steps for stacking one foam block flat on another with the real left Metal arm, one small move at a time.

Each call does one thing (snapshot, find, one move, a grip change, a look) and appends to the run's step log, so a
person (or an agent) can look at the wrist and overhead frames between steps and decide the next one. Built on
calib/tower_plan.py (Kin, SagModel, WristView, frame_down) and calib/tower_safety.py (TowerSafety); the tower
runner's flip is not used.

Server: the team's left-arm server through the tunnel http://127.0.0.1:8691 (GET /state, /wrist.jpg, /overhead.jpg;
POST /move {q, grip, rate}). Nothing is ever posted to the team page (18081 / 8081); no stop_on_stall.

Before every move:
  - the team page (18081) says the left arm is enabled, not assigned to a leader, not recording; the Dream caller
    (18084) says left is not running. Otherwise /stop and refuse.
  - the path (joint-space line) passes TowerSafety.check_path from both the last target and the measured joints
    (model = measured + calib/arm_left.json dq_rad), plus: fingertips at y >= -0.15 m, the camera box at least 1 cm
    from every known block (the one being gripped excepted, below its top).
  - rate at most 0.3 rad/s (0.1 for descents).
Commands go out as sag.command(q_target) - dq.

    python -m calib.opstack RUN gate
    python -m calib.opstack RUN snap TAG
    python -m calib.opstack RUN find TAG                   # every block in the wrist view (no guess)
    python -m calib.opstack RUN look COLOUR TAG [Z0]       # one block near its stored estimate; Z0 its bottom
    python -m calib.opstack RUN to X Y TIPZ YAWDEG GRIP RATE TAG [home]   # fingertips to x, y, z (m), jaws at yaw
    python -m calib.opstack RUN grip GRIP TAG
    python -m calib.opstack RUN sheet "STEM:colour@z0[~held],...|caption" ...   # overlay sheet of saved steps
    python -m calib.opstack RUN joints RATE TAG q0..q5 [GRIP]  # straight to measured-frame joints (the way home)
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time
import urllib.request

import cv2
import numpy as np

from calib.tower_plan import FACE, H, Kin, SagModel, Shape, WristView, frame_down
from calib.tower_safety import BOX_HI, BOX_LO, TowerSafety

ROOT = pathlib.Path(__file__).resolve().parent.parent
URL = "http://127.0.0.1:8691"
TEAM = "http://127.0.0.1:18081/api/state"
DREAM = "http://127.0.0.1:18084/api/state"
TABLE_Z = -0.008
MAX_RATE = 0.3
TIP_Y_MIN = -0.15
CYAN, AMBER = (255, 224, 63), (40, 170, 245)
COLOURS = ("green", "blue", "purple")


def _get(url, timeout=5):
    return urllib.request.urlopen(url, timeout=timeout)


def _post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=5).read())


def arm_state():
    return json.loads(_get(URL + "/state").read())


def gate():
    """(ok, why): the left arm is on and nobody else drives it."""
    try:
        a = json.loads(_get(TEAM).read())
        b = json.loads(_get(DREAM).read())
    except Exception as e:  # noqa: BLE001
        return False, f"could not read the team page or Dream caller: {e}"
    left = (a.get("arms") or {}).get("left") or {}
    why = (f"enabled={left.get('enabled')!r} assigned_side={a.get('assigned_side')!r} recording={a.get('recording')!r} "
           f"dream left running={(b.get('left') or {}).get('running')!r}")
    ok = left.get("enabled") is True and a.get("assigned_side") != "left" and a.get("recording") is False \
        and (b.get("left") or {}).get("running") is False
    return ok, why


class Op:
    def __init__(self, run: pathlib.Path):
        self.run = run
        run.mkdir(parents=True, exist_ok=True)
        self.f = run / "op.json"
        self.st = json.loads(self.f.read_text()) if self.f.exists() else {"k": 0, "blocks": {}, "target": None, "start": None}
        self.dq = np.array(json.loads((ROOT / "calib" / "arm_left.json").read_text())["dq_rad"])
        self.kin = Kin()
        self.sag = SagModel(self.kin)
        self.view = WristView(self.kin)
        self.safe = TowerSafety()
        if self.st["start"] is None:
            self.st["start"] = arm_state()["q"]
            self.save()

    def save(self):
        self.f.write_text(json.dumps(self.st, indent=1))

    def log(self, tag, saw="", decided="", command=None):
        with open(self.run / "steps.jsonl", "a") as fh:
            fh.write(json.dumps({"t": time.strftime("%H:%M:%S"), "k": self.st["k"], "tag": tag, "saw": saw, "decided": decided,
                                 "command": command}) + "\n")

    # cameras
    def snap(self, tag):
        """Both cameras and the state, saved as NN-tag-*. Returns the wrist frame at 640x360, its model joints, grip."""
        self.st["k"] += 1
        k = self.st["k"]
        stem = f"{k:02d}-{tag}"
        meta = {}
        wrist = None
        for cam in ("wrist", "overhead"):
            r = _get(f"{URL}/{cam}.jpg")
            data = r.read()
            (self.run / f"{stem}-{cam}.jpg").write_bytes(data)
            meta[cam] = {"joints": json.loads(r.headers.get("X-Joints") or "null"), "grip": json.loads(r.headers.get("X-Grip") or "null"),
                         "stamp": r.headers.get("X-Stamp")}
            if cam == "wrist":
                wrist = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        meta["state"] = arm_state()
        (self.run / f"{stem}.json").write_text(json.dumps(meta))
        self.save()
        q = np.array(meta["wrist"]["joints"], float) + self.dq
        grip = float(meta["wrist"]["grip"] or 0.0)
        return cv2.resize(wrist, (640, 360), interpolation=cv2.INTER_AREA), q, grip, stem

    def overlay(self, img, q, opening, fits, label):
        """Jaw CAD and the fitted blocks: thin cyan outlines (silhouette and top face), amber labels in cm."""
        out = img.copy()
        cs, _ = cv2.findContours(self.view.jaws_mask(q, opening), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, CYAN, 1, cv2.LINE_AA)
        cam = self.view.pose(q, opening)
        for name, (fit, shape) in fits.items():
            if fit is None:
                continue
            hull = self.view.silhouette(cam, shape, fit["x"], fit["y"], fit["yaw"])
            if hull is None:
                continue
            cv2.polylines(out, [hull], True, CYAN, 1, cv2.LINE_AA)
            top = shape.corners(fit["x"], fit["y"], fit["yaw"])[[1, 3, 7, 5]]
            uv, ok = self.view.project(cam, top)
            if ok.all():
                cv2.polylines(out, [np.round(uv).astype(np.int32)], True, CYAN, 1, cv2.LINE_AA)
            at = tuple(int(v) for v in hull.reshape(-1, 2).min(0))
            cv2.putText(out, f"{name} {fit['x'] * 100:.1f}, {fit['y'] * 100:.1f} cm {math.degrees(fit['yaw']):+.0f} deg",
                        (max(2, at[0]), max(28, at[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, AMBER, 1, cv2.LINE_AA)
        cv2.putText(out, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, AMBER, 1, cv2.LINE_AA)
        return out

    def find_all(self, tag):
        img, q, grip, stem = self.snap(tag)
        opening = grip * 0.1
        flat = Shape((FACE, FACE, H), TABLE_Z)
        fits = {}
        for c in COLOURS:
            fit = self.view.find(img, c, q, opening, flat)
            fits[c] = (fit, flat)
            if fit is not None:
                self.st["blocks"][c] = {"x": fit["x"], "y": fit["y"], "yaw": fit["yaw"], "iou": fit["iou"], "edge": fit["edge"], "z0": TABLE_Z}
        cv2.imwrite(str(self.run / f"{stem}-ov.jpg"), self.overlay(img, q, opening, fits, stem))
        self.save()
        tip = self.tip(q)
        return {c: (None if f is None else {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()}) for c, (f, _) in fits.items()}, tip, stem

    def look(self, colour, tag, z0=TABLE_Z, guess=None, occlude=None):
        """occlude: the colour of a held block; its pixels (grown 4 px) are masked out with the jaws, so a block
        partly hidden behind it is fitted on what shows."""
        img, q, grip, stem = self.snap(tag)
        opening = grip * 0.1
        shape = Shape((FACE, FACE, H), z0)
        b = self.st["blocks"].get(colour)
        g = guess or ((b["x"], b["y"], b["yaw"]) if b else None)
        jaws_mask = self.view.jaws_mask
        if occlude:
            def masked(q_, o_):
                j = jaws_mask(q_, o_)
                held = self.view.colour_mask(img, occlude, j)
                return np.maximum(j, cv2.dilate(held * 255, np.ones((9, 9), np.uint8)))
            self.view.jaws_mask = masked
        try:
            fit = self.view.find(img, colour, q, opening, shape, g)
        finally:
            self.view.jaws_mask = jaws_mask
        cv2.imwrite(str(self.run / f"{stem}-ov.jpg"), self.overlay(img, q, opening, {colour: (fit, shape)}, stem))
        tip = self.tip(q)
        return fit, tip, stem, grip

    # kinematics
    def tip(self, q):
        """Fingertip point (between the pad ends) of model joints q, and the jaw yaw."""
        pos, R = self.kin.fk(q)
        return pos + self.kin.tip * R[:, 2], math.degrees(math.atan2(R[1, 0], R[0, 0]))

    def solve(self, xyz_tip, yaw, opening):
        ref = np.array(self.st["target"] if self.st["target"] else np.array(arm_state()["q"]) + self.dq)
        frames = [frame_down(yaw + k * math.pi / 2) for k in range(4)]
        from calib.tower_safety import DOWN_MIN, outward_ok
        frames = [R for R in frames if outward_ok(R, xyz_tip[:2])]
        cands = []
        for R in frames:
            q, _ = self.kin.ik(np.asarray(xyz_tip, float) + [0, 0, self.kin.tip], [R], ref)
            if q is not None:
                cands.append((float(np.abs(q - ref).sum()), q, R))
        if not cands:
            return None, "no IK"
        why = ""
        for _, q, R in sorted(cands, key=lambda c: c[0]):
            ok, why = self.safe.check_pose(q, opening)
            if ok:
                return q, "ok"
        return None, why

    def extra_checks(self, q0, q1, skip=None):
        """Fingertips y >= TIP_Y_MIN and the camera box 1 cm clear of every known block along the path."""
        n = max(2, int(math.ceil(np.max(np.abs(q1 - q0)) / math.radians(2))) + 1)
        corners = np.array([[x, y, z] for x in (BOX_LO[0], BOX_HI[0]) for y in (BOX_LO[1], BOX_HI[1]) for z in (BOX_LO[2], BOX_HI[2])])
        for t in np.linspace(0, 1, n):
            q = q0 + (q1 - q0) * t
            (p, _yaw) = self.tip(q)
            if p[1] < TIP_Y_MIN:
                return False, f"at {t * 100:.0f}%: fingertips at y {p[1] * 1000:.0f} mm (min {TIP_Y_MIN * 1000:.0f})"
            Rc, pc = self.safe.camera(q)
            cw = corners @ Rc.T + pc
            for c, b in self.st["blocks"].items():
                if c == skip:
                    continue
                h = FACE / 2
                top = b.get("z0", TABLE_Z) + H
                dx = np.maximum(np.abs(cw[:, 0] - b["x"]) - h, 0)
                dy = np.maximum(np.abs(cw[:, 1] - b["y"]) - h, 0)
                dz = np.maximum(cw[:, 2] - top, 0)
                if np.sqrt(dx ** 2 + dy ** 2 + dz ** 2).min() < 0.01:
                    return False, f"at {t * 100:.0f}%: camera within 1 cm of the {c} block"
        return True, "ok"

    def move(self, q_target, grip, rate, tag, home=False, skip=None, note="", raw=False):
        """raw: send q_target - dq as is, no sag compensation (the way back to the measured start joints)."""
        ok, why = gate()
        if not ok:
            _post("/stop", {})
            raise SystemExit(f"GATE: {why}: stopped")
        q_target = np.asarray(q_target, float)
        opening = float(grip) * 0.1
        q_meas = np.array(arm_state()["q"]) + self.dq
        q_last = np.array(self.st["target"]) if self.st["target"] else q_meas
        for name, q0 in (("from the last target", q_last), ("from the measured joints", q_meas)):
            ok, why = self.safe.check_path(q0, q_target, opening, home=home)
            if ok:
                ok, why = self.extra_checks(q0, q_target, skip)
            if not ok:
                self.log(tag, decided=f"REFUSED {name}: {why}", command={"q_model": q_target.tolist(), "grip": grip})
                raise SystemExit(f"REFUSED {name}: {why}")
        rate = float(min(rate, MAX_RATE))
        cmd = (q_target if raw else self.sag.command(q_target)) - self.dq
        body = {"q": [float(v) for v in cmd], "grip": float(np.clip(grip, 0, 1)), "rate": rate}
        _post("/move", body)
        t0 = time.time()
        time.sleep(0.3)
        while True:
            s = arm_state()
            if s.get("error"):
                raise SystemExit(f"arm server error: {s['error']}")
            if not s["moving"]:
                break
            if time.time() - t0 > 60:
                _post("/stop", {})
                raise SystemExit("move did not finish in 60 s: stopped")
            ok, why = gate()
            if not ok:
                _post("/stop", {})
                raise SystemExit(f"GATE during move: {why}: stopped")
            time.sleep(0.15)
        time.sleep(0.8)  # the servos lag 135-215 ms; let the sag settle
        s = arm_state()
        q_now = np.array(s["q"]) + self.dq
        self.st["target"] = q_target.tolist()
        self.save()
        err = q_now - q_target
        tip_t, _ = self.tip(q_target)
        tip_m, yaw_m = self.tip(q_now)
        res = {"sec": round(time.time() - t0, 1), "grip": round(s["grip"], 3), "goal_grip": round(s["goal_grip"], 3),
               "joint_err_deg": np.round(np.degrees(err), 1).tolist(), "tip_target_mm": np.round(tip_t * 1000).tolist(),
               "tip_measured_mm": np.round(tip_m * 1000).tolist(), "yaw_measured_deg": round(yaw_m, 1)}
        self.log(tag, decided=note, command={"q_model": q_target.tolist(), "cmd": body, "result": res})
        return res


def sheet(op: Op, specs, out_name="sheet.jpg"):
    """Overlay sheet from saved steps. specs: "STEM:colour@z0[~occluder],colour@z0,...|caption"; each frame with the jaw
    CAD and the fitted blocks (thin cyan outlines, amber labels), three to a row."""
    tiles = []
    for spec in specs:
        body, caption = (spec.split("|", 1) + [""])[:2]
        stem, rest = body.split(":", 1)
        meta = json.loads((op.run / f"{stem}.json").read_text())
        q = np.array(meta["wrist"]["joints"], float) + op.dq
        opening = float(meta["wrist"]["grip"] or 0.0) * 0.1
        img = cv2.resize(cv2.imread(str(op.run / f"{stem}-wrist.jpg")), (640, 360), interpolation=cv2.INTER_AREA)
        fits = {}
        for item in rest.split(","):
            occ = None
            if "~" in item:
                item, occ = item.split("~")
            colour, z0 = item.split("@")
            shape = Shape((FACE, FACE, H), float(z0))
            b = op.st["blocks"].get(colour)
            jm = op.view.jaws_mask
            if occ:
                op.view.jaws_mask = lambda q_, o_, jm=jm, occ=occ: np.maximum(jm(q_, o_), cv2.dilate(
                    op.view.colour_mask(img, occ, jm(q_, o_)) * 255, np.ones((9, 9), np.uint8)))
            try:
                fit = op.view.find(img, colour, q, opening, shape, (b["x"], b["y"], b["yaw"]) if b else None)
            finally:
                op.view.jaws_mask = jm
            fits[colour] = (fit, shape)
        tiles.append(op.overlay(img, q, opening, fits, f"{stem} {caption}"))
    while len(tiles) % 3:
        tiles.append(np.zeros((360, 640, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    path = op.run / out_name
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


# ---------------------------------------------------------------------------------------------------------------
# One look, then one streamed motion each for the pick and the place (Nov 7, evening).
RECORDS = ROOT / "runs" / "station" / "opstack" / "calib_records.jsonl"
# measured minus commanded fingertips (m) at the end of a straight-down approach. Commands aim at target - offset.
# x, y: run 182628's slow stepped descents (pick 8 moves, place 5, spread about 1 mm) and run 184543's streamed ones
# agree (-2.9..-4.5, -0.6..-1.2 mm). z: stepped descents end 2.8 (pick) and 3.8 mm (place) high, but the streamed
# ones of 184543 ended +0.8 and -0.4 mm (one each), so the streamed values are used: the set landed about 1 mm into
# the foam when aimed with the stepped offset.
PICK_DOWN = np.array([-0.0031, -0.0006, 0.0008])
PLACE_DOWN = np.array([-0.0040, -0.0012, -0.0004])
GRIP_TIP = 0.007  # measured fingertips this high (arm frame) when gripping a block on the table (15 mm over it)
CARRY_Z = 0.150  # fingertips over everything by more than 8 cm (the tower top is 61 mm)
LOOK_TIP = (0.28, 0.0, 0.19, 90.0)
TICK = 1 / 30


def record(kind, **d):
    RECORDS.parent.mkdir(parents=True, exist_ok=True)
    with open(RECORDS, "a") as fh:
        fh.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, **d}, default=lambda v: np.asarray(v).tolist()) + "\n")


def fetch(cam):
    r = _get(f"{URL}/{cam}.jpg")
    data = r.read()
    return data, json.loads(r.headers.get("X-Joints") or "null"), json.loads(r.headers.get("X-Grip") or "null")


def overhead_blocks(data):
    from calib.fit_overhead import TABLE_TOP_ROW
    from vision.locate import Camera, track
    from vision.track_real import IGNORE
    cam = Camera.from_calib(json.loads((ROOT / "calib" / "cam_mid.json").read_text()))
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    out = {}
    for p in track(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cam, TABLE_Z, ignore=IGNORE, table_row=TABLE_TOP_ROW):
        if p["kind"] == "block":
            out.setdefault(p["colour"], []).append({"x": p["xyz"][0], "y": p["xyz"][1], "yaw": p["yaw"], "px": p["px"]})
    return out


class GateWatch:
    """The team page and Dream caller polled in the background while a motion streams."""

    def __init__(self):
        import threading
        self.ok, self.why, self.stop = True, "", False
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()

    def run(self):
        misses = 0  # the team page's tunnel drops a request now and then; a closed gate stops at once, 1.5 s of no reply stops too
        while not self.stop:
            ok, why = gate()
            misses = misses + 1 if why.startswith("could not read") else 0
            if not ok and (misses == 0 or misses >= 6):
                self.ok, self.why = False, why
            time.sleep(0.25)


class Plan:
    """Waypoints solved to joints, every piece checked before anything moves."""

    def __init__(self, op: Op, q0, opening):
        self.op, self.nodes, self.opening = op, [(np.asarray(q0, float), 0.0, "start")], opening
        self.R = None

    def solve_from(self, ref, xyz, yaw_deg, keep):
        from calib.tower_safety import outward_ok
        frames = [self.R] if keep and self.R is not None else \
            [R for R in (frame_down(math.radians(yaw_deg) + k * math.pi / 2) for k in range(4)) if outward_ok(R, xyz[:2])]
        best = None
        for R in frames:
            q, ok = self.op.kin.ik_one(np.asarray(xyz, float) + [0, 0, self.op.kin.tip], R, ref)
            if not ok:
                q, _ = self.op.kin.ik(np.asarray(xyz, float) + [0, 0, self.op.kin.tip], [R], ref)
                if q is None:
                    continue
            if not self.op.safe.check_pose(q, self.opening)[0]:
                continue
            c = float(np.abs(q - ref).sum())
            if best is None or c < best[0]:
                best = (c, q, R)
        if best is None:
            raise SystemExit(f"no safe pose for fingertips {np.round(np.asarray(xyz) * 1000).tolist()} yaw {yaw_deg:.0f}")
        self.R = best[2]
        return best[1]

    def to(self, xyz, yaw_deg, v, tag, line=None, keep=False):
        """Fingertips to xyz. line: Cartesian steps (m) along the straight line from the last node (fingertips),
        else one joint-space piece."""
        q_prev = self.nodes[-1][0]
        if line:
            p0 = self.op.tip(q_prev)[0]
            n = max(1, int(math.ceil(np.linalg.norm(np.asarray(xyz) - p0) / line)))
            for i in range(1, n + 1):
                q_prev = self.solve_from(q_prev, p0 + (np.asarray(xyz) - p0) * i / n, yaw_deg, keep=True)
                self.nodes.append((q_prev, v, tag))
        else:
            self.nodes.append((self.solve_from(q_prev, xyz, yaw_deg, keep), v, tag))

    def check(self, skip=None, floor=None):
        for (q0, _, _), (q1, v, tag) in zip(self.nodes[:-1], self.nodes[1:]):
            ok, why = self.op.safe.check_path(q0, q1, self.opening)
            if ok:
                ok, why = self.op.extra_checks(q0, q1, skip)
            if ok and floor is not None and tag in floor:
                for t in np.linspace(0, 1, 9):
                    z = self.op.tip(q0 + (q1 - q0) * t)[0][2]
                    if z < floor[tag]:
                        ok, why = False, f"fingertips {z * 1000:.0f} mm, under the {floor[tag] * 1000:.0f} mm floor"
            if not ok:
                raise SystemExit(f"plan refused at {tag}: {why}")


def stream(op: Op, plan: Plan, grip, watch: GateWatch, tag):
    """Post joint goals along the plan in real time (30 Hz target), constant joint speed per piece, ramped in and
    out over 0.3 s; then wait for the arm to stop. Returns the timing and the end reading."""
    nodes = plan.nodes
    dur = [max(float(np.abs(q1 - q0).max()) / v, 1e-3) for (q0, _, _), (q1, v, _) in zip(nodes[:-1], nodes[1:])]
    T, ta = sum(dur), 0.3
    cum = np.concatenate([[0], np.cumsum(dur)])
    t0 = time.time()
    tau, last = 0.0, t0
    posts = 0
    while True:
        if not watch.ok:
            _post("/stop", {})
            raise SystemExit(f"GATE during {tag}: {watch.why}: stopped")
        now = time.time()
        t = now - t0
        # ramp in and out; slow posting can leave tau short of T when the ramp-out ends: then finish at 30 %
        f = max(0.05, min(1.0, t / ta)) if t < T + ta - ta else max(0.3, min(1.0, (T + ta - t) / ta))
        tau = min(T, tau + f * (now - last))
        last = now
        i = min(int(np.searchsorted(cum, tau, side="right")) - 1, len(dur) - 1)
        a = (tau - cum[i]) / dur[i]
        q = nodes[i][0] + (nodes[i + 1][0] - nodes[i][0]) * a
        v = nodes[i + 1][1]
        cmd = op.sag.command(q) - op.dq
        _post("/move", {"q": [float(x) for x in cmd], "grip": float(grip), "rate": float(min(0.6, 1.3 * v))})
        posts += 1
        if tau >= T:
            break
        time.sleep(max(0.0, TICK - (time.time() - now)))
    t_stream = time.time() - t0
    while arm_state()["moving"]:
        if not watch.ok:
            _post("/stop", {})
            raise SystemExit(f"GATE during {tag}: {watch.why}: stopped")
        if time.time() - t0 > T + 10:
            _post("/stop", {})
            raise SystemExit(f"{tag}: the arm did not reach the end in time: stopped")
        time.sleep(0.03)
    op.st["target"] = nodes[-1][0].tolist()
    return {"planned_s": round(T + ta, 2), "streamed_s": round(t_stream, 2), "done_s": round(time.time() - t0, 2), "posts": posts}


def end_record(op: Op, plan: Plan, tag, direction, run, settle=0.25):
    time.sleep(settle)
    s = arm_state()
    q_t = plan.nodes[-1][0]
    q_m = np.array(s["q"]) + op.dq
    tip_t, tip_m = op.tip(q_t)[0], op.tip(q_m)[0]
    rec = {"run": run, "tag": tag, "direction": direction, "q_target_model": q_t, "q_cmd_sent": op.sag.command(q_t) - op.dq,
           "q_goal_server": s["goal"], "q_measured": s["q"], "joint_err_deg": np.degrees(q_m - q_t).round(2),
           "tip_target_mm": (tip_t * 1000).round(1), "tip_measured_mm": (tip_m * 1000).round(1),
           "tip_err_mm": ((tip_m - tip_t) * 1000).round(1), "grip": s["grip"]}
    record("move_end", **rec)
    return tip_m, s


def fast_look(op: Op, tag="F-look"):
    """From the arm's present pose: overhead blocks (arm folded: nothing hidden), then to the high look pose, one
    wrist frame: blue on the table, purple on green (its bottom at the green's top)."""
    watch = GateWatch()
    oh, _, _ = fetch("overhead")
    (op.run / f"{tag}-rest-overhead.jpg").write_bytes(oh)
    ohb = overhead_blocks(oh)
    q0 = np.array(arm_state()["q"]) + op.dq
    pl = Plan(op, q0, 0.08)
    pl.to(LOOK_TIP[:3], LOOK_TIP[3], 0.4, "look")
    for (a, _, _), (b, _, _) in zip(pl.nodes[:-1], pl.nodes[1:]):
        ok, why = op.safe.check_path(a, b, 0.08, home=True)
        if not ok:
            raise SystemExit(f"look refused: {why}")
    t = stream(op, pl, 0.8, watch, "look")
    end_record(op, pl, "look", "unfold", op.run.name, settle=0.4)
    watch.stop = True
    img, q, grip, stem = op.snap(tag)
    flat = Shape((FACE, FACE, H), TABLE_Z)
    top = Shape((FACE, FACE, H), TABLE_Z + H)
    fits = {}
    for c, shape in (("blue", flat), ("purple", top), ("green", flat)):
        b = op.st["blocks"].get(c)
        g = (b["x"], b["y"], b["yaw"]) if b else None
        fits[c] = (op.view.find(img, c, q, grip * 0.1, shape, g), shape)
    cv2.imwrite(str(op.run / f"{stem}-ov.jpg"), op.overlay(img, q, grip * 0.1, fits, stem))
    out = {c: f for c, (f, _) in fits.items()}
    record("look", run=op.run.name, stem=stem, wrist={c: f for c, f in out.items()}, overhead_at_rest=ohb,
           q_measured=(q - op.dq), tip_mm=(op.tip(q)[0] * 1000).round(1))
    return out, ohb, t


def fast_go(op: Op, held, pick, tower, tower_top, grasp_ref=None, pick_levels=0):
    """pick: (x, y, yaw) of the block on the table (wrist look); tower: (x, y, yaw) of the tower's top block;
    tower_top: its top face height. One streamed motion to grip, close, lift; one to the tower, set, open, back off."""
    run = op.run.name
    watch = GateWatch()
    T0 = time.time()
    times = {}
    bx, by, byaw = pick
    jy = math.degrees(byaw)
    cmd_xy = np.array([bx, by]) - PICK_DOWN[:2]
    # pick_levels: blocks under the one picked (0: on the table). Off a stack the real fingertips ended 10 mm lower on
    # the block than planned (4.7 mm over its bottom, Nov 7, from=2), so aim 10 mm higher there: then the block is held
    # as a table pick holds it and a set-down on the table keeps the jaws clear of the table rule
    z_grip = GRIP_TIP - PICK_DOWN[2] + pick_levels * H + (0.010 if pick_levels else 0.0)
    q0 = np.array(op.st["target"]) if op.st["target"] else np.array(arm_state()["q"]) + op.dq
    pl = Plan(op, q0, 0.08)
    pl.to([cmd_xy[0], cmd_xy[1], TABLE_Z + (pick_levels + 1) * H + 0.08], jy, 0.5, "over")
    pl.to([cmd_xy[0], cmd_xy[1], z_grip + 0.03], jy, 0.3, "down", line=0.01)
    pl.to([cmd_xy[0], cmd_xy[1], z_grip], jy, 0.2, "last 3 cm", line=0.005)
    pl.check(skip=held)
    t_plan = time.time() - T0
    times["pick_stream"] = stream(op, pl, 0.8, watch, "pick")
    tip_m, s = end_record(op, pl, "grip height", "down", run, settle=0.2)
    # close, wait for the jaws to stop
    _post("/move", {"q": s["goal"], "grip": 0.44, "rate": 0.1})
    t_c = time.time()
    g_prev, still = None, 0
    while time.time() - t_c < 1.5:
        g = arm_state()["grip"]
        still = still + 1 if g_prev is not None and abs(g - g_prev) < 0.002 else 0
        g_prev = g
        if still >= 3 and time.time() - t_c > 0.3:
            break
        time.sleep(0.05)
    s = arm_state()
    q_m = np.array(s["q"]) + op.dq
    tip_grasp, yaw_grasp = op.tip(q_m)
    times["close_s"] = round(time.time() - t_c, 2)
    grasp = {"run": run, "block": held, "grip": s["grip"], "tip_fk_mm": (tip_grasp * 1000).round(1), "jaw_yaw_deg": round(yaw_grasp, 1),
             "wrist_look": {"x": bx, "y": by, "yaw": byaw}, "q_measured": s["q"]}
    if s["grip"] < 0.46:
        record("grasp_failed", **grasp)
        _post("/move", {"q": s["goal"], "grip": 0.8, "rate": 0.1})
        time.sleep(0.6)
        raise SystemExit(f"grip closed to {s['grip']:.3f} (no block between the jaws): opened, stopped at grip height")
    # the held block in the wrist frame, fetched while the lift streams
    import threading
    held_frame = {}
    threading.Thread(target=lambda: held_frame.update(data=fetch("wrist")), daemon=True).start()
    # fingertips over the picked block's own bottom (it sat on pick_levels blocks): Nov 7, measuring this from the
    # table let a block picked off a stack go 3.5-7 cm above its place
    grasp_h = tip_grasp[2] - (TABLE_Z + pick_levels * H)
    # place: over the tower, straight down until the block presses the surface (2 mm into the foam), then open:
    # the block is set down, never dropped
    tx, ty, tyaw = tower
    set_tip = tower_top - 0.002 + grasp_h
    pxy = np.array([tx, ty]) - PLACE_DOWN[:2]
    z_set = set_tip - PLACE_DOWN[2]
    pl2 = Plan(op, np.array(op.st["target"]), 0.044)
    pl2.R = pl.R
    tip_c = op.tip(pl.nodes[-1][0])[0]  # the commanded grip point: lift straight up from it
    pl2.to(tip_c + [0, 0, 0.03], jy, 0.25, "lift", line=0.01, keep=True)
    pl2.to([tip_c[0], tip_c[1], CARRY_Z], jy, 0.5, "up", line=0.03, keep=True)
    pl2.to([pxy[0], pxy[1], CARRY_Z], math.degrees(tyaw), 0.5, "carry")
    pl2.to([pxy[0], pxy[1], z_set + 0.03], math.degrees(tyaw), 0.3, "lower", line=0.01, keep=True)
    pl2.to([pxy[0], pxy[1], z_set + 0.01], math.degrees(tyaw), 0.15, "near", line=0.005, keep=True)
    pl2.to([pxy[0], pxy[1], z_set], math.degrees(tyaw), 0.06, "set on the surface", line=0.002, keep=True)
    pl2.check(skip=held, floor={"carry": CARRY_Z - 0.015})
    times["place_stream"] = stream(op, pl2, 0.44, watch, "place")
    tip_set, s = end_record(op, pl2, "set height", "down", run, settle=0.2)
    _post("/move", {"q": s["goal"], "grip": 0.8, "rate": 0.1})
    time.sleep(0.35)
    q_rel = np.array(arm_state()["q"]) + op.dq
    tip_rel = op.tip(q_rel)[0]
    pl3 = Plan(op, np.array(op.st["target"]), 0.08)
    pl3.R = pl2.R
    pl3.to(op.tip(pl2.nodes[-1][0])[0] + [0, 0, 0.05], math.degrees(tyaw), 0.3, "back off", line=0.01, keep=True)
    pl3.check(skip=held)
    times["back_stream"] = stream(op, pl3, 0.8, watch, "back off")
    watch.stop = True
    T1 = time.time()
    times["block_s"] = round(T1 - T0, 2)
    times["plan_s"] = round(t_plan, 2)
    # the held block as the wrist saw it after the close: the truth across the pads
    held_fit = None
    if "data" in held_frame:
        data, jq, jg = held_frame["data"]
        (op.run / "F-held-wrist.jpg").write_bytes(data)
        (op.run / "F-held.json").write_text(json.dumps({"wrist": {"joints": jq, "grip": jg}}))
        img = cv2.resize(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), (640, 360), interpolation=cv2.INTER_AREA)
        qh = np.array(jq) + op.dq
        tip_h = op.tip(qh)[0]
        held_fit = op.view.find(img, held, qh, jg * 0.1, Shape((FACE, FACE, H), tip_h[2] - grasp_h), None)
        if held_fit:
            held_fit["minus_tip_mm"] = [round((held_fit["x"] - tip_h[0]) * 1000, 1), round((held_fit["y"] - tip_h[1]) * 1000, 1)]
    grasp["held_in_wrist"] = held_fit
    truth = tip_grasp[:2] + (np.array(held_fit["minus_tip_mm"]) / 1000 if held_fit else 0)
    grasp["truth_mm"] = (truth * 1000).round(1)
    grasp["wrist_err_mm"] = ((np.array([bx, by]) - truth) * 1000).round(1)
    if grasp_ref:
        grasp["overhead"] = grasp_ref
        grasp["overhead_err_mm"] = ((np.array([grasp_ref["x"], grasp_ref["y"]]) - truth) * 1000).round(1)
    grasp["grasp_h_mm"] = round(grasp_h * 1000, 1)
    record("grasp", **grasp)
    return times, grasp, tip_set, tip_rel


def fast_check(op: Op, held, tower, tag="F-check"):
    """One photo from over the tower, 7 cm back (the top fully in view): the placed block's offset from the tower."""
    watch = GateWatch()
    pl = Plan(op, np.array(op.st["target"]), 0.08)
    pl.to([tower[0] - 0.065, tower[1], 0.13], math.degrees(tower[2]), 0.4, "check")
    pl.check()
    stream(op, pl, 0.8, watch, "check")
    watch.stop = True
    time.sleep(0.3)
    img, q, grip, stem = op.snap(tag)
    shape = Shape((FACE, FACE, H), TABLE_Z + 2 * H)
    fit = op.view.find(img, held, q, grip * 0.1, shape, (tower[0], tower[1], tower[2]))
    cv2.imwrite(str(op.run / f"{stem}-ov.jpg"), op.overlay(img, q, grip * 0.1, {held: (fit, shape)}, stem))
    return fit, stem


def report():
    rows = [json.loads(x) for x in open(RECORDS)]
    moves = [r for r in rows if r["kind"] == "move_end"]
    by = {}
    for r in moves:
        by.setdefault(r["direction"], []).append(r["tip_err_mm"])
    print("command vs actual, fingertips (measured - target, mm):")
    for k, v in by.items():
        v = np.array(v)
        print(f"  {k}: {len(v)} moves, mean {v.mean(0).round(1).tolist()}, sd {v.std(0).round(1).tolist()}")
    for r in rows:
        if r["kind"] == "grasp":
            print(f"  grasp {r['block']}: truth {r.get('truth_mm')} mm, wrist err {r.get('wrist_err_mm')}, overhead err {r.get('overhead_err_mm')}")
        if r["kind"] == "place":
            print(f"  place {r['block']}: offset {r.get('offset_mm')} mm, yaw {r.get('yaw_off_deg')} deg")


def main():
    run = pathlib.Path(sys.argv[1])
    cmd, a = sys.argv[2], sys.argv[3:]
    if cmd == "gate":
        print(gate())
        return
    op = Op(run)
    if cmd == "snap":
        _, q, grip, stem = op.snap(a[0])
        print(stem, "tip", np.round(op.tip(q)[0] * 1000).tolist(), "grip", round(grip, 3))
    elif cmd == "find":
        fits, tip, stem = op.find_all(a[0])
        print(stem, "tip", np.round(tip[0] * 1000).tolist(), "yaw", round(tip[1], 1))
        for c, f in fits.items():
            print(" ", c, f)
        op.log(a[0], saw=json.dumps(fits))
    elif cmd == "look":
        z0 = float(a[2]) if len(a) > 2 and "=" not in a[2] else TABLE_Z
        occ = next((v.split("=")[1] for v in a[2:] if v.startswith("occlude=")), None)
        fit, tip, stem, grip = op.look(a[0], a[1], z0, occlude=occ)
        print(stem, "tip", np.round(tip[0] * 1000).tolist(), "yaw", round(tip[1], 1), "grip", round(grip, 3))
        print(" ", a[0], fit)
        if fit is not None:
            dx, dy = fit["x"] - tip[0][0], fit["y"] - tip[0][1]
            dyaw = (math.degrees(fit["yaw"]) - tip[1] + 45) % 90 - 45
            print(f"  block minus fingertips: dx {dx * 1000:+.1f} mm, dy {dy * 1000:+.1f} mm, dyaw {dyaw:+.1f} deg")
        op.log(a[1], saw=json.dumps({a[0]: fit, "tip_mm": np.round(tip[0] * 1000).tolist(), "grip": grip}))
    elif cmd == "to":
        x, y, z, yawd, grip, rate, tag = float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4]), float(a[5]), a[6]
        home = "home" in a[7:]
        skip = next((v.split("=")[1] for v in a[7:] if v.startswith("skip=")), None)
        q, why = op.solve([x, y, z], math.radians(yawd), grip * 0.1)
        if q is None:
            op.log(tag, decided=f"no pose: {why}")
            raise SystemExit(f"no pose: {why}")
        print("q_model", np.round(q, 3).tolist(), "tip", np.round(op.tip(q)[0] * 1000).tolist(), "yaw", round(op.tip(q)[1], 1))
        if "dry" in a[7:]:
            q_meas = np.array(arm_state()["q"]) + op.dq
            q_last = np.array(op.st["target"]) if op.st["target"] else q_meas
            for q0 in (q_last, q_meas):
                print("  path", op.safe.check_path(q0, q, grip * 0.1, home=home), op.extra_checks(q0, q, skip),
                      "max joint change", round(float(np.abs(q - q0).max()), 2))
            return
        res = op.move(q, grip, rate, tag, home=home, skip=skip, note=f"fingertips to {x:.3f} {y:.3f} {z:.3f} yaw {yawd:.0f}")
        print(json.dumps(res))
    elif cmd == "grip":
        q = np.array(op.st["target"] if op.st["target"] else np.array(arm_state()["q"]) + op.dq)
        skip = next((v.split("=")[1] for v in a[2:] if v.startswith("skip=")), None)
        res = op.move(q, float(a[0]), 0.1, a[1], skip=skip, note=f"grip {a[0]}")
        print(json.dumps(res))
    elif cmd == "streamtest":
        dx, v = float(a[0]), float(a[1])
        watch = GateWatch()
        q0 = np.array(op.st["target"])
        p0, yaw = op.tip(q0)
        pl = Plan(op, q0, 0.08)
        pl.to(p0 + [0, dx, 0], yaw, v, "side", line=0.01)
        pl.to(p0, yaw, v, "back", line=0.01, keep=True)
        pl.check()
        t = stream(op, pl, 0.8, watch, "test")
        tip_m, s = end_record(op, pl, "streamtest", "level", op.run.name, settle=0.3)
        watch.stop = True
        print(t, "tip target", np.round(p0 * 1000, 1).tolist(), "measured", np.round(tip_m * 1000, 1).tolist())
    elif cmd == "report":
        report()
    elif cmd == "fastlook":
        out, ohb, t = fast_look(op)
        print(json.dumps({"wrist": out, "overhead_at_rest": ohb, "time": t}, default=float))
        op.save()
    elif cmd == "fastgo":
        held = a[0]
        pick = tuple(float(v) for v in a[1:4])
        tower = tuple(float(v) for v in a[4:7])
        kw = dict(v.split("=", 1) for v in a[7:] if "=" in v and not v.startswith("{"))
        ref = next((json.loads(v) for v in a[7:] if v.startswith("{")), None)
        # top=<blocks under the place spot> (0: on the table), from=<blocks under the picked block>
        top = TABLE_Z + int(kw.get("top", 2)) * H
        times, grasp, tip_set, tip_rel = fast_go(op, held, pick, tower, top, ref, pick_levels=int(kw.get("from", 0)))
        op.save()
        print(json.dumps({"times": times, "grasp": grasp, "tip_set_mm": tip_set * 1000, "tip_release_mm": tip_rel * 1000}, default=lambda v: np.asarray(v).tolist()))
    elif cmd == "fastcheck":
        held = a[0]
        tower = tuple(float(v) for v in a[1:4])
        fit, stem = fast_check(op, held, tower)
        op.save()
        off = None if fit is None else [round((fit["x"] - tower[0]) * 1000, 1), round((fit["y"] - tower[1]) * 1000, 1)]
        yo = None if fit is None else round((math.degrees(fit["yaw"] - tower[2]) + 45) % 90 - 45, 1)
        record("place", run=op.run.name, block=held, tower=tower, fit=fit, offset_mm=off, yaw_off_deg=yo, stem=stem)
        print(stem, fit, "offset mm", off, "yaw off deg", yo)
    elif cmd == "sheet":
        print(sheet(op, a))
    elif cmd == "joints":
        rate, tag = float(a[0]), a[1]
        q = np.array([float(v) for v in a[2:8]]) + op.dq
        home = True
        grip = float(a[8]) if len(a) > 8 else float(arm_state()["goal_grip"])
        res = op.move(q, grip, rate, tag, home=home, note="straight to joints", raw=True)
        print(json.dumps(res))


if __name__ == "__main__":
    main()
