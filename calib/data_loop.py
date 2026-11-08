"""Pick and place data collection on the real left Metal arm: scatter the three foam blocks, restack them, again.

Built on calib/opstack.py's look-once, move-once routine (Op, Plan, stream, GateWatch, end_record, the gate, the sag
compensation, the joint offsets) and its pick and place numbers (PICK_DOWN, PLACE_DOWN, GRIP_TIP, CARRY_Z, LOOK_TIP).
The pick and the place are split into their own streamed motions so each is its own episode.

One cycle (the arm shared through runs/station/ARM_OWNER, taken with mkdir):
  1. one wrist look from the high pose: every green, blue and purple blob fitted as a flat block at table level, one
     and two blocks up; the levels chosen so each block above the table sits on one below it (2 cm) and no two share
     a level within 4.5 cm; other coloured blobs (sticks, balls) and tag cubes are obstacles.
  2. scatter: each block, top first, to a random free spot (x 25-42 cm, y -10..25 cm, any heading, 11 cm between
     block centres, 8 cm from every obstacle, in view of the look pose).
  3. restack: three flat at a random free spot, random colour order.
  4. a check photo after each place (from 6.5 cm back): more than 1 cm off or tipped gets one recovery.
Every path: TowerSafety and opstack's extra checks (Plan.check), plus fingertips (and a held block's bottom) 12 mm
over every block they pass within 8.5 cm of.

Recording: calib/loop_rec.py on the station keeps 30 Hz ticks (both cameras, joints, goals); each pick and each
place is saved as one LeRobot episode (dataset/ or failures/) under ~/pah/loop/<session>/ on the station and synced
to runs/station/loop/<session>/; labels.jsonl beside each; calibration records in calib_records.jsonl.

    python -m calib.data_loop run [CYCLES]
    python -m calib.data_loop report SESSION
"""
from __future__ import annotations

import http.client
import itertools
import json
import math
import os
import pathlib
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

import cv2
import numpy as np

from calib import opstack as O
from calib.opstack import (CARRY_Z, GRIP_TIP, LOOK_TIP, PICK_DOWN, PLACE_DOWN, TABLE_Z, GateWatch, Op, Plan, _post,
                           arm_state, end_record, fetch, gate, overhead_blocks, stream)
from calib.tower_plan import FACE, H, WORKSPACE, Shape

ROOT = O.ROOT
STATION = ROOT / "runs" / "station"
TOKEN = STATION / "ARM_OWNER"
LOOP = STATION / "loop"
STATUS = LOOP / "STATUS.txt"
STOPFILE = LOOP / "STOP"
REC = "http://127.0.0.1:8697"
AREA = ((0.22, 0.45), (-0.14, 0.30))  # as wide as the safety checks and IK allow (y over -15 cm)
GAP = 0.08  # block centres apart (blocks are 5 cm), with the jaws checked against every neighbour (finger_clear)
CLEAR = 0.06  # from sticks, balls and other non-block objects
FINGER = 0.045  # an open finger's distance from the block centre, along either face axis
CANDIDATES = 60
OFF_TOL = 0.010  # table set-downs: spot error allowed
NEAT_MM, NEAT_DEG = 4.0, 4.0  # stack set-downs, relative to the measured block below: worse gets one lift and re-place
CARRY = (0.13, 0.17)  # carry height, randomised per block
IOU_MIN = 0.70
GRIP_HELD = 0.46
COLOURS = ("green", "blue", "purple")
GOOD = ((0.3667, -0.0592), (0.268, -0.006), (0.387, -0.073), (0.330, 0.140))  # spots that worked on Nov 7
STACK_PHRASES = ("place the {a} block on the {b} block", "stack the {a} block on top of the {b} block")


if "Fatal" not in globals():  # kept across a hot reload
    class Fatal(Exception):
        pass


if "CycleFail" not in globals():  # kept across a hot reload
    class CycleFail(Exception):
        pass


if "NoSpot" not in globals():  # kept across a hot reload
    class NoSpot(CycleFail):
        pass


if "NoPlan" not in globals():  # kept across a hot reload
    class NoPlan(CycleFail):
        pass


def status(line):
    msg = f"{time.strftime('%H:%M:%S')} {line}"
    with open(STATUS, "a") as fh:
        fh.write(msg + "\n")
    print(msg, flush=True)


def rec(path, **q):
    url = f"{REC}{path}" + (("?" + urllib.parse.urlencode(q)) if q else "")
    return json.loads(urllib.request.urlopen(url, timeout=60).read())


def now():
    return rec("/now")["t"]


def jdump(v):
    return json.loads(json.dumps(v, default=lambda x: np.asarray(x).tolist()))


def take_token():
    while True:
        try:
            TOKEN.mkdir()
            break
        except FileExistsError:
            time.sleep(5)
    (TOKEN / "who.txt").write_text(f"data-loop {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def release_token():
    shutil.rmtree(TOKEN, ignore_errors=True)


def arm_state_retry(tries=20):
    for i in range(tries):
        try:
            return arm_state()
        except (OSError, http.client.HTTPException) as e:
            if i == tries - 1:
                raise Fatal(f"arm server unreadable: {e!r}")
            time.sleep(1.0)


# streamed joint speeds (rad/s): high up, lowering, the last 3 cm and the set-down (Nov 7 20:40: 1.5x the 18:45
# values; the server's walk caps every goal at 0.6 rad/s, opstack.stream)
V_HIGH, V_DOWN, V_LAST, V_SET = 0.6, 0.45, 0.3, 0.09
REACH_Z = 0.238  # highest legal fingertip height, fingers down (Kin + TowerSafety sweep, x 16-38 cm)
LOOK_MAX_Z = 0.22  # close looks from 6.5 cm back stop here (the camera rule tightens near the reach limit)


def bot(b):
    """A block's bottom: its measured z0, else its level of flat blocks."""
    return b["z0"] if b.get("z0") is not None else TABLE_Z + b["level"] * H


def hgt(b):
    return FACE if b.get("lying") else H  # on its side: 50.3 mm tall, hole across


def top_of(b):
    return bot(b) + hgt(b)


def shape_of(b, z0=None):
    z = bot(b) if z0 is None else z0
    return Shape((H, FACE, FACE), z, period=math.pi) if b.get("lying") else Shape((FACE, FACE, H), z)


def look_z(level):
    """Fingertip height for a close look at a block at this level from 6.5 cm back: 13 cm, or 6 cm over its top."""
    return max(0.13, TABLE_Z + (level + 1) * H + 0.06)


def wait_gate():
    t0 = time.time()
    while True:
        ok, why = gate()
        if ok:
            return
        if time.time() - t0 > 600:
            raise Fatal(f"gate closed for 10 minutes: {why}")
        print("gate closed, waiting:", why, flush=True)
        time.sleep(5)


def health():
    s = arm_state()
    if s.get("error"):
        raise Fatal(f"arm server error: {s['error']}")
    hot = []

    def walk(d, path=""):
        if isinstance(d, dict):
            for k, v in d.items():
                walk(v, f"{path}.{k}")
        elif isinstance(d, (int, float)) and "temp" in path.lower() and d > 60:
            hot.append((path, d))
    walk(s)
    if hot:
        raise Fatal(f"motor temperature over 60 C: {hot}")
    return s


class Loop:
    relook_iou = 0.75  # a re-look must fit this well
    def __init__(self, session: pathlib.Path):
        self.dir = session
        session.mkdir(parents=True, exist_ok=True)
        O.RECORDS = session / "calib_records.jsonl"
        self.op = Op(session / "op")
        self.rng = np.random.default_rng(int(time.time()))
        self.belief = {}  # colour -> dict(x, y, yaw, level, iou)
        self.obstacles = np.zeros((0, 2))
        self.q_look = None
        self.jaws_look = None
        self.saves = queue.Queue()
        self.nstack = 0
        self.counts = {"pick": [0, 0], "place": [0, 0]}  # [success, failure]
        self.dr = {"carry_z": CARRY_Z, "over_dz": 0.0}
        self.cyc = {}
        self.bias = self.load_bias()
        self.block_times = []
        threading.Thread(target=self.saver, daemon=True).start()

    # ---- place bias, learned online per level (arm frame): the mean of the last 5 measured (fit - aimed) offsets
    def load_bias(self):
        """Raw place errors (measured minus aimed, mm, arm frame) with where they happened, per level, from every loop
        session's calib_records."""
        hist = {0: [], 1: [], 2: []}
        for f in sorted(LOOP.glob("*/calib_records.jsonl")):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r.get("kind") == "place" and r.get("offset_mm") and r.get("target") and r.get("fit"):
                    corr = r.get("correction_mm") or [0.0, 0.0]
                    lvl = int(r["target"][3])
                    if r["fit"].get("level", lvl) == lvl:
                        hist[lvl].append([r["target"][0], r["target"][1], r["offset_mm"][0] + corr[0], r["offset_mm"][1] + corr[1]])
        return hist

    def correction(self, level, x=0.33, y=0.07):
        """Fewer than 3 samples at this level: none. 3 to 5: half their mean. 6 or more: offset = a + B [x, y], least
        squares regularised toward 0 (ridge, lambda 2 on the scaled features). Clipped to 8 mm."""
        h = np.array(self.bias.get(level) or [], float).reshape(-1, 4)
        if len(h) < 3:
            return [0.0, 0.0]
        if len(h) < 6:
            return [float(np.clip(v, -8.0, 8.0)) for v in 0.5 * h[:, 2:].mean(0)]
        h = h[-20:]
        F = np.stack([np.ones(len(h)), (h[:, 0] - 0.33) / 0.1, (h[:, 1] - 0.07) / 0.1], 1)
        W = np.linalg.solve(F.T @ F + 2.0 * np.eye(3), F.T @ h[:, 2:])
        out = np.array([1.0, (x - 0.33) / 0.1, (y - 0.07) / 0.1]) @ W
        return [float(np.clip(v, -8.0, 8.0)) for v in out]

    # ---- recording
    def saver(self):
        while True:
            job = self.saves.get()
            try:
                root = job.get("root") or f"{self.dir.name}/" + ("failures" if not job["label"]["success"] else "dataset")
                r = rec("/save", t0=job["t0"], t1=job["t1"], root=root, task=job["label"]["prompt"])
                lab = dict(job["label"], dataset=root, **{k: r.get(k) for k in ("episode_index", "frames", "seconds", "max_gap_s", "error")})
                d = LOOP / root
                d.mkdir(parents=True, exist_ok=True)
                with open(d / "labels.jsonl", "a") as fh:
                    fh.write(json.dumps(jdump(lab)) + "\n")
                if r.get("error"):
                    print("save error:", r, flush=True)
            except Exception as e:  # noqa: BLE001
                print("save failed:", e, flush=True)
            finally:
                self.saves.task_done()

    def episode(self, t0, t1, root=None, **label):
        self.counts.setdefault(label["skill"], [0, 0])[0 if label["success"] else 1] += 1
        self.saves.put({"t0": t0 - 0.3, "t1": t1 + 0.25, "root": root,
                        "label": dict(label, session=self.dir.name, randomisation=dict(self.dr), cycle_setup=dict(self.cyc))})

    def sync(self):
        """Pull the recorded episodes from the station's loop recorder. The station's address is not kept here: set
        STATION_LOOP (rsync source, user@host:path/loop) to sync; unset, nothing is pulled (sim runs)."""
        self.saves.join()
        remote = os.environ.get("STATION_LOOP")
        if not remote:
            return
        for src, dst in ((f"{self.dir.name}/", self.dir), ("dream_ft/", LOOP / "dream_ft"), ("dream_ft_failures/", LOOP / "dream_ft_failures")):
            pathlib.Path(dst).mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-a", "--bwlimit=1500", f"{remote}/{src}", f"{dst}/"], timeout=600, check=False)

    # ---- belief helpers
    def put_blocks(self):
        self.op.st["blocks"] = {c: {"x": b["x"], "y": b["y"], "yaw": b["yaw"], "z0": top_of(b) - H} for c, b in self.belief.items()}

    def below(self, colour):
        """The colour and every block under it in its stack."""
        b = self.belief[colour]
        return {colour} | {c for c, o in self.belief.items() if bot(o) < bot(b) - 0.005 and math.hypot(o["x"] - b["x"], o["y"] - b["y"]) < 0.03}

    def sweep(self, plan, exempt, held_h=0.0):
        for (q0, _, _), (q1, _, tag) in zip(plan.nodes[:-1], plan.nodes[1:]):
            n = max(2, int(math.ceil(np.max(np.abs(q1 - q0)) / math.radians(2))) + 1)
            for t in np.linspace(0, 1, n):
                p = self.op.tip(q0 + (q1 - q0) * t)[0]
                for c, b in self.belief.items():
                    if c in exempt:
                        continue
                    if math.hypot(p[0] - b["x"], p[1] - b["y"]) < (0.075 if held_h > 0 else 0.06):
                        top = top_of(b)
                        if p[2] - held_h < top + 0.012:
                            raise SystemExit(f"sweep refused at {tag}: {(p[2] - held_h - top) * 1000:.0f} mm over the {c} block")

    # ---- looking
    def fits_all(self, img, q, grip):
        """Every candidate blob of each block colour fitted flat at levels 0, 1, 2."""
        view = self.op.view
        opening = grip * 0.1
        cam = view.pose(q, opening)
        jaws = view.jaws_mask(q, opening)
        cands = {}
        leftover = []
        for c in COLOURS:
            m = view.colour_mask(img, c, jaws)
            n, lab, stats, cen = cv2.connectedComponentsWithStats(m)
            comps = sorted([i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 150], key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:3]
            cands[c] = []
            for i in comps:
                for lvl in range(3):
                    shape = Shape((FACE, FACE, H), TABLE_Z + lvl * H)
                    f = view._fit(cam, jaws, lab, stats, i, shape, None)
                    if f is None or f["iou"] < 0.55:
                        continue
                    if not (WORKSPACE[0][0] < f["x"] < WORKSPACE[0][1] and WORKSPACE[1][0] < f["y"] < WORKSPACE[1][1]):
                        continue
                    cands[c].append(dict(f, level=lvl, blob=int(i)))
                ys, xs = np.nonzero(lab == i)
                sel = np.stack([xs, ys], 1)[:: max(1, len(xs) // 200)]
                leftover.append((c, int(i), view.ray_on_plane(cam, sel, TABLE_Z + 0.015)[:, :2]))
        # a colour not seen at all may be hidden under a block seen one level up: that block's pose, one level down
        for c in COLOURS:
            if not cands[c]:
                cands[c] = [dict(A, level=A["level"] - 1, iou=0.0, blob=-1, hidden=True) for o in COLOURS if o != c
                            for A in cands[o] if A["level"] >= 1]
        # other colours (red, orange sticks and balls): all obstacles
        from vision.locate import HUES, SAT_MIN, VAL_MIN
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        obst = []
        for c in ("red", "orange"):
            bins = HUES.get(c, [(10, 24)])
            mm = np.zeros(hsv.shape[:2], bool)
            for lo, hi in bins:
                mm |= (hsv[..., 0] >= lo) & (hsv[..., 0] < hi)
            mm = (mm & (hsv[..., 1] >= SAT_MIN) & (hsv[..., 2] >= VAL_MIN) & (jaws == 0)).astype(np.uint8)
            mm = cv2.morphologyEx(mm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(mm)
            for i in range(1, n):
                if stats[i, 4] >= 120:
                    ys, xs = np.nonzero(lab == i)
                    sel = np.stack([xs, ys], 1)[:: max(1, len(xs) // 200)]
                    obst.append(view.ray_on_plane(cam, sel, TABLE_Z + 0.011)[:, :2])
        return cands, leftover, obst, jaws

    def choose(self, cands, prior):
        best = None
        for combo in itertools.product(*[cands[c] for c in COLOURS]):
            byc = dict(zip(COLOURS, combo))
            ok = True
            for a, b in itertools.combinations(COLOURS, 2):
                A, B = byc[a], byc[b]
                d = math.hypot(A["x"] - B["x"], A["y"] - B["y"])
                if A["level"] == B["level"] and d < 0.045:
                    ok = False
            for c, A in byc.items():
                if A["level"] > 0 and not any(B["level"] == A["level"] - 1 and math.hypot(A["x"] - B["x"], A["y"] - B["y"]) < 0.02
                                              for k, B in byc.items() if k != c):
                    ok = False
            if not ok:
                continue
            score = sum(A["iou"] - (0.1 if A["edge"] else 0) for A in byc.values())
            for c, A in byc.items():
                p = (prior or {}).get(c)
                if p and p["level"] == A["level"] and math.hypot(p["x"] - A["x"], p["y"] - A["y"]) < 0.03:
                    score += 0.05
            if best is None or score > best[0]:
                best = (score, byc)
        return None if best is None else best[1]

    def jitter_look(self, dxy=0.03, dyaw=10.0):
        return (LOOK_TIP[0] + self.rng.uniform(-dxy, dxy), LOOK_TIP[1] + self.rng.uniform(-dxy, dxy), LOOK_TIP[2],
                LOOK_TIP[3] + self.rng.uniform(-dyaw, dyaw))

    def look(self, tag, tip=LOOK_TIP):
        self.to_pose(tip, "look")
        time.sleep(0.25)
        img, q, grip, stem = self.op.snap(tag)
        oh = (self.op.run / f"{stem}-overhead.jpg").read_bytes()
        try:
            ohb = overhead_blocks(oh)
        except Exception as e:  # noqa: BLE001
            ohb = {"error": repr(e)}
        cands, leftover, obst, jaws = self.fits_all(img, q, grip)
        byc = self.choose(cands, self.belief)
        if byc is None:
            return None, stem
        used = {(c, A["blob"]) for c, A in byc.items()}
        pts = [p for c, i, p in leftover if (c, i) not in used] + obst
        self.obstacles = np.concatenate(pts) if pts else np.zeros((0, 2))
        self.obstacles = self.obstacles[(self.obstacles[:, 0] > 0.15) & (self.obstacles[:, 0] < 0.55) & (self.obstacles[:, 1] > -0.2)
                                        & (self.obstacles[:, 1] < 0.35)]
        # the blocks' own side faces (a stack's lower blocks show as strips around it) are not obstacles
        for A in byc.values():
            if len(self.obstacles):
                self.obstacles = self.obstacles[np.hypot(self.obstacles[:, 0] - A["x"], self.obstacles[:, 1] - A["y"]) > 0.06]
        self.belief = {c: {k: A[k] for k in ("x", "y", "yaw", "level", "iou", "edge")} for c, A in byc.items()}
        self.put_blocks()
        self.q_look, self.jaws_look = q, jaws
        self.look_overhead = ohb
        fits = {c: (dict(A), Shape((FACE, FACE, H), TABLE_Z + A["level"] * H)) for c, A in byc.items()}
        out = self.op.overlay(img, q, grip * 0.1, fits, stem)
        for p in self.obstacles[::5]:
            uv, ok = self.op.view.project(self.op.view.pose(q, grip * 0.1), [[p[0], p[1], TABLE_Z + 0.011]])
            if ok.all():
                cv2.circle(out, tuple(int(v) for v in uv[0]), 2, (0, 0, 255), -1)
        cv2.imwrite(str(self.op.run / f"{stem}-ov.jpg"), out)
        O.record("look", run=self.dir.name, stem=stem, wrist=self.belief, overhead=ohb, q_measured=(q - self.op.dq),
                 obstacles=len(self.obstacles))
        return self.belief, stem

    # ---- motion
    def start_q(self):
        """The last target, else the arm's reading; when the sagging reading itself breaks a rule but the server's
        command (within 0.2 rad of it) does not, the command: paths are checked from the command (calib/safety.py)."""
        if self.op.st["target"]:
            return np.array(self.op.st["target"])
        s = arm_state()
        q_m, q_g = np.array(s["q"]) + self.op.dq, np.array(s["goal"]) + self.op.dq
        if not self.op.safe.check_pose(q_m, 0.08)[0] and self.op.safe.check_pose(q_g, 0.08)[0] and np.abs(q_g - q_m).max() < 0.2:
            return q_g
        return q_m

    def to_pose(self, tip, tag, v=0.6):
        """Fingertips to a high pose (look or end), from wherever the arm is."""
        q0 = self.start_q()
        if np.linalg.norm(self.op.tip(q0)[0] - np.array(tip[:3])) < 0.01:
            return
        pl = Plan(self.op, q0, 0.08)
        pl.to(tip[:3], tip[3], v, tag)
        try:
            pl.check()
        except SystemExit:
            s = arm_state()
            near = [np.abs(np.array(s["q"]) - np.array(s[k])).max() for k in ("start_joints", "zero_joints") if s.get(k)]
            if not near or min(near) > 0.15:  # only the folded rest (start or saved zero) may unfold under the home rules
                raise
            for (a, _, _), (b, _, _) in zip(pl.nodes[:-1], pl.nodes[1:]):
                ok, why = self.op.safe.check_path(a, b, 0.08, home=True)
                if not ok:
                    raise SystemExit(f"{tag} refused: {why}")
        self.sweep(pl, exempt=set())
        wait_gate()
        watch = GateWatch()
        try:
            stream(self.op, pl, 0.8, watch, tag)
        finally:
            watch.stop = True
        end_record(self.op, pl, tag, "level", self.dir.name, settle=0.3)

    def plan_pick(self, colour, q0, regrip=0.0):
        b = self.belief[colour]
        jy = math.degrees(b["yaw"])
        cxy = np.array([b["x"], b["y"]]) - PICK_DOWN[:2]
        # off a stack the real fingertips end about 10 mm lower on the block than planned (Nov 7): aim 10 mm higher
        z_grip = GRIP_TIP + (bot(b) - TABLE_Z) - PICK_DOWN[2] + (0.010 if bot(b) > TABLE_Z + 0.005 or b.get("lying") else 0.0) + regrip
        z_over = min(top_of(b) + 0.08 + self.dr["over_dz"], REACH_Z)
        ex = self.below(colour)
        last = None
        for route in ("direct", "high"):
            pl = Plan(self.op, q0, 0.08)
            try:
                if route == "high":
                    p0 = self.op.tip(q0)[0]
                    pl.to([p0[0], p0[1], max(self.dr["carry_z"], p0[2])], self.op.tip(q0)[1], 0.4, "up", line=0.02)
                    pl.to([cxy[0], cxy[1], max(self.dr["carry_z"], z_over)], jy, 0.5, "over high")
                    pl.to([cxy[0], cxy[1], z_over], jy, 0.4, "over", line=0.02, keep=True)
                else:
                    pl.to([cxy[0], cxy[1], z_over], jy, V_HIGH, "over")
                pl.to([cxy[0], cxy[1], z_grip + 0.03], jy, V_DOWN, "down", line=0.01, keep=True)
                pl.to([cxy[0], cxy[1], z_grip], jy, V_LAST, "last 3 cm", line=0.005, keep=True)
                pl.check(skip=colour)
                self.sweep(pl, ex)
                return pl
            except SystemExit as e:
                last = e
        raise SystemExit(f"pick {colour}: {last}")

    def plan_place(self, colour, q0, R, target, grasp_h):
        """target: (x, y, yaw, support_top, support colour or None)."""
        tx, ty, tyaw, top, sup = target
        pxy = np.array([tx, ty]) - PLACE_DOWN[:2]
        z_set = top - 0.002 + grasp_h - PLACE_DOWN[2]  # pressed 2 mm into the surface, then opened: set down, never dropped
        pl = Plan(self.op, q0, 0.044)
        pl.R = R
        pl.to([pxy[0], pxy[1], self.dr["carry_z"]], math.degrees(tyaw), V_HIGH, "carry")
        pl.to([pxy[0], pxy[1], z_set + (0.03 if z_set + 0.03 <= REACH_Z else 0.015)], math.degrees(tyaw), V_DOWN, "lower", line=0.01, keep=True)
        pl.to([pxy[0], pxy[1], z_set + 0.01], math.degrees(tyaw), 0.22, "near", line=0.005, keep=True)
        pl.to([pxy[0], pxy[1], z_set], math.degrees(tyaw), V_SET, "set on the surface", line=0.002, keep=True)
        pl.check(skip=colour, floor={"carry": self.dr["carry_z"] - 0.015})
        self.sweep(pl, {colour} | (self.below(sup) if sup else set()), held_h=grasp_h)
        return pl

    def plan_lift(self, q0, R, jy, colour=None):
        pl = Plan(self.op, q0, 0.044)
        pl.R = R
        tip_c = self.op.tip(q0)[0]
        pl.to(tip_c + [0, 0, 0.03], jy, 0.35, "lift", line=0.01, keep=True)
        pl.to([tip_c[0], tip_c[1], self.dr["carry_z"]], jy, V_HIGH, "up", line=0.03, keep=True)
        pl.check(skip=colour)
        return pl

    def go(self, pl, grip, tag, direction):
        wait_gate()
        health()
        watch = GateWatch()
        try:
            t = stream(self.op, pl, grip, watch, tag)
        finally:
            watch.stop = True
        tip_m, s = end_record(self.op, pl, tag, direction, self.dir.name, settle=0.15)
        return t, tip_m, s

    def release(self, s):
        """Open slowly, partly first, to cut the twist on release."""
        _post("/move", {"q": s["goal"], "grip": 0.6, "rate": 0.1})
        time.sleep(0.2)
        _post("/move", {"q": s["goal"], "grip": 0.8, "rate": 0.1})
        time.sleep(0.2)

    def support_look(self, sup):
        """A close look at the block a place goes onto (6.5 cm back, 13 cm up), fitted at its own level."""
        b = self.belief[sup]
        pl = Plan(self.op, self.start_q(), 0.08)
        pl.to([b["x"] - 0.065, b["y"], min(max(0.13, top_of(b) + 0.06), LOOK_MAX_Z)], math.degrees(b["yaw"]), 0.4, "support look")
        pl.check()
        self.sweep(pl, exempt=set())
        self.go(pl, 0.8, "support look", "level")
        time.sleep(0.25)
        img, q, grip, stem = self.op.snap(f"under-{sup}")
        shape = shape_of(b)
        f = self.op.view.find(img, sup, q, grip * 0.1, shape, (b["x"], b["y"], b["yaw"]))
        cv2.imwrite(str(self.op.run / f"{stem}-ov.jpg"), self.op.overlay(img, q, grip * 0.1, {sup: (f, shape)}, stem))
        if f is None or f["iou"] < 0.8 or math.hypot(f["x"] - b["x"], f["y"] - b["y"]) > 0.02:
            status(f"support look {sup}: no good fit ({None if f is None else round(f['iou'], 2)}); placing on its last measured pose")
            return b
        d = [round((f["x"] - b["x"]) * 1000, 1), round((f["y"] - b["y"]) * 1000, 1)]
        b.update(x=f["x"], y=f["y"], yaw=f["yaw"], iou=f["iou"], edge=f["edge"])
        self.put_blocks()
        O.record("support_look", run=self.dir.name, block=sup, fit=f, moved_from_last_mm=d, stem=stem)
        return b

    def grip_to(self, s, g, wait_close=False):
        _post("/move", {"q": s["goal"], "grip": g, "rate": 0.1})
        t_c = time.time()
        if not wait_close:
            time.sleep(0.35)
            return
        g_prev, still = None, 0
        while time.time() - t_c < 1.5:
            gg = arm_state()["grip"]
            still = still + 1 if g_prev is not None and abs(gg - g_prev) < 0.002 else 0
            g_prev = gg
            if still >= 3 and time.time() - t_c > 0.3:
                break
            time.sleep(0.05)

    # ---- one block
    def check_place(self, colour, target, level, z0=None):
        tx, ty, tyaw = target[:3]
        z0 = TABLE_Z + level * H if z0 is None else z0
        held = dict(self.belief.get(colour, {}), z0=z0)
        pl = Plan(self.op, np.array(self.op.st["target"]), 0.08)
        pl.to([tx - 0.065, ty, min(max(0.13, top_of(held) + 0.06), LOOK_MAX_Z)], math.degrees(tyaw), 0.4, "check")
        pl.check()
        self.sweep(pl, exempt=set())
        self.go(pl, 0.8, "check", "level")
        time.sleep(0.25)
        img, q, grip, stem = self.op.snap(f"check-{colour}")
        best = None
        for lvl, zz in [(level, z0)] + [(lv, TABLE_Z + lv * H) for lv in range(level - 1, -1, -1)]:
            for lying in (False, True):  # flat or on its side: the better fit says which
                shape = shape_of({"lying": lying}, zz)
                f = self.op.view.find(img, colour, q, grip * 0.1, shape, (tx, ty, tyaw))
                if f is not None and (best is None or f["iou"] > best[0]["iou"] + (0.05 if lvl != level else 0)):
                    best = (dict(f, level=lvl, z0=zz, lying=lying), shape)
            if best is not None and best[0]["level"] == level and best[0]["iou"] >= 0.85:
                break
        fits = {colour: best} if best else {}
        cv2.imwrite(str(self.op.run / f"{stem}-ov.jpg"), self.op.overlay(img, q, grip * 0.1, fits, stem))
        # the next blocks, refreshed when this frame shows them well
        for c, b in self.belief.items():
            if c == colour:
                continue
            f = self.op.view.find(img, c, q, grip * 0.1, shape_of(b), (b["x"], b["y"], b["yaw"]))
            if f and f["iou"] >= 0.88 and not f["edge"] and math.hypot(f["x"] - b["x"], f["y"] - b["y"]) < 0.015:
                b.update(x=f["x"], y=f["y"], yaw=f["yaw"])
        return (best[0] if best else None), stem

    def legal_place(self, colour, q0, R, tgt, grasp_h):
        """The place plan with the block pressed 2 mm into the surface; if the jaws rule refuses it, the set may stop up
        to 4 mm higher (the block's bottom at most 2 mm over the surface: set down, never dropped). (plan, mm higher)."""
        why = None
        for k_up in range(5):
            try:
                return self.plan_place(colour, q0, R, tgt, grasp_h + 0.001 * k_up), k_up, None
            except SystemExit as e:
                why = str(e)
        return None, None, why

    def block(self, colour, target, cycle, k, phase, recovery=False, regrip=0.0):
        """Pick colour (belief) and place it at target (x, y, yaw, level, support colour or None). Returns ok."""
        op = self.op
        tx, ty, tyaw, tlevel, sup = target
        # carry high enough that the held block's bottom clears the stack it goes onto by 2 cm
        sup_top = top_of(self.belief[sup]) if sup else TABLE_Z
        # carry high enough that the held block clears the stack it goes onto, inside the arm's reach
        self.dr = {"carry_z": round(min(max(float(self.rng.uniform(*CARRY)), sup_top + 0.05), max(REACH_Z, sup_top + 0.035)), 4),
                   "over_dz": round(float(self.rng.uniform(0, 0.02)), 4)}
        if sup and not recovery:
            sb = self.support_look(sup)  # the block below, measured at its own level: place squared and centred on it
            tx, ty, tyaw = sb["x"], sb["y"], sb["yaw"]
            target = (tx, ty, tyaw, tlevel, sup)
        if not recovery:
            self.relook(colour)
        src = dict(self.belief[colour])
        top = top_of(self.belief[sup]) if sup else TABLE_Z
        q0 = self.start_q()
        corr = self.correction(tlevel, tx, ty)  # aim off by the learned bias at this level and place (mm, arm frame)
        ax, ay = tx - corr[0] / 1000, ty - corr[1] / 1000
        try:
            pl = self.plan_pick(colour, q0, regrip)
            # the place must solve too, before anything moves (nominal grasp height)
            lift = self.plan_lift(pl.nodes[-1][0], pl.R, math.degrees(src["yaw"]), colour)
            self.plan_place(colour, lift.nodes[-1][0], lift.R, (ax, ay, tyaw, top, sup), 0.015)
        except SystemExit as e:
            raise NoPlan(f"{colour} to ({tx * 100:.1f}, {ty * 100:.1f}) L{tlevel}: {e}")
        T0 = now()
        t_wall = time.time()
        self.go(pl, getattr(self, "pick_open", {}).get(colour, 0.8), "pick", "down")
        s = arm_state()
        self.grip_to(s, 0.44, wait_close=True)
        s = arm_state()
        q_m = np.array(s["q"]) + op.dq
        tip_grasp, yaw_grasp = op.tip(q_m)
        grasp = {"run": self.dir.name, "block": colour, "grip": s["grip"], "tip_fk_mm": (tip_grasp * 1000).round(1),
                 "jaw_yaw_deg": round(yaw_grasp, 1), "wrist_look": src, "q_measured": s["q"], "cycle": cycle}
        prompt = f"pick up the {colour} block"
        if s["grip"] < GRIP_HELD:
            O.record("grasp_failed", **grasp)
            self.grip_to(s, 0.8)
            back = Plan(op, np.array(op.st["target"]), 0.08)
            back.R = pl.R
            back.to(op.tip(pl.nodes[-1][0])[0] + [0, 0, 0.06], math.degrees(src["yaw"]), 0.3, "back up", line=0.01, keep=True)
            back.check(skip=colour)
            self.go(back, 0.8, "back up", "up")
            self.episode(T0, now(), skill="pick", prompt=prompt, success=False, colour=colour, start_pose=src, grasp_pose=None,
                         end_pose=None, error="grip closed empty", duration_s=round(time.time() - t_wall, 2), cycle=cycle,
                         block=k, recovery=recovery, phase=phase)
            status(f"cycle {cycle} block {k} {phase} {colour}: grasp missed (grip {s['grip']:.3f})")
            if recovery:
                raise CycleFail(f"{colour}: grasp missed twice")
            # one recovery: look from right over it
            img, q, grip, stem = op.snap(f"relook-{colour}")
            f = op.view.find(img, colour, q, grip * 0.1, shape_of(src), (src["x"], src["y"], src["yaw"]))
            if f is None or f["iou"] < self.relook_iou:
                # not seen from right over it (jaws and neighbours hide it): the high look pose and a refit by id
                status(f"{colour}: not seen from over it after the miss; re-looking from the look pose")
                self.relook(colour)
            else:
                self.belief[colour].update(x=f["x"], y=f["y"], yaw=f["yaw"])
                self.put_blocks()
            return self.block(colour, target, cycle, k, phase, recovery=True)
        # held: the wrist frame of the block between the pads, while the lift streams
        held_frame = {}
        threading.Thread(target=lambda: held_frame.update(data=fetch("wrist")), daemon=True).start()
        z_bottom = bot(src)
        grasp_h = tip_grasp[2] - z_bottom
        lift = self.plan_lift(np.array(op.st["target"]), pl.R, math.degrees(src["yaw"]), colour)
        self.go(lift, 0.44, "lift", "up")
        s = arm_state()
        T1 = now()
        held_ok = s["grip"] >= GRIP_HELD
        truth, held_fit = None, None
        for _ in range(20):
            if "data" in held_frame:
                break
            time.sleep(0.05)
        if "data" in held_frame:
            data, jq, jg = held_frame["data"]
            (op.run / f"held-{cycle}-{k}-wrist.jpg").write_bytes(data)
            img = cv2.resize(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), (640, 360), interpolation=cv2.INTER_AREA)
            qh = np.array(jq) + op.dq
            tip_h = op.tip(qh)[0]
            held_fit = op.view.find(img, colour, qh, jg * 0.1, shape_of(src, tip_h[2] - grasp_h), None)
            if held_fit:
                held_fit["minus_tip_mm"] = [round((held_fit["x"] - tip_h[0]) * 1000, 1), round((held_fit["y"] - tip_h[1]) * 1000, 1)]
        truth = tip_grasp[:2] + (np.array(held_fit["minus_tip_mm"]) / 1000 if held_fit and held_fit["iou"] > 0.6 else 0)
        grasp.update(held_in_wrist=held_fit, truth_mm=(truth * 1000).round(1), grasp_h_mm=round(grasp_h * 1000, 1),
                     wrist_err_mm=((np.array([src["x"], src["y"]]) - truth) * 1000).round(1))
        oh = [o for o in (self.look_overhead.get(colour) or [])] if isinstance(self.look_overhead, dict) else []
        if oh:
            o = min(oh, key=lambda o: math.hypot(o["x"] - truth[0], o["y"] - truth[1]))
            grasp["overhead"] = o
            grasp["overhead_err_mm"] = ((np.array([o["x"], o["y"]]) - truth) * 1000).round(1)
        O.record("grasp", **grasp)
        grasp_pose = {"x": truth[0], "y": truth[1], "yaw": math.radians(yaw_grasp), "level": src["level"], "z_tip": tip_grasp[2]}
        self.episode(T0, T1, skill="pick", prompt=prompt, success=bool(held_ok), colour=colour, start_pose=src, grasp_pose=grasp_pose,
                     end_pose={"tip_mm": (op.tip(np.array(s["q"]) + op.dq)[0] * 1000).round(1), "grip": s["grip"]},
                     error={"tracker_minus_truth_mm": grasp["wrist_err_mm"]}, duration_s=round(time.time() - t_wall, 2),
                     cycle=cycle, block=k, recovery=recovery, phase=phase)
        if not held_ok:
            raise CycleFail(f"{colour} slipped out during the lift (grip {s['grip']:.3f})")
        # place
        t_place = time.time()
        pl3, k_up, why = self.legal_place(colour, np.array(op.st["target"]), lift.R, (ax, ay, tyaw, top, sup), grasp_h)
        if pl3 is not None and k_up:
            status(f"{colour}: grasp {grasp_h * 1000:.1f} mm over its bottom; set {k_up} mm over the 2 mm press (bottom within 2 mm of contact)")
        if pl3 is None:
            # too low to set down on contact here: set it back where it came from (same level), open, regrip 10 mm higher
            if regrip > 0:
                raise Fatal(f"{colour} held: still no legal set-down after a regrip ({why})")
            src_sup = next((c for c, o in self.belief.items() if c != colour and o["level"] == src["level"] - 1
                            and math.hypot(o["x"] - src["x"], o["y"] - src["y"]) < 0.03), None)
            status(f"{colour}: grasp {grasp_h * 1000:.1f} mm over its bottom is too low for a contact set-down at the target ({why}); "
                   f"setting it back at L{src['level']} to regrip 10 mm higher")
            plb, kb, whyb = self.legal_place(colour, np.array(op.st["target"]), lift.R, (src["x"], src["y"], src["yaw"], z_bottom, src_sup), grasp_h)
            if plb is None:
                raise Fatal(f"{colour} held: no contact set-down at the target or back at its source ({whyb})")
            Tb = now()
            self.go(plb, 0.44, "set back", "down")
            self.grip_to(arm_state(), 0.8)
            back = Plan(op, np.array(op.st["target"]), 0.08)
            back.R = plb.R
            back.to(op.tip(plb.nodes[-1][0])[0] + [0, 0, 0.05], math.degrees(src["yaw"]), 0.3, "back off", line=0.01, keep=True)
            back.check(skip=colour)
            self.go(back, 0.8, "back off", "up")
            pp = STACK_PHRASES[0].format(a=colour, b=src_sup) if src_sup else f"put the {colour} block down on the table"
            self.episode(Tb, now(), skill="place", prompt=pp, success=False, colour=colour, start_pose=grasp_pose, target=src,
                         end_pose=None, error="set back unchecked, to regrip higher", duration_s=None, cycle=cycle, block=k,
                         recovery=True, phase=phase + "-setback")
            return self.block(colour, target, cycle, k, phase, recovery=recovery, regrip=0.010)
        self.go(pl3, 0.44, "place", "down")
        s = arm_state()
        self.release(s)
        back = Plan(op, np.array(op.st["target"]), 0.08)
        back.R = pl3.R
        back.to(op.tip(pl3.nodes[-1][0])[0] + [0, 0, 0.05], math.degrees(tyaw), 0.3, "back off", line=0.01, keep=True)
        back.check(skip=colour)
        self.go(back, 0.8, "back off", "up")
        T2 = now()
        t_block = time.time() - t_wall
        # check
        t_c = time.time()
        fit, stem = self.check_place(colour, (tx, ty, tyaw), tlevel, top)
        t_check = time.time() - t_c
        if sup:
            phrase = STACK_PHRASES[self.nstack % 2]
            self.nstack += 1
            pprompt = phrase.format(a=colour, b=sup)
        else:
            pprompt = f"put the {colour} block down on the table"
        off = yaw_off = None
        ok = False
        if fit is not None:
            off = [round((fit["x"] - tx) * 1000, 1), round((fit["y"] - ty) * 1000, 1)]
            yaw_off = round((math.degrees(fit["yaw"] - tyaw) + 45) % 90 - 45, 1)
            ok = fit["level"] == tlevel and fit["iou"] >= IOU_MIN
            if sup:  # neatness: relative to the measured block directly below
                ok = ok and math.hypot(*off) <= NEAT_MM and abs(yaw_off) <= NEAT_DEG
            else:
                ok = ok and math.hypot(*off) <= OFF_TOL * 1000
        O.record("place", run=self.dir.name, block=colour, target=[tx, ty, tyaw, tlevel], support=sup, fit=fit, offset_mm=off,
                 yaw_off_deg=yaw_off, stem=stem, cycle=cycle, recovery=recovery,
                 neat=({"rel_offset_mm": off, "rel_yaw_deg": yaw_off, "under": sup} if sup else None),
                 correction_mm=[round(c, 2) for c in corr], aimed=[ax, ay])
        if fit is not None and off is not None and fit["level"] == tlevel:
            self.bias.setdefault(tlevel, []).append([tx, ty, off[0] + corr[0], off[1] + corr[1]])
        self.episode(T1, T2, skill="place", prompt=pprompt, success=bool(ok), colour=colour, start_pose=grasp_pose,
                     target={"x": tx, "y": ty, "yaw": tyaw, "level": tlevel, "support": sup}, end_pose=fit,
                     error={"offset_mm": off, "yaw_off_deg": yaw_off, "relative_to": sup or "target spot"},
                     role=({0: "base", 1: "middle", 2: "top"}.get(tlevel, f"level {tlevel}") if phase == "stack" else "scatter"),
                     duration_s=round(time.time() - t_place - t_check, 2),
                     cycle=cycle, block=k, recovery=recovery, phase=phase)
        # the whole block move as one task episode (Dream fine-tuning): still at the look (or check) pose, pick, carry,
        # place, back off, still; stack moves and table moves in their own folders, failures apart
        task = "stack" if sup else "place_on_table"
        tprompt = f"stack the {colour} block on top of the {sup} block" if sup else f"put the {colour} block down on the table"
        whole_ok = bool(ok and held_ok)
        self.episode(T0, T2, root=f"dream_ft/{task}" if whole_ok else f"dream_ft_failures/{task}", skill="task", prompt=tprompt,
                     success=whole_ok, colour=colour, start_pose=src, grasp_pose=grasp_pose,
                     target={"x": tx, "y": ty, "yaw": tyaw, "level": tlevel, "support": sup}, end_pose=fit,
                     error={"offset_mm": off, "yaw_off_deg": yaw_off, "relative_to": sup or "target spot"},
                     role=({0: "base", 1: "middle", 2: "top"}.get(tlevel, f"level {tlevel}") if phase == "stack" else phase),
                     duration_s=round(t_block, 2), cycle=cycle, block=k, recovery=recovery, phase=phase, task=task)
        self.block_times.append(t_block)
        status(f"cycle {cycle} block {k} {phase} {colour}: ({src['x'] * 100:.1f}, {src['y'] * 100:.1f}) L{src['level']} -> "
               f"({tx * 100:.1f}, {ty * 100:.1f}) L{tlevel}{' on ' + sup if sup else ''}: grip {grasp['grip']:.3f}, tracker err "
               f"{grasp['wrist_err_mm'].tolist()} mm; placed {'OK' if ok else 'FAIL'} off {off} mm yaw {yaw_off} deg iou "
               f"{None if fit is None else round(fit['iou'], 2)}; {t_block:.1f} s (+{t_check:.1f} s check){' [re-place]' if recovery else ''}; carry {self.dr['carry_z'] * 100:.1f} cm; aim corrected by {[-round(c, 1) for c in corr]} mm")
        if fit is not None:
            self.belief[colour] = {"x": fit["x"], "y": fit["y"], "yaw": fit["yaw"], "level": fit["level"], "iou": fit["iou"], "edge": fit["edge"],
                                   "z0": fit.get("z0"), "lying": fit.get("lying", src.get("lying", False))}
        else:
            self.belief[colour] = {"x": tx, "y": ty, "yaw": tyaw, "level": tlevel, "iou": 0.0, "edge": False, "z0": top, "lying": src.get("lying", False)}
        self.put_blocks()
        if ok:
            return True
        if fit is None or fit["iou"] < IOU_MIN:
            raise CycleFail(f"{colour}: not found flat after the place (tipped or lost)")
        if recovery:
            if sup:  # neat target missed twice: leave it (still stacked), note it, go on
                status(f"{colour}: still {off} mm / {yaw_off} deg off the {sup} block after one re-place; kept")
                return False
            raise CycleFail(f"{colour}: still off after one recovery ({off} mm)")
        return self.block(colour, target, cycle, k, phase, recovery=True)

    # ---- spots
    def visible(self, x, y, yaw, levels):
        view = self.op.view
        cam = view.pose(self.q_look, 0.08)
        for lvl in levels:
            uv, ok = view.project(cam, Shape((FACE, FACE, H), TABLE_Z + lvl * H).corners(x, y, yaw))
            if not ok.all() or (uv[:, 0] < 12).any() or (uv[:, 0] > view.W - 12).any() or (uv[:, 1] < 12).any() or (uv[:, 1] > view.H - 12).any():
                return False
            c = np.round(uv.mean(0)).astype(int)
            if self.jaws_look[min(c[1], view.H - 1), min(c[0], view.W - 1)]:
                return False
        return True

    def finger_clear(self, x, y, yaw, moving):
        """Open fingers (along either face axis) of this block and of each neighbour keep 4 cm from the other's centre."""
        def fingers(cx, cy, a):
            return [(cx + s * FINGER * math.cos(a + k * math.pi / 2), cy + s * FINGER * math.sin(a + k * math.pi / 2)) for s in (-1, 1) for k in (0, 1)]
        for c, b in self.belief.items():
            if c == moving:
                continue
            if any(math.hypot(fx - b["x"], fy - b["y"]) < 0.04 for fx, fy in fingers(x, y, yaw)) or \
                    any(math.hypot(fx - x, fy - y) < 0.04 for fx, fy in fingers(b["x"], b["y"], b["yaw"])):
                return False
        return True

    def spot(self, moving, levels, avoid=(), gap=GAP):
        """Candidates: the spots that worked today (jittered 2 cm) first, then uniform in the area."""
        for i in range(3000):
            if i < len(GOOD) and AREA[0][0] <= GOOD[i][0] <= AREA[0][1] and AREA[1][0] <= GOOD[i][1] <= AREA[1][1]:
                x, y = GOOD[i][0] + self.rng.uniform(-0.02, 0.02), GOOD[i][1] + self.rng.uniform(-0.02, 0.02)
            else:
                x, y = self.rng.uniform(*AREA[0]), self.rng.uniform(*AREA[1])
            yaw = self.rng.uniform(-math.pi / 4, math.pi / 4)
            if any(math.hypot(x - b["x"], y - b["y"]) < gap for c, b in self.belief.items() if c != moving) or \
                    any(math.hypot(x - a[0], y - a[1]) < gap for a in avoid):
                self.why["block gap"] += 1
                continue
            if moving and math.hypot(x - self.belief[moving]["x"], y - self.belief[moving]["y"]) < 0.06:
                self.why["too close to where it is"] += 1
                continue
            if not self.finger_clear(x, y, yaw, moving):
                self.why["jaws near a neighbour"] += 1
                continue
            if len(self.obstacles) and np.min(np.hypot(self.obstacles[:, 0] - x, self.obstacles[:, 1] - y)) < CLEAR:
                self.why["obstacle keep-out"] += 1
                continue
            if not self.visible(x, y, yaw, levels):
                self.why["out of the look's view"] += 1
                continue
            yield x, y, yaw

    def feasible(self, colour, target):
        """The whole block, planned from where the arm is (no motion)."""
        try:
            pl = self.plan_pick(colour, self.start_q())
            lift = self.plan_lift(pl.nodes[-1][0], pl.R, math.degrees(self.belief[colour]["yaw"]), colour)
            tx, ty, tyaw, tl, sup = target
            p3 = self.plan_place(colour, lift.nodes[-1][0], lift.R, (tx, ty, tyaw, TABLE_Z + tl * H, sup), 0.015)
            ck = Plan(self.op, p3.nodes[-1][0], 0.08)
            ck.to([tx - 0.065, ty, look_z(tl)], math.degrees(tyaw), 0.4, "check")
            ck.check(skip=colour)
            return True
        except SystemExit as e:
            self.last_refusal = str(e)
            return False

    def pick_spot(self, colour, levels, avoid=()):
        import collections
        self.why = collections.Counter()
        n = 0
        for gap in (GAP,):
            m = 0
            for x, y, yaw in self.spot(colour, levels, avoid, gap):
                n, m = n + 1, m + 1
                if self.feasible(colour, (x, y, yaw, 0, None)):
                    return x, y, yaw
                self.why["plan: " + self.last_refusal.split(":")[0][:40]] += 1
                if m > CANDIDATES:
                    break
        raise NoSpot(f"no feasible free spot for {colour} ({n} planned; rejections {dict(self.why)}; last plan refusal: {getattr(self, 'last_refusal', '')})")

    # ---- a cycle
    def cycle(self, n):
        t0 = time.time()
        lt = self.jitter_look()
        bel, stem = self.look(f"look-c{n}", lt)
        if bel is None:
            bel, stem = self.look(f"look2-c{n}", (0.31, 0.06, 0.21, 90.0))
        if bel is None:
            raise CycleFail("the look did not find three flat blocks")
        status(f"cycle {n} look {stem}: " + ", ".join(f"{c} ({b['x'] * 100:.1f}, {b['y'] * 100:.1f}) L{b['level']} iou {b['iou']:.2f}"
                                                   for c, b in bel.items()) + f"; {len(self.obstacles)} obstacle points")
        k = 0
        old_tower = [(b["x"], b["y"]) for c, b in bel.items() if any(o["level"] > b["level"] and math.hypot(o["x"] - b["x"], o["y"] - b["y"]) < 0.03
                                                                    for o in bel.values())]
        order = sorted(COLOURS, key=lambda c: (-bel[c]["level"], self.rng.random()))
        self.cyc = {"look_tip": [round(v, 4) for v in lt], "scatter_order": order, "start_levels": {c: bel[c]["level"] for c in COLOURS},
                    "fallbacks": []}
        left = set()
        for c in order:
            k += 1
            if any(o["level"] > self.belief[c]["level"] and math.hypot(o["x"] - self.belief[c]["x"], o["y"] - self.belief[c]["y"]) < 0.03
                   for o2, o in self.belief.items() if o2 != c):
                self.fallback(n, f"scatter {c}: a block still sits on it; left in place")
                left.add(c)
                continue
            if self.belief[c]["iou"] < 0.5:
                self.relook(c)  # hidden in the look: seen before its spot is chosen
            try:
                x, y, yaw = self.pick_spot(c, [0])
                self.block(c, (x, y, yaw, 0, None), n, k, "scatter")
            except (NoSpot, NoPlan) as e:
                done = False
                if self.belief[c]["level"] > 0:  # (b) onto a loose block: a 2-stack
                    for o in self.loose(exclude={c}):
                        try:
                            b = self.belief[o]
                            self.block(c, (b["x"], b["y"], b["yaw"], 1, o), n, k, "scatter-2stack")
                            self.fallback(n, f"scatter {c}: {e}; (b) set on the loose {o} block")
                            done = True
                            break
                        except NoPlan as e2:
                            e = e2
                if not done:
                    self.fallback(n, f"scatter {c}: {e}; (a) left where it is")
                    left.add(c)
            self.stop_check()
        # restack: on the tallest stack if one is left, else a new base spot, else the base where it is, else the old spot
        stacks = self.stacks()
        tower = max(stacks, key=len)
        if len(tower) >= 2:
            self.fallback(n, f"restack on the existing stack {' / '.join(tower)}")
        else:
            base = str(self.rng.choice(COLOURS))
            try:
                x, y, yaw = self.pick_spot(base, [0, 1, 2], avoid=old_tower)  # a new spot: not where the old tower stood
                k += 1
                self.block(base, (x, y, yaw, 0, None), n, k, "stack")
            except (NoSpot, NoPlan) as e:
                self.fallback(n, f"tower base {base}: {e}; (a) built on {base} where it is")
            tower = [base]
        rest = [c for c in self.rng.permutation(COLOURS) if c not in tower]
        self.cyc["tower_order"] = {"base": tower[0], "middle": (tower + rest)[1], "top": (tower + rest)[2]}
        b0 = self.belief[tower[0]]
        self.cyc["tower_spot"] = [round(b0["x"], 4), round(b0["y"], 4), round(b0["yaw"], 4)]
        for c in rest:
            k += 1
            s_ = self.belief[tower[-1]]
            self.block(str(c), (s_["x"], s_["y"], s_["yaw"], len(tower), tower[-1]), n, k, "stack")
            tower.append(str(c))
            self.stop_check()
        self.to_pose(LOOK_TIP, "end")
        status(f"cycle {n} done: tower {' / '.join(tower)} at ({b0['x'] * 100:.1f}, {b0['y'] * 100:.1f}); {time.time() - t0:.0f} s")

    def fallback(self, n, what):
        self.cyc.setdefault("fallbacks", []).append(what)
        status(f"cycle {n} fallback: {what}")

    def stacks(self):
        out = []
        for c, b in self.belief.items():
            if b["level"] != 0:
                continue
            st = [c]
            while True:
                nxt = [o for o, ob in self.belief.items() if ob["level"] == len(st)
                       and math.hypot(ob["x"] - self.belief[st[-1]]["x"], ob["y"] - self.belief[st[-1]]["y"]) < 0.03]
                if not nxt:
                    break
                st.append(nxt[0])
            out.append(st)
        return out

    def loose(self, exclude=()):
        return [st[0] for st in self.stacks() if len(st) == 1 and st[0] not in exclude]

    def relook(self, colour):
        """A fresh look at the block about to be picked, from a jittered high pose (then the plain one if it fails)."""
        b = self.belief[colour]
        for tip in (self.jitter_look(0.015, 5.0), LOOK_TIP):
            self.to_pose(tip, "look")
            time.sleep(0.25)
            img, q, grip, stem = self.op.snap(f"relook-{colour}")
            f = self.op.view.find(img, colour, q, grip * 0.1, shape_of(b), (b["x"], b["y"], b["yaw"]))
            if f is not None and f["iou"] >= self.relook_iou and not f["edge"] and math.hypot(f["x"] - b["x"], f["y"] - b["y"]) <= 0.04:
                break
        if f is None or f["iou"] < self.relook_iou or math.hypot(f["x"] - b["x"], f["y"] - b["y"]) > 0.04:
            raise CycleFail(f"{colour}: not found where it was hidden ({f})")
        b.update(x=f["x"], y=f["y"], yaw=f["yaw"], iou=f["iou"], edge=f["edge"])
        self.put_blocks()
        status(f"relook {colour}: ({f['x'] * 100:.1f}, {f['y'] * 100:.1f}) iou {f['iou']:.2f}")

    def stop_check(self):
        if STOPFILE.exists():
            raise Fatal("STOP file")


def run(cycles=20):
    LOOP.mkdir(parents=True, exist_ok=True)
    session = LOOP / time.strftime("%m%d-%H%M%S")
    take_token()
    lp = Loop(session)
    lp.op.st["target"] = None
    status(f"data loop session {session.name} started; token taken")
    rec("/start")
    done, fails, reason = 0, 0, "done"
    held_ok = True
    try:
        s = health()
        if s.get("grip_holding"):
            raise Fatal("the jaws are holding something at the start")
        mtime = pathlib.Path(__file__).stat().st_mtime
        for n in range(1, cycles + 1):
            if pathlib.Path(__file__).stat().st_mtime != mtime:  # edited: new methods from the next cycle on
                import importlib
                mtime = pathlib.Path(__file__).stat().st_mtime
                try:
                    mod = importlib.reload(sys.modules[__name__])
                    lp.__class__ = mod.Loop
                    lp.bias = lp.load_bias()
                    status("data_loop.py reloaded at the cycle boundary")
                except Exception as e:  # noqa: BLE001
                    status(f"reload failed, keeping the running code: {e!r}")
            try:
                lp.cycle(n)
                done += 1
                fails = 0
            except (CycleFail, SystemExit, OSError, http.client.HTTPException) as e:
                link = isinstance(e, (OSError, http.client.HTTPException)) or "could not read" in str(e)
                if isinstance(e, SystemExit) and not link:
                    raise  # a refused path or a closed gate: not retried here
                fails += 0 if link else 1
                status(f"cycle {n} STOPPED: {type(e).__name__}: {e}" + ("; link hiccup: waiting for the gate, then the next cycle" if link else ""))
                if link:
                    wait_gate()
                st = arm_state_retry()
                if st["grip"] < 0.75 and st["goal_grip"] < 0.75:
                    raise Fatal(f"stopped holding a block: {e}")
                lp.belief = {}
                lp.op.st["target"] = None
                lp.to_pose(LOOK_TIP, "end")
                if fails >= 2:
                    raise Fatal("two cycles in a row stopped")
            if n % 3 == 0 and n < cycles:
                lp.saves.join()
                status("pause 30 s (3 cycles)")
                time.sleep(30)
            lp.stop_check()
    except (Fatal, SystemExit, KeyboardInterrupt, Exception) as e:  # noqa: BLE001
        reason = f"{type(e).__name__}: {e}"
        try:
            st = arm_state()
            held_ok = st["grip"] > 0.75 or st["goal_grip"] > 0.75 and not st.get("grip_holding")
            if not st["moving"] and held_ok and "GATE" not in str(e):
                lp.op.st["target"] = None
                lp.to_pose(LOOK_TIP, "end")
        except BaseException as e2:  # noqa: BLE001
            reason += f"; end pose failed: {e2}"
            held_ok = False
    finally:
        lp.saves.join()
        rec("/stop")
        if held_ok:
            release_token()
            status(f"session {session.name} ended ({reason}); {done} cycles; token released")
        else:
            (TOKEN / "who.txt").write_text(f"data-loop HELD {time.strftime('%H:%M:%S')}: stopped in an unclean state ({reason})\n")
            status(f"session {session.name} ended ({reason}); {done} cycles; TOKEN KEPT (arm not in a clean state)")
        lp.sync()
        report(session)


def report(session):
    session = pathlib.Path(session)
    if not session.is_absolute():
        session = LOOP / session
    labs = []
    for d in ("dataset", "failures"):
        p = session / d / "labels.jsonl"
        if p.exists():
            labs += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    for sk in ("pick", "place"):
        r = [x for x in labs if x["skill"] == sk]
        ok = sum(x["success"] for x in r)
        print(f"{sk}: {len(r)} episodes, {ok} success, {len(r) - ok} failure" + (f", {ok / len(r) * 100:.0f}%" if r else ""))
    rows = [json.loads(x) for x in open(session / "calib_records.jsonl")] if (session / "calib_records.jsonl").exists() else []
    g = [r for r in rows if r["kind"] == "grasp"]
    for key in ("wrist_err_mm", "overhead_err_mm"):
        e = np.array([r[key] for r in g if r.get(key) is not None])
        if len(e):
            print(f"{key}: {len(e)} grasps, mean {e.mean(0).round(1).tolist()}, sd {e.std(0).round(1).tolist()}, rms "
                  f"{np.sqrt((e ** 2).sum(1).mean()):.1f} mm")
            for name, lo, hi, ax in (("x<33", 0, 0.33, 0), ("x>=33", 0.33, 1, 0), ("y<8", -1, 0.08, 1), ("y>=8", 0.08, 1, 1)):
                sel = [i for i, r in enumerate(r_ for r_ in g if r_.get(key) is not None) if lo <= r["truth_mm"][ax] / 1000 < hi]
                sel = [i for i, r in enumerate([r_ for r_ in g if r_.get(key) is not None]) if lo <= r["truth_mm"][ax] / 1000 < hi]
                if sel:
                    print(f"   {name}: {len(sel)}, mean {e[sel].mean(0).round(1).tolist()} mm")
    m = [r for r in rows if r["kind"] == "move_end"]
    by = {}
    for r in m:
        by.setdefault(r["direction"], []).append(r["tip_err_mm"])
    for k, v in by.items():
        v = np.array(v)
        print(f"command vs actual {k}: {len(v)} moves, mean {v.mean(0).round(1).tolist()} sd {v.std(0).round(1).tolist()}")
    pl = [r for r in rows if r["kind"] == "place" and r.get("offset_mm")]
    if pl:
        o = np.array([r["offset_mm"] for r in pl])
        print(f"place offsets: {len(o)}, mean |off| {np.hypot(*o.T).mean():.1f} mm, max {np.hypot(*o.T).max():.1f}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    elif sys.argv[1] == "report":
        report(sys.argv[2])
