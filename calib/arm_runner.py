"""Always-on runner for the real LEFT Metal arm: one process owns the arm and keeps it working while agents think.

The only thing that moves the arm while it runs. It holds runs/station/ARM_OWNER (mkdir, the token calib/data_loop.py
uses) from start to stop and talks to the team's left server on http://127.0.0.1:8691 through calib/opstack.py and
calib/data_loop.py, unchanged: the same gate before every motion (left enabled, not assigned, not recording, Dream not
running the left arm), the same TowerSafety and extra checks on every planned piece, the same background GateWatch
while a motion streams. Nothing is posted to 8081 and nothing to /quit.

What it does, in order, forever:
  1. a queued job, if one was accepted (highest priority first, first in first out within a priority);
  2. otherwise one data_loop cycle (look, scatter the three blocks, restack them; every pick and place recorded and
     labelled), unless the default loop is paused. Between cycles the queue is read, so a job waits at most one
     cycle; a priority 9 job also stops a cycle at its next block boundary (the jaws empty, the arm over the table).

Jobs are JSON files in runs/station/queue/ ({id, kind, args, priority 0-9, submitted_by}); kinds:
  cycle       {"n": 1}                         data_loop cycles, recorded
  pick_place  {"held": "blue", "place": [x, y, yaw], "top": 0, "pick": [x, y, yaw]?}   opstack.fast_go's arguments
              (place is fast_go's tower, top its blocks under the place spot); the block is picked from where a fresh
              look finds it (pick is only a hint), through data_loop's block(): the same recording, check and recovery
  look        {"pose": [x, y, z, yaw_deg]?}    one wrist look from the high pose; the fitted blocks come back
  views       {"poses": [[6 joints], ...], "grip": 0.8, "rate": 0.3}   joints as the server reports them; both
              frames saved at each, for calibration
  home        {"pose": "rest" | "look", "resume": false}   rest: the saved zero; then the default loop is paused
              unless resume is true
A job is accepted only if its dry run passes: the same code (RLoop.run_job) against calib/station_sim.py, a sim of
the station seeded with the runner's latest belief (blocks from its last look, the arm at the look pose). Rejected
jobs go to done/<id>.json with the reason. The real run still checks every piece again before it moves.

Recovery instead of stopping: a missed grasp gets a look from where it is and one retry (data_loop); a block that
slipped, dropped or came off its spot gets a fresh look and is picked from where it is; no free spot tries the next
candidates (data_loop) and then a new cycle; a block on its side is pushed over flat with closed jaws, then the cycle
starts again; a gate closing holds the arm still (the token and any held block kept), ALERT lines, and the motion
resumes from where it stopped once the gate opens. The runner stops only for: a gate closed over 5 minutes, a joint
lag abort or other arm server error, a block off the table, a motor over 60 C (when the server reports temperatures),
a block held in a state it cannot account for, or STOP.

Files (runs/station/queue/): STATUS.txt (a line per block, job and alert), ALERTS.txt, latest.jpg (the last check
frame), stats.json, world.json (the last belief), runner.json, PAUSE and STOP (control), accepted/ running/ done/
cancel/ validate/.

    python -m calib.arm_runner run [--cycles N] [--sim [--speed K] [--seed S]]
    python -m calib.arm_runner submit KIND ['{"args": ...}'] [--priority P] [--by NAME] [--id ID]
    python -m calib.arm_runner list | cancel ID | pause | resume | stop | status
    python -m calib.arm_runner inject '{"kind": "gate", "seconds": 20}'          (--sim runs only)
Every command takes --sim to use runs/sim_station/ instead of runs/station/.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

import numpy as np

from calib import data_loop as D
from calib import opstack as O
from calib.opstack import LOOK_TIP, TABLE_Z, Plan, arm_state, gate
from calib.tower_plan import FACE, H, WORKSPACE, Shape

ROOT = O.ROOT
PY = sys.executable
KINDS = ("cycle", "pick_place", "look", "views", "home")
COLOURS = D.COLOURS
GATE_MAX_S = 300
LOOK2 = (0.31, 0.06, 0.21, 90.0)  # data_loop's second look
REAL = {"arm": "http://127.0.0.1:8691", "team": "http://127.0.0.1:18081/api/state", "dream": "http://127.0.0.1:18084/api/state",
        "rec": "http://127.0.0.1:8697"}
REAL_PORTS = (8691, 8081, 18081, 18084, 8697)
_wall = time  # the real clock, for the files' timestamps and the validator's polling


class Preempt(Exception):
    pass


class Cancelled(Exception):
    pass


class StopRequested(Exception):
    pass


class Tipped(D.CycleFail):
    def __init__(self, colour, fit):
        super().__init__(f"{colour} block on its side at ({fit['x'] * 100:.1f}, {fit['y'] * 100:.1f}) cm")
        self.colour, self.fit = colour, fit


def jdump(v):
    return json.loads(json.dumps(v, default=lambda x: np.asarray(x).tolist()))


def write_json(path: pathlib.Path, obj):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(jdump(obj), indent=1))
    os.replace(tmp, path)


def sim_urls(base):
    return {"arm": base, "team": base + "/team/api/state", "dream": base + "/dream/api/state", "rec": base + "/rec"}


def use_backend(urls, clock):
    """Point opstack and data_loop at a station (the real one or a sim) and a clock. The real station's URLs are
    the modules' own; a sim must be on none of the real ports."""
    if urls["arm"] != REAL["arm"]:
        for u in urls.values():
            port = int(u.split(":")[2].split("/")[0])
            if port in REAL_PORTS:
                raise SystemExit(f"a sim URL on a real station port: {u}")
    O.URL, O.TEAM, O.DREAM, D.REC = urls["arm"], urls["team"], urls["dream"], urls["rec"]
    O.time = D.time = clock
    globals()["time"] = clock


def sim_clock(base):
    from calib.station_sim import Clock
    c = json.loads(urllib.request.urlopen(base + "/clock", timeout=5).read())
    return Clock(c["k"], c["real0"])


def wait_up(base, proc=None, timeout=90):
    t0 = _wall.time()
    while _wall.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise SystemExit(f"the sim at {base} exited ({proc.returncode})")
        try:
            urllib.request.urlopen(base + "/clock", timeout=2).read()
            return
        except Exception:  # noqa: BLE001
            _wall.sleep(0.3)
    raise SystemExit(f"the sim at {base} did not come up")


def start_sim(port, speed, mode, seed, log):
    proc = subprocess.Popen([PY, "-m", "calib.station_sim", "--port", str(port), "--speed", str(speed), "--mode", mode, "--seed", str(seed)],
                            cwd=ROOT, stdout=open(log, "a"), stderr=subprocess.STDOUT, start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    wait_up(base, proc)
    return proc, base


def free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Paths:
    def __init__(self, base: pathlib.Path):
        self.base = base
        self.q = base / "queue"
        self.accepted, self.running, self.done = self.q / "accepted", self.q / "running", self.q / "done"
        self.cancel, self.work = self.q / "cancel", self.q / "validate"
        self.status, self.alerts, self.latest = self.q / "STATUS.txt", self.q / "ALERTS.txt", self.q / "latest.jpg"
        self.stats, self.world, self.info = self.q / "stats.json", self.q / "world.json", self.q / "runner.json"
        self.pause, self.stop = self.q / "PAUSE", self.q / "STOP"
        self.token, self.loop = base / "ARM_OWNER", base / "loop"

    def make(self):
        for d in (self.q, self.accepted, self.running, self.done, self.cancel, self.work, self.loop):
            d.mkdir(parents=True, exist_ok=True)


def paths_for(a) -> Paths:
    if getattr(a, "root", None):
        return Paths(pathlib.Path(a.root).resolve())
    return Paths(ROOT / "runs" / ("sim_station" if a.sim else "station"))


# ---------------------------------------------------------------------------------------------------------------
# validation without motion
def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def static_check(job):
    """(ok, why) for a job's shape: kind, priority, arguments in range."""
    if not isinstance(job, dict):
        return False, "not a JSON object"
    jid = job.get("id")
    if not isinstance(jid, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", jid):
        return False, f"bad id {jid!r} (letters, digits, _ . -, at most 64)"
    k = job.get("kind")
    if k not in KINDS:
        return False, f"unknown kind {k!r} (one of {', '.join(KINDS)})"
    p = job.get("priority", 5)
    if not isinstance(p, int) or isinstance(p, bool) or not 0 <= p <= 9:
        return False, f"priority {p!r} must be an integer 0-9"
    a = job.get("args") or {}
    if not isinstance(a, dict):
        return False, "args must be an object"
    if k == "cycle":
        n = a.get("n", 1)
        if not isinstance(n, int) or not 1 <= n <= 5:
            return False, "cycle n must be 1-5"
    elif k == "pick_place":
        if a.get("held") not in COLOURS:
            return False, f"held must be one of {COLOURS}"
        pl = a.get("place", a.get("tower"))
        if not (isinstance(pl, list) and len(pl) == 3 and all(_num(v) for v in pl)):
            return False, "place (or tower) must be [x, y, yaw] in m, m, rad"
        if not (WORKSPACE[0][0] < pl[0] < WORKSPACE[0][1] and WORKSPACE[1][0] < pl[1] < WORKSPACE[1][1]):
            return False, f"place ({pl[0]:.3f}, {pl[1]:.3f}) outside the block workspace {WORKSPACE}"
        top = a.get("top", 0)
        if not isinstance(top, int) or not 0 <= top <= 2:
            return False, "top must be 0, 1 or 2 (blocks under the place spot)"
        if "pick" in a and not (isinstance(a["pick"], list) and len(a["pick"]) == 3 and all(_num(v) for v in a["pick"])):
            return False, "pick must be [x, y, yaw]"
    elif k == "look":
        pose = a.get("pose")
        if pose is not None and not (isinstance(pose, list) and len(pose) == 4 and all(_num(v) for v in pose) and 0.15 <= pose[2] <= 0.30):
            return False, "look pose must be [x, y, z, yaw_deg] with z 0.15-0.30 m"
    elif k == "views":
        poses = a.get("poses")
        if not (isinstance(poses, list) and 1 <= len(poses) <= 30):
            return False, "views needs 1-30 poses"
        for i, q in enumerate(poses):
            if not (isinstance(q, list) and len(q) == 6 and all(_num(v) and abs(v) < math.pi for v in q)):
                return False, f"pose {i} must be 6 joint angles (rad, |q| < pi)"
        if not _num(a.get("grip", 0.8)) or not 0 <= a.get("grip", 0.8) <= 1:
            return False, "grip 0-1"
        if not _num(a.get("rate", 0.3)) or not 0.02 <= a.get("rate", 0.3) <= 0.3:
            return False, "rate 0.02-0.3 rad/s"
    elif k == "home":
        if a.get("pose", "rest") not in ("rest", "look"):
            return False, "home pose is rest or look"
    return True, "ok"


# ---------------------------------------------------------------------------------------------------------------
class RLoop(D.Loop):
    """data_loop's Loop with the runner's hooks: block boundaries (stop, cancel, preempt), gate holds that resume,
    recovery from a slipped, dropped or tipped block, the check frame published, and the jobs."""

    def __init__(self, session, hooks):
        super().__init__(session)
        self.hooks = hooks
        self.pics = []
        self.cur = None
        self.recov = collections.Counter()

    # ---- hooks into data_loop
    def stop_check(self):
        self.hooks.boundary(self)

    # every override passes data_loop's own arguments through (*a, **kw): that file changes under us
    def go(self, pl, grip, tag, *a, **kw):
        for attempt in range(4):
            try:
                return super().go(pl, grip, tag, *a, **kw)
            except SystemExit as e:
                if "GATE" not in str(e) or attempt == 3:
                    raise
                self.hooks.alert(f"gate closed during '{tag}': the arm stopped and holds there ({e})")
                self.hooks.wait_gate()
                pl = self.resume_plan(pl, tag)
                self.recov["gate resume"] += 1
                D.status(f"gate open again: '{tag}' resumes from where it stopped ({len(pl.nodes) - 1} pieces left)")

    def resume_plan(self, pl, tag):
        """The rest of a stopped plan, from the measured joints: the nearest node on, every piece checked again."""
        op = self.op
        s = arm_state()
        qm = np.array(s["q"]) + op.dq
        best = (1e9, 0)
        for j, ((a, _, _), (b, _, _)) in enumerate(zip(pl.nodes[:-1], pl.nodes[1:])):  # nearest point on each piece
            ab = b - a
            t = float(np.clip(np.dot(qm - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0, 1))
            d = float(np.abs(a + ab * t - qm).max())
            if d < best[0]:
                best = (d, j)
        d, j = best
        if d > 0.10:
            raise D.CycleFail(f"after the gate the arm is {d:.2f} rad from its plan for '{tag}' (moved by someone?): not resuming it")
        new = Plan(op, qm, pl.opening)
        new.R = pl.R
        new.nodes += pl.nodes[j + 1:]
        new.check(skip=self.cur)
        return new

    def to_pose(self, tip, tag, *a, **kw):
        for attempt in range(4):
            try:
                return super().to_pose(tip, tag, *a, **kw)
            except SystemExit as e:
                if "GATE" not in str(e) or attempt == 3:
                    raise
                self.hooks.alert(f"gate closed during '{tag}': the arm stopped and holds there ({e})")
                self.hooks.wait_gate()
                self.op.st["target"] = None
                self.recov["gate resume"] += 1

    def grip_to(self, s, g, *a, **kw):
        self.hooks.wait_gate()
        return super().grip_to(s, g, *a, **kw)

    def look(self, tag, *a, **kw):
        bel, stem = super().look(tag, *a, **kw)
        self.pics.append(str(self.op.run / f"{stem}-ov.jpg"))
        if bel is not None:
            self.hooks.world(self)
        return bel, stem

    def check_place(self, colour, *a, **kw):
        fit, stem = super().check_place(colour, *a, **kw)
        ov = self.op.run / f"{stem}-ov.jpg"
        self.pics.append(str(ov))
        self.hooks.picture(ov)
        return fit, stem

    def pick_spot(self, colour, levels, *a, **kw):
        self._levels = levels
        return super().pick_spot(colour, levels, *a, **kw)

    def feasible(self, colour, target, *a, **kw):
        """data_loop's check (pick from where it is, place at the spot, the check photo), plus: the block can be picked
        again from the spot, at every level it may end up at there. Without this the loop set blocks where no pick pose
        solves (x 40 cm, y 16 cm, yaw 37 deg in the sim) and every later cycle failed on them."""
        if not super().feasible(colour, target, *a, **kw):
            return False
        tx, ty, tyaw = target[:3]
        saved = self.belief[colour]
        q0 = self.q_look if self.q_look is not None else self.start_q()
        try:
            for lvl in getattr(self, "_levels", None) or [target[3]]:
                self.belief[colour] = dict(saved, x=tx, y=ty, yaw=tyaw, level=lvl)
                self.plan_pick(colour, q0)
            return True
        except SystemExit as e:
            self.last_refusal = f"pick back from the spot: {e}"
            return False
        finally:
            self.belief[colour] = saved

    def release(self, s, *a, **kw):
        self.hooks.wait_gate()
        return super().release(s, *a, **kw)

    def block(self, colour, target, cycle, k, phase, recovery=False, **kw):
        prev, self.cur = self.cur, colour
        try:
            return super().block(colour, target, cycle, k, phase, recovery, **kw)
        except D.CycleFail as e:
            msg = str(e)
            if recovery or isinstance(e, Tipped) or not any(w in msg for w in ("slipped out", "not found flat after the place", "not found where it was hidden",
                                                                               "not found after a missed grasp")):
                raise
            D.status(f"recovery for {colour}: {msg}; a fresh look, then a pick from where it is")
            self.recov["re-look and pick from where it is"] += 1
            self.find_again(colour)
            return super().block(colour, target, cycle, k, phase, recovery=True)
        finally:
            self.cur = prev

    # ---- finding a block that moved
    def find_again(self, colour):
        """A fresh look (all three blocks, flat, any level); if that fails, the blocks on their sides; nothing:
        off the table."""
        for tag, tip in (("search", LOOK_TIP), ("search2", LOOK2)):
            bel, stem = self.look(f"{tag}-{colour}", tip)
            if bel is not None and bel[colour]["iou"] < 0.8:
                standing, _ = self.sides()
                if colour in standing:
                    raise Tipped(colour, standing[colour])
            if bel is not None:
                D.status(f"found {colour} at ({bel[colour]['x'] * 100:.1f}, {bel[colour]['y'] * 100:.1f}) L{bel[colour]['level']}")
                return bel
        standing, missing = self.sides()
        if colour in standing:
            raise Tipped(colour, standing[colour])
        if standing:
            c = next(iter(standing))
            raise Tipped(c, standing[c])
        if missing:
            raise D.Fatal(f"block off the table: {', '.join(missing)} not seen from two looks")
        raise D.CycleFail(f"{colour}: the looks see every block but cannot fit three flat ones")

    def sides(self):
        """From the high pose (and the second look pose if needed): each colour seen flat, on its side (a standing fit
        beats the flat one), possibly hidden under a block one level up (data_loop's rule), or not seen at all."""
        view = self.op.view
        flat0 = Shape((FACE, FACE, H), TABLE_Z)
        side = Shape((FACE, H, FACE), TABLE_Z, math.pi)
        standing, seen, hidden = {}, set(), set()
        for tag, tip in (("sides", LOOK_TIP), ("sides2", LOOK2)):
            self.to_pose(tip, "look")
            time.sleep(0.25)
            img, q, grip, stem = self.op.snap(tag)
            self.pics.append(str(self.op.run / f"{stem}-wrist.jpg"))
            cands, _, _, _ = self.fits_all(img, q, grip)
            for c in COLOURS:
                if any(not A.get("hidden") for A in cands[c]):
                    seen.add(c)
                elif cands[c]:
                    hidden.add(c)
                fs = view.find(img, c, q, grip * 0.1, side)
                if fs is None:
                    continue
                seen.add(c)
                ff = view.find(img, c, q, grip * 0.1, flat0)
                if fs["iou"] >= 0.7 and (ff is None or fs["iou"] > ff["iou"] + 0.05):
                    standing[c] = fs
            if set(COLOURS) <= seen | hidden:
                break
        missing = [c for c in COLOURS if c not in seen and c not in hidden]
        D.status("side check: " + ", ".join(f"{c} " + ("on its side" if c in standing else "missing" if c in missing else
                                                        "maybe under another" if c not in seen else "flat") for c in COLOURS))
        return standing, missing

    def push_over(self, colour, fit):
        """A block on its side: closed jaws pushed through its top along the hole (its 34.5 mm base), so it falls
        flat. The push starts 3.5 cm before the face, 4.2 cm over the table, and ends 2 cm past the far face."""
        op = self.op
        x, y, yaw = fit["x"], fit["y"], fit["yaw"]
        u = np.array([-math.sin(yaw), math.cos(yaw)])  # the hole axis (the shape's local y)
        z = TABLE_Z + 0.042  # 42 mm up a 50 mm tall block: over b / (2 mu) = 34.5 mm, so it tips rather than slides
        last = None
        for sgn in (1, -1):
            d = sgn * u
            land = np.array([x, y]) + d * 0.035
            if not (D.AREA[0][0] - 0.03 < land[0] < D.AREA[0][1] + 0.03 and D.AREA[1][0] - 0.03 < land[1] < D.AREA[1][1] + 0.03):
                last = "it would land outside the area"
                continue
            if any(math.hypot(land[0] - b["x"], land[1] - b["y"]) < 0.09 for c, b in self.belief.items() if c != colour):
                last = "it would land on another block"
                continue
            p0 = np.array([x, y]) - d * (H / 2 + 0.035)
            p1 = np.array([x, y]) + d * (H / 2 + 0.020)
            jy = math.degrees(math.atan2(d[1], d[0]))  # the jaws travel along the push: the closed pads lead with their faces
            pl = Plan(op, self.start_q(), 0.03)
            try:
                pl.to([p0[0], p0[1], 0.10], jy, 0.4, "over")
                pl.to([p0[0], p0[1], z], jy, 0.25, "down", line=0.01, keep=True)
                pl.to([p1[0], p1[1], z], jy, 0.15, "push", line=0.005, keep=True)
                pl.to([p1[0], p1[1], 0.10], jy, 0.3, "up", line=0.01, keep=True)
                pl.check(skip=colour)
                self.sweep(pl, exempt={colour})
            except SystemExit as e:
                last = str(e)
                continue
            s = arm_state()
            self.grip_to(s, 0.25)
            self.go(pl, 0.25, "push over", "level")
            self.recov["pushed over"] += 1
            D.status(f"pushed the {colour} block over along ({d[0]:+.2f}, {d[1]:+.2f})")
            return True
        D.status(f"no safe push for the {colour} block: {last}")
        return False

    # ---- jobs (the runner and its dry runs both come here)
    def run_job(self, job):
        k, a = job["kind"], job.get("args") or {}
        op = self.op
        if k == "look":
            bel, stem = self.look("job-look", tuple(a.get("pose", LOOK_TIP)))
            if bel is None:
                raise D.CycleFail("the look did not find three flat blocks")
            return {"blocks": bel, "stem": stem, "obstacle_points": len(self.obstacles)}
        if k == "cycle":
            for _ in range(int(a.get("n", 1))):
                self.cycle(self.hooks.next_cycle())
            return {"cycles": int(a.get("n", 1))}
        if k == "pick_place":
            return self.pick_place(a)
        if k == "views":
            out = []
            grip, rate = float(a.get("grip", 0.8)), float(min(a.get("rate", 0.3), 0.3))
            for i, q in enumerate(a["poses"]):
                q_model = np.array(q, float) + op.dq
                q0 = self.start_q()
                fake = Plan(op, q0, grip * 0.1)
                fake.nodes.append((q_model, rate, f"view {i}"))
                try:
                    self.sweep(fake, exempt=set())
                except SystemExit as e:
                    raise D.CycleFail(f"view {i} refused: {e}")
                for attempt in range(4):
                    self.hooks.wait_gate()
                    try:
                        res = op.move(q_model, grip, rate, f"view{i:02d}", raw=True, note="calibration view")
                        break
                    except SystemExit as e:
                        if "GATE" not in str(e) or attempt == 3:
                            raise D.CycleFail(f"view {i} refused: {e}")
                        self.hooks.alert(f"gate closed during view {i}: holding ({e})")
                time.sleep(0.3)
                img, qq, g, stem = op.snap(f"view{i:02d}")
                pics = [str(op.run / f"{stem}-wrist.jpg"), str(op.run / f"{stem}-overhead.jpg")]
                self.pics += pics
                out.append({"pose": i, "stem": stem, "pictures": pics, "joints_measured": (qq - op.dq).round(4).tolist(),
                            "joint_err_deg": res["joint_err_deg"], "tip_measured_mm": res["tip_measured_mm"]})
                self.stop_check()
            self.to_pose(LOOK_TIP, "views end")
            return {"views": out}
        if k == "home":
            self.to_pose(LOOK_TIP, "home look")
            if a.get("pose", "rest") == "rest":
                s = arm_state()
                start = s.get("start_joints") or op.st["start"]
                q_model = np.array(start, float) + op.dq
                for attempt in range(4):
                    self.hooks.wait_gate()
                    try:
                        op.move(q_model, float(a.get("grip", 0.27)), 0.3, "home", home=True, raw=True, note="back to the saved zero")
                        break
                    except SystemExit as e:
                        if "GATE" not in str(e) or attempt == 3:
                            raise D.CycleFail(f"home refused: {e}")
                        self.hooks.alert(f"gate closed on the way home: holding ({e})")
            self.hooks.after_home(job)
            return {"pose": a.get("pose", "rest")}
        raise D.CycleFail(f"unknown kind {k}")

    def pick_place(self, a):
        held = a["held"]
        px, py, pyaw = (float(v) for v in a.get("place", a.get("tower")))
        top = int(a.get("top", 0))
        bel, stem = self.look("job-pp-look")
        if bel is None:
            bel, stem = self.look("job-pp-look2", LOOK2)
        if bel is None:
            raise D.CycleFail("the look did not find three flat blocks")
        b = bel[held]
        notes = []
        if a.get("pick"):
            dd = math.hypot(a["pick"][0] - b["x"], a["pick"][1] - b["y"])
            if dd > 0.03:
                notes.append(f"the look found {held} {dd * 1000:.0f} mm from the pick hint; picked where the look saw it")
        above = [c for c, o in bel.items() if c != held and o["level"] == b["level"] + 1 and math.hypot(o["x"] - b["x"], o["y"] - b["y"]) < 0.03]
        if above:
            raise D.CycleFail(f"the {held} block has the {above[0]} block on it")
        sup = None
        if top > 0:
            under = [c for c, o in bel.items() if c != held and o["level"] == top - 1 and math.hypot(o["x"] - px, o["y"] - py) < 0.02]
            if not under:
                raise D.CycleFail(f"nothing at level {top - 1} within 2 cm of the place spot ({px:.3f}, {py:.3f})")
            sup = under[0]
            on_sup = [c for c, o in bel.items() if c not in (held, sup) and o["level"] == top and math.hypot(o["x"] - px, o["y"] - py) < 0.03]
            if on_sup:
                raise D.CycleFail(f"the {sup} block already has the {on_sup[0]} block on it")
        else:
            near = [(c, math.hypot(o["x"] - px, o["y"] - py)) for c, o in bel.items() if c != held and c not in self.below(held)]
            near = [(c, d) for c, d in near if d < 0.095]
            if near:
                raise D.CycleFail(f"the place spot is not free: the {near[0][0]} block is {near[0][1] * 100:.1f} cm away (9.5 cm needed)")
        self.block(held, (px, py, pyaw, top, sup), 0, 1, "job")
        rows = [json.loads(x) for x in open(O.RECORDS)] if O.RECORDS.exists() else []
        place = next((r for r in reversed(rows) if r["kind"] == "place" and r.get("block") == held), None)
        grasp = next((r for r in reversed(rows) if r["kind"] == "grasp" and r.get("block") == held), None)
        self.to_pose(LOOK_TIP, "end")
        return {"placed": {"offset_mm": place and place.get("offset_mm"), "yaw_off_deg": place and place.get("yaw_off_deg"),
                           "fit": place and place.get("fit")}, "grasp_wrist_err_mm": grasp and grasp.get("wrist_err_mm"),
                "picked_from": b, "support": sup, "notes": notes}


# ---------------------------------------------------------------------------------------------------------------
class Runner:
    def __init__(self, p: Paths, sim_base=None, val_base=None, max_cycles=None):
        self.p, self.sim_base, self.val_base, self.max_cycles = p, sim_base, val_base, max_cycles
        self.ncycle = 0
        self.job = None
        self.in_default = False
        self.state = "starting"
        self.c = collections.Counter()
        self.fails_in_row = 0
        self.stopping = False
        self.offset = 0.0
        self.world_ready = threading.Event()
        self.session = None
        self.lp = None
        self.gate_t0 = None

    # ---- lines
    def alert(self, msg):
        self.c["alerts"] += 1
        D.status(f"ALERT {msg}")
        with open(self.p.alerts, "a") as fh:
            fh.write(f"{_wall.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

    def sim_truth(self, why):
        """Sim runs only: where the blocks really are, for the log (the runner itself never reads it)."""
        if not self.sim_base:
            return
        try:
            t = json.loads(urllib.request.urlopen(self.sim_base + "/truth", timeout=10).read())
            D.status(f"[sim truth after {why[:60]}] " + ", ".join(
                f"{c} ({b['x'] * 100:.1f}, {b['y'] * 100:.1f}, z {b['z'] * 100:.1f}) {'flat L' + str(b['level']) if b['flat'] else 'ON ITS SIDE'}"
                f"{'' if b['on_table'] else ' OFF THE TABLE'}" for c, b in t.items()))
        except Exception as e:  # noqa: BLE001
            D.status(f"[sim truth unavailable: {e!r}]")

    # ---- hooks for RLoop
    def boundary(self, lp):
        self.write_stats()
        if self.p.stop.exists():
            raise StopRequested("STOP file")
        if self.job is not None:
            if (self.p.cancel / self.job["id"]).exists():
                raise Cancelled("cancelled")
            return
        if self.in_default:
            top = self.waiting()
            if top and top[0]["priority"] >= 9:
                raise Preempt(f"priority {top[0]['priority']} job {top[0]['id']}")
            if self.p.pause.exists():
                raise Preempt("default loop paused")

    def wait_gate(self):
        ok, why = gate()
        if ok:
            if self.gate_t0 is not None:
                D.status(f"gate open after {time.time() - self.gate_t0:.0f} s: resuming")
                self.gate_t0 = None
            return
        t0 = self.gate_t0 = time.time()
        prev, last_alert = self.state, -1e9
        self.state = "holding: gate closed"
        self.write_stats()
        while True:
            ok, why = gate()
            if ok:
                D.status(f"gate open after {time.time() - t0:.0f} s: resuming")
                self.gate_t0 = None
                self.state = prev
                return
            if time.time() - t0 > GATE_MAX_S:
                raise D.Fatal(f"gate closed for over {GATE_MAX_S // 60} minutes: {why}")
            if time.time() - last_alert > 60:
                self.alert(f"gate closed ({why}): holding still, token kept, {time.time() - t0:.0f} s so far (stop at {GATE_MAX_S} s)")
                last_alert = time.time()
            if self.p.stop.exists():
                raise StopRequested("STOP file while the gate was closed")
            time.sleep(2.0)

    def world(self, lp):
        s = arm_state()
        w = {"blocks": lp.belief, "q": s["goal"], "q_measured": s["q"], "t": _wall.strftime("%H:%M:%S"), "session": self.session.name}
        write_json(self.p.world, w)
        self.world_ready.set()

    def picture(self, path):
        try:
            shutil.copyfile(path, self.p.latest.with_suffix(".tmp.jpg"))
            os.replace(self.p.latest.with_suffix(".tmp.jpg"), self.p.latest)
        except OSError:
            pass
        self.write_stats()

    def next_cycle(self):
        self.ncycle += 1
        return self.ncycle

    def after_home(self, job):
        if (job.get("args") or {}).get("pose", "rest") == "rest" and not (job.get("args") or {}).get("resume"):
            self.p.pause.write_text(f"paused by home job {job['id']} at {_wall.strftime('%H:%M:%S')}\n")
            D.status(f"home: at the saved zero; default loop paused (resume: python -m calib.arm_runner resume)")

    def now(self):
        return time.time() + self.offset

    # ---- the queue
    def waiting(self):
        jobs = []
        for f in self.p.accepted.glob("*.json"):
            try:
                j = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            jobs.append(j)
        jobs.sort(key=lambda j: (-j["priority"], j.get("accepted_seq", 0)))
        return jobs

    def pending(self):
        return [f for f in self.p.q.glob("*.json") if f.name not in ("stats.json", "world.json", "runner.json")]

    def finish(self, job, status, **kw):
        out = dict(job, status=status, **kw)
        write_json(self.p.done / f"{job['id']}.json", out)
        for d in (self.p.running, self.p.accepted):
            (d / f"{job['id']}.json").unlink(missing_ok=True)
        (self.p.cancel / job["id"]).unlink(missing_ok=True)
        self.c[f"jobs {status}"] += 1

    # ---- validation (a thread; dry runs in a subprocess against the validation sim)
    def validator(self):
        seq = 0
        while not self.stopping:
            try:
                files = sorted(self.p.q.glob("*.json"), key=lambda f: f.stat().st_mtime)
            except OSError:
                files = []
            for f in files:
                if self.stopping:
                    break
                if f.name in ("stats.json", "world.json", "runner.json"):
                    continue
                seq += 1
                try:
                    self.validate(f, seq)
                except Exception as e:  # noqa: BLE001
                    D.status(f"validator error on {f.name}: {e!r}")
                    traceback.print_exc()
            _wall.sleep(0.5)

    def validate(self, f: pathlib.Path, seq):
        try:
            job = json.loads(f.read_text())
        except ValueError:
            if _wall.time() - f.stat().st_mtime < 3:
                return  # still being written
            job = {"id": f.stem, "raw": f.read_text()[:400]}
            f.unlink(missing_ok=True)
            return self.finish(job, "rejected", reason="not valid JSON", finished=_wall.strftime("%H:%M:%S"))
        except OSError:
            return
        if isinstance(job, dict):
            job.setdefault("id", f.stem)
            job.setdefault("priority", 5)
            job.setdefault("args", {})
            job.setdefault("submitted_by", "?")
            job.setdefault("submitted_at", _wall.strftime("%Y-%m-%d %H:%M:%S", _wall.localtime(f.stat().st_mtime)))
        jid = job.get("id") if isinstance(job, dict) else f.stem
        ok, why = static_check(job)
        if ok and any((d / f"{jid}.json").exists() for d in (self.p.accepted, self.p.running, self.p.done)):
            ok, why = False, f"duplicate id {jid}"
        if ok and (self.p.cancel / jid).exists():
            f.unlink(missing_ok=True)
            return self.finish(job, "cancelled", reason="cancelled before it was validated")
        if not ok:
            f.unlink(missing_ok=True)
            job = job if isinstance(job, dict) else {"id": f.stem, "raw": job}
            D.status(f"job {jid} rejected: {why}")
            return self.finish(job, "rejected", reason=why, stage="static check", finished=_wall.strftime("%H:%M:%S"))
        if not self.world_ready.is_set():
            return  # dry runs start from the first look's belief
        t0 = _wall.time()
        work = self.p.work / jid
        work.mkdir(parents=True, exist_ok=True)
        write_json(work / "job.json", job)
        shutil.copyfile(self.p.world, work / "world.json")
        try:
            r = subprocess.run([PY, "-m", "calib.arm_runner", "dryrun", "--job", str(work / "job.json"), "--world", str(work / "world.json"),
                                "--url", self.val_base, "--out", str(work / "verdict.json"), "--work", str(work)],
                               cwd=ROOT, capture_output=True, text=True, timeout=900)
            verdict = json.loads((work / "verdict.json").read_text()) if (work / "verdict.json").exists() else \
                {"ok": False, "reason": f"dry run crashed: {(r.stderr or r.stdout)[-600:]}"}
        except subprocess.TimeoutExpired:
            verdict = {"ok": False, "reason": "dry run took over 15 minutes"}
        verdict["wall_s"] = round(_wall.time() - t0, 1)
        if (self.p.cancel / jid).exists():
            f.unlink(missing_ok=True)
            return self.finish(job, "cancelled", reason="cancelled while it was validated", validation=verdict)
        if verdict.get("ok"):
            job["validation"] = verdict
            job["accepted_seq"] = seq
            job["accepted_at"] = _wall.strftime("%Y-%m-%d %H:%M:%S")
            write_json(self.p.accepted / f"{jid}.json", job)
            f.unlink(missing_ok=True)
            D.status(f"job {jid} ({job['kind']}, priority {job['priority']}, from {job['submitted_by']}) accepted: dry run passed in "
                     f"{verdict['wall_s']:.0f} s")
        else:
            f.unlink(missing_ok=True)
            D.status(f"job {jid} rejected: dry run: {verdict.get('reason')}")
            self.finish(job, "rejected", reason=f"dry run: {verdict.get('reason')}", stage="dry run", validation=verdict,
                        finished=_wall.strftime("%H:%M:%S"))

    # ---- stats
    def write_stats(self):
        lp = self.lp
        if lp is None:
            return
        rows = []
        if O.RECORDS.exists():
            rows = [json.loads(x) for x in open(O.RECORDS) if x.strip()]
        g = [r for r in rows if r["kind"] == "grasp"]
        err = {}
        for key, name in (("wrist_err_mm", "wrist"), ("overhead_err_mm", "overhead")):
            e = np.array([r[key] for r in g if r.get(key) is not None], float).reshape(-1, 2)
            if len(e):
                err[name] = {"n": len(e), "mean_mm": e.mean(0).round(1).tolist(), "sd_mm": e.std(0).round(1).tolist(),
                             "rms_mm": round(float(np.sqrt((e ** 2).sum(1).mean())), 1)}
        pl = [r for r in rows if r["kind"] == "place" and r.get("offset_mm")]
        bt = np.array(lp.block_times) if lp.block_times else None
        cyc = {k: self.c[f"cycles {k}"] for k in ("completed", "failed", "preempted")}
        tried = cyc["completed"] + cyc["failed"]
        pick, place = lp.counts["pick"], lp.counts["place"]
        st = {"updated": _wall.strftime("%Y-%m-%d %H:%M:%S"), "mode": "sim" if self.sim_base else "real", "session": self.session.name,
              "state": self.state, "cycles": cyc, "cycle_success_rate": round(cyc["completed"] / tried, 3) if tried else None,
              "picks": {"ok": pick[0], "failed": pick[1]}, "places": {"ok": place[0], "failed": place[1]},
              "block_success_rate": round(place[0] / (place[0] + place[1]), 3) if place[0] + place[1] else None,
              "seconds_per_block": None if bt is None else {"n": len(bt), "mean": round(float(bt.mean()), 1), "median": round(float(np.median(bt)), 1),
                                                           "last": round(float(bt[-1]), 1)},
              "grasp_truth_err": err,
              "place_offset_mm": {"n": len(pl), "mean": round(float(np.mean([math.hypot(*r["offset_mm"]) for r in pl])), 1)} if pl else None,
              "jobs": {k.split(" ", 1)[1]: v for k, v in self.c.items() if k.startswith("jobs ")},
              "recoveries": dict(lp.recov), "alerts": self.c["alerts"], "queue": [j["id"] for j in self.waiting()],
              "paused": self.p.pause.exists()}
        try:
            write_json(self.p.stats, st)
        except OSError:
            pass

    # ---- token
    def take_token(self):
        t0 = time.time()
        while True:
            try:
                self.p.token.mkdir()
                break
            except FileExistsError:
                who = (self.p.token / "who.txt").read_text().strip() if (self.p.token / "who.txt").exists() else "?"
                if time.time() - t0 < 1:
                    self.alert(f"waiting for the arm token, held by: {who}")
                if self.p.stop.exists():
                    raise SystemExit("STOP while waiting for the token")
                _wall.sleep(5)
        (self.p.token / "who.txt").write_text(f"arm-runner ({'sim' if self.sim_base else 'real'}) pid {os.getpid()} since "
                                              f"{_wall.strftime('%Y-%m-%d %H:%M:%S')}; queue at {self.p.q}\n")

    # ---- the loop
    def holding(self):
        s = arm_state()
        return s["grip"] < 0.75 and s["goal_grip"] < 0.75 and s["grip"] >= D.GRIP_HELD - 0.02

    def recover(self, lp, why):
        """After a cycle or job stopped: the jaws empty, back to the look pose from wherever the arm is."""
        s = arm_state()
        if s.get("error"):
            raise D.Fatal(f"arm server error: {s['error']}")
        if self.holding():
            raise D.Fatal(f"holding a block after: {why}")
        lp.op.st["target"] = None
        try:
            lp.to_pose(LOOK_TIP, "end")
        except SystemExit as e:
            if "GATE" in str(e):
                raise
            D.status(f"the way to the look pose was refused with the old belief ({e}); planned without it")
            lp.belief = {}
            lp.put_blocks()
            lp.to_pose(LOOK_TIP, "end")

    def tipped(self, lp, e: Tipped):
        self.alert(f"{e}: pushing it over flat")
        for attempt in range(2):
            if not lp.push_over(e.colour, e.fit):
                break
            standing, missing = lp.sides()
            if e.colour not in standing:
                D.status(f"{e.colour} is flat again")
                self.recover(lp, "push over")
                return
            e.fit = standing[e.colour]
        self.alert(f"{e.colour} block still on its side: it needs a hand; default loop paused")
        self.p.pause.write_text(f"paused: {e.colour} block on its side ({_wall.strftime('%H:%M:%S')})\n")
        self.recover(lp, "push over failed")

    def run_cycle(self, lp):
        n = self.next_cycle()
        self.in_default = True
        self.state = f"cycle {n}"
        t0 = time.time()
        nb = len(lp.block_times)
        try:
            lp.cycle(n)
            self.c["cycles completed"] += 1
            self.fails_in_row = 0
            moved = len(lp.block_times) - nb
            self.idle_cycles = 0 if moved else getattr(self, "idle_cycles", 0) + 1
            D.status(f"cycle {n} took {time.time() - t0:.0f} s, {moved} blocks moved")
            if self.idle_cycles >= 2:
                self.alert(f"{self.idle_cycles} cycles in a row moved no block (data_loop found nothing it could pick or place): "
                           f"the arm is not collecting; see the fallback lines")
        except Preempt as e:
            self.c["cycles preempted"] += 1
            D.status(f"cycle {n} stopped at a block boundary for {e}")
            self.recover(lp, "preempt")
        except Tipped as e:
            self.c["cycles failed"] += 1
            self.tipped(lp, e)
        except (D.Fatal, StopRequested, Cancelled):
            raise
        except (D.CycleFail, SystemExit, Exception) as e:  # noqa: BLE001
            if isinstance(e, SystemExit) and "GATE" in str(e):
                raise
            self.c["cycles failed"] += 1
            self.fails_in_row += 1
            D.status(f"cycle {n} failed: {type(e).__name__}: {e}")
            self.sim_truth(f"cycle {n} failed")
            self.recover(lp, str(e))
            if self.fails_in_row >= 2:
                standing, missing = lp.sides()
                if missing:
                    raise D.Fatal(f"block off the table: {', '.join(missing)} not seen")
                if standing:
                    c = next(iter(standing))
                    self.tipped(lp, Tipped(c, standing[c]))
            if self.fails_in_row >= 3:
                self.alert(f"{self.fails_in_row} cycles in a row failed (last: {e}); waiting 60 s before the next")
                time.sleep(60)
        finally:
            self.in_default = False
            self.write_stats()

    def run_job(self, lp, job):
        jid = job["id"]
        os.replace(self.p.accepted / f"{jid}.json", self.p.running / f"{jid}.json")
        self.job = job
        self.state = f"job {jid} ({job['kind']})"
        lp.pics = []
        t0, started = time.time(), _wall.strftime("%Y-%m-%d %H:%M:%S")
        D.status(f"job {jid} ({job['kind']}, priority {job['priority']}, from {job['submitted_by']}) started")
        status, extra = "done", {}
        try:
            if (self.p.cancel / jid).exists():
                raise Cancelled("cancelled")
            lp.to_pose(LOOK_TIP, "job start")
            extra["result"] = lp.run_job(job)
        except Cancelled:
            status = "cancelled"
            self.recover(lp, "cancel")
        except Tipped as e:
            status, extra["error"] = "failed", str(e)
            self.tipped(lp, e)
        except (D.Fatal, StopRequested, KeyboardInterrupt) as e:
            self.finish(job, "failed" if isinstance(e, D.Fatal) else "stopped", error=f"runner stopped: {type(e).__name__}: {e}", started=started,
                        finished=_wall.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 1), pictures=lp.pics)
            raise
        except (D.CycleFail, SystemExit, Exception) as e:  # noqa: BLE001
            if isinstance(e, SystemExit) and "GATE" in str(e):
                self.finish(job, "failed", error=f"runner stopped: {e}", started=started, finished=_wall.strftime("%Y-%m-%d %H:%M:%S"),
                            seconds=round(time.time() - t0, 1), pictures=lp.pics)
                raise
            status, extra["error"] = "failed", f"{type(e).__name__}: {e}"
            self.recover(lp, str(e))
        finally:
            self.job = None
        self.finish(job, status, started=started, finished=_wall.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 1),
                    pictures=list(lp.pics), **extra)
        D.status(f"job {jid} {status} in {time.time() - t0:.0f} s" + (f": {extra['error']}" if "error" in extra else ""))

    def run(self):
        p = self.p
        p.make()
        p.stop.unlink(missing_ok=True)
        # data_loop's files go under this station's loop/ (labels, the learned place bias), never the real one's in a sim
        D.STATUS, D.STOPFILE, D.LOOP = p.status, p.stop, p.loop
        self.session = p.loop / _wall.strftime("%m%d-%H%M%S")
        self.take_token()
        D.status(f"arm runner started ({'sim ' + self.sim_base if self.sim_base else 'real arm on ' + O.URL}); token taken; session {self.session.name}")
        write_json(p.info, {"pid": os.getpid(), "mode": "sim" if self.sim_base else "real", "sim_url": self.sim_base, "validation_url": self.val_base,
                            "session": str(self.session), "started": _wall.strftime("%Y-%m-%d %H:%M:%S")})
        # data_loop's waits and streams through the runner: 5 minute gate holds, a server error after every stream
        D.wait_gate = self.wait_gate
        orig_stream = O.stream

        def stream(op, plan, grip, watch, tag):
            out = orig_stream(op, plan, grip, watch, tag)
            s = arm_state()
            if s.get("error"):
                raise D.Fatal(f"arm server error after '{tag}': {s['error']}")
            return out
        D.stream = stream
        try:
            ts = []
            for _ in range(3):
                a = time.time()
                t = D.rec("/now")["t"]
                ts.append(t - (a + time.time()) / 2)
            self.offset = float(np.median(ts))
            D.rec("/start")
            recording = True
        except Exception as e:  # noqa: BLE001
            recording = False
            self.alert(f"the loop recorder ({D.REC}) does not answer ({e}): cycles run unrecorded")
        D.now = self.now
        lp = self.lp = RLoop(self.session, self)
        if self.sim_base:
            lp.sync = lambda: lp.saves.join()
        lp.op.st["target"] = None
        threading.Thread(target=self.validator, daemon=True).start()
        reason, clean = "stopped", True
        try:
            s = D.health()
            if s.get("grip_holding"):
                raise D.Fatal("the jaws are holding something at the start")
            lp.to_pose(LOOK_TIP, "start")
            bel, _ = lp.look("look-start")
            if bel is None:
                lp.look("look-start2", LOOK2)
            done_cycles = 0
            while True:
                if p.stop.exists():
                    raise StopRequested("STOP file")
                jobs = self.waiting()
                if jobs:
                    self.run_job(lp, jobs[0])
                    continue
                if self.max_cycles is not None and self.c["cycles completed"] + self.c["cycles failed"] >= self.max_cycles:
                    if not self.pending():
                        raise StopRequested(f"{self.max_cycles} cycles done")
                    time.sleep(1.0)  # jobs still in validation: wait for them
                    continue
                if p.pause.exists():
                    if self.state != "paused":
                        D.status(f"default loop paused ({p.pause.read_text().strip()}): holding at the look pose")
                    self.state = "paused"
                    self.write_stats()
                    time.sleep(1.0)
                    continue
                self.run_cycle(lp)
                done_cycles += 1
                if not self.sim_base and done_cycles % 5 == 0:
                    threading.Thread(target=lp.sync, daemon=True).start()
        except StopRequested as e:
            reason = f"stop: {e}"
        except KeyboardInterrupt:
            reason = "interrupted"
        except BaseException as e:  # noqa: BLE001
            reason = f"{type(e).__name__}: {e}"
            self.alert(f"RUNNER STOPPED: {reason}")
            self.sim_truth("the stop")
            traceback.print_exc()
            try:
                st = arm_state()
                if not st["moving"] and not self.holding() and "GATE" not in str(e) and "gate closed" not in str(e) and not st.get("error"):
                    lp.op.st["target"] = None
                    lp.to_pose(LOOK_TIP, "end")
                elif self.holding() or st.get("error"):
                    clean = False
            except BaseException as e2:  # noqa: BLE001
                reason += f"; end pose failed: {e2}"
                clean = False
        finally:
            self.stopping = True
            self.state = f"stopped ({reason})"
            self.write_stats()
            lp.saves.join()
            if recording:
                try:
                    D.rec("/stop")
                except Exception:  # noqa: BLE001
                    pass
            if clean:
                shutil.rmtree(p.token, ignore_errors=True)
                D.status(f"arm runner ended ({reason}); {self.c['cycles completed']} cycles; token released")
            else:
                (p.token / "who.txt").write_text(f"arm-runner HELD {_wall.strftime('%H:%M:%S')}: stopped in an unclean state ({reason})\n")
                D.status(f"arm runner ended ({reason}); TOKEN KEPT (arm not in a clean state)")
            if not self.sim_base:
                lp.sync()
            p.info.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------------------------
class DryHooks:
    """A dry run's hooks: no queue, no preemption; the gate is the sim's."""

    def __init__(self):
        self.n = 0
        self.alerts = []

    def boundary(self, lp):
        pass

    def wait_gate(self):
        t0 = time.time()
        while not gate()[0]:
            if time.time() - t0 > 30:
                raise D.Fatal("the dry run's gate stayed closed")
            time.sleep(1)

    def alert(self, msg):
        self.alerts.append(msg)

    def world(self, lp):
        pass

    def picture(self, path):
        pass

    def next_cycle(self):
        self.n += 1
        return self.n

    def after_home(self, job):
        pass


def dryrun(a):
    out = pathlib.Path(a.out)
    t0 = _wall.time()
    verdict = {"ok": False}
    try:
        base = a.url.rstrip("/")
        use_backend(sim_urls(base), sim_clock(base))
        job = json.loads(pathlib.Path(a.job).read_text())
        world = json.loads(pathlib.Path(a.world).read_text())
        body = json.dumps({"blocks": world["blocks"], "q": world["q"], "grip": 0.8}).encode()
        urllib.request.urlopen(urllib.request.Request(base + "/reset", body, {"Content-Type": "application/json"}), timeout=60).read()
        work = pathlib.Path(a.work)
        D.STATUS, D.LOOP = work / "STATUS.txt", work / "loop"
        hooks = DryHooks()
        D.wait_gate = hooks.wait_gate
        D.now = lambda: time.time()
        lp = RLoop(work / "session", hooks)
        lp.sync = lambda: None
        lp.op.st["target"] = None
        lp.belief = {c: dict(b) for c, b in world["blocks"].items()}
        lp.put_blocks()
        D.rec("/start")
        res = lp.run_job(job)
        lp.saves.join()
        verdict = {"ok": True, "reason": "ok", "result": jdump(res), "recoveries": dict(lp.recov)}
    except BaseException as e:  # noqa: BLE001
        verdict = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    verdict["sim_s"] = None
    verdict["dry_wall_s"] = round(_wall.time() - t0, 1)
    try:
        verdict["status_tail"] = (pathlib.Path(a.work) / "STATUS.txt").read_text().splitlines()[-6:]
    except OSError:
        pass
    write_json(out, verdict)


# ---------------------------------------------------------------------------------------------------------------
def cli_submit(p: Paths, a):
    p.make()
    args = json.loads(a.args) if a.args else {}
    jid = a.id or f"{_wall.strftime('%H%M%S')}-{a.kind}-{os.urandom(2).hex()}"
    job = {"id": jid, "kind": a.kind, "args": args, "priority": a.priority, "submitted_by": a.by,
           "submitted_at": _wall.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(p.q / f"{jid}.json", job)
    print(jid)


def cli_list(p: Paths):
    def rows(d):
        out = []
        for f in sorted(d.glob("*.json"), key=lambda f: f.stat().st_mtime):
            if f.name in ("stats.json", "world.json", "runner.json"):
                continue
            try:
                out.append(json.loads(f.read_text()))
            except ValueError:
                out.append({"id": f.stem, "kind": "?"})
        return out
    for name, d in (("validating", p.q), ("accepted", p.accepted), ("running", p.running)):
        for j in rows(d):
            print(f"{name:10s} {j.get('id')}  {j.get('kind')}  p{j.get('priority', '?')}  by {j.get('submitted_by', '?')}")
    for j in rows(p.done)[-12:]:
        extra = j.get("reason") or j.get("error") or ""
        print(f"{j.get('status', '?'):10s} {j.get('id')}  {j.get('kind')}  p{j.get('priority', '?')}  {j.get('seconds', '')}s  {extra[:110]}")


def cli_cancel(p: Paths, jid):
    p.make()
    for d in (p.q, p.accepted):
        f = d / f"{jid}.json"
        if f.exists():
            dst = p.cancel / f"{jid}.moving"
            try:
                os.replace(f, dst)
            except OSError:
                continue
            job = json.loads(dst.read_text())
            write_json(p.done / f"{jid}.json", dict(job, status="cancelled", reason="cancelled before it ran",
                                                     finished=_wall.strftime("%Y-%m-%d %H:%M:%S")))
            dst.unlink()
            print(f"{jid}: cancelled before it ran")
            return
    (p.cancel / jid).write_text(_wall.strftime("%H:%M:%S"))
    print(f"{jid}: cancel requested (a running job stops at its next block boundary)")


def main():
    ap = argparse.ArgumentParser(prog="python -m calib.arm_runner")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--sim", action="store_true")
        sp.add_argument("--root")
        return sp
    r = common(sub.add_parser("run"))
    r.add_argument("--cycles", type=int)
    r.add_argument("--speed", type=float, default=3.0)
    r.add_argument("--vspeed", type=float, default=6.0)
    r.add_argument("--seed", type=int, default=1)
    s = common(sub.add_parser("submit"))
    s.add_argument("kind")
    s.add_argument("args", nargs="?")
    s.add_argument("--priority", type=int, default=5)
    s.add_argument("--by", default=os.environ.get("USER", "?"))
    s.add_argument("--id")
    common(sub.add_parser("list"))
    c = common(sub.add_parser("cancel"))
    c.add_argument("id")
    for name in ("pause", "resume", "stop", "status"):
        common(sub.add_parser(name))
    i = common(sub.add_parser("inject"))
    i.add_argument("body")
    d = sub.add_parser("dryrun")
    for k in ("job", "world", "url", "out", "work"):
        d.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    if a.cmd == "dryrun":
        return dryrun(a)
    p = paths_for(a)
    if a.cmd == "run":
        p.make()
        procs = []
        try:
            vport = free_port()
            vproc, vbase = start_sim(vport, a.vspeed, "true", 99, p.q / "validation_sim.log")
            procs.append(vproc)
            sim_base = None
            if a.sim:
                sproc, sim_base = start_sim(free_port(), a.speed, "real", a.seed, p.q / "station_sim.log")
                procs.append(sproc)
                use_backend(sim_urls(sim_base), sim_clock(sim_base))
            else:
                assert O.URL == REAL["arm"] and O.TEAM == REAL["team"] and O.DREAM == REAL["dream"]
            def _term(*_):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, _term)
            Runner(p, sim_base, vbase, a.cycles).run()
        finally:
            for pr in procs:
                pr.terminate()
    elif a.cmd == "submit":
        cli_submit(p, a)
    elif a.cmd == "list":
        cli_list(p)
    elif a.cmd == "cancel":
        cli_cancel(p, a.id)
    elif a.cmd == "pause":
        p.make()
        p.pause.write_text(f"paused from the command line at {_wall.strftime('%H:%M:%S')}\n")
        print("default loop paused (it stops at the next block boundary; jobs still run)")
    elif a.cmd == "resume":
        p.pause.unlink(missing_ok=True)
        print("default loop resumed")
    elif a.cmd == "stop":
        p.make()
        p.stop.write_text(_wall.strftime("%H:%M:%S"))
        print("stop requested (at the next block boundary)")
    elif a.cmd == "status":
        print(p.stats.read_text() if p.stats.exists() else "no stats yet")
        if p.status.exists():
            print("\n".join(p.status.read_text().splitlines()[-15:]))
    elif a.cmd == "inject":
        info = json.loads(p.info.read_text())
        if not info.get("sim_url"):
            raise SystemExit("inject is for --sim runs only")
        body = a.body.encode()
        print(urllib.request.urlopen(urllib.request.Request(info["sim_url"] + "/inject", body, {"Content-Type": "application/json"}), timeout=30).read().decode())


if __name__ == "__main__":
    main()
