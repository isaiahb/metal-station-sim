"""The station, offline: one HTTP port that stands in for everything calib/arm_runner.py talks to on the real station,
backed by the CPU MuJoCo sim (sim/physics.py through sim/tower.py's SimRobot: the Metal arm with the fitted sag, the
real table, the three foam blocks, the wrist camera through the calibrated mount and lens).

    /state /wrist.jpg /overhead.jpg, POST /move /stop     the team's left arm server (station/armserver.py's walk:
                                                          30 Hz, at most `rate` rad/s, grip 1.5/s, joint lag abort 0.3)
    /team/api/state                                       the team page (left enabled, assigned_side, recording)
    /dream/api/state                                      the Dream caller (left running)
    /rec/now /rec/start /rec/stop /rec/save               the loop recorder (calib/loop_rec.py); saves only count frames
    GET /clock                                            {"real0", "k"}: the shared clock (see Clock)
    POST /reset {"blocks": {colour: {x, y, yaw, level}}, "q": [6] | "look" | "start", "grip"}   (dry runs)
    POST /inject {"kind": ...}                            faults for tests: now (gate, nudge, tip, remove, place, drop,
                                                          temp, lag) or armed for an event (nudge_on_close: the block
                                                          under the jaws moves as they close; drop_on_lift: the block falls
                                                          out 3 cm into a lift; tip_on_release: the block set down on the
                                                          table falls on its side); "fired" lists what fired
    GET /truth                                            the blocks' true poses (tests only; the runner never reads it)

Frames are the server's: joints as the arm reports them (the model's joints less calib/arm_left.json dq_rad, less the
sim's own joint zero errors in the "real" mode). Nothing here can reach the real station: it binds 127.0.0.1 on its own
port and posts nowhere.

    python -m calib.station_sim --port 8790 [--speed 4] [--mode real|noise|true] [--seed 1]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import queue
import threading
import time as _time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
COLOURS = ("green", "blue", "purple")  # sim/tower.py colours its props in this order
START_JOINTS = [0.018, 0.014, 0.242, -0.418, 0.008, 0.141]  # the left arm's saved zero (runs/station/ARM_FREE, Nov 7)
TICK_STEPS = 17  # physics steps (2 ms) per server tick: 34 ms, the real server's 30 Hz walk
MAX_RATE, ABORT_RAD = 0.6, 0.3


class Clock:
    """Time for a sim run: k times faster than the wall clock, the same in every process that reads /clock.
    time() is virtual seconds (it starts at the server's wall time); sleep(s) sleeps s / k real seconds. Anything else
    (strftime, ...) is the time module's."""

    def __init__(self, k=1.0, real0=None):
        self.k, self.real0 = float(k), float(real0 if real0 is not None else _time.time())

    def time(self):
        return self.real0 + (_time.time() - self.real0) * self.k

    def sleep(self, s):
        if s > 0:
            _time.sleep(s / self.k)

    def __getattr__(self, name):
        return getattr(_time, name)


class Station:
    def __init__(self, seed=1, mode="real", speed=4.0):
        self.seed, self.mode, self.clock = seed, mode, Clock(speed)
        self.jobs: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.snap = {}
        self.team = {"enabled": True, "assigned_side": None, "recording": False}
        self.dream_running = False
        self.gate_until = {}  # field -> virtual time it stays closed until
        self.temps = [34.0] * 7
        self.rec = {"on": False, "n": 0}
        self.behind_max = 0.0
        self.armed = []  # faults that fire on an event: nudge_on_close, drop_on_lift, tip_on_release
        self.fired = []
        self.z_close = None
        threading.Thread(target=self.sim_thread, daemon=True).start()
        self.ready.wait(60)

    # ---- the sim thread owns MuJoCo (its renderer belongs to the thread that made it)
    def sim_thread(self):
        try:
            self._sim_thread()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            raise

    def _sim_thread(self):
        import mujoco
        from sim.tower import SIDE, SimRobot, setup
        props, kw = setup(self.seed, self.mode)
        rng = np.random.default_rng(self.seed + 77)
        # blocks where the station's are: inside the look's view (x 26-40 cm, y -8..18 cm), 11 cm apart
        pts, misses = [], 0
        while len(pts) < 3:
            p = np.array([rng.uniform(0.26, 0.36), rng.uniform(-0.06, 0.16)])
            if all(np.linalg.norm(p - q) > 0.11 for q in pts):
                pts.append(p)
                misses = 0
            else:
                misses += 1
                if misses > 200:  # the first two left no room for a third (seed 1 looped forever): start over
                    pts, misses = [], 0
        for p, xy in zip(props, pts):  # where data_loop's plans solve (a block at x 40 cm, yaw -31 deg had no pick pose)
            p["at"] = [float(xy[0]), float(xy[1])]
            p["yaw"] = float(rng.uniform(-0.35, 0.35))
        self.robot = r = SimRobot(props, **kw)
        # SimRobot resizes the desk to the real table but leaves the compiled bounding box, so blocks past y 13 cm fell
        # through it: the box and sphere bounds follow the new size
        g = mujoco.mj_name2id(r.m, mujoco.mjtObj.mjOBJ_GEOM, "desk")
        r.m.geom_aabb[g] = [0, 0, 0, *r.m.geom_size[g]]
        r.m.geom_rbound[g] = float(np.linalg.norm(r.m.geom_size[g]))
        self.side = SIDE
        self.mj = mujoco
        for c, pp in zip(COLOURS, props):  # back onto the table (SimRobot settled them before the fix)
            self.set_block(c, pp["at"][0], pp["at"][1], pp.get("yaw", 0.0))
        self.op_dq = np.array(json.loads((ROOT / "calib" / "arm_left.json").read_text())["dq_rad"])
        self.oh = mujoco.Renderer(r.m, 480, 640)
        self.ohcam = mujoco.MjvCamera()
        self.ohcam.lookat[:] = [0.32, 0.05, 0.0]
        self.ohcam.distance, self.ohcam.azimuth, self.ohcam.elevation = 0.95, 180, -62
        self.reset_arm(np.array(START_JOINTS), 0.27)
        self.sim_t = self.clock.time()
        self.ready.set()
        while True:
            while True:
                try:
                    fn, box, ev = self.jobs.get_nowait()
                except queue.Empty:
                    break
                try:
                    box["out"] = fn()
                except Exception as e:  # noqa: BLE001
                    box["err"] = repr(e)
                    traceback.print_exc()
                ev.set()
            target = self.clock.time()
            if self.sim_t < target:
                self.tick()
                self.sim_t += TICK_STEPS * r.m.opt.timestep
                self.behind_max = max(self.behind_max, target - self.sim_t)
            else:
                _time.sleep(0.001)

    def call(self, fn, timeout=30):
        box, ev = {}, threading.Event()
        self.jobs.put((fn, box, ev))
        if not ev.wait(timeout):
            raise TimeoutError("sim thread busy")
        if "err" in box:
            raise RuntimeError(box["err"])
        return box.get("out")

    # ---- the arm server's walk (station/armserver.py control())
    def reset_arm(self, q_server, grip):
        r = self.robot
        q_model = np.asarray(q_server, float) + self.op_dq
        r.s.set_arm(self.side, r.plant(q_model), grip, snap=True)
        self.mj.mj_forward(r.m, r.d)
        r.s.step(60)
        self.cmd = np.asarray(q_server, float).copy()
        self.goal = self.cmd.copy()
        self.grip_cmd = self.goal_grip = float(grip)
        self.rate = 0.3
        self.error = None
        self.publish()

    def tick(self):
        r = self.robot
        with self.lock:
            goal, goal_grip, rate = self.goal.copy(), self.goal_grip, min(self.rate, MAX_RATE)
        d = goal - self.cmd
        step = rate * TICK_STEPS * r.m.opt.timestep
        n = float(np.max(np.abs(d)))
        self.cmd = goal.copy() if n <= step else self.cmd + d * (step / n)
        self.grip_cmd += float(np.clip(goal_grip - self.grip_cmd, -1.5 * 0.034, 1.5 * 0.034))
        r.s.set_arm(self.side, r.plant(self.cmd + self.op_dq), self.grip_cmd)
        r.s.step(TICK_STEPS)
        meas = self.meas()
        lag = float(np.max(np.abs(self.cmd - meas)))
        if lag > ABORT_RAD:
            with self.lock:
                self.goal = meas.copy()
                self.error = f"joint lag {lag:.2f} rad: stopped where the arm is"
            self.cmd = meas.copy()
        self.fire_tick()
        self.publish(lag)

    def tcp(self):
        return self.robot.s.tcp(self.side)

    def nearest(self):
        t = self.truth()
        p = self.tcp()
        return min(t, key=lambda c: math.hypot(t[c]["x"] - p[0], t[c]["y"] - p[1])), t

    def fire(self, kind, **info):
        self.armed = [a for a in self.armed if a["kind"] != kind]
        self.fired.append(dict(info, kind=kind, t=self.clock.time()))
        print("fault fired:", kind, info, flush=True)

    def fire_tick(self):
        kinds = {a["kind"]: a for a in self.armed}
        if "drop_on_lift" in kinds and self.z_close is not None and self.goal_grip < 0.5 and self.holding() \
                and self.tcp()[2] - self.z_close > 0.03:
            # a slip: the block falls out of the jaws onto the table under them (1.5 cm toward the base, where it
            # stays reachable), flat; the jaws close on air (0.44)
            c, t = self.nearest()
            p = self.tcp()
            u = -p[:2] / max(float(np.linalg.norm(p[:2])), 1e-6)
            self.set_block(c, float(p[0] + 0.015 * u[0]), float(p[1] + 0.015 * u[1]), t[c]["yaw"], 0)
            self.fire("drop_on_lift", colour=c, z=round(float(p[2]), 3))
        rel = getattr(self, "released", None)
        if rel is not None and self.clock.time() > rel[1]:
            self.released = None
            c, t = rel[0], self.truth()[rel[0]]
            if t["flat"] and t["level"] == 0:
                self.set_block(c, t["x"], t["y"], t["yaw"], standing=True)
                self.robot.s.step(50)
                self.fire("tip_on_release", colour=c)

    def on_grip(self, old, new):
        """From the /move handler: the jaws told to close or open."""
        kinds = {a["kind"]: a for a in self.armed}
        if old >= 0.7 and new < 0.5:
            self.z_close = float(self.tcp()[2])
            if "nudge_on_close" in kinds:
                a = kinds["nudge_on_close"]

                def fn():  # moved `toward` m toward the arm base (stays reachable, clear of the jaws), or by dx, dy
                    c, t = self.nearest()
                    x, y = t[c]["x"], t[c]["y"]
                    if "toward" in a:
                        r = math.hypot(x, y)
                        x, y = x - float(a["toward"]) * x / r, y - float(a["toward"]) * y / r
                    else:
                        x, y = x + float(a.get("dx", 0.03)), y + float(a.get("dy", 0.0))
                    self.set_block(c, x, y, t[c]["yaw"], t[c]["level"] or 0)
                    self.fire("nudge_on_close", colour=c)
                self.jobs.put((fn, {}, threading.Event()))
        if old < 0.5 and new >= 0.55 and "tip_on_release" in kinds:
            def fn():
                c, t = self.nearest()
                if t[c]["level"] == 0 and math.hypot(t[c]["x"] - self.tcp()[0], t[c]["y"] - self.tcp()[1]) < 0.03:
                    self.released = (c, self.clock.time() + 0.6)
            self.jobs.put((fn, {}, threading.Event()))

    def meas(self):
        return self.robot.q() - self.op_dq

    def grip_now(self):
        return float(self.robot.s.arm_state(self.side)["grip"])

    def holding(self):
        r = self.robot
        pads = r.s.pad_geoms[self.side]
        bodies = {p["body"] for p in r.s.props}
        for c in r.d.contact[: r.d.ncon]:
            g = {c.geom1, c.geom2}
            if g & pads and any(r.m.geom_bodyid[x] in bodies for x in g):
                return True
        return False

    def publish(self, lag=0.0):
        meas = self.meas()
        moving = bool(np.max(np.abs(self.goal - self.cmd)) > 1e-4 or abs(self.goal_grip - self.grip_cmd) > 1e-3)
        with self.lock:
            self.snap = {"q": meas.tolist(), "grip": self.grip_now(), "goal": self.goal.tolist(), "goal_grip": self.goal_grip,
                         "rate": self.rate, "moving": moving, "stalled": False, "lag": lag, "stamp": self.clock.time(),
                         "cams": {"wrist": True, "overhead": True}, "error": self.error, "start_joints": START_JOINTS,
                         "grip_holding": self.holding(),
                         "temps_c": {f"m{i + 1}": t for i, t in enumerate(self.temps)}}

    # ---- pictures
    def wrist(self):
        img = self.robot.wrist_image()
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes(), self.meas().tolist(), self.grip_now()

    def overhead(self):
        self.oh.update_scene(self.robot.d, self.ohcam)
        img = self.oh.render()[..., ::-1]
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(), self.meas().tolist(), self.grip_now()

    # ---- blocks
    def _block_adr(self, colour):
        r = self.robot
        b = r.s.props[COLOURS.index(colour)]["body"]
        return r.m.jnt_qposadr[r.m.body_jntadr[b]], r.m.jnt_dofadr[r.m.body_jntadr[b]], b

    def set_block(self, colour, x, y, yaw, level=0, standing=False):
        from calib.tower_plan import FACE, H
        from calib.tower_safety import TABLE_Z
        r = self.robot
        qa, da, _ = self._block_adr(colour)
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        if standing:  # on its side: the hole horizontal, along the block's local y after the turn about x
            R = Rz @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            z = TABLE_Z + FACE / 2 + 0.0015
        else:
            R = Rz
            z = TABLE_Z + level * H + H / 2 + 0.0015
        qq = np.zeros(4)
        self.mj.mju_mat2Quat(qq, R.reshape(9))
        r.d.qpos[qa:qa + 3] = [x, y, z]
        r.d.qpos[qa + 3:qa + 7] = qq
        r.d.qvel[da:da + 6] = 0
        self.mj.mj_forward(r.m, r.d)

    def truth(self):
        from calib.tower_plan import H
        from calib.tower_safety import TABLE, TABLE_Z
        out = {}
        for k, c in enumerate(COLOURS):
            p, R = self.robot.pose(k)
            flat = abs(R[2, 2]) > 0.7
            on = TABLE["x"][0] < p[0] < TABLE["x"][1] and TABLE["y"][0] < p[1] < TABLE["y"][1] and p[2] > TABLE_Z - 0.01
            out[c] = {"x": round(float(p[0]), 4), "y": round(float(p[1]), 4), "z": round(float(p[2]), 4),
                      "yaw": round(math.atan2(R[1, 0], R[0, 0]), 3), "flat": bool(flat), "on_table": bool(on),
                      "level": int(round((p[2] - TABLE_Z - H / 2) / H)) if flat else None}
        return out

    def reset(self, body):
        def fn():
            blocks = body.get("blocks") or {}
            for i, c in enumerate(COLOURS):
                b = blocks.get(c)
                if b is None:  # a block the belief does not have: far out of the way
                    self.set_block(c, 0.6, -0.5 + 0.1 * i, 0.0)
            for c, b in sorted(blocks.items(), key=lambda kv: kv[1].get("level", 0)):
                self.set_block(c, b["x"], b["y"], b["yaw"], int(b.get("level", 0)))
            q = body.get("q", "start")
            self.reset_arm(np.array(START_JOINTS if q == "start" else q, float), float(body.get("grip", 0.8)))
            self.robot.s.step(150)
            self.publish()
            return self.truth()
        return self.call(fn)

    def inject(self, body):
        kind = body["kind"]
        now = self.clock.time()
        if kind == "gate":  # one gate field closed for `seconds` (virtual)
            self.gate_until[body.get("field", "recording")] = now + float(body.get("seconds", 30))
            return {"closed_until": self.gate_until}
        if kind == "temp":
            self.temps = [float(body.get("value", 65))] * 7
            return {"temps": self.temps}
        if kind == "lag":
            def fn():
                meas = self.meas()
                with self.lock:
                    self.goal = meas.copy()
                    self.error = "joint lag 0.34 rad: stopped where the arm is"
                self.cmd = meas.copy()
                self.publish()
            return self.call(fn)
        if kind == "drop":  # the jaws open (a slip)
            with self.lock:
                self.goal_grip = 0.8
            return {"ok": True}
        if kind in ("nudge_on_close", "drop_on_lift", "tip_on_release"):
            self.armed.append(dict(body))
            return {"armed": self.armed}
        if kind == "fired":
            return {"fired": self.fired, "armed": self.armed}

        def fn():
            c = body["colour"]
            t = self.truth()[c]
            if kind == "nudge":
                self.set_block(c, t["x"] + float(body.get("dx", 0)), t["y"] + float(body.get("dy", 0)), t["yaw"] + float(body.get("dyaw", 0)),
                               t["level"] or 0)
            elif kind == "tip":
                self.set_block(c, float(body.get("x", t["x"])), float(body.get("y", t["y"])), t["yaw"], standing=True)
            elif kind == "remove":  # off the table (it falls to the floor)
                self.set_block(c, 0.80, 0.30, 0.0)
            elif kind == "place":
                self.set_block(c, float(body["x"]), float(body["y"]), float(body.get("yaw", 0)), int(body.get("level", 0)))
            self.robot.s.step(100)
            return self.truth()[c]
        return self.call(fn)

    def team_state(self):
        now = self.clock.time()
        closed = {k for k, t in self.gate_until.items() if now < t}
        return {"arms": {"left": {"enabled": "enabled" not in closed and self.team["enabled"]}},
                "assigned_side": "left" if "assigned_side" in closed else self.team["assigned_side"],
                "recording": True if "recording" in closed else self.team["recording"]}

    def dream_state(self):
        now = self.clock.time()
        running = self.dream_running or (now < self.gate_until.get("dream", 0))
        return {"left": {"running": running}, "right": {"running": False}}


def serve(st: Station, port: int):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):
            pass

        def reply(self, code, body, ctype="application/json", headers=()):
            if not isinstance(body, (bytes, bytearray)):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            a = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
            p = u.path
            try:
                if p in ("/wrist.jpg", "/overhead.jpg"):
                    jpg, q, g = st.call(st.wrist if p.startswith("/wrist") else st.overhead)
                    return self.reply(200, jpg, "image/jpeg", [("X-Joints", json.dumps(q)), ("X-Grip", json.dumps(g)),
                                                               ("X-Stamp", f"{st.clock.time():.3f}")])
                if p == "/state":
                    with st.lock:
                        return self.reply(200, dict(st.snap))
                if p == "/team/api/state":
                    return self.reply(200, st.team_state())
                if p == "/dream/api/state":
                    return self.reply(200, st.dream_state())
                if p == "/clock":
                    return self.reply(200, {"real0": st.clock.real0, "k": st.clock.k, "t": st.clock.time(), "behind_max_s": round(st.behind_max, 3)})
                if p == "/truth":
                    return self.reply(200, st.call(st.truth))
                if p == "/rec/now":
                    return self.reply(200, {"t": st.clock.time()})
                if p in ("/rec/start", "/rec/stop"):
                    st.rec["on"] = p.endswith("start")
                    return self.reply(200, {"on": st.rec["on"]})
                if p == "/rec/save":
                    if not st.rec["on"]:
                        return self.reply(200, {"error": "recorder off"})
                    i = st.rec["n"]
                    st.rec["n"] += 1
                    sec = float(a["t1"]) - float(a["t0"])
                    return self.reply(200, {"episode_index": i, "frames": int(sec * 30), "seconds": round(sec, 2), "max_gap_s": 0.034,
                                            "error": None, "sim": True})
                self.reply(404, {"error": "unknown"})
            except Exception as e:  # noqa: BLE001
                self.reply(500, {"error": repr(e)})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                if self.path == "/move":
                    with st.lock:
                        if body.get("q") is not None:
                            st.goal = np.array([float(v) for v in body["q"]])
                        if body.get("grip") is not None:
                            old = st.goal_grip
                            st.goal_grip = float(np.clip(body["grip"], 0, 1))
                            st.on_grip(old, st.goal_grip)
                        st.rate = float(body.get("rate", 0.3))
                        st.error = None
                    return self.reply(200, {"ok": True})
                if self.path == "/stop":
                    with st.lock:
                        st.goal = np.array(st.snap["q"])
                    return self.reply(200, {"ok": True})
                if self.path == "/reset":
                    return self.reply(200, st.reset(body))
                if self.path == "/inject":
                    return self.reply(200, st.inject(body))
                self.reply(404, {"error": "unknown (this stand-in has no /quit)"})
            except Exception as e:  # noqa: BLE001
                self.reply(500, {"error": repr(e)})

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.daemon_threads = True
    print(f"station sim on 127.0.0.1:{port} (mode {st.mode}, seed {st.seed}, speed x{st.clock.k})", flush=True)
    srv.serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--mode", default="real")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.port in (8691, 8081, 18081, 18084, 8697):
        raise SystemExit("that port belongs to the real station")
    serve(Station(a.seed, a.mode, a.speed), a.port)


if __name__ == "__main__":
    main()
