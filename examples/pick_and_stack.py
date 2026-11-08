"""Worked example: look with the wrist camera, pick one foam block and stack it flat on another.

Built only on calib/opstack.py (Op, Plan, stream, fast_go): every waypoint is solved by the arm model's IK and every
path piece goes through calib/tower_safety.py (and opstack's extra checks) before anything moves. The script talks
to the arm only through the arm-server API (GET /state, /wrist.jpg, /overhead.jpg; POST /move, /stop), so the same
code drives the sim or the real arm.

    # terminal 1: the sim station
    python -m calib.station_sim --port 8790
    # terminal 2: this example against it
    python -m examples.pick_and_stack --url http://127.0.0.1:8790 [--pick blue --onto green]

Against the real arm, pass --real instead of --url: opstack's own URLs are used (the arm server and the station's
gate pages, see README.md). Run it on the real arm only when the station is yours to drive.

Frames and pictures go to runs/example/<time>/ (wrist and overhead frames, overlays of the fitted blocks, step log).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time as _wall
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calib import arm_runner as AR  # noqa: E402
from calib import opstack as O  # noqa: E402
from calib.tower_plan import H  # noqa: E402


def to_look(op):
    """From wherever the arm is (the folded rest at the start) to the high look pose, one streamed motion."""
    watch = O.GateWatch()
    q0 = np.array(op.st["target"]) if op.st["target"] else np.array(O.arm_state()["q"]) + op.dq
    pl = O.Plan(op, q0, 0.08)
    pl.to(O.LOOK_TIP[:3], O.LOOK_TIP[3], 0.4, "look")
    for (a, _, _), (b, _, _) in zip(pl.nodes[:-1], pl.nodes[1:]):
        ok, why = op.safe.check_path(a, b, 0.08, home=True)  # the move out of the folded rest: height rules only
        if not ok:
            raise SystemExit(f"look refused: {why}")
    O.stream(op, pl, 0.8, watch, "look")
    watch.stop = True
    O.time.sleep(0.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8790", help="the sim's arm-server URL")
    ap.add_argument("--real", action="store_true", help="drive the real arm with opstack's own URLs instead")
    ap.add_argument("--pick", default="blue")
    ap.add_argument("--onto", default="green")
    a = ap.parse_args()
    sim = not a.real
    if sim:
        base = a.url.rstrip("/")
        AR.use_backend(AR.sim_urls(base), AR.sim_clock(base))  # refuses a sim URL on a real station port
    print(f"arm server: {O.URL} ({'sim' if sim else 'REAL ARM'})", flush=True)

    run = ROOT / "runs" / "example" / _wall.strftime("%m%d-%H%M%S")
    op = O.Op(run)
    op.st["target"] = None

    to_look(op)
    fits, tip, stem = op.find_all("look")
    print("wrist look:", json.dumps(fits), flush=True)
    for c in (a.pick, a.onto):
        if fits.get(c) is None:
            raise SystemExit(f"the {c} block is not in the wrist view (overlay: {run / (stem + '-ov.jpg')})")
    p, t = fits[a.pick], fits[a.onto]

    # one streamed pick (over, down, close, lift) and one streamed place (carry, lower, set on the surface, open,
    # back off); tower_top is the top face of the block it goes on
    times, grasp, tip_set, tip_rel = O.fast_go(op, a.pick, (p["x"], p["y"], p["yaw"]), (t["x"], t["y"], t["yaw"]), O.TABLE_Z + H)
    op.save()
    print("placed:", json.dumps({"block_s": times["block_s"], "grip": round(grasp["grip"], 3)}), flush=True)

    # a check from the look pose: the stacked block's fit at one block up
    to_look(op)
    fit, tip, stem, grip = op.look(a.pick, "check", z0=O.TABLE_Z + H, guess=(t["x"], t["y"], t["yaw"]))
    off = None if fit is None else [round((fit["x"] - t["x"]) * 1000, 1), round((fit["y"] - t["y"]) * 1000, 1)]
    print(f"check: {a.pick} on {a.onto}, offset from the block below as the wrist sees it: {off} mm", flush=True)
    print(f"pictures: {run}", flush=True)

    if sim:  # the sim knows the true poses; a real run has only the check photo
        truth = json.loads(urllib.request.urlopen(a.url.rstrip("/") + "/truth", timeout=10).read())
        up, down = truth[a.pick], truth[a.onto]
        d = math.hypot(up["x"] - down["x"], up["y"] - down["y"]) * 1000
        ok = up["level"] == 1 and up["flat"] and d < 10
        print(f"sim truth: {a.pick} level {up['level']}, {d:.1f} mm from the {a.onto} centre: {'STACKED' if ok else 'NOT STACKED'}",
              flush=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
