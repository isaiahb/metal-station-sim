"""One tower, as tall as it goes, from every foam block on the table (real left Metal arm).

Built on calib/data_loop.py's Loop (re-look before every pick, close look at the block below before every place,
contact set-downs, check photo, re-place once, the place bias, episodes recorded on the station) and so on opstack.py.
What is new here:
  - several blocks of one colour: ids like yellow#1, the colour from the prefix; hue bins for the new blocks (yellow
    with its own saturation floor, orange split from the red sticks at H 5); round blobs are balls, poor box fits are
    sticks, both obstacles;
  - picks choose the jaw axis whose open fingers clear the neighbours (blocks can touch), checked along every path
    point low enough to meet a neighbour;
  - the tower spot: every level's place, close look and check photo plan from a carry above it, before anything moves.

    python -m calib.tall_tower look                 # the overlay of every block on the current wrist look (no motion)
    python -m calib.tall_tower dry                  # the whole tower planned on the current look (no motion)
    python -m calib.tall_tower run                  # build it
"""
from __future__ import annotations

import json
import math
import re
import sys
import time

import cv2
import numpy as np

from calib import data_loop as D
from calib import opstack as O
from calib.opstack import GRIP_TIP, LOOK_TIP, PICK_DOWN, TABLE_Z, Op, Plan, arm_state
from calib.tower_plan import FACE, H, Shape, WristView, frame_down
from calib.data_loop import V_DOWN, V_HIGH, V_LAST, bot, hgt, shape_of, top_of

ROOT = O.ROOT
TALL = ROOT / "runs" / "station" / "tall"
STICK = Shape((0.1005, 0.02265, 0.02265), -0.008, period=math.pi)  # a stick lying on the table
BLOCK_COLOURS = ("yellow", "blue", "orange", "purple", "green")  # sticks and balls of these colours fail the box fit
# OpenCV HSV: hue bins, saturation floor (measured on the station's light, Nov 7 19:40)
HUES = {"yellow": ([(22, 38)], 35), "orange": ([(5, 20)], 70), "blue": ([(88, 125)], 70), "red": ([(0, 5), (176, 181)], 70),
        "purple": ([(135, 176)], 55), "green": ([(40, 88)], 55)}
VAL_MIN = 60
PAD_HALF = (0.007, 0.016)  # a pad's half extent (along the closing axis, across it) with 3 mm margin
FINGER = 0.043  # open fingers from the jaw centre at grip 0.8
D.NEAT_MM, D.NEAT_DEG = 6.0, 5.0


def colour_of(bid):
    return bid.split("#")[0]


def colour_mask(bgr, colour, jaws):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    bins, sat = HUES[colour_of(colour)]
    m = np.zeros(hsv.shape[:2], bool)
    for lo, hi in bins:
        m |= (hsv[..., 0] >= lo) & (hsv[..., 0] < hi)
    m = (m & (hsv[..., 1] >= sat) & (hsv[..., 2] >= VAL_MIN) & (jaws == 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


WristView.colour_mask = staticmethod(colour_mask)


def roundness(blob):
    cs, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    a = cv2.contourArea(c)
    (_, _), r = cv2.minEnclosingCircle(c)
    return a / max(math.pi * r * r, 1), a / max(cv2.contourArea(cv2.convexHull(c)), 1)


class AxisPlan(Plan):
    """A Plan whose free frame choice keeps to the jaw axes allowed (k parity: 0 closes along the block's yaw, 1
    across it)."""
    parity = None

    def solve_from(self, ref, xyz, yaw_deg, keep):
        if keep and self.R is not None or self.parity is None:
            return super().solve_from(ref, xyz, yaw_deg, keep)
        from calib.tower_safety import outward_ok
        best = None
        for k in range(4):
            if k % 2 not in self.parity:
                continue
            R = frame_down(math.radians(yaw_deg) + k * math.pi / 2)
            if not outward_ok(R, xyz[:2]):
                continue
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
            raise SystemExit(f"no safe pose for fingertips {np.round(np.asarray(xyz) * 1000).tolist()} yaw {yaw_deg:.0f} on jaw axes {self.parity}")
        self.R = best[2]
        return best[1]


class Tall(D.Loop):
    relook_iou = 0.65  # blocks touching a neighbour fit about 0.7
    # ---- labels: prompts by colour, not id
    def episode(self, t0, t1, root=None, **label):
        label["prompt"] = re.sub(r"#\d+", "", label["prompt"])
        label["block_id"] = label.get("colour")
        if root and root.startswith("dream_ft"):
            root = root.replace("dream_ft", "dream_ft_tall", 1)
        return super().episode(t0, t1, root=root, **label)

    def relook(self, colour):
        b = self.belief[colour]
        recent = [pl for pl in getattr(self, "places", []) if pl[2] > b.get("t_seen", 0) and math.hypot(pl[0] - b["x"], pl[1] - b["y"]) < 0.10]
        if time.time() - b.get("t_seen", 0) < 30 and not recent:
            return
        super().relook(colour)
        self.belief[colour]["t_seen"] = time.time()

    def block(self, colour, target, *a, **k):
        out = super().block(colour, target, *a, **k)
        self.places = getattr(self, "places", []) + [(target[0], target[1], time.time())]
        return out

    def sync(self):
        super().sync()
        import os
        import subprocess
        remote = os.environ.get("STATION_LOOP")  # see data_loop.Loop.sync
        if not remote:
            return
        for src in ("dream_ft_tall/", "dream_ft_tall_failures/"):
            (D.LOOP / src).mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-a", "--bwlimit=1500", f"{remote}/{src}", f"{D.LOOP / src}"], timeout=600, check=False)

    # ---- finding every block
    def find_blocks(self, img, q, grip):
        view = self.op.view
        opening = grip * 0.1
        cam = view.pose(q, opening)
        jaws = view.jaws_mask(q, opening)
        blocks, obst, rejected = [], [], []
        flat = Shape((FACE, FACE, H), TABLE_Z)
        lying = shape_of({"lying": True}, TABLE_Z)
        for c in BLOCK_COLOURS + ("red",):
            m = colour_mask(img, c, jaws)
            n, lab, stats, cen = cv2.connectedComponentsWithStats(m)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] < 150:
                    continue
                blob = lab == i
                ys, xs = np.nonzero(blob)
                pts = view.ray_on_plane(cam, np.stack([xs, ys], 1)[:: max(1, len(xs) // 200)], TABLE_Z + 0.015)[:, :2]
                fill, solid = roundness(blob)
                kind = "ball" if (fill > 0.78 and solid > 0.93) else None
                f = None
                (_, _), (rw, rh), _ = cv2.minAreaRect((pts * 1000).astype(np.float32))
                aspect = max(rw, rh) / max(min(rw, rh), 1e-3)
                if c in BLOCK_COLOURS and kind is None and aspect <= 2.6:
                    f = view._fit(cam, jaws, lab, stats, i, flat, None)
                    fl = view._fit(cam, jaws, lab, stats, i, lying, None)
                    fs = view._fit(cam, jaws, lab, stats, i, STICK, None)
                    on_side = fl is not None and (f is None or fl["iou"] > f["iou"] + 0.03)
                    if on_side:
                        f = fl
                    if fs is not None and f is not None and fs["iou"] > f["iou"]:  # a lying stick fits it better
                        f = None
                    # 0.62: a block partly hidden behind a touching neighbour still fits about 0.7
                    if f is not None and f["iou"] >= 0.62 and 0.07 < f["x"] < 0.55 and -0.30 < f["y"] < 0.40:
                        blocks.append(dict(f, colour=c, level=0, z0=TABLE_Z, lying=bool(on_side), px=int(stats[i, cv2.CC_STAT_AREA])))
                        continue
                    kind = "stick or cut block"
                elif aspect > 2.6:
                    kind = f"stick (aspect {aspect:.1f})"
                obst.append(pts)
                rejected.append({"colour": c, "kind": kind or "stick", "px": int(stats[i, cv2.CC_STAT_AREA]),
                                 "iou": None if f is None else round(f["iou"], 2), "at_mm": (pts.mean(0) * 1000).round().tolist()})
        # ids: per colour, nearest the arm first
        out = {}
        for c in BLOCK_COLOURS:
            for j, b in enumerate(sorted([b for b in blocks if b["colour"] == c], key=lambda b: (b["x"], b["y"]))):
                out[f"{c}#{j + 1}"] = {k: b[k] for k in ("x", "y", "yaw", "level", "iou", "edge", "z0", "lying")}
        o = np.concatenate(obst) if obst else np.zeros((0, 2))
        o = o[(o[:, 0] > 0.15) & (o[:, 0] < 0.55) & (o[:, 1] > -0.2) & (o[:, 1] < 0.35)]
        for b in out.values():
            o = o[np.hypot(o[:, 0] - b["x"], o[:, 1] - b["y"]) > 0.045]
        return out, o, rejected, jaws

    def look_all(self, tag, tip=LOOK_TIP, move=True):
        if move:
            self.to_pose(tip, "look")
            time.sleep(0.25)
        img, q, grip, stem = self.op.snap(tag)
        self.last_look_img = self.op.run / f"{stem}-wrist.jpg"
        blocks, obst, rejected, jaws = self.find_blocks(img, q, grip)
        for b in blocks.values():
            b["t_seen"] = time.time()
        self.belief, self.obstacles = blocks, obst
        self.q_look, self.jaws_look = q, jaws
        self.put_blocks()
        try:
            self.look_overhead = O.overhead_blocks((self.op.run / f"{stem}-overhead.jpg").read_bytes())
        except Exception:  # noqa: BLE001
            self.look_overhead = {}
        out = self.op.overlay(img, q, grip * 0.1, {bid: (dict(b), shape_of(b)) for bid, b in blocks.items()}, stem)
        cam = self.op.view.pose(q, grip * 0.1)
        for p in obst[::4]:
            uv, ok = self.op.view.project(cam, [[p[0], p[1], TABLE_Z + 0.011]])
            if ok.all():
                cv2.circle(out, tuple(int(v) for v in uv[0]), 2, (0, 0, 255), -1)
        for bid, b in blocks.items():
            uv, ok = self.op.view.project(cam, [[b["x"], b["y"], TABLE_Z + H]])
            if ok.all():
                cv2.putText(out, bid, (int(uv[0][0]) - 20, int(uv[0][1]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(out, bid, (int(uv[0][0]) - 20, int(uv[0][1]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        path = self.op.run / f"{stem}-blocks.jpg"
        cv2.imwrite(str(path), out)
        O.record("look", run=self.dir.name, stem=stem, wrist=blocks, rejected=rejected, overhead=self.look_overhead)
        return blocks, rejected, path

    # ---- picks that keep the open fingers off touching neighbours
    def pad_hits(self, p, xaxis, exempt, finger=FINGER):
        """Neighbour blocks a pad would overlap at fingertips p with the jaws closing along xaxis."""
        hits = []
        yaxis = np.array([-xaxis[1], xaxis[0]])
        for c, b in self.belief.items():
            if c in exempt:
                continue
            if p[2] > top_of(b) + 0.015 or bot(b) > p[2] + 0.06:
                continue
            cb, sb = math.cos(b["yaw"]), math.sin(b["yaw"])
            hu, hv = ((H / 2, FACE / 2) if b.get("lying") else (FACE / 2, FACE / 2))
            for s in (-1, 1):
                ctr = p[:2] + s * finger * xaxis[:2]
                corners = [ctr + a * PAD_HALF[0] * xaxis[:2] + w * PAD_HALF[1] * yaxis for a in (-1, 1) for w in (-1, 1)] + [ctr]
                for pt in corners:
                    d = pt - [b["x"], b["y"]]
                    u, v = d[0] * cb + d[1] * sb, -d[0] * sb + d[1] * cb
                    if abs(u) < hu + 0.003 and abs(v) < hv + 0.003:
                        hits.append(c)
                        break
        # sticks and balls lie up to 43 mm high: no pad within 5 mm of one of their points while that low
        if len(self.obstacles) and p[2] < TABLE_Z + 0.06:
            for s in (-1, 1):
                ctr = p[:2] + s * finger * xaxis[:2]
                d = self.obstacles - ctr
                u, v = np.abs(d @ xaxis[:2]), np.abs(d @ yaxis)
                if np.any((u < PAD_HALF[0]) & (v < PAD_HALF[1])):  # foam sticks: 3 mm past the pad's outline
                    hits.append("an obstacle")
                    break
        return hits

    def finger_sweep(self, plan, exempt, finger=FINGER):
        for (q0, _, _), (q1, _, tag) in zip(plan.nodes[:-1], plan.nodes[1:]):
            n = max(2, int(math.ceil(np.max(np.abs(q1 - q0)) / math.radians(2))) + 1)
            for t in np.linspace(0, 1, n):
                pos, R = self.op.kin.fk(q0 + (q1 - q0) * t)
                p = pos + self.op.kin.tip * R[:, 2]
                hits = self.pad_hits(p, R[:, 0], exempt, finger)
                if hits:
                    raise SystemExit(f"fingers refused at {tag}: a pad meets {hits}")

    def plan_pick(self, colour, q0, regrip=0.0):
        b = self.belief[colour]
        jy = math.degrees(b["yaw"])
        cxy = np.array([b["x"], b["y"]]) - PICK_DOWN[:2]
        z_grip = GRIP_TIP + (bot(b) - TABLE_Z) - PICK_DOWN[2] + (0.010 if bot(b) > TABLE_Z + 0.005 or b.get("lying") else 0.0) + regrip
        z_over = min(top_of(b) + 0.08 + self.dr["over_dz"], D.REACH_Z)
        ex = self.below(colour)
        last = None
        # the widest opening (80, 70 or 63 mm) with a jaw axis whose open fingers clear the neighbours at the grip
        axes = []
        for g in (0.8, 0.7, 0.63):
            finger = g * 0.05 + 0.003
            # on its side the block is 34.5 mm along its yaw (the hole) and 50.3 across: the pads close across, on the
            # 50 mm span, or the 0.44 close would stop short of a 34.5 mm grip
            for par in ((1,) if b.get("lying") else (0, 1)):
                a = math.radians(jy) + par * math.pi / 2
                if not self.pad_hits(np.array([b["x"], b["y"], z_grip]), np.array([math.cos(a), math.sin(a), 0.0]), ex, finger):
                    axes.append(par)
            if axes:
                break
        if not axes:
            raise SystemExit(f"pick {colour}: neighbours close on both jaw axes even at a 63 mm opening")
        self.pick_open = getattr(self, "pick_open", {})
        self.pick_open[colour] = g
        z_high = z_over
        for route in ("direct", "high", "low"):
            # low: off a tower near the reach limit, over at the top + 3 cm (the place's own approach height)
            z_over = top_of(b) + 0.03 if route == "low" else z_high
            pl = AxisPlan(self.op, q0, 0.08)
            pl.parity = axes
            try:
                if route in ("high", "low"):
                    p0 = self.op.tip(q0)[0]
                    pl.to([p0[0], p0[1], max(self.dr["carry_z"], p0[2])], self.op.tip(q0)[1], 0.4, "up", line=0.02)
                    pl.R = None
                    pl.to([cxy[0], cxy[1], max(self.dr["carry_z"], z_over)], jy, 0.5, "over high")
                    pl.to([cxy[0], cxy[1], z_over], jy, 0.4, "over", line=0.02, keep=True)
                else:
                    pl.to([cxy[0], cxy[1], z_over], jy, V_HIGH, "over")
                pl.to([cxy[0], cxy[1], z_grip + 0.03], jy, V_DOWN, "down", line=0.01, keep=True)
                pl.to([cxy[0], cxy[1], z_grip], jy, V_LAST, "last 3 cm", line=0.005, keep=True)
                pl.check(skip=colour)
                self.finger_sweep(pl, ex, finger)
                self.sweep_high(pl, ex)
                return pl
            except SystemExit as e:
                last = e
        raise SystemExit(f"pick {colour}: {last}")

    def sweep_high(self, plan, exempt):
        """The data loop's sweep for the parts of a pick path above the neighbours' grip zone (fingertips more than
        1.5 cm over a block's top are never in the way; lower, the pad check above decides)."""
        for (q0, _, _), (q1, _, tag) in zip(plan.nodes[:-1], plan.nodes[1:]):
            n = max(2, int(math.ceil(np.max(np.abs(q1 - q0)) / math.radians(2))) + 1)
            for t in np.linspace(0, 1, n):
                p = self.op.tip(q0 + (q1 - q0) * t)[0]
                for c, b in self.belief.items():
                    if c in exempt:
                        continue
                    top = top_of(b)
                    if math.hypot(p[0] - b["x"], p[1] - b["y"]) < 0.03 and p[2] < top + 0.012:
                        raise SystemExit(f"sweep refused at {tag}: fingertips over the {c} block")

    # ---- the tower
    def tower_spot(self, order, levels=5):
        """A spot where the most levels plan (all of them if it can). Returns (x, y, yaw, order that plans)."""
        import collections
        best = None
        self.why = collections.Counter()
        tried = 0
        for x, y, yaw in self.spot(None, list(range(levels)), avoid=[(LOOK_TIP[0], LOOK_TIP[1])], gap=0.09):
            tried += 1
            got = self.plan_tower((x, y, yaw), order)
            if best is None or len(got) > len(best[3]):
                best = (x, y, yaw, got, getattr(self, "last_refusal", None))
            if len(got) == levels or tried >= 40:
                break
        if best:
            self.last_refusal = best[4]
            return best[:4]
        return None

    def plan_level(self, bid, lvl, spot, sup, q):
        x, y, yaw = spot
        st = top_of(self.belief[sup]) if sup else TABLE_Z
        self.dr = {"carry_z": min(max(0.15, st + 0.05), max(D.REACH_Z, st + 0.035)), "over_dz": 0.01}
        self.op.st["target"] = q.tolist()
        pl = self.plan_pick(bid, q)
        lift = self.plan_lift(pl.nodes[-1][0], pl.R, math.degrees(self.belief[bid]["yaw"]), bid)
        grasp_h = 0.025 if self.belief[bid].get("lying") else 0.015
        p3 = self.plan_place(bid, lift.nodes[-1][0], lift.R, (x, y, yaw, st, sup), grasp_h)
        for zt, tag in ((st + hgt(self.belief[bid]), "check"),) + (((st, "support look"),) if sup else ()):
            ck = Plan(self.op, p3.nodes[-1][0], 0.08)
            ck.to([x - 0.065, y, min(max(0.13, zt + 0.06), D.LOOK_MAX_Z)], math.degrees(yaw), 0.4, tag)
            ck.check(skip=bid)

    def plan_tower(self, spot, order, base=()):
        """Greedy: at each level the first block (in this order) whose pick, place, close look and check photo all plan,
        with the blocks already placed moved onto the tower. base: the tower already standing (bottom first).
        Returns the order that planned (after base)."""
        saved = {k: dict(v) for k, v in self.belief.items()}
        q = self.q_look if self.q_look is not None else self.start_q()
        got, why = [], {}
        stack = list(base)
        try:
            for lvl in range(len(stack), len(stack) + len(order)):
                ok = None
                top = self.belief[stack[-1]] if stack else None
                at = (top["x"], top["y"], top["yaw"]) if top else spot
                for bid in [b for b in order if b not in got]:
                    try:
                        self.plan_level(bid, lvl, at, stack[-1] if stack else None, q)
                        ok = bid
                        break
                    except SystemExit as e:
                        why[bid] = str(e)[:160]
                if ok is None:
                    self.last_refusal = f"level {lvl} (block {lvl + 1}): " + "; ".join(f"{b}: {w}" for b, w in why.items() if b not in got)
                    break
                st = top_of(self.belief[stack[-1]]) if stack else TABLE_Z
                self.belief[ok] = {"x": at[0], "y": at[1], "yaw": at[2], "level": lvl, "iou": 0.95, "edge": False, "z0": st,
                                   "lying": self.belief[ok].get("lying", False)}
                self.put_blocks()
                got.append(ok)
                stack.append(ok)
                why = {}
        finally:
            self.belief = saved
            self.put_blocks()
            self.op.st["target"] = None
        return got

    def refresh_free(self, tower):
        """A full look from the look pose: the blocks not in the tower refitted and keyed by their nearest id."""
        old = {k: dict(v) for k, v in self.belief.items()}
        blocks, rejected, path = self.look_all("look-refresh")
        new = {k: v for k, v in old.items() if k in tower}
        tx, ty = old[tower[0]]["x"], old[tower[0]]["y"]
        free = [b for b in blocks.values() if math.hypot(b["x"] - tx, b["y"] - ty) > 0.05]
        cols = {k: v for k, v in blocks.items() if v in free}
        prev = {k: v for k, v in old.items() if k not in tower}
        for k, b in sorted(cols.items(), key=lambda kv: kv[0]):
            c = colour_of(k)
            cands = [p for p in prev if colour_of(p) == c and p not in new]
            bid = min(cands, key=lambda p: math.hypot(prev[p]["x"] - b["x"], prev[p]["y"] - b["y"])) if cands else None
            if bid is None:
                n = 1
                while f"{c}#{n}" in new or f"{c}#{n}" in tower:
                    n += 1
                bid = f"{c}#{n}"
            new[bid] = b
        self.belief = new
        self.put_blocks()
        D.status("tall tower full look: " + ", ".join(f"{k} ({b['x'] * 100:.1f}, {b['y'] * 100:.1f}) L{b['level']} iou {b['iou']:.2f}"
                                                     for k, b in new.items()) + f"; overlay {path}")

    def move_tower(self, tower, pile, parked=()):
        """Unstack the tower top-down to spread spots (18-30 cm out, 8 cm apart), rebuild it in the same order at the
        spot near x 18-22 cm where the most levels (tower then pile) plan, then add the pile blocks top first."""
        import collections
        area = D.AREA
        try:
            # where it goes: the most levels planned with every block at its parking spot unknown yet: plan from
            # the blocks where they stand (the tower's own blocks moved off it first in the plan)
            D.AREA = ((0.16, 0.26), (-0.14, 0.14))
            self.why = collections.Counter()
            best = None
            cands = []
            for x, y, yaw in self.spot(None, [0, 1, 2], avoid=[(self.belief[tower[0]]["x"], self.belief[tower[0]]["y"])], gap=0.06):
                cands.append((x, y, yaw))
                if len(cands) >= 40:
                    break
            if cands:  # the reach is highest near x 18-22, y -5..10
                x, y, yaw = min(cands, key=lambda c: math.hypot(c[0] - 0.20, c[1] - 0.025))
                best = (x, y, yaw, [])
            if best is None:
                raise D.Fatal(f"no new tower spot near x 18-22 cm: {dict(self.why)}")
            x, y, yaw, got = best
            D.status(f"tall tower: new spot ({x * 100:.1f}, {y * 100:.1f}) yaw {math.degrees(yaw):.0f}: {len(got)} levels plan ({got}); limit {getattr(self, 'last_refusal', None)}")
            # park the tower blocks top-down, 8 cm apart, 18-30 cm out, clear of the new spot
            D.AREA = ((0.18, 0.30), (-0.10, 0.20))
            park = []
            for k in reversed(tower):
                wait_no_hands(self)
                px, py, pyaw = self.pick_spot(k, [0], avoid=[(x, y)] + park)
                self.block(k, (px, py, pyaw, 0, None), 1, 0, "unstack")
                park.append((px, py))
            # rebuild in the same order at the new spot
            built = []
            for k in list(tower) + list(parked):
                wait_no_hands(self)
                if built:
                    s_ = self.belief[built[-1]]
                    tgt = (s_["x"], s_["y"], s_["yaw"], len(built), built[-1])
                else:
                    tgt = (x, y, yaw, 0, None)
                self.block(k, tgt, 1, len(built) + 1, "stack")
                built.append(k)
                D.status(f"tall tower: {len(built)} high at the new spot ({' / '.join(built)}), top at {(top_of(self.belief[k]) - TABLE_Z) * 1000:.0f} mm")
        finally:
            D.AREA = area
        built = self.build_seq(pile, built)
        D.status(f"tall tower done: {len(built)} blocks ({' / '.join(built)}), top at {(top_of(self.belief[built[-1]]) - TABLE_Z) * 1000:.0f} mm over the table")
        return built

    def build_seq(self, order, base):
        """Strictly in this order (a pile comes off top first), each onto the tower's top; stops at the first failure."""
        built = list(base)
        self.cyc = {"tower_order": list(base) + list(order), "fallbacks": []}
        for bid in order:
            wait_no_hands(self)
            s_ = self.belief[built[-1]]
            try:
                self.block(bid, (s_["x"], s_["y"], s_["yaw"], len(built), built[-1]), 1, len(built) + 1, "stack")
            except (D.NoPlan, D.NoSpot) as e:
                D.status(f"tall tower: {bid} cannot go on at level {len(built)}: {str(e)[:300]}")
                break
            built.append(bid)
            D.status(f"tall tower: {len(built)} high ({' / '.join(built)}), top at {(top_of(self.belief[bid]) - TABLE_Z) * 1000:.0f} mm")
        return built

    def build(self, order, spot, base=()):
        """Places in the planned order; a block whose plan or grasp fails now gives way to the next one (each block
        retried at most twice). When none is left that plans, one full look for blocks moved within reach."""
        x, y, yaw = spot
        self.cyc = {"tower_spot": [x, y, yaw], "tower_order": list(base) + list(order), "fallbacks": []}
        built, left = list(base), list(order)
        tries = {}
        looked = False
        while True:
            lvl = len(built)
            sup = built[-1] if built else None
            if sup:
                s_ = self.belief[sup]
                target = (s_["x"], s_["y"], s_["yaw"], lvl, sup)
            else:
                target = (x, y, yaw, 0, None)
            done = None
            for bid in list(left):
                if tries.get(bid, 0) >= 3:
                    continue
                try:
                    tries[bid] = tries.get(bid, 0) + 1
                    self.block(bid, target, 1, lvl + 1, "stack")
                    done = bid
                    break
                except D.NoPlan as e:
                    D.status(f"tall tower: {bid} does not plan for level {lvl} now ({str(e)[:200]}); trying the next block")
                except D.CycleFail as e:
                    st = arm_state()
                    if st["grip"] < 0.75 and st["goal_grip"] < 0.75:
                        raise
                    D.status(f"tall tower: {bid} failed ({e}); trying the next block, then {bid} again")
                    self.op.st["target"] = None
            if done is None:
                if looked:
                    D.status(f"tall tower: no remaining block plans for level {lvl}: stopping at {lvl} high")
                    break
                looked = True
                self.refresh_free(built)
                left = [k for k in self.belief if k not in built]
                tries = {}
                continue
            left.remove(done)
            built.append(done)
            looked = False
            D.status(f"tall tower: {len(built)} high ({' / '.join(built)})")
            if not left:
                self.refresh_free(built)  # a block slid within reach gets added too
                left = [k for k in self.belief if k not in built]
                looked = True
                if not left:
                    break
        return built


def hands_px(bgr):
    """Skin-coloured pixels along the wrist frame's bottom edge (where a hand reaching in shows), saturated orange
    and red foam left out."""
    h = bgr.shape[0]
    w = bgr.shape[1]
    hsv = cv2.cvtColor(bgr[int(h * 0.45):int(h * 0.80), int(w * 0.05):int(w * 0.95)], cv2.COLOR_BGR2HSV)
    skin = ((hsv[..., 0] < 25) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 35) & (hsv[..., 1] < 150) & (hsv[..., 2] > 70)
    return int(skin.sum())


def wait_no_hands(t, limit=2500, timeout=300):
    t0 = time.time()
    while True:
        data, _, _ = O.fetch("wrist")
        n = hands_px(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR))
        if n < limit:
            return n
        if time.time() - t0 > timeout:
            raise D.Fatal(f"a hand stayed in the wrist view for {timeout} s ({n} px)")
        D.status(f"hand in the wrist view ({n} px): waiting")
        time.sleep(3)


def use_sim(port):
    """The same script against calib/station_sim.py on PORT: its URLs and clock, runs/sim_station/ for the token,
    status and sessions, no rsync."""
    global TALL, time
    from calib import arm_runner as AR
    base = f"http://127.0.0.1:{port}"
    clock = AR.sim_clock(base)
    AR.use_backend(AR.sim_urls(base), clock)
    time = clock
    sim = ROOT / "runs" / "sim_station"
    D.TOKEN, D.LOOP = sim / "ARM_OWNER", sim / "loop"
    D.STATUS = D.LOOP / "STATUS.txt"
    D.LOOP.mkdir(parents=True, exist_ok=True)
    TALL = sim / "tall"
    Tall.sync = lambda self: None


def main():
    if "--sim" in sys.argv:
        i = sys.argv.index("--sim")
        use_sim(int(sys.argv[i + 1]))
        del sys.argv[i:i + 2]
    cmd = sys.argv[1]
    TALL.mkdir(parents=True, exist_ok=True)
    session = TALL / time.strftime("%m%d-%H%M%S")
    if cmd in ("look", "dry"):
        t = Tall(session)
        t.q_look = None
        blocks, rejected, path = t.look_all("look", move=False)
        print(json.dumps({"blocks": {k: {kk: round(v, 4) if isinstance(v, float) else v for kk, v in b.items()} for k, b in blocks.items()},
                          "not_blocks": rejected, "overlay": str(path)}, indent=1))
        if cmd == "dry":
            order = sorted(blocks, key=lambda k: -min([math.hypot(blocks[k]["x"] - o["x"], blocks[k]["y"] - o["y"]) for j, o in blocks.items() if j != k] or [1]))
            t.dr = {"carry_z": 0.15, "over_dz": 0.01}
            res = t.tower_spot(order, len(order))
            print("ORDER", order, "SPOT", None if res is None else [round(v, 4) for v in res[:3]], "PLANS", None if res is None else res[3])
            print("LIMIT", getattr(t, "last_refusal", None))
            print("SPOT REJECTIONS", dict(t.why))
        return
    if cmd == "continue":  # continue SESSION ID [ID ...]: the tower standing from SESSION (bottom first)
        prev = TALL / sys.argv[2]
        lying_ids = {a.split(":")[0] for a in sys.argv[3:] if a.endswith(":lying")}
        tower = [a.split(":")[0] for a in sys.argv[3:]]
        fits, looks = {}, None
        for f in sorted(TALL.glob("*/calib_records.jsonl")):  # every tall session, the latest fit of each block wins
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r["kind"] == "place" and r.get("fit") and r["block"] in tower:
                    fits[r["block"]] = r["fit"]
                if r["kind"] == "look":
                    looks = r["wrist"]
        D.take_token()
        t = Tall(session)
        t.op.st["target"] = None
        D.status(f"tall tower session {session.name} (continuing {prev.name}, standing {' / '.join(tower)}): token taken")
        D.rec("/start")
        clean = True
        try:
            t.belief = {**{k: dict(v) for k, v in (looks or {}).items() if k not in tower},
                        **{k: {"x": f["x"], "y": f["y"], "yaw": f["yaw"], "level": f["level"], "iou": f["iou"], "edge": f["edge"]} for k, f in fits.items()}}
            z = TABLE_Z
            for k in tower:  # the standing tower's bottoms from its blocks' heights, bottom first
                t.belief[k]["z0"] = z
                t.belief[k]["lying"] = k in lying_ids
                z = top_of(t.belief[k])
            t.refresh_free(tower)
            for k in tower:
                if k in lying_ids:  # refit on its side at its real bottom from the fresh look
                    b = t.belief[k]
                    img = cv2.resize(cv2.imread(str(t.last_look_img)), (640, 360), interpolation=cv2.INTER_AREA)
                    f = t.op.view.find(img, k, t.q_look, 0.08, shape_of(b), (b["x"], b["y"], b["yaw"]))
                    D.status(f"tall tower: {k} refit on its side at {bot(b) * 1000:.0f} mm: {None if f is None else {kk: round(v, 4) for kk, v in f.items() if isinstance(v, float)}}")
                    if f is not None and f["iou"] >= 0.6 and math.hypot(f["x"] - b["x"], f["y"] - b["y"]) < 0.03:
                        b.update(x=f["x"], y=f["y"], yaw=f["yaw"], iou=f["iou"])
            t.put_blocks()
            free = [k for k in t.belief if k not in tower]
            order = sorted(free, key=lambda k: -min([math.hypot(t.belief[k]["x"] - o["x"], t.belief[k]["y"] - o["y"]) for j, o in t.belief.items() if j != k] or [1]))
            t.dr = {"carry_z": 0.15, "over_dz": 0.01}
            base = t.belief[tower[0]]
            got = t.plan_tower((base["x"], base["y"], base["yaw"]), order, tower)
            D.status(f"tall tower plan: {len(got)} more on {len(tower)} plan, order {got}; limit: {getattr(t, 'last_refusal', None) if len(got) < len(order) else 'none'}")
            built = t.build(got + [k for k in order if k not in got], (base["x"], base["y"], base["yaw"]), tower)
            D.status(f"tall tower done: {len(built)} blocks ({' / '.join(built)}), top at {(top_of(t.belief[built[-1]]) - TABLE_Z) * 1000:.0f} mm over the table")
        except BaseException as e:  # noqa: BLE001
            D.status(f"tall tower stopped: {type(e).__name__}: {e}")
            st = arm_state()
            clean = st["grip"] > 0.75 and not st.get("grip_holding")
        finally:
            finish(t, clean)
        return
    if cmd == "pile":  # pile SESSION TOWER_IDS... -- X Y COLOUR:SHAPE:LEVEL ...: unstack a pile at (X, Y) onto the tower
        i = sys.argv.index("--")
        prev, tower_args = TALL / sys.argv[2], sys.argv[3:i]
        px, py = float(sys.argv[i + 1]), float(sys.argv[i + 2])
        j = sys.argv.index("--then") if "--then" in sys.argv else len(sys.argv)
        then_args = sys.argv[j + 1:] if j < len(sys.argv) else []
        pile = [a.split(":") for a in sys.argv[i + 3:j] if not a.startswith("--")]
        lying_ids = {a.split(":")[0].split("@")[0] for a in tower_args + then_args if a.endswith(":lying")}
        parked = [a.split(":")[0] for a in then_args]
        tower = [a.split(":")[0].split("@")[0] for a in tower_args]
        fits = {}
        for f in sorted(TALL.glob("*/calib_records.jsonl")):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r["kind"] == "place" and r.get("fit") and r["block"] in tower + parked:
                    fits[r["block"]] = r["fit"]
        for a in tower_args:  # id@x,y,yaw gives a tower block's pose directly (sim runs)
            if "@" in a:
                k, xyz = a.split(":")[0].split("@")
                x_, y_, w_ = (float(v) for v in xyz.split(","))
                fits[k] = {"x": x_, "y": y_, "yaw": w_, "iou": 0.9, "edge": False}
        dry = "--dry" in sys.argv
        if "--sim" in sys.argv:
            sys.argv = [a for a in sys.argv if a not in ("--sim",)]
        if not dry:
            D.take_token()
        t = Tall(session)
        t.op.st["target"] = None
        clean = True
        try:
            if not dry:
                D.status(f"tall tower session {session.name}: unstacking the pile at ({px * 100:.0f}, {py * 100:.0f}) onto {' / '.join(tower)}; token taken")
                D.rec("/start")
            t.belief = {k: {"x": f["x"], "y": f["y"], "yaw": f["yaw"], "level": n, "iou": f["iou"], "edge": f["edge"]} for n, (k, f) in enumerate((k, fits[k]) for k in tower)}
            z = TABLE_Z
            for k in tower:
                t.belief[k]["z0"], t.belief[k]["lying"] = z, k in lying_ids
                z = top_of(t.belief[k])
            for k in parked:
                f = fits[k]
                t.belief[k] = {"x": f["x"], "y": f["y"], "yaw": f["yaw"], "level": 0, "iou": f["iou"], "edge": f["edge"], "z0": TABLE_Z, "lying": k in lying_ids}
            tb = {k: dict(v) for k, v in t.belief.items()}
            wait_no_hands(t)
            blocks, rejected, path = t.look_all("look-pile", move=not dry)
            img = cv2.resize(cv2.imread(str(t.last_look_img)), (640, 360), interpolation=cv2.INTER_AREA)
            for k in tower:  # the tower from this look, each at its own bottom and shape
                b = tb[k]
                f = t.op.view.find(img, k, t.q_look, 0.08, shape_of(b), (b["x"], b["y"], b["yaw"]))
                if f is not None and f["iou"] >= 0.8 and math.hypot(f["x"] - b["x"], f["y"] - b["y"]) < 0.02:
                    b.update(x=f["x"], y=f["y"], yaw=f["yaw"], iou=f["iou"])
            names = []
            gx, gy = px, py
            for colour, shape, lvl in pile:
                bid = f"{colour}#pile"
                b = {"x": gx, "y": gy, "yaw": 0.0, "level": int(lvl), "z0": TABLE_Z + int(lvl) * H, "lying": shape == "lying", "iou": 0.0, "edge": False}
                f = t.op.view.find(img, bid, t.q_look, 0.08, shape_of(b), (gx, gy, 0.0))
                D.status(f"pile {bid}: {shape} at level {lvl}: fit {None if f is None else {kk: round(v, 3) for kk, v in f.items() if isinstance(v, float)}}")
                if f is None or f["iou"] < 0.55 or math.hypot(f["x"] - gx, f["y"] - gy) > 0.08:
                    if not names:
                        raise D.Fatal(f"pile top {bid} not found as {shape} at level {lvl}")
                    # hidden under the blocks above it: placed under them, re-looked once they are off
                    D.status(f"pile {bid}: hidden for now; re-looked when its turn comes")
                    tb[bid] = b
                    names.append(bid)
                    continue
                b.update(x=f["x"], y=f["y"], yaw=f["yaw"], iou=f["iou"], edge=f["edge"])
                gx, gy = f["x"], f["y"]  # the blocks under it sit right below
                tb[bid] = b
                names.append(bid)
            t.belief = tb
            # the known blocks' own side faces (a tower's lower blocks fail the flat fit) are not obstacles
            for b in tb.values():
                if len(t.obstacles):
                    t.obstacles = t.obstacles[np.hypot(t.obstacles[:, 0] - b["x"], t.obstacles[:, 1] - b["y"]) > 0.045]
            t.put_blocks()
            t.dr = {"carry_z": 0.15, "over_dz": 0.01}
            # dry run: each pile block in turn onto the tower top, as high as reach plans
            b0 = t.belief[tower[0]]
            if "--move" in sys.argv:  # move the tower near x 18-22 cm, where the reach goes highest, then add the pile
                t.move_tower(tower, names, parked)
                return
            got = t.plan_tower((b0["x"], b0["y"], b0["yaw"]), names[:1], tower)
            # the pile comes off top first; the hidden ones plan for real once they are seen
            order = names if got else []
            D.status(f"pile plan: {order} onto the tower (planned {got}); limit: {getattr(t, 'last_refusal', None)}")
            if dry:
                return
            built = t.build_seq(order, tower)
            D.status(f"tall tower done: {len(built)} blocks ({' / '.join(built)}), top at {(top_of(t.belief[built[-1]]) - TABLE_Z) * 1000:.0f} mm over the table")
        except BaseException as e:  # noqa: BLE001
            D.status(f"tall tower stopped: {type(e).__name__}: {e}")
            if dry:
                raise
            st = arm_state()
            clean = st["grip"] > 0.75 and not st.get("grip_holding")
        finally:
            if not dry:
                finish(t, clean)
        return
    if cmd == "run":
        D.take_token()
        t = Tall(session)
        t.op.st["target"] = None
        D.status(f"tall tower session {session.name}: token taken")
        D.rec("/start")
        clean = True
        try:
            blocks, rejected, path = t.look_all("look")
            D.status(f"tall tower look: {len(blocks)} blocks " + ", ".join(f"{k} ({b['x'] * 100:.1f}, {b['y'] * 100:.1f}) iou {b['iou']:.2f}"
                                                                          for k, b in blocks.items()) + f"; overlay {path}")
            order = sorted(blocks, key=lambda k: -min([math.hypot(blocks[k]["x"] - o["x"], blocks[k]["y"] - o["y"]) for j, o in blocks.items() if j != k] or [1]))
            t.dr = {"carry_z": 0.15, "over_dz": 0.01}
            res = t.tower_spot(order, len(order))
            if res is None:
                raise D.Fatal(f"no tower spot: {dict(t.why)}")
            x, y, yaw, got = res
            D.status(f"tall tower plan: {len(got)} of {len(order)} levels plan at ({x * 100:.1f}, {y * 100:.1f}) yaw {math.degrees(yaw):.0f}; "
                     f"order {got}; limit: {getattr(t, 'last_refusal', None) if len(got) < len(order) else 'none'}")
            built = t.build(got, (x, y, yaw))
            D.status(f"tall tower done: {len(built)} blocks, top at {(TABLE_Z + len(built) * H + 0.008) * 1000:.0f} mm over the table")
        except BaseException as e:  # noqa: BLE001
            D.status(f"tall tower stopped: {type(e).__name__}: {e}")
            st = arm_state()
            clean = st["grip"] > 0.75 and not st.get("grip_holding")
        finally:
            finish(t, clean)


def finish(t, clean):
    try:
        t.op.st["target"] = None
        if clean:
            t.to_pose(LOOK_TIP, "end")
            time.sleep(0.4)
            img, q, grip, stem = t.op.snap("final")
            D.status(f"tall tower final photos: {t.op.run / (stem + '-wrist.jpg')} and {t.op.run / (stem + '-overhead.jpg')}")
    except BaseException as e:  # noqa: BLE001
        clean = False
        D.status(f"end pose failed: {e}")
    t.saves.join()
    D.rec("/stop")
    if clean:
        D.release_token()
        D.status("tall tower: token released")
    else:
        (D.TOKEN / "who.txt").write_text(f"data-loop HELD {time.strftime('%H:%M:%S')}: tall tower stopped unclean\n")
    t.sync()


if __name__ == "__main__":
    main()
