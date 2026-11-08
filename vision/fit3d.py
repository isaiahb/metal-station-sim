"""Fits each piece's CAD to its SAM 3 mask: the pose on the table that makes the model's outline match.

    python -m vision.fit3d <frames dir> <sam dir> [--calib calib/cam_mid.json]

The table is flat and the plate lies flush on it, so a piece lying on it has three unknowns: x, y and
heading. The model (block 50.7 x 49.9 x 34.5 mm; stick 22.65 mm across, 100.5 mm long, from the
calipers) is projected through the calibrated overhead camera; its outline is the convex hull of the
projected corners (block) or end circles (stick), exact for convex solids. The pose maximises the
overlap (IoU) of that outline with the SAM 3 mask, ignoring model pixels on dark things in front of it
(the black arm), so a part hidden by the arm does not pull the fit.

SAM 3 masks come from vision/sam_label.py (on a pod); masks from all prompts are pooled, duplicates
merged, and each is named by its median hue (purple, red, blue, green) and by which prompts found it.

Writes runs/vision/real/<stamp>/: per frame an overlay (green: SAM 3 mask outline; orange: the fitted
CAD outline; blue: the fast colour tracker's outline, vision/locate.track) and tracks.json with the
fitted pieces (the vision page's station tab shows them in 3D) and how far the fast tracker is from them.
"""

import json
import math
import pathlib
import sys
import time

import cv2
import numpy as np
from scipy.optimize import minimize

from calib.fit_overhead import SPACING, TABLE_TOP_ROW
from vision.locate import BLOCK, PEG, Camera, track
from vision.track_real import IGNORE, TABLE_Z

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "vision" / "real"
BLOCK_KEYS = ("block", "cube", "purple_block", "blue_block", "green_block")
STICK_KEYS = ("stick", "cylinder", "purple_stick", "red_stick")
MIN_SCORE, MIN_PX, MERGE_IOU = 0.4, 150, 0.5
COLOURS = {"red": [(0, 10), (176, 181)], "purple": [(135, 176)], "blue": [(88, 125)], "green": [(40, 88)]}
KINDS = {"red": {"stick"}, "purple": {"block", "stick"}, "blue": {"block"}, "green": {"block"}}  # the station's five pieces
GREEN, ORANGE, BLUE = (60, 200, 60), (40, 140, 255), (230, 140, 40)


def hue_name(h: float) -> str | None:
    return next((n for n, bins in COLOURS.items() if any(lo <= h < hi for lo, hi in bins)), None)


def model_points(kind: str, x: float, y: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    if kind == "block":
        pts = [(sx * BLOCK["x"] / 2, sy * BLOCK["y"] / 2, z) for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, BLOCK["h"])]
    else:  # lying along its heading, centreline one radius up
        r, h = PEG["d"] / 2, PEG["length"] / 2
        pts = [(sx * h, r * math.cos(a), r + r * math.sin(a)) for sx in (-1, 1) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
    return np.array([[x + c * u - s * v, y + s * u + c * v, TABLE_Z + z] for u, v, z in pts])


def outline(cam: Camera, kind: str, pose) -> np.ndarray:
    uv = cam.project(model_points(kind, *pose)).astype(np.float32)
    return cv2.convexHull(uv).reshape(-1, 2)


def fill(poly: np.ndarray, shape, off) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round((poly - off) * 8).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=3)
    return m.astype(bool)


def fit_one(cam: Camera, kind: str, mask: np.ndarray, dark: np.ndarray):
    """Best (x, y, yaw) and its IoU for one mask."""
    v, u = np.nonzero(mask)
    x0, y0, x1, y1 = u.min() - 40, v.min() - 40, u.max() + 40, v.max() + 40
    x0, y0 = max(0, x0), max(0, y0)
    M, D = mask[y0:y1, x0:x1], dark[y0:y1, x0:x1]
    off = np.array([x0, y0], float)
    # start: the mask's pixels on the plane of the piece's top, their middle and long axis
    top = TABLE_Z + (BLOCK["h"] if kind == "block" else PEG["d"])
    xy = cam.on_plane(np.stack([u, v], 1), top)[:, :2]
    cxy = np.median(xy, 0)
    w, vecs = np.linalg.eigh(np.cov((xy - xy.mean(0)).T))
    yaw0 = math.atan2(vecs[1, -1], vecs[0, -1])

    def cost(p):
        R = fill(outline(cam, kind, p), M.shape, off)
        inter = (R & M).sum()
        union = ((R & ~D) | M).sum()
        return 1.0 - inter / max(union, 1)

    best = None
    for dyaw in ((0.0, math.pi / 4) if kind == "block" else (0.0, 0.15, -0.15)):
        r = minimize(cost, [cxy[0], cxy[1], yaw0 + dyaw], method="Nelder-Mead",
                     options={"initial_simplex": np.array([[cxy[0], cxy[1], yaw0 + dyaw], [cxy[0] + 0.01, cxy[1], yaw0 + dyaw],
                                                           [cxy[0], cxy[1] + 0.01, yaw0 + dyaw], [cxy[0], cxy[1], yaw0 + dyaw + 0.25]]),
                              "xatol": 2e-4, "fatol": 1e-4, "maxiter": 400})
        if best is None or r.fun < best.fun:
            best = r
    x, y, yaw = best.x
    yaw = yaw % (math.pi / 2) if kind == "block" else yaw % math.pi
    return (float(x), float(y), float(yaw)), float(1 - best.fun)


AREA = ((0.05, 0.85), (-0.95, 0.3))  # where pieces can be on the table, left arm frame (x, y ranges)


def elongation(cam: Camera, mask: np.ndarray) -> float:
    """Long over short side of the mask's pixels met with the table plane: a block about 1-1.6, a lying stick 3-5."""
    v, u = np.nonzero(mask)
    xy = cam.on_plane(np.stack([u, v], 1), TABLE_Z + PEG["d"])[:, :2]
    (_, _), (w, h), _ = cv2.minAreaRect(xy.astype(np.float32))
    return max(w, h) / max(min(w, h), 1e-4)


def instances(sam: dict, hsv: np.ndarray, cam: Camera) -> list[dict]:
    """SAM 3 masks from all prompts, duplicates merged, on the table, named by hue."""
    cands = []
    for keys, kind in ((BLOCK_KEYS, "block"), (STICK_KEYS, "stick")):
        for k in keys:
            for m, s in zip(sam.get(f"masks_{k}", []), sam.get(f"scores_{k}", [])):
                if s >= MIN_SCORE and m.sum() >= MIN_PX:
                    cands.append({"mask": m, "score": float(s), "votes": {kind: 1}, "prompts": [k]})
    cands.sort(key=lambda c: -c["score"])
    kept = []
    for c in cands:
        for k in kept:
            inter = (k["mask"] & c["mask"]).sum()
            if inter / ((k["mask"] | c["mask"]).sum()) > MERGE_IOU:
                for kind, n in c["votes"].items():
                    k["votes"][kind] = k["votes"].get(kind, 0) + n
                k["prompts"] += c["prompts"]
                break
        else:
            kept.append(c)
    out = []
    for c in kept:
        v, u = np.nonzero(c["mask"])
        if np.median(v) < TABLE_TOP_ROW:
            continue  # the room beyond the table
        h, s = hsv[..., 0][c["mask"]], hsv[..., 1][c["mask"]]
        if np.median(s) < 50:
            continue  # grey: the arm, the power strip, the phone
        sat = s >= 50
        hue = np.median(np.where(h[sat] > 175, h[sat] - 180, h[sat]))  # red wraps around 0
        name = hue_name(hue % 180)
        if name is None:
            continue
        kinds = KINDS[name]
        # both kinds come in this colour: the shape on the table decides (the prompts' votes favour "block")
        kind = ("stick" if elongation(cam, c["mask"]) > 2.2 else "block") if len(kinds) > 1 else next(iter(kinds))
        xy = np.median(cam.on_plane(np.stack([u, v], 1), TABLE_Z)[:, :2], 0)
        if not (AREA[0][0] <= xy[0] <= AREA[0][1] and AREA[1][0] <= xy[1] <= AREA[1][1]):
            continue  # off the table: the room, a bottle
        out.append({**c, "colour": name, "kind": kind, "hue": float(hue % 180)})
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    calib = sys.argv[sys.argv.index("--calib") + 1] if "--calib" in sys.argv else str(ROOT / "calib" / "cam_mid.json")
    if calib in args:
        args.remove(calib)
    frames_dir, sam_dir = map(pathlib.Path, args[:2])
    c = json.load(open(calib))
    cam = Camera.from_calib(c)
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    folder = OUT / stamp
    folder.mkdir(parents=True, exist_ok=True)
    frames, gaps = [], []
    for npz in sorted(sam_dir.glob("*.npz")):
        name = npz.stem + ".jpg"
        src = next((p for p in (frames_dir / name, *frames_dir.rglob(name)) if p.exists()), None)
        if src is None:
            continue
        bgr = cv2.imread(str(src))
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        dark = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0) < 60
        sam = dict(np.load(npz))
        t0 = time.perf_counter()
        pieces = []
        for inst in instances(sam, hsv, cam):
            pose, iou = fit_one(cam, inst["kind"], inst["mask"], dark)
            z = TABLE_Z + (BLOCK["h"] / 2 if inst["kind"] == "block" else PEG["d"] / 2)
            pieces.append({"kind": inst["kind"], "colour": inst["colour"], "xyz": [pose[0], pose[1], z], "yaw": pose[2], "standing": False,
                           "iou": round(iou, 3), "sam_score": round(inst["score"], 3), "prompts": sorted(set(inst["prompts"])), "_mask": inst["mask"]})
        ms = (time.perf_counter() - t0) * 1000
        fast = track(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cam, TABLE_Z, ignore=IGNORE, table_row=TABLE_TOP_ROW)
        # the overlay: SAM 3 mask outline, the fitted CAD outline, the fast tracker's CAD outline
        img = bgr.copy()
        for p in pieces:
            cs, _ = cv2.findContours(p.pop("_mask").astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, GREEN, 1, cv2.LINE_AA)
            cv2.polylines(img, [np.round(outline(cam, p["kind"], (p["xyz"][0], p["xyz"][1], p["yaw"]))).astype(np.int32)], True, ORANGE, 2, cv2.LINE_AA)
            match = next((f for f in fast if f["colour"] == p["colour"] and f["kind"] == p["kind"]), None)
            if match and not match.get("standing"):
                cv2.polylines(img, [np.round(outline(cam, p["kind"], (match["xyz"][0], match["xyz"][1], match["yaw"]))).astype(np.int32)], True, BLUE, 1, cv2.LINE_AA)
                d = float(np.hypot(match["xyz"][0] - p["xyz"][0], match["xyz"][1] - p["xyz"][1]))
                p["tracker_gap_mm"] = round(d * 1000, 1)
                gaps.append((p["colour"], p["kind"], d * 1000))
            p["outline"] = np.round(outline(cam, p["kind"], (p["xyz"][0], p["xyz"][1], p["yaw"])), 1).tolist()
            uv = cam.project(np.array([p["xyz"]]))[0]
            label = f"{p['colour']} {p['kind']} {p['xyz'][0] * 100:.1f},{p['xyz'][1] * 100:.1f} cm  IoU {p['iou']:.2f}"
            org = (int(uv[0]) - 60, int(uv[1]) - 34)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(img, (org[0] - 3, org[1] - th - 4), (org[0] + tw + 3, org[1] + 4), (40, 170, 245), -1)
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(img, "green: SAM 3 mask   orange: fitted CAD   blue: fast colour tracker", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(folder / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(folder / f"raw-{name}"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])  # the page draws its own outlines on this
        joints = None
        pj = src.parent / "poses.json"
        if pj.exists():
            joints = next((r["measured"] for r in json.load(open(pj)) if r.get("frame") == name), None)
        frames.append({"file": name, "source": f"runs/vision/real/{stamp}/raw-{name}", "left_joints": joints, "ms": round(ms, 1), "pieces": pieces,
                       "tracker": [{k: v for k, v in f.items() if k != "box"} for f in fast]})
        print(f"{name}: " + ", ".join(f"{p['colour']} {p['kind']} at {p['xyz'][0] * 100:.1f},{p['xyz'][1] * 100:.1f} cm yaw {math.degrees(p['yaw']):.0f} IoU {p['iou']:.2f}"
                                      + (f" (tracker {p['tracker_gap_mm']:.0f} mm off)" if "tracker_gap_mm" in p else "") for p in pieces), flush=True)
    out = {"stamp": stamp, "source": "SAM 3 masks + CAD fit (vision/fit3d.py)", "frame": "left arm base (x forward, y left, z up; origin on the plate top)",
           "camera": {k: c[k] for k in ("pos", "R_cols_xyz", "fovy_deg", "width", "height", "k1", "err_px") if k in c},
           "bases": {"left": [0, 0, 0], "right": [0, -SPACING, 0]}, "table_z": TABLE_Z, "frames": frames}
    (folder / "tracks.json").write_text(json.dumps(out, indent=1))
    (OUT / "latest.json").write_text(json.dumps({"stamp": stamp}))
    if gaps:
        g = np.array([x[2] for x in gaps])
        print(f"fast tracker vs CAD fit: median {np.median(g):.1f} mm, worst {g.max():.1f} mm over {len(g)} pieces")
    print("->", folder)


if __name__ == "__main__":
    main()
