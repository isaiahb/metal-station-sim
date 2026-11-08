"""Runs vision/locate.track on real overhead frames: block and peg poses in the left arm's frame.

    python -m vision.track_real runs/station/calib/poses-1/pose03.jpg [more frames or a folder] [--calib calib/cam_mid.json]

Writes runs/vision/real/<stamp>/: each frame with the found pieces drawn on it (their outlines projected
back through the calibrated camera), and tracks.json (camera, arm bases, pieces per frame) for the
vision page's 3D view. Frames are 1280x720 from the station's overhead camera.
"""

import json
import math
import pathlib
import sys
import time

import cv2
import numpy as np

from calib.fit_overhead import SPACING, TABLE_TOP_ROW
from vision.locate import BLOCK, PEG, Camera, track

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "vision" / "real"
TABLE_Z = -0.008  # the table is flat and the 8 mm base plate lies flush on it (MakerMods station drawing); the arm frame starts on the plate top
# not pieces: skin and the laptop's stickers in this station's view (full-resolution boxes)
IGNORE = [(0, 440, 200, 640), (1140, 330, 1280, 480)]
INK = {"green": (60, 200, 60), "blue": (230, 120, 30), "purple": (200, 70, 150), "red": (40, 40, 230)}


def outline(cam: Camera, piece: dict) -> list[np.ndarray]:
    x, y, z = piece["xyz"]
    if piece["kind"] == "block":
        c, s = math.cos(piece["yaw"]), math.sin(piece["yaw"])
        sq = [(BLOCK["x"] / 2 * a, BLOCK["y"] / 2 * b) for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        rings = []
        for zz in (z - BLOCK["h"] / 2, z + BLOCK["h"] / 2):
            rings.append(np.array([[x + c * u - s * v, y + s * u + c * v, zz] for u, v in sq]))
        return [cam.project(r) for r in rings]
    if piece["standing"]:
        ring = lambda zz: np.array([[x + PEG["d"] / 2 * math.cos(a), y + PEG["d"] / 2 * math.sin(a), zz] for a in np.linspace(0, 2 * math.pi, 20)])  # noqa: E731
        return [cam.project(ring(z + PEG["length"] / 2)), cam.project(np.array([[x, y, z - PEG["length"] / 2], [x, y, z + PEG["length"] / 2]]))]
    h = PEG["length"] / 2
    a = piece["yaw"]
    ends = np.array([[x - h * math.cos(a), y - h * math.sin(a), z], [x + h * math.cos(a), y + h * math.sin(a), z]])
    return [cam.project(ends)]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    calib = sys.argv[sys.argv.index("--calib") + 1] if "--calib" in sys.argv else str(ROOT / "calib" / "cam_mid.json")
    if calib in args:
        args.remove(calib)
    c = json.load(open(calib))
    cam = Camera.from_calib(c)
    files = []
    for a in args:
        p = pathlib.Path(a)
        files += sorted(p.glob("pose*.jpg")) + sorted(p.glob("overhead-*.jpg")) if p.is_dir() else [p]
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    folder = OUT / stamp
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for f in files:
        bgr = cv2.imread(str(f))
        t0 = time.perf_counter()
        pieces = track(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cam, TABLE_Z, ignore=IGNORE, table_row=TABLE_TOP_ROW)
        ms = (time.perf_counter() - t0) * 1000
        for p in pieces:
            col = INK[p["colour"]]
            for line in outline(cam, p):
                cv2.polylines(bgr, [line.round().astype(np.int32)], p["kind"] == "block" or p.get("standing", False), col, 2, cv2.LINE_AA)
            x0, y0 = p["box"][:2]
            label = f"{p['colour']} {p['kind']}{' in' if p.get('standing') else ''} {p['xyz'][0] * 100:.0f},{p['xyz'][1] * 100:.0f} cm"
            cv2.putText(bgr, label, (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(bgr, label, (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        from vision.fit3d import outline as cad_outline
        for p in pieces:
            if not p.get("standing"):
                p["outline"] = np.round(cad_outline(cam, p["kind"], (p["xyz"][0], p["xyz"][1], p["yaw"])), 1).tolist()
        name = f.stem + ".jpg"
        cv2.imwrite(str(folder / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        # the left arm's measured joints, when the frame is one of a station/calib_poses.py run
        joints = None
        pj = f.parent / "poses.json"
        if pj.exists():
            joints = next((r["measured"] for r in json.load(open(pj)) if r.get("frame") == f.name), None)
        frames.append({"file": name, "left_joints": joints, "source": str(f.relative_to(ROOT) if f.is_absolute() and ROOT in f.parents else f), "ms": round(ms, 1), "pieces": pieces})
        print(f"{f.name}: {len(pieces)} pieces in {ms:.0f} ms: " + ", ".join(f"{p['colour']} {p['kind']}{'(in)' if p.get('standing') else ''} at {np.round(np.array(p['xyz']) * 100, 1).tolist()} cm" for p in pieces))
    out = {"stamp": stamp, "frame": "left arm base (x forward, y left, z up; origin on the plate top)",
           "camera": {k: c[k] for k in ("pos", "R_cols_xyz", "fovy_deg", "width", "height", "k1", "err_px") if k in c},
           "bases": {"left": [0, 0, 0], "right": [0, -SPACING, 0]}, "table_z": TABLE_Z, "frames": frames}
    (folder / "tracks.json").write_text(json.dumps(out, indent=1))
    (OUT / "latest.json").write_text(json.dumps({"stamp": stamp}))
    print("->", folder)


if __name__ == "__main__":
    main()
