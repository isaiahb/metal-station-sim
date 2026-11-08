"""MuJoCo scene for the Dream-1 sim: one or two arms of a kind Dream knows, a desk, props, cameras.

Built the way the calibrated Maker Arm pick model is (robot-studio sim/pick/model.py, the physics
the PPO pick policies trained on and that picked 44 of 45 real cubes), for every arm:

- Only boxes collide: the desk, the props, one pad over each fingertip, a palm box under the hand,
  and the arm's own base. The arm meshes are drawn only.
- Pads as in sim/pick: 26 mm across, 8 mm thick, 64 mm long ending at the fingertip, centred
  3.35 mm off the jaw so the gap between them is the jaw reading minus 1.3 mm (the real Maker jaw
  read 40.5 mm on the 39.2 mm cube). Palm: 60 mm across, the open jaws' span along them, 24 mm
  deep, 8.9 cm behind the fingertips. On the Maker Arm these rules give sim/pick's boxes exactly.
- Contact: stiff (solref 0.008, solimp 0.95 0.99), friction 0.8 on pads and props, 0.5 on the
  desk, pyramidal cone, impratio 10: at MuJoCo's defaults a squeezed cube crept out and a pressed
  one sank into the desk (sim/pick check_hold.py).
- Motors: MIT position servos at each vendor's gains, clipped to the motor's torque; kd goes in as
  joint damping, which the Euler integrator treats implicitly (explicit kd chattered, sim/pick);
  armature and Coulomb friction per joint; gravity compensated, as the drivers' feedforward does.
- Jaws: one servo drives the first jaw, the second mirrors it through a stiff equality (soft, the
  left jaw ran 20 mm ahead and shoved the cube). The station drivers' holding law: the jaws move at
  the gripper motor's full torque, and while a pad touches something the squeeze is capped at the
  holding torque (0.5 N m at the motor, Dream's robots/makers_hold.py).
- Link masses from the vendors' URDFs (assets/dream/<arm>/inertials.json); the Maker Arm's scaled
  1.21x as sim/pick fitted to its gravity sweep.

What is measured and what is not: the Maker Arm's numbers are measured on the real arm. The Metal
Arm's gains are MakerMods' (Dream's robots/metal.py port of MakerMods Lab), its torques and masses
MakerMods' URDF; the YAM's gains and friction i2rt's (yam.yml). Neither has been checked against
a real arm here; armature on both is a guess of the Maker's order.
"""

from __future__ import annotations

import json
import math
import os
import pathlib

import mujoco
import numpy as np

ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets" / "dream"

RS00, RS02 = dict(t=14.0, arm=0.002), dict(t=17.0, arm=0.004)  # RobStride: torque N m, armature (sim/throw model.py)
DM4340, DM4310 = dict(t=27.0, arm=0.004), dict(t=7.0, arm=0.002)  # Damiao peak torque; armature a guess of the same order

# Per arm: Dream's joint names and flange; the tool point (offset in the flange, rows of the rotation
# whose columns are the TCP axes x jaw travel, y across, z approach, from dream_ik/specs.py); where an
# episode starts; Dream's rest posture (an IK reference); the gripper's opening at grip 1; the joints'
# (kp, kd, torque limit, armature, Coulomb friction); the gripper motor's (kp, kd, torque limit) and
# metres of each jaw per motor radian; the mass scale.
ARMS = {
    "makers": {
        "joints": [f"link_00{i}_joint" for i in range(2, 8)], "flange": "link_007",
        "tcp": ((0, 0, -0.1696387), ((0, 1, 0), (1, 0, 0), (0, 0, -1))),
        "start": (0, -1.2, 0.81, -0.1, 0, 0), "rest": (0, -1.2, 0.81, -0.1, 0, 0), "stroke": 0.08,
        # robot-studio driver VENDOR_GAINS, RobStride models per joint, frictionloss 0.15 (sim/pick)
        "motors": [(60, 4, RS00, 0.15), (150, 4.5, RS02, 0.15), (90, 3, RS02, 0.15), (30, 2, RS00, 0.15), (30, 2, RS00, 0.15), (30, 2, RS00, 0.15)],
        "grip_motor": (20.0, 0.5, 14.0), "jaw_per_rad": 0.0524125 / (0.0262 + 2.0961), "mass_scale": 1.21,
    },
    "metal": {
        "joints": [f"joint{i}" for i in range(1, 7)], "flange": "link6",
        "tcp": ((0.135, 0, 0), ((0, 0, 1), (1, 0, 0), (0, 1, 0))),
        "start": (0,) * 6, "rest": (0, -1.2, 1.5, 0, 0, 0), "stroke": 0.1,
        # MakerMods' follow gains (Dream's robots/metal.py); torque limits from MakerMods' URDF efforts; friction unmeasured, the Maker's
        "motors": [(100, 5, dict(t=30.0, arm=0.004), 0.15), (100, 5, dict(t=100.0, arm=0.004), 0.15), (100, 5, dict(t=30.0, arm=0.004), 0.15),
                   (50, 5, dict(t=10.0, arm=0.002), 0.15), (20, 0.6, dict(t=10.0, arm=0.002), 0.15), (20, 0.6, dict(t=10.0, arm=0.002), 0.15)],
        # gripper 20/0.6; 100 mm opening over its 137.5 deg of motor travel (Dream's robots/metal_gripper.py)
        "grip_motor": (20.0, 0.6, 20.0), "jaw_per_rad": 0.1 / 2 / math.radians(137.5), "mass_scale": 1.0,
    },
    "yam": {
        "joints": [f"joint{i}" for i in range(1, 7)], "flange": "link_6",
        "tcp": ((0, 0, 0.1347), ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
        "start": (0,) * 6, "rest": (0, 0.5, 0.5, -0.35, 0, 0), "stroke": 0.095,
        # i2rt yam.yml: kp, kd, DM4340 x3 then DM4310 x3, coulomb_friction
        "motors": [(80, 5, DM4340, 0.3), (80, 5, DM4340, 0.3), (80, 5, DM4340, 0.3), (40, 1.5, DM4310, 0.06), (10, 1.5, DM4310, 0.06), (10, 1.5, DM4310, 0.06)],
        # i2rt linear_4310 motor_kp 20; its jaw gearing is not in Dream's files: the Metal's ratio, assumed
        "grip_motor": (20.0, 0.5, 7.0), "jaw_per_rad": 0.1 / 2 / math.radians(137.5), "mass_scale": 1.0,
    },
}
HOLD_TORQUE_NM = 0.5  # the station drivers' default gripper holding torque (robots/makers_hold.py)

COLORS = [(0.56, 0.65, 0.75), (0.91, 0.66, 0.56), (0.66, 0.80, 0.69), (0.89, 0.79, 0.54), (0.73, 0.66, 0.86), (0.61, 0.76, 0.89), (0.42, 0.42, 0.44)]

HAND, PROP, STATIC = 2, 4, 8  # collision groups: props touch everything; hand boxes, desk and base touch props; hand touches the base
CONTACT = dict(solref=[0.008, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2])
PAD = dict(across=0.026, thick=0.008, length=0.064, inset=0.00065)  # sim/pick: pad gap = jaw reading - 1.3 mm
PALM = dict(across=0.06, deep=0.024, behind_tip=0.0894)
CUBE_EDGE, CUBE_MASS = 0.0392, 0.018  # the real tag cube C (sim/pick/model.py)
WRIST_CAM = {"pos": (0, 0.055, -0.07), "tilt": -0.45, "fovy": 78}


def quat_of(rot: np.ndarray) -> list[float]:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(rot, dtype=float).reshape(9))
    return q.tolist()


def _mesh_points(m, d, g) -> np.ndarray:
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


def _measure(kind: str) -> dict:
    """The hand's and base's collision boxes, by sim/pick's rules, from the CAD with the jaws shut."""
    m = _arm_spec(kind, boxes=None).compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    Rt, pt = d.site_xmat[sid].reshape(3, 3), d.site_xpos[sid]
    pads, tips = [], []
    for i in range(2):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"jaw{i}")
        g = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == b)
        t = (_mesh_points(m, d, g) - pt) @ Rt  # tool frame: x jaw travel, y across, z approach
        tip = t[:, 2].max()
        near = t[t[:, 2] > tip - 0.02]
        side = 1.0 if near[:, 0].mean() > 0 else -1.0
        across = PAD["across"]  # sim/pick's 26 mm, wider than the CAD finger near its tip
        centre = np.array([side * (PAD["thick"] / 2 - PAD["inset"]), (near[:, 1].max() + near[:, 1].min()) / 2, tip - PAD["length"] / 2])
        Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
        pads.append({"pos": (Rb.T @ (pt + Rt @ centre - pb)).tolist(), "quat": quat_of(Rb.T @ Rt),
                     "size": [PAD["thick"] / 2, across / 2, PAD["length"] / 2]})
        tips.append(tip)
    fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ARMS[kind]["flange"])
    Rf, pf = d.xmat[fl].reshape(3, 3), d.xpos[fl]
    palm_tool = np.array([0.0, 0.0, max(tips) - PALM["behind_tip"]])
    palm = {"pos": (Rf.T @ (pt + Rt @ palm_tool - pf)).tolist(), "quat": quat_of(Rf.T @ Rt),
            "size": [ARMS[kind]["stroke"] / 2 - 0.005, PALM["across"] / 2, PALM["deep"] / 2]}
    # the base: what is welded to the world, as one box, and a column over the pan axis up to the
    # top of the first moving link (sim/pick: the clamp plate and a box over base and shoulder)
    static, first = [], []
    first_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, [m.body(b).name for b in range(m.nbody) if m.body_jntnum[b] > 0][0])
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if m.body_weldid[m.geom_bodyid[g]] == 0:
            static.append(_mesh_points(m, d, g))
        elif m.geom_bodyid[g] == first_body:
            first.append(_mesh_points(m, d, g))
    s, f = np.concatenate(static), np.concatenate(first)
    lo, hi = s.min(0), s.max(0)
    r = max(np.abs(f[:, :2]).max(), 0.03)
    base = [{"pos": ((lo + hi) / 2).tolist(), "size": ((hi - lo) / 2).tolist()},
            {"pos": [0.0, 0.0, (lo[2] + f[:, 2].max()) / 2], "size": [r, r, (f[:, 2].max() - lo[2]) / 2]}]
    return {"pads": pads, "palm": palm, "base": base, "desk_z": float(lo[2])}


_BOXES: dict[str, dict] = {}


def boxes(kind: str) -> dict:
    if kind not in _BOXES:
        _BOXES[kind] = _measure(kind)
    return _BOXES[kind]


def _arm_spec(kind: str, boxes: dict | None | bool = True, draw: bool = True) -> mujoco.MjSpec:
    """One arm as its own spec: Dream's IK URDF, vendor masses, the maker's meshes (drawn), servos,
    sliding jaws, collision boxes, tool site, wrist camera. draw=False leaves the meshes out (training)."""
    if boxes is True:
        boxes = globals()["boxes"](kind)
    A = ARMS[kind]
    folder = ASSETS / kind
    spec = mujoco.MjSpec.from_file(str(folder / "ik.urdf"))
    spec.meshdir = str(folder)
    spec.compiler.discardvisual = False  # the URDF importer turns this on; the drawn-only meshes need it off
    for geom in list(spec.geoms):  # any visuals the URDF carried; the meshes come from view.json
        spec.delete(geom)
    for old in list(spec.meshes):
        spec.delete(old)
    view = json.loads((folder / "view.json").read_text())
    inert = json.loads((folder / "inertials.json").read_text())
    bodies = {b.name: b for b in spec.bodies}
    for b in bodies.values():
        if b.name == "world":
            continue
        link = inert["links"].get(b.name)
        b.explicitinertial = True
        if link:
            b.mass = (link["mass"] + inert["add"].get(b.name, 0.0)) * A["mass_scale"]
            b.ipos = link["com"]
            b.inertia = [max(v, 1e-6) * A["mass_scale"] for v in link["diag"]]
            b.fullinertia = [math.nan] * 6
        elif not (b.mass > 0):
            b.mass, b.inertia = 0.01, [1e-6, 1e-6, 1e-6]
        b.gravcomp = 1.0
    for name, (kp, kd, motor, fric) in zip(A["joints"], A["motors"]):
        j = spec.joint(name)
        j.armature = motor["arm"]
        j.frictionloss = fric
        j.damping = [kd, 0, 0]  # implicit under Euler
        spec.add_actuator(name=f"q_{name}", target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[kp] + [0] * 9,
                          biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -kp, 0] + [0] * 7,
                          forcelimited=True, forcerange=[-motor["t"], motor["t"]])
    links = [n for n in bodies if n != "world"]
    seen = set()

    def mesh(name):
        if name not in seen:
            spec.add_mesh(name=f"{kind}_{name}", file=name)
            seen.add(name)
        return f"{kind}_{name}"

    for part in view["parts"] if draw else []:
        p = part["pose"]
        k = links.index(part["link"]) if part["link"] in links else 0
        bodies[part["link"]].add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh(part["mesh"]), pos=p[4:7], quat=p[0:4],
                                      rgba=[*COLORS[min(k, len(COLORS) - 1)], 1], contype=0, conaffinity=0, group=1)
    r = A["jaw_per_rad"]
    kp_m, kd_m, t_m = A["grip_motor"]
    for i, jaw in enumerate(view["jaws"]):
        p = jaw["pose"]
        body = bodies[jaw["link"]].add_body(name=f"jaw{i}", explicitinertial=True, mass=inert["jaw_mass"], inertia=[2e-5] * 3, gravcomp=1.0)
        sign = math.copysign(1.0, jaw["travel"])
        jj = body.add_joint(name=f"jaw{i}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[sign * a for a in jaw["axis"]],
                            range=[0, A["stroke"] / 2], armature=1e-4 / r ** 2 / 2)
        jj.damping = [kd_m / r ** 2 if i == 0 else 0.5 / r ** 2 * 0.02, 0, 0]  # the motor's kd on the driven jaw, implicit
        if draw or boxes is None:  # _measure needs the finger meshes
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh(jaw["mesh"]), pos=p[4:7], quat=p[0:4],
                          rgba=[0.18, 0.18, 0.2, 1], contype=0, conaffinity=0, group=1)
        if boxes is not None:
            pad = boxes["pads"][i]
            body.add_geom(name=f"pad{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=pad["size"], pos=pad["pos"], quat=pad["quat"],
                          rgba=[0.9, 0.4, 0.9, 0], contype=HAND, conaffinity=STATIC, friction=[0.8, 0.005, 0.0001], condim=4, group=3, **CONTACT)
    # one servo on the first jaw at the gripper motor's stiffness and torque; Scene.step caps it at the
    # holding torque while the jaws are stalled on something (the drivers' holding law)
    kp_lin = kp_m / r ** 2
    spec.add_actuator(name="jaw", target="jaw0", trntype=mujoco.mjtTrn.mjTRN_JOINT, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                      gainprm=[kp_lin] + [0] * 9, biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -kp_lin, 0] + [0] * 7,
                      forcelimited=True, forcerange=[-t_m / r, t_m / r], ctrlrange=[0, A["stroke"] / 2])
    spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="jaw1", name2="jaw0", data=[0, 1] + [0] * 10, solref=[0.008, 1])
    offset, rows = A["tcp"]
    R = np.array(rows, dtype=float)  # columns are the TCP axes in the flange
    flange = bodies[A["flange"]]
    flange.add_site(name="tcp", pos=list(offset), quat=quat_of(R), size=[0.004, 0, 0], rgba=[1, 0.4, 0.2, 0.6], group=4)
    if boxes is not None:
        palm = boxes["palm"]
        flange.add_geom(name="palm", type=mujoco.mjtGeom.mjGEOM_BOX, size=palm["size"], pos=palm["pos"], quat=palm["quat"],
                        rgba=[0.9, 0.4, 0.9, 0], contype=HAND, conaffinity=STATIC, friction=[0.5, 0.005, 0.0001], group=3, **CONTACT)
    tilt = WRIST_CAM["tilt"]
    Rx = np.array([[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)]])
    cam_in_tcp = np.diag([-1.0, 1.0, -1.0]) @ Rx
    flange.add_camera(name="wrist", pos=(np.array(offset) + R @ np.array(WRIST_CAM["pos"])).tolist(), quat=quat_of(R @ cam_in_tcp), fovy=WRIST_CAM["fovy"])
    return spec


# The hackathon's foam pieces (sizes in sim/measured.json, from calipers on Nov 7): the green block
# 50.7 x 49.9 mm and 34.5 mm tall, a 23.1 mm hole straight through; the purple peg 22.65 mm across
# and 100.5 mm long.
# The hole is a 23.1 mm square opening (four boxes), the peg a capsule: MJX has no cylinder-box
# contact. Square against round, the peg has the real 0.2 mm a side on four faces. Foam is softer than
# the printed cubes (solref 0.015) and grippier against itself; masses are estimates (EVA foam
# about 0.1 g/cm3) until the pieces are weighed.
MEASURED = json.loads((pathlib.Path(__file__).resolve().parent / "measured.json").read_text())
_mv = lambda group, key: MEASURED[group][key]["value"]  # noqa: E731
BLOCK = dict(x=_mv("block", "x_m"), y=_mv("block", "y_m"), h=_mv("block", "h_m"), hole=_mv("block", "hole_d_m"),
             hole_dx=_mv("block", "hole_offset_x_m"), hole_dy=_mv("block", "hole_offset_y_m"), mass=_mv("block", "mass_kg"))
PEG = dict(d=_mv("peg", "d_m"), length=_mv("peg", "length_m"), mass=_mv("peg", "mass_kg"))
BASE_SPACING = _mv("station", "base_spacing_m")
FOAM = dict(solref=[0.015, 1], solimp=[0.9, 0.97, 0.001, 0.5, 2])


# BLOCK_HOLE=poly12 (environment): the hole's contact walls as a 12-sided ring of thin boxes (inscribed radius the
# hole's), not the four slabs' square alone. Off by default (the PPO runs train on the square hole).
BLOCK_HOLE = os.environ.get("BLOCK_HOLE", "square")


def block_mesh(segments: int = 48):
    """The foam block as drawn: the 50.7 x 49.9 x 34.5 mm box with the round 23.1 mm hole straight through along its
    height, centred on the hole (vertices, triangles; outward faces)."""
    hx, hy, hz, r = BLOCK["x"] / 2, BLOCK["y"] / 2, BLOCK["h"] / 2, BLOCK["hole"] / 2
    ox, oy = BLOCK["hole_dx"], BLOCK["hole_dy"]
    x0, x1, y0, y1 = -hx - ox, hx - ox, -hy - oy, hy - oy  # the box's sides around the hole's centre

    def ray(th):  # where the ray from the hole's centre at angle th leaves the box
        c, s_ = math.cos(th), math.sin(th)
        t = min((x1 if c > 0 else x0) / c if abs(c) > 1e-12 else 1e9, (y1 if s_ > 0 else y0) / s_ if abs(s_) > 1e-12 else 1e9)
        return t * c, t * s_

    corners = [(x1, y1), (x0, y1), (x0, y0), (x1, y0)]
    ca = [math.atan2(y, x) % (2 * math.pi) for x, y in corners]
    verts, faces = [], []

    def v(x, y, z):
        verts.append((x, y, z))
        return len(verts) - 1

    th = [2 * math.pi * k / segments for k in range(segments)]
    for z, up in ((hz, True), (-hz, False)):  # the top and bottom faces: the ring between hole and box edge
        inner = [v(r * math.cos(t), r * math.sin(t), z) for t in th]
        outer = [v(*ray(t), z) for t in th]
        for k in range(segments):
            k2 = (k + 1) % segments
            a0, a1 = th[k], th[k] + 2 * math.pi / segments
            between = [i for i, a in enumerate(ca) if a0 < a < a1 or a0 < a + 2 * math.pi < a1]
            poly = [outer[k]] + [v(*corners[i], z) for i in between] + [outer[k2]]
            tris = [(inner[k], inner[k2], poly[-1])] + [(inner[k], poly[j + 1], poly[j]) for j in range(len(poly) - 1)]
            for t3 in tris:
                faces.append(t3[::-1] if up else t3)
    for (xa, ya), (xb, yb) in zip(corners, corners[1:] + corners[:1]):  # the four sides
        a, b, c, d = v(xa, ya, -hz), v(xb, yb, -hz), v(xb, yb, hz), v(xa, ya, hz)
        faces += [(a, b, c), (a, c, d)]
    for k in range(segments):  # the hole's wall, facing in
        t0, t1 = th[k], th[(k + 1) % segments]
        a, b = v(r * math.cos(t0), r * math.sin(t0), -hz), v(r * math.cos(t1), r * math.sin(t1), -hz)
        c, d = v(r * math.cos(t1), r * math.sin(t1), hz), v(r * math.cos(t0), r * math.sin(t0), hz)
        faces += [(a, c, b), (a, d, c)]
    return np.array(verts, float), np.array(faces, int)


def _foam_block(body, rgba) -> None:
    """Contact: four slabs around a square opening (plus, with BLOCK_HOLE=poly12, a 12-sided ring of thin boxes
    rounding the opening to the hole's radius); not drawn. Drawn: the block's own mesh with its round hole (contact
    off). The body's origin is the hole's centre, so a plugged peg shares the block's xy (the hole may sit off the
    block's centre: hole_offset_*)."""
    hx, hy, hz, a = BLOCK["x"] / 2, BLOCK["y"] / 2, BLOCK["h"] / 2, BLOCK["hole"] / 2
    ox, oy = BLOCK["hole_dx"], BLOCK["hole_dy"]  # the hole's centre from the block's
    common = dict(contype=PROP, conaffinity=HAND | PROP | STATIC, friction=[1.0, 0.005, 0.0001], condim=4, rgba=[*rgba[:3], 0], group=3, **FOAM)
    m = BLOCK["mass"]
    ring = BLOCK_HOLE == "poly12"
    ms = 0.85 if ring else 1.0  # the ring takes the rest of the mass
    xp, xn = (hx - ox - a) / 2, (hx + ox - a) / 2  # slab half widths on each side along x
    body.add_geom(name=f"{body.name}_xp", type=mujoco.mjtGeom.mjGEOM_BOX, size=[xp, hy, hz], pos=[a + xp, -oy, 0], mass=m * 0.3 * ms, **common)
    body.add_geom(name=f"{body.name}_xn", type=mujoco.mjtGeom.mjGEOM_BOX, size=[xn, hy, hz], pos=[-a - xn, -oy, 0], mass=m * 0.3 * ms, **common)
    yp, yn = (hy - oy - a) / 2, (hy + oy - a) / 2  # between them, in front of and behind the hole
    body.add_geom(name=f"{body.name}_yp", type=mujoco.mjtGeom.mjGEOM_BOX, size=[a, yp, hz], pos=[0, a + yp, 0], mass=m * 0.2 * ms, **common)
    body.add_geom(name=f"{body.name}_yn", type=mujoco.mjtGeom.mjGEOM_BOX, size=[a, yn, hz], pos=[0, -a - yn, 0], mass=m * 0.2 * ms, **common)
    if ring:  # 12 walls, inner faces at the hole's radius, 2 mm thick, each one side of the inscribed 12-gon
        t = 0.001
        half = a * math.tan(math.pi / 12)
        for k in range(12):
            th = k * math.pi / 6
            body.add_geom(name=f"{body.name}_r{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[t, half, hz],
                          pos=[(a + t) * math.cos(th), (a + t) * math.sin(th), 0], quat=[math.cos(th / 2), 0, 0, math.sin(th / 2)],
                          mass=m * 0.15 / 12, **common)
    body.add_geom(name=f"{body.name}_draw", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=BLOCK_MESH, rgba=rgba, contype=0, conaffinity=0,
                  density=0, group=1)


BLOCK_MESH = "foam_block_mesh"  # added to the spec by Scene._build when a block is in the scene


def _prop(world, i: int, kind: str, at, yaw: float = 0.0) -> None:
    upright = kind.endswith(":up")
    body = world.add_body(name=f"prop{i}", pos=[at[0], at[1], at[2]], quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    body.add_freejoint(name=f"prop{i}")
    common = dict(contype=PROP, conaffinity=HAND | PROP | STATIC, friction=[0.8, 0.005, 0.0001], condim=4, **CONTACT)
    if kind.startswith("block"):
        _foam_block(body, [0.62, 0.8, 0.2, 1])
        return
    if kind.startswith("peg"):
        r, half = PEG["d"] / 2, PEG["length"] / 2 - PEG["d"] / 2
        quat = [1, 0, 0, 0] if upright else [0.7071068, 0.7071068, 0, 0]  # standing, or lying along y
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[r, half], quat=quat, mass=PEG["mass"], rgba=[0.78, 0.4, 0.75, 1],
                      contype=PROP, conaffinity=HAND | PROP | STATIC, friction=[1.0, 0.005, 0.0001], condim=4, **FOAM)
        # the real stick's ends are flat: a 16 mm square foot at each (inside the 23.1 mm hole's
        # opening), or the capsule's round end will not stand
        for sgn in (1, -1):
            at_end = [0, 0, sgn * (PEG["length"] / 2 - 0.001)] if upright else [0, sgn * (PEG["length"] / 2 - 0.001), 0]
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.008, 0.008, 0.001] if upright else [0.008, 0.001, 0.008], pos=at_end,
                          mass=0.0005, rgba=[0.78, 0.4, 0.75, 1], contype=PROP, conaffinity=HAND | PROP | STATIC,
                          friction=[1.0, 0.005, 0.0001], condim=4, **FOAM)
        return
    if kind.startswith("cube"):
        hexc = kind.split(":")[1] if ":" in kind else "#d9473b"
        rgb = [int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_EDGE / 2] * 3, mass=CUBE_MASS, rgba=[*rgb, 1], **common)
    elif kind == "duck":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.03, 0.025, 0.02], mass=0.02, rgba=[0.95, 0.76, 0.19, 1], **common)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.014], pos=[0.012, 0, 0.024], mass=0.005, rgba=[0.95, 0.76, 0.19, 1], **common)
    else:  # marker, lying across the desk
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.008, 0.057], quat=[0.7071068, 0.7071068, 0, 0], mass=0.012, rgba=[0.91, 0.45, 0.16, 1], **common)


PROP_HALF_HEIGHT = {"cube": CUBE_EDGE / 2, "duck": 0.025, "marker": 0.008, "block": BLOCK["h"] / 2, "peg": PEG["d"] / 2}


def prop_half_height(kind: str) -> float:
    """How far a prop's centre sits above whatever it rests on."""
    if kind.startswith("peg") and kind.endswith(":up"):
        return PEG["length"] / 2
    return PROP_HALF_HEIGHT[kind.split(":")[0]]


class Scene:
    """A compiled scene and the handles the loop needs: arm joint addresses, jaw servos, cameras, props."""

    def __init__(self, arm: str = "metal", sides: str = "both", spacing: float = BASE_SPACING, props=None, cam: str = "near", draw: bool = True):
        self.kind, self.sides, self.spacing, self.draw = arm, sides, spacing, draw
        self.side_names = ["right", "left"] if sides == "both" else ["right"]
        props = props if props is not None else [{"kind": "cube:#d9473b", "at": [0.27, spacing / 2 if sides == "both" else 0.04]}]
        self.desk_z = boxes(arm)["desk_z"]  # the base's bottom
        self.model = self._build(props, cam).compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.arm_qadr, self.arm_dadr, self.arm_act, self.jaw_act, self.jaw_qadr = {}, {}, {}, {}, {}
        for side in self.side_names:
            joints = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{n}") for n in self.joint_names]
            self.arm_qadr[side] = [m.jnt_qposadr[j] for j in joints]
            self.arm_dadr[side] = [m.jnt_dofadr[j] for j in joints]
            self.arm_act[side] = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_q_{n}") for n in self.joint_names]
            self.jaw_act[side] = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_jaw")]
            self.jaw_qadr[side] = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_jaw{i}")] for i in range(2)]
            self.pad_geoms = getattr(self, "pad_geoms", {})
            self.pad_geoms[side] = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_pad{i}") for i in range(2)}
        r, t_m = ARMS[arm]["jaw_per_rad"], ARMS[arm]["grip_motor"][2]
        self.jaw_force = (t_m / r, HOLD_TORQUE_NM / r)  # moving, holding
        self.props = [{"kind": p["kind"], "body": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"prop{i}")} for i, p in enumerate(props)]
        self.reset()

    @property
    def joint_names(self) -> list[str]:
        return ARMS[self.kind]["joints"]

    def _build(self, props, cam: str) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.option.timestep = 0.002
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
        spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        spec.option.impratio = 10  # with CONTACT: a held cube does not creep (sim/pick check_hold.py)
        spec.visual.global_.offwidth = 640
        spec.visual.global_.offheight = 480
        w = spec.worldbody
        z = self.desk_z
        mid = self.spacing / 2 if self.sides == "both" else 0.0
        w.add_light(pos=[0.3, mid, z + 1.5], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8], castshadow=True)
        w.add_light(pos=[-0.5, mid - 0.8, z + 1.2], dir=[0.5, 0.5, -1], diffuse=[0.4, 0.4, 0.4], castshadow=False)
        width = self.spacing + 0.9 if self.sides == "both" else 0.9
        w.add_geom(name="desk", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5, width / 2, 0.015], pos=[0.3, mid, z - 0.015],
                   rgba=[0.23, 0.21, 0.2, 1], contype=STATIC, conaffinity=0, friction=[0.5, 0.005, 0.0001], **CONTACT)
        w.add_geom(name="paper", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.15, 0.21, 0.0005], pos=[0.3, mid, z + 0.0005],
                   rgba=[0.95, 0.94, 0.92, 1], contype=0, conaffinity=0)
        w.add_geom(name="wall", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 1.5, 1.0], pos=[1.0, mid, z + 0.6], rgba=[0.73, 0.76, 0.8, 1], contype=0, conaffinity=0)
        w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], pos=[0.3, mid, z - 0.75], rgba=[0.65, 0.55, 0.43, 1], contype=0, conaffinity=0)
        for side in self.side_names:
            y = self.spacing if side == "left" else 0.0
            for k, b in enumerate(boxes(self.kind)["base"]):
                w.add_geom(name=f"{side}_base{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=b["size"], pos=[b["pos"][0], b["pos"][1] + y, b["pos"][2]],
                           rgba=[0.9, 0.4, 0.9, 0], contype=STATIC, conaffinity=0, group=3, **CONTACT)
            frame = w.add_frame(pos=[0, y, 0])
            frame.attach_body(_arm_spec(self.kind, draw=self.draw).worldbody.first_body(), f"{side}_", "")
        if any(p["kind"].startswith("block") for p in props):
            vv, ff = block_mesh()
            spec.add_mesh(name=BLOCK_MESH, uservert=vv.reshape(-1).tolist(), userface=ff.reshape(-1).tolist())
        for i, p in enumerate(props):
            _prop(w, i, p["kind"], [p["at"][0], p["at"][1], z + prop_half_height(p["kind"]) + p.get("lift", 0.0)], p.get("yaw", 0.0))
        near = cam == "near"
        if self.sides == "both":
            pos, look = [-0.05 if near else -0.15, mid, z + (0.5 if near else 0.65)], [0.32, mid, z - 0.03]
        else:
            pos, look = [0.0 if near else -0.05, 0.2 if near else 0.25, z + (0.43 if near else 0.58)], [0.28, 0.04, z - 0.03]
        f = np.subtract(look, pos)
        f /= np.linalg.norm(f)
        right = np.cross(f, [0, 0, 1])
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        w.add_camera(name="overhead", pos=pos, quat=quat_of(np.column_stack([right, up, -f])), fovy=62)
        return spec

    def step(self, n: int = 1) -> None:
        """n physics steps, with the drivers' holding law on each gripper."""
        m, d = self.model, self.data
        move, hold = self.jaw_force
        for _ in range(n):
            touching = {g for c in d.contact[: d.ncon] for g in (c.geom1, c.geom2)}
            for side in self.side_names:
                lim = hold if touching & self.pad_geoms[side] else move  # a pad on something: holding
                m.actuator_forcerange[self.jaw_act[side][0]] = [-lim, lim]
            mujoco.mj_step(m, d)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        for side in self.side_names:
            self.set_arm(side, ARMS[self.kind]["start"], 1.0, snap=True)
        mujoco.mj_forward(self.model, self.data)

    def set_arm(self, side: str, q, grip: float, dt: float | None = None, snap: bool = False) -> None:
        """Arm servos toward q, the jaw servo toward the grip's opening; snap puts the joints there now."""
        d = self.data
        for k, (qa, da, act) in enumerate(zip(self.arm_qadr[side], self.arm_dadr[side], self.arm_act[side])):
            d.ctrl[act] = q[k]
            if snap:
                d.qpos[qa], d.qvel[da] = q[k], 0.0
        target = max(0.0, min(1.0, grip)) * ARMS[self.kind]["stroke"] / 2  # each jaw's travel
        for a in self.jaw_act[side]:
            d.ctrl[a] = target
        if snap:
            for qa in self.jaw_qadr[side]:
                d.qpos[qa] = target

    def arm_state(self, side: str) -> dict:
        d = self.data
        opening = sum(d.qpos[qa] for qa in self.jaw_qadr[side])
        return {"side": side, "q": [float(d.qpos[qa]) for qa in self.arm_qadr[side]], "grip": float(opening / ARMS[self.kind]["stroke"])}

    def prop_states(self) -> list[dict]:
        d = self.data
        return [{"kind": p["kind"], "pos": d.xpos[p["body"]].tolist(), "quat": d.xquat[p["body"]].tolist()} for p in self.props]

    def contact_boxes(self) -> list[dict]:
        """The collision boxes of the hands and bases as placed now: centre, rotation, half sizes, in the world."""
        m, d = self.model, self.data
        out = []
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name.endswith(("pad0", "pad1", "palm", "base0", "base1")):
                out.append({"name": name, "pos": d.geom_xpos[g].tolist(), "mat": d.geom_xmat[g].tolist(), "size": m.geom_size[g].tolist()})
        return out

    def tcp(self, side: str) -> np.ndarray:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")
        return self.data.site_xpos[sid].copy()
