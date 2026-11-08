# Metal station sim

An offline stand-in for our hackathon station (a MakerMods Metal Arm Station: two 6-DoF Metal arms, an overhead
camera, and a wrist camera on the left arm). It serves the same HTTP API as the left arm's server, so a script that
drives the real arm drives the sim by changing one URL. Use it to test motion, planning and perception scripts
without the arm.

Everything is in the arm's model frame: x forward, y left, z up, metres, the base plate's top at z = 0, the table
top at z = -0.008. The sim's arm is our left arm (it is called `right` inside the MuJoCo scene).

## Quick start for teammates

You need Python 3.12. Everything runs on a laptop CPU; no GPU, no station access.

```bash
git clone https://github.com/isaiahb/metal-station-sim.git
cd metal-station-sim
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Start the sim station (leave it running; it serves the arm API on port 8790):

```bash
.venv/bin/python -m calib.station_sim --port 8790
```

In a second terminal, run the worked example (a wrist look, then pick a block and stack it):

```bash
.venv/bin/python -m examples.pick_and_stack --url http://127.0.0.1:8790
```

Then point your own script at the same URL. The API:

| Call | What it does |
| --- | --- |
| `GET /state` | joints `q` (6, rad), `grip` (0 closed .. 1 open), `moving`, `goal` |
| `GET /wrist.jpg`, `GET /overhead.jpg` | camera frames; the `X-Joints` header gives the joints at that frame |
| `POST /move {"q": [6], "grip": g, "rate": r}` | move to joints `q` at up to `r` rad/s |
| `POST /stop` | hold where it is |

Before anything you write goes to the real arm:

1. Run it here first and make sure every move passes `calib/tower_safety.py` (`check_path`, `check_pose`); the
   larger scripts already do.
2. The real arm stops for a person in the work area, a gate (someone else assigned the arm), or a safety refusal.
   Plan for that: a script must never leave the arm low over a block, and its first move must be checked like
   every other move.
3. Keep moves on the left side of the table; the other arm shares it.

Ask in the team chat before running on the real arm, so two scripts never drive it at once.

## What is here

| Path | What it is |
| --- | --- |
| `calib/station_sim.py` | The sim station: one HTTP port with the arm server API, backed by MuJoCo. |
| `sim/physics.py` | The MuJoCo scene: Metal arm from its URDF and CAD, servos, jaws, table, foam blocks. |
| `sim/tower.py` | `SimRobot`: the scene plus the sag, the joint zero errors and the calibrated wrist camera. |
| `calib/tower_plan.py` | Kinematics (`Kin`: FK, IK), the sag model, block shapes, the wrist-view block tracker (`WristView`). |
| `calib/safety.py`, `calib/tower_safety.py` | The safety checks every pose and path goes through (rules below). |
| `calib/opstack.py` | Planner and streamed motions: `Op`, `Plan`, `stream`, `fast_go` (one pick and one place). |
| `calib/data_loop.py`, `calib/tall_tower.py`, `calib/arm_runner.py` | Our larger scripts: scatter and restack, tallest tower, and the always-on job runner. All take `--sim`. |
| `calib/fit_overhead.py`, `vision/` | The overhead camera's colour tracker (used by the scripts above on the overhead frame). |
| `calib/arm_left.json` | Joint offsets: model joints = measured joints + `dq_rad`. |
| `calib/cam_wrist_l.json` | Wrist camera calibration: mount on the tool, focal length, k1/k2 distortion (1280x720). |
| `calib/cam_mid.json` | Overhead camera calibration (real frames). |
| `calib/sag_left.json`, `sim/servo_fit.json` | The fitted sag model and the servo fit to the real left arm's tracking. |
| `sim/measured.json` | Measured sizes of the station and the foam pieces, each with its source. |
| `assets/dream/metal/` | The Metal arm's URDF, meshes and inertials (MIT, MakerMods; see `assets/dream/`). |
| `examples/pick_and_stack.py` | The worked example below. |

## What the sim models

- **The arm.** The Metal arm's kinematics from its URDF, its CAD meshes drawn, link masses from the vendor URDF,
  collision boxes for the pads, palm and base. Position servos at MakerMods' gains, torque-limited, with joint
  friction and armature.
- **The arm server's behaviour.** A 30 Hz loop that walks the command toward the goal at most `rate` rad/s (capped
  at 0.6), the jaws at 1.5 per second, and stops the move where the arm is when a joint lags its command by more
  than 0.3 rad. `/state` reports joints in the server's frame, as the real arm does (model joints less `dq_rad`).
- **Sag.** The real driver has no gravity feedforward, so loaded joints sit below their command. The sim injects
  the fitted sag (`calib/sag_left.json`, `kp` from `sim/servo_fit.json`), and the planner compensates for it the
  same way it does on the real arm.
- **The wrist camera.** Rendered through the calibrated mount and lens in `calib/cam_wrist_l.json` (position on the
  tool, focal length, k1/k2 distortion), at 640x360. The tracker finds blocks by colour in that image exactly as it
  does on real frames.
- **Blocks.** Three foam blocks (green, blue, purple), 50.7 x 49.9 x 34.5 mm with the 23.1 mm hole drawn round,
  foam contact (soft, grippy), on a table the size of the real one. They can be picked, stacked, tipped and knocked.
- **Mismatch, on purpose.** The default mode `real` adds pixel noise and brightness jitter, the camera mount off by
  up to 2 mm and 1 deg, the sag at 0.8 to 1.2 times the model, and joint zero errors up to 0.75 deg. `--mode true`
  turns all of that off; `--mode noise` keeps only the image noise and mount error.
- **Faults for tests.** `POST /inject` can nudge, tip, remove or drop a block, close the gate, raise motor
  temperatures, or arm a fault for an event (the block slips on close, falls out on lift, tips on release).

## What it does not model, or only roughly

- **The real tracker error.** Sim renders are clean, flat-coloured CAD, so the wrist fit is better in sim than on
  real frames, where blocks touching each other, glare and partial views are the hard cases. The sim does not
  reproduce the real error. A look that works in sim is a necessary check, not proof.
- **Real lighting.** One headlight and flat colours. No shadows that move, no glare, no white balance drift, no
  motion blur. The table is a plain brown box with a white sheet.
- **Sag beyond the model.** Sag is the fitted gravity model with a random scale. The real arm's sag also depends on
  direction of approach, load and temperature, and the final 1 to 3 mm of a set-down differ on the real arm.
- **The servo fit.** `sim/servo_fit.json` holds a fit to the real arm's tracking (80 ms command delay, per-joint
  friction, speed caps). `calib/station_sim.py` uses only its `kp` (through the sag model); the motors in
  `sim/physics.py` run at MakerMods' default gains. Real moves lag more than sim moves.
- **The overhead camera.** `/overhead.jpg` is a plain MuJoCo view from above and behind the arm, not rendered
  through `calib/cam_mid.json`. Overhead tracking results in sim mean nothing.
- **Other things on the table.** Only the three blocks. No sticks, golf balls, tag cubes, hands, or the right arm.
  The scripts that classify extra colours (`calib/tall_tower.py`) see parts of the sim's table as yellow "sticks".
- **The real server's extras.** The sim ignores `stop_on_stall` and has no `/quit`. The gate pages (team page and
  Dream caller) always say the left arm is free unless you close them with `/inject`.

## Setup

Python 3.12 (tested on 3.12.13). The only packages are MuJoCo, NumPy, SciPy and OpenCV (`requirements.txt`).

```bash
git clone https://github.com/isaiahb/metal-station-sim.git
cd metal-station-sim
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run everything from the repo root (the code imports `calib.*`, `sim.*` and `vision.*`). Output goes to `runs/`,
which git ignores.

## Start the sim server

```bash
.venv/bin/python -m calib.station_sim --port 8790
```

It prints `station sim on 127.0.0.1:8790 ...` when ready (a few seconds; MuJoCo may print a few RuntimeWarnings
while it measures the hand, which are harmless). Options:

- `--speed 4` (default): sim time runs 4 times faster than the wall clock. Scripts read `/clock` and use the same
  clock, so their sleeps and timeouts scale with it. Use `--speed 1` for real time.
- `--mode real|noise|true`: how far the sim is from its own model (see above).
- `--seed N`: block positions, yaws and the random errors.

It binds 127.0.0.1 only and refuses the real station's ports (8691, 8081, 18081, 18084, 8697).

### The API

The same as the real left arm server:

| Call | What it does |
| --- | --- |
| `GET /state` | `{"q": [6], "grip", "goal", "goal_grip", "rate", "moving", "lag", "error", "grip_holding", "temps_c", ...}` |
| `GET /wrist.jpg`, `GET /overhead.jpg` | The latest frame; headers `X-Joints`, `X-Grip`, `X-Stamp`. |
| `POST /move {"q": [6] or null, "grip": 0..1 or null, "rate": rad/s}` | New goal. grip 0 is closed, 1 is open (0.1 m). |
| `POST /stop` | Goal = where the arm is. |

And sim-only calls for tests:

| Call | What it does |
| --- | --- |
| `GET /clock` | The shared clock: `{"real0", "k", "t"}`. |
| `GET /truth` | Every block's true pose, `level` (0 on the table, 1 on one block), `flat`, `on_table`. Never use it inside a script you want to run on the real arm. |
| `POST /reset {"blocks": {colour: {x, y, yaw, level}}, "q": [6] or "start", "grip"}` | Put the blocks and the arm where you want them. |
| `POST /inject {"kind": ...}` | Faults: `nudge`, `tip`, `remove`, `place`, `drop`, `gate`, `temp`, `lag`, or armed ones `nudge_on_close`, `drop_on_lift`, `tip_on_release`. |
| `GET /team/api/state`, `GET /dream/api/state` | The gate pages (see the safety rules). |

Joints: `/state` and `/move` use the server's frame. The planner's model frame is
`q_model = q_server + dq_rad` (`calib/arm_left.json`), and `calib/opstack.py` sends
`sag.command(q_model) - dq_rad`. If you post raw joints yourself, do the same.

## Sim or real arm: same API, different URL

`calib/opstack.py` keeps the URLs it talks to in three module variables:

- `URL`: the arm server. Real: `http://127.0.0.1:8691` (the left arm server through its SSH tunnel).
- `TEAM`, `DREAM`: the gate pages, read before and during every motion. Real: ports 18081 and 18084 on the same
  tunnel.

To point a script at the sim, set all three (and the clock) with one call before anything moves:

```python
from calib import arm_runner as AR
base = "http://127.0.0.1:8790"
AR.use_backend(AR.sim_urls(base), AR.sim_clock(base))  # refuses a sim URL on a real station port
```

Leave that call out and the same script drives the real arm with opstack's own URLs. Ask before you do: only one
program may drive the real arm at a time.

The bigger scripts take the flag themselves:

```bash
.venv/bin/python -m calib.tall_tower dry --sim 8790        # plan the tallest tower on the current look, no motion
.venv/bin/python -m calib.tall_tower run --sim 8790        # build it
.venv/bin/python -m calib.arm_runner run --sim --cycles 1  # the job runner; starts its own sims on free ports
```

## Worked example: pick a block and stack it

`examples/pick_and_stack.py` looks at the table with the wrist camera, fits all three blocks, picks the blue one
and sets it flat on the green one, then looks again to check. It uses only `calib/opstack.py`:

1. `Plan.to(...)` solves each waypoint (fingertip xyz and jaw yaw) with the arm model's IK, fingers down;
2. every piece of the plan is checked against the safety rules before anything moves;
3. `stream(...)` posts the joint goals at 30 Hz, compensating sag and the joint offsets;
4. `fast_go(...)` is one streamed pick (over, down in 1 cm steps, close, lift) and one streamed place (carry at
   15 cm, lower, set the block 2 mm into the surface below, open, back off).

```bash
# terminal 1
.venv/bin/python -m calib.station_sim --port 8790
# terminal 2
.venv/bin/python -m examples.pick_and_stack --url http://127.0.0.1:8790
```

Output (seed 1, about 5 s at speed 4; the mm figures change a little from run to run with the image noise):

```
arm server: http://127.0.0.1:8790 (sim)
wrist look: {"green": {"x": 0.3389, "y": 0.0673, "yaw": 0.2286, "iou": 0.9653, ...}, "blue": {"x": 0.344, "y": -0.0474, ...}, ...}
placed: {"block_s": 9.96, "grip": 0.509}
check: blue on green, offset from the block below as the wrist sees it: [-2.8, 0.0] mm
pictures: .../runs/example/<time>
sim truth: blue level 1, 2.5 mm from the green centre: STACKED
```

`runs/example/<time>/` holds every wrist and overhead frame, `*-ov.jpg` overlays (the jaw CAD and the fitted
blocks drawn on the wrist frame, positions in cm), and `steps.jsonl`. Pick other colours with
`--pick purple --onto blue`. Run it again from a fresh layout by restarting the sim with another `--seed`, or with
`POST /reset`.

## Safety rules

Every pose and every path piece is checked before the arm moves, in sim and on the real arm. A path is the
straight line in joint space the server walks, sampled every 2 deg of the largest joint change. A refusal raises
`SystemExit` with the rule that failed. From `calib/safety.py`, `calib/tower_safety.py` and `calib/opstack.py`:

1. **The wrist camera never goes upside down.** Its image-up keeps a component along world up of at least 0
   (sideways is fine). With the fingers pointing down (approach dot down >= 0.6) the camera may face down, but its
   image-up must not point back toward the arm base (dot outward >= -0.05) and never be rolled under
   (dot up >= -0.5).
2. **Wrist limits.** wrist_roll within 1.6 rad of camera-up; wrist_flex at least 0.3 rad inside its limits.
3. **The camera box clears the arm.** The camera as a box on the gripper (its real size plus 2 cm) stays clear of
   every other link, the base and the plate (5 mm for the two wrist links next to it), and four tag cubes at fixed
   spots on the real table (`CUBES` in `calib/safety.py`; the sim does not draw them but the check still holds).
4. **Over the table.** The camera box stays 4 cm over the table; the forearm, wrist, hand and both jaws stay 1 cm
   over it.
5. **Inside the table.** Every point of the arm from the forearm out lower than 10 cm over the table stays in front
   of the base plate's front edge (x >= 0.05 m) and between the table's side edges (y -0.30 to 0.42 m). The only
   exception is the move out of and back to the folded rest pose (`home=True`), where only the height rules hold.
6. **Fingertips** at y >= -0.15 m, and the camera box at least 1 cm from every known block (the block being
   gripped excepted) along every path piece.
7. **Fingers down for picks and places**, the jaws turned (by 90 deg steps) so the camera faces outward.
8. **Speed.** Single moves (`Op.move`) at most 0.3 rad/s; streamed plans at most 0.6 rad/s, and the place's last
   2 mm steps at 0.06 rad/s. A carry may not dip under its floor (13.5 cm).
9. **The gate, before and during every motion.** The team page must say the left arm is enabled, not assigned to a
   leader and not recording, and the Dream caller must say it is not running the left arm. Otherwise the arm is
   stopped (`/stop`) and the script exits. In sim the gate is always open unless you `POST /inject {"kind": "gate"}`.
10. **Server side.** The arm server itself only caps speed and stops the move when a joint lags its command by
    over 0.3 rad. It checks nothing else: a raw `POST /move` skips rules 1 to 9. Write your own motions through
    `Plan` and `Plan.check()` (or `TowerSafety().check_path(q0, q1, opening)`) so they are checked.

## Notes

- `calib/data_loop.py` pulls recorded episodes from the station with rsync only when `STATION_LOOP` is set; sim
  runs never need it.
- The scripts write a lock directory (`runs/sim_station/ARM_OWNER` in sim) while they hold the arm. If a sim run is
  killed, remove it before the next one.
- One sim at a time is plenty. It uses one core.
