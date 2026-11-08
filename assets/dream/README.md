Copied from the `dreamlabs` 1.0.37 wheel (PyPI), Nov 7 2026, for the local sim.

- `<arm>/*.stl`, `<arm>/view.json`, `<arm>/LICENSE`: robots/setup_view/<arm> (YAM meshes MIT, I2RT Robotics; Metal meshes MIT, MakerMods; Makers Arm meshes Apache-2.0, MakerMods). view.json says which Dream IK link each mesh rides and how the jaws move.
- `<arm>/ik.urdf`: dream_ik/assets (maker_arm_ik.urdf, metal.urdf, yam.urdf), the kinematic models Dream's client IK solves with. Apache-2.0 (LICENSE-dreamlabs, NOTICE).
- Tool points (TCP) per arm are restated in viewer/sim.html from dream_ik/specs.py.

Only the Metal arm's folder (`metal/`) is kept in this repo; it is the only arm the station sim loads.
