"""Follow a right Wuji Glove in the Jacobi v1028 MuJoCo viewer."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Event, Thread
from time import perf_counter, sleep

import mujoco as mj
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from retarget_wuji_glove import LIVE_SCENE, MARKERS, load_solver, projected_bend, robot_offsets, unit
from wuji_pad_orientation import (
    DEFAULT_NORMAL_WEIGHT,
    DEFAULT_POSITION_WEIGHT,
    PadOrientationSolver,
    angle_errors_deg,
    sdk_pad_normals,
)

INPUT_DT = 1 / 120
DISPLAY_DT = 1 / 60
INPUT_BUFFER_FRAMES = 4
STALE_S = .15
MIN_CONFIDENCE = .35
KP = np.tile([5.2, 5.8, 4.6, 4.0], 3)
KD = np.tile([.11, .13, .065, .038], 3)
JOINT_NAMES = tuple(f"finger{finger}joint{joint}" for finger in (1, 3, 2) for joint in range(1, 5))


@dataclass(frozen=True)
class Skeleton:
    points_m: np.ndarray
    scale_m: float
    thumb_ip: float
    index_pip: float | None
    pad_normals: np.ndarray


def read_skeleton(record):
    if record["header"]["frame_id"] != "r_wrist":
        raise ValueError("Expected a right-wrist skeleton.")
    joints = {joint["name"]: joint for joint in record["joints"]}
    chosen = [[joints[name] for name in chain] for chain in MARKERS]
    confidence = np.asarray([[joint["confidence"] for joint in chain] for chain in chosen])
    if not np.isfinite(confidence).all() or confidence.min() < MIN_CONFIDENCE:
        raise ValueError("Low-confidence skeleton.")
    raw = np.asarray([[joint["pose"]["position"] for joint in chain] for chain in chosen], dtype=float)
    wrist = np.asarray(joints["wrist"]["pose"]["position"], dtype=float)
    if raw.shape != (3, 4, 3) or wrist.shape != (3,) or not np.isfinite(raw).all() or not np.isfinite(wrist).all():
        raise ValueError("Invalid skeleton coordinates.")
    origin = raw[1:, 0].mean(axis=0)
    scale = float(np.linalg.norm(origin - wrist))
    if scale < .01 or np.linalg.norm(np.diff(raw, axis=1), axis=-1).min() <= .003:
        raise ValueError("Degenerate hand geometry.")
    points = (raw - origin)[..., [0, 2, 1]] * [1, -1, 1]
    thumb_ip = measured_hinge(points[0], joints["thumb_ip"], 1)
    try:
        index_pip = measured_hinge(points[1], joints["index_finger_dip"], 0)
    except (ValueError, KeyError, TypeError):
        index_pip = None
    return Skeleton(points, scale, thumb_ip, index_pip, sdk_pad_normals(joints, points))


def measured_hinge(points, joint, first_bone):
    quat = joint["pose"]["orientation"]
    axis = Rotation.from_quat([quat[key] for key in "xyzw"]).apply([1., 0., 0.])[[0, 2, 1]] * [1, -1, 1]
    directions = unit(np.diff(points, axis=0))[first_bone:first_bone + 2]
    if np.max(abs(directions @ axis)) > .15:
        raise ValueError("SDK hinge axis disagrees with the measured bone plane.")
    return float(projected_bend(*directions, axis))


def read_index_pip(record):
    """Measure the raw index hinge without running the three-finger retargeter."""
    if record["header"]["frame_id"] != "r_wrist":
        raise ValueError("Expected a right-wrist skeleton.")
    joints = {joint["name"]: joint for joint in record["joints"]}
    chain = [joints[name] for name in MARKERS[1][:3]]
    confidence = np.asarray([joint["confidence"] for joint in chain], dtype=float)
    raw = np.asarray([joint["pose"]["position"] for joint in chain], dtype=float)
    if (not np.isfinite(confidence).all() or confidence.min() < MIN_CONFIDENCE
            or raw.shape != (3, 3) or not np.isfinite(raw).all()
            or np.linalg.norm(np.diff(raw, axis=0), axis=1).min() <= .003):
        raise ValueError("Invalid index finger coordinates or confidence.")
    points = raw[..., [0, 2, 1]] * [1, -1, 1]
    return measured_hinge(points, chain[2], 0)


class SkeletonFeed:
    """Absorb brief IK stalls without building an unbounded backlog."""

    def __init__(self, lines, *, realtime_replay=False, pip_sink=None):
        self.stop = Event()
        self.done = Event()
        self.ready = Condition()
        self.frames = deque(maxlen=INPUT_BUFFER_FRAMES)
        self.thread = Thread(target=self.read, args=(lines, realtime_replay, pip_sink), daemon=True)
        self.thread.start()

    def read(self, lines, replay, pip_sink):
        first_tick = None
        last_tick = -1
        start = perf_counter()
        try:
            for line in lines:
                if self.stop.is_set():
                    break
                received = perf_counter()
                try:
                    record = json.loads(line)
                    tick = int(record["header"]["timestamp_us"])
                    if tick <= last_tick:
                        continue
                except (ValueError, KeyError, TypeError, OverflowError):
                    continue
                if first_tick is None:
                    first_tick = tick
                if replay:
                    delay = start + (tick - first_tick) / 1e6 - received
                    if self.stop.wait(max(0., delay)):
                        break
                    received = perf_counter()
                last_tick = tick
                with self.ready:
                    self.frames.append((received, tick, record))
                    self.ready.notify()
                if pip_sink is not None:
                    try:
                        angle = read_index_pip(record)
                    except (ValueError, KeyError, TypeError, OverflowError):
                        pass
                    else:
                        pip_sink.write(json.dumps({
                            "device_timestamp_us": tick, "glove_index_pip_rad": angle,
                            "state": "TRACKING RIGHT GLOVE",
                        }, allow_nan=False) + "\n")
        finally:
            with self.ready:
                self.done.set()
                self.ready.notify()

    def take(self):
        with self.ready:
            self.ready.wait_for(lambda: self.frames or self.done.is_set(), INPUT_DT)
            return self.frames.popleft() if self.frames else None


def wuji_executable(override):
    if override is not None:
        return override
    on_path = shutil.which("wuji")
    if on_path:
        return on_path
    return Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Wuji" / "bin" / "wuji.exe"


@contextmanager
def skeleton_source(args, pip_sink=None):
    if args.replay is not None:
        with args.replay.open("rb") as lines:
            feed = SkeletonFeed(lines, realtime_replay=True, pip_sink=pip_sink)
            try:
                yield feed
            finally:
                feed.stop.set()
                feed.thread.join()
        return

    device = ["--sn", args.sn] if args.sn else ["--handedness", "right"]
    command = [str(wuji_executable(args.wuji_cli)), "sub", "hand_skeleton", *device, "--jsonl"]
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, env={**os.environ, "WUJI_NO_UPDATE_CHECK": "1"},
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ) as process:
        feed = SkeletonFeed(process.stdout, pip_sink=pip_sink)
        try:
            yield feed
        finally:
            feed.stop.set()
            if process.poll() is None:
                process.terminate()
            feed.thread.join()


class LiveMapping:
    def __init__(self, solver, normal_weight=DEFAULT_NORMAL_WEIGHT, position_weight=DEFAULT_POSITION_WEIGHT):
        self.solver = solver
        self.orientation = PadOrientationSolver(solver, normal_weight, position_weight)
        self.thumb_offset = robot_offsets(solver.model)[0][0, 1]
        self.target = np.clip(np.zeros(12), solver.lo, solver.hi)
        if solver.contacts(self.target)[0].min(initial=0) < -solver.penetration_tol:
            raise ValueError("The scene's neutral posture has self-penetration.")
        self.desired_pad_normals = None
        self.ik_target = self.target.copy()
        self.target_pad_normals = self.orientation.normals_at(self.target)
        self.target_pad_error_deg = None
        self.solver_evaluations = 0
        self.contact_correction = "none"
        bodies = solver.geom_bodies
        self.hand_geom = (bodies >= solver.landmarks.base) & (bodies <= solver.landmarks.tips.max())
        self.geom_joints = solver.joint_affects[bodies].astype(np.int8)
        sleep_flag = int(getattr(mj.mjtEnableBit, "mjENBL_SLEEP", 0))
        self.collision_only = hasattr(mj, "mj_fwdKinematics") and not (solver.model.opt.enableflags & sleep_flag)

    def _hand_contacts_at(self, q):
        solver = self.solver
        model, data = solver.model, solver.data
        data.qpos[:12] = q
        if self.collision_only:
            mj.mj_fwdKinematics(model, data)
            mj.mj_collision(model, data)
        else:
            mj.mj_forward(model, data)
        geoms = data.contact.geom[:data.ncon]
        return np.flatnonzero(((geoms >= 0) & self.hand_geom[np.maximum(geoms, 0)]).all(axis=1))

    def contact_depth(self, q):
        ids = self._hand_contacts_at(q)
        return float(-self.solver.data.contact.dist[ids].min(initial=0.))

    def _deepest_violation(self, q):
        ids = self._hand_contacts_at(q)
        if ids.size:
            contacts = self.solver.data.contact
            worst = ids[np.argmin(contacts.dist[ids])]
            if contacts.dist[worst] < -self.solver.penetration_tol:
                return contacts[worst]
        return None

    def _relative_joints(self, contact):
        # A joint moving both bodies rigidly cannot change their separation.
        first, second = contact.geom
        return self.geom_joints[second] - self.geom_joints[first]

    def accept_target(self, proposed, previous):
        """Preserve the requested motion on axes independent of a collision."""
        g = self.solver
        target = np.clip(proposed, g.lo, g.hi)
        contact = self._deepest_violation(target)
        if contact is None:
            return target, "none"

        # Project the request out of the contact along its distance gradient.
        # Only retract a requested displacement that actually closes the gap.
        # Every iterate is rechecked on the entire MuJoCo hand.
        data = g.data
        for _ in range(6):
            arms = contact.pos - data.xanchor[:12]
            gradient = self._relative_joints(contact) * (np.cross(data.xaxis[:12], arms) @ contact.frame[:3])
            retreat = previous - target
            gradient[gradient * retreat <= 0] = 0.
            norm_sq = gradient @ gradient
            if norm_sq < 1e-16:
                break
            step = -contact.dist * gradient / norm_sq
            target += np.clip(step, np.minimum(0., retreat), np.maximum(0., retreat))
            contact = self._deepest_violation(target)
            if contact is None:
                return target, "contact_projection"

        return self._backtrack_contacts(target, previous, contact)

    def _backtrack_contacts(self, target, previous, contact):
        # Build a feasible anchor from the current target, restoring only the
        # ancestors of each penetrating pair. Mixing poses can create a new
        # contact, so rediscover pairs until the full anchor is feasible.
        anchor = target.copy()
        groups = []
        while contact is not None:
            axes = np.flatnonzero((self._relative_joints(contact) != 0) & (anchor != previous))
            if not axes.size:
                # At a numerically ambiguous boundary start with the known
                # feasible hand, then put back each as-yet untested axis.
                if self._deepest_violation(previous) is not None:
                    raise ValueError("No feasible previous target for Wuji contact repair.")
                groups.extend(np.array([i]) for i in np.flatnonzero(anchor != previous))
                anchor = previous.copy()
                break
            groups.append(axes)
            anchor[axes] = previous[axes]
            contact = self._deepest_violation(anchor)

        # Restore feasible motion independently for each group. Save only
        # tested full poses, including when two different fingers meet.
        for axes in reversed(groups):
            trial = anchor.copy()
            trial[axes] = target[axes]
            if self._deepest_violation(trial) is None:
                anchor = trial
                continue
            start = anchor[axes].copy()
            delta = target[axes] - start
            lo, hi = 0., 1.
            for _ in range(12):
                mid = (lo + hi) / 2
                trial[axes] = start + mid * delta
                if self._deepest_violation(trial) is None:
                    lo = mid
                    anchor[axes] = trial[axes]
                else:
                    hi = mid
        return anchor, "local_contact_backtrack"

    def update(self, skeleton, dt):
        # dt remains part of the interface for replay callers. No target rate
        # bound, temporal penalty, or input low-pass is applied.
        previous = self.target
        result = self.orientation.solve(
            skeleton.points_m / skeleton.scale_m, skeleton.pad_normals, previous,
            skeleton.thumb_ip - self.thumb_offset, nfev=6,
        )
        self.ik_target = result.x.copy()
        self.target, self.contact_correction = self.accept_target(self.ik_target, previous)
        self.desired_pad_normals = skeleton.pad_normals.copy()
        self.target_pad_normals = self.orientation.normals_at(self.target)
        self.target_pad_error_deg = angle_errors_deg(self.target_pad_normals, self.desired_pad_normals)
        self.solver_evaluations = result.nfev
        return self.target


def configure_viewer(viewer):
    with viewer.lock():
        viewer.opt.geomgroup[:] = 0
        viewer.opt.geomgroup[1] = 1
        viewer.cam.lookat[:] = viewer.d.xpos[viewer.m.body("base_link").id]
        viewer.cam.lookat[2] += .04
        viewer.cam.distance = .43
        viewer.cam.azimuth = 125
        viewer.cam.elevation = -22


def advance_physics(model, data, target, n_steps):
    effort = model.jnt_actfrcrange[:12]
    clipped = 0
    for _ in range(n_steps):
        torque = KP * (target - data.qpos[:12]) - KD * data.qvel[:12] + data.qfrc_bias[:12]
        applied = np.clip(torque, effort[:, 0], effort[:, 1])
        clipped += int(np.count_nonzero(applied != torque))
        data.qfrc_applied[:12] = applied
        mj.mj_step(model, data)
    mj.mj_forward(model, data)
    return clipped / (12 * n_steps)


def draw_pad_directions(viewer, orientation, desired):
    viewer.user_scn.ngeom = 0
    if desired is None:
        return
    centers, actual = orientation.pad_frames(viewer.d)
    desired_world = desired @ orientation.geometry.transform.T
    for directions, color in ((actual, [.15, .85, .45, 1.]), (desired_world, [1., .55, .10, 1.])):
        for center, normal in zip(centers, directions, strict=True):
            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mj.mjv_initGeom(geom, mj.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(), color)
            mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_ARROW, .0015, center, center + .025 * normal)
            viewer.user_scn.ngeom += 1
    return angle_errors_deg(actual, desired_world)


def display_loop(viewer, snapshots, simulation, orientation):
    previous = None
    while viewer.is_running() and not simulation.done():
        start = perf_counter()
        if snapshots and snapshots[-1] is not previous:
            previous = snapshots[-1]
            qpos, status, age, desired = previous
            with viewer.lock():
                viewer.d.qpos[:] = qpos
                mj.mj_forward(viewer.m, viewer.d)
                pad_error = draw_pad_directions(viewer, orientation, desired)
            age_text = "waiting" if age is None else f"{max(0., age*1000):.0f} ms since valid frame"
            if pad_error is not None:
                age_text += "\nPad angle T/I/M: " + " / ".join(f"{x:.1f}" for x in pad_error) + " deg"
                age_text += "\nGreen: robot pad / orange: glove"
            viewer.set_texts((mj.mjtFontScale.mjFONTSCALE_150, mj.mjtGridPos.mjGRID_TOPLEFT, status, age_text))
            viewer.sync()
        remaining = DISPLAY_DT - (perf_counter() - start)
        if remaining > 0.:
            sleep(remaining)


def live_loop(args, feed, mapper, data, trace, stop, snapshots=None, target_sink=None):
    model = mapper.solver.model
    timestep = model.opt.timestep
    tick_start = last_step_at = perf_counter()
    pending_time = 0.
    last_good_device = None
    last_good_at = None
    index_pip = None
    status = "WAITING FOR RIGHT GLOVE"
    last_status = None
    updated = rejected = 0
    while not stop.is_set():
        frame = feed.take()
        if stop.is_set():
            break
        if feed.done.is_set():
            if args.replay is None:
                raise RuntimeError("Wuji skeleton stream exited. See the preceding CLI diagnostic.")
            if frame is None:
                stop.wait(INPUT_DT)
        start = perf_counter()
        if args.seconds is not None and start - tick_start >= args.seconds:
            break
        processed_ms = None
        if frame is not None:
            received, timestamp, record = frame
            if start - received <= STALE_S:
                try:
                    skeleton = read_skeleton(record)
                except (ValueError, KeyError, TypeError, OverflowError):
                    rejected += 1
                else:
                    dt = STALE_S if last_good_device is None else (timestamp - last_good_device) / 1e6
                    mapper.update(skeleton, dt)
                    if target_sink is not None:
                        target_sink(timestamp, mapper)
                    last_good_at, last_good_device = received, timestamp
                    index_pip = skeleton.index_pip
                    processed_ms = (perf_counter() - received) * 1000
                    updated += 1

        age = None if last_good_at is None else start - last_good_at
        if age is not None:
            status = "TRACKING RIGHT GLOVE" if age <= STALE_S else "HOLDING / INPUT STALE"
        if status != last_status:
            print(status, flush=True)
            last_status = status
        # Keep the native integration step and discard wall time lost during
        # a long suspension so it does not trigger a burst of old steps.
        pending_time += min(start - last_step_at, STALE_S)
        last_step_at = start
        steps = int(pending_time / timestep)
        pending_time -= steps * timestep
        clipped = advance_physics(model, data, mapper.target, steps) if steps else 0.
        if not np.isfinite(data.qpos[:12]).all():
            raise RuntimeError("MuJoCo returned a nonfinite joint position.")
        if snapshots is not None:
            snapshots.append((data.qpos.copy(), status, age, mapper.desired_pad_normals))
        if trace is not None:
            actual_normals = mapper.orientation.pad_frames(data)[1] @ mapper.solver.transform
            pad_error = (None if mapper.desired_pad_normals is None
                         else angle_errors_deg(actual_normals, mapper.desired_pad_normals).tolist())
            trace.write(json.dumps({
                "elapsed_s": start - tick_start, "device_timestamp_us": last_good_device,
                "glove_index_pip_rad": index_pip,
                "state": status, "updated": processed_ms is not None, "app_input_to_target_ms": processed_ms,
                "q_target_rad": mapper.target.tolist(), "q_rad": data.qpos[:12].tolist(),
                "qvel_rad_s": data.qvel[:12].tolist(), "sim_time_s": data.time,
                "q_ik_rad": mapper.ik_target.tolist(),
                "glove_thumb_ip_rad": None if frame is None or processed_ms is None else skeleton.thumb_ip,
                "glove_pad_normals": None if mapper.desired_pad_normals is None else mapper.desired_pad_normals.tolist(),
                "target_pad_normals": mapper.target_pad_normals.tolist(),
                "actual_pad_normals": actual_normals.tolist(),
                "target_pad_error_deg": None if mapper.target_pad_error_deg is None else mapper.target_pad_error_deg.tolist(),
                "actual_pad_error_deg": pad_error,
                "pad_orientation_weight": mapper.orientation.normal_weight,
                "tip_position_weight": mapper.orientation.position_weight,
                "solver_evaluations": mapper.solver_evaluations,
                "contact_correction": mapper.contact_correction,
                "max_contact_depth_mm": float(-1000 * data.contact.dist[:data.ncon].min(initial=0)),
                "contact_count": data.ncon,
                "solver_warning_counts": data.warning.number.tolist(),
                "effort_clipped_fraction": clipped,
            }, allow_nan=False) + "\n")
        if args.replay and feed.done.is_set() and (age is None or age > STALE_S):
            break
    print(f"Mapped {updated} frames, rejected {rejected}; {perf_counter()-tick_start:.2f} seconds.", flush=True)


def main(*, target_sink=None, stop=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wuji-cli", type=Path, help="Path to the installed Wuji CLI.")
    parser.add_argument("--sn", help="Wuji Glove serial number.")
    parser.add_argument("--scene", type=Path, default=LIVE_SCENE)
    parser.add_argument("--replay", type=Path, help="Play right-glove JSONL at its original device cadence.")
    parser.add_argument("--headless", action="store_true", help="Run the identical mapping and physics without a viewer.")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--pad-orientation-weight", type=float, default=DEFAULT_NORMAL_WEIGHT,
                        help="Relative weight of fingertip-pad direction alignment (default: %(default)s).")
    parser.add_argument("--tip-position-weight", type=float, default=DEFAULT_POSITION_WEIGHT,
                        help="Relative weight of fingertip position alignment (default: %(default)s).")
    parser.add_argument("--trace", type=Path, help="Optional JSONL of simulation states and timing.")
    parser.add_argument("--pip-stream", type=Path,
                        help="Optional live index PIP JSONL at the raw SDK frame rate.")
    args = parser.parse_args()
    if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be positive and finite.")
    solver = load_solver(args.scene.resolve())
    model = solver.model
    if tuple(model.joint(i).name for i in range(12)) != JOINT_NAMES:
        parser.error("Scene's 12 hand joints do not match the v1028 order.")
    mapper = LiveMapping(solver, args.pad_orientation_weight, args.tip_position_weight)
    data = mj.MjData(model)
    data.qpos[:] = solver.qpos0
    data.qpos[:12] = mapper.target
    for name in ("cube", "support"):
        model.geom_group[model.geom_bodyid == model.body(name).id] = 0
    model.geom_group[model.geom_bodyid == 0] = 0
    mj.mj_forward(model, data)
    if args.trace and args.pip_stream and args.trace.resolve() == args.pip_stream.resolve():
        parser.error("--trace and --pip-stream need different paths.")
    for path in (args.trace, args.pip_stream):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
    trace_file = nullcontext() if args.trace is None else args.trace.open("x", encoding="utf-8", buffering=1)
    pip_file = nullcontext() if args.pip_stream is None else args.pip_stream.open("x", encoding="utf-8", buffering=1)
    if stop is None:
        stop = Event()
    with trace_file as trace, pip_file as pip_sink:
        if args.headless:
            with skeleton_source(args, pip_sink) as feed:
                live_loop(args, feed, mapper, data, trace, stop, target_sink=target_sink)
            return
        # Keep viewer events and model options separate from the physics loop.
        display_model = copy(model)
        display_data = mj.MjData(display_model)
        display_data.qpos[:] = data.qpos
        mj.mj_forward(display_model, display_data)
        with mujoco.viewer.launch_passive(display_model, display_data, show_left_ui=False, show_right_ui=False) as viewer:
            configure_viewer(viewer)
            with skeleton_source(args, pip_sink) as feed, ThreadPoolExecutor(max_workers=1) as executor:
                snapshots = deque(maxlen=1)
                simulation = executor.submit(live_loop, args, feed, mapper, data, trace, stop, snapshots, target_sink)
                try:
                    display_loop(viewer, snapshots, simulation, mapper.orientation)
                finally:
                    stop.set()
                simulation.result()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
