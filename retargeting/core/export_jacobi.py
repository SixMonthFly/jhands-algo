"""Export three-finger landmarks from v1028 MuJoCo replay and empty-hand motion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np


BASE_TO_RIGHT_PALM = np.array([-1.0, -1.0, 1.0])
CHAINS = (("link2", "link3", "link4"),
          ("link6", "link7", "link8"),
          ("link10", "link11", "link12"))
PAD_GEOMS = ("component_29_structure", "component_34_structure", "component_38_structure")
DEFAULT_SOURCE = Path(__file__).resolve().parents[2] / "runs" / "mjwarp-wuji-cube-face-sm-cycle-g2u25-dev-20261116"


class Landmarks:
    """Canonical palm axes are +x thumbward, +y distal and +z toward finger flexion."""

    def __init__(self, model: mj.MjModel):
        self.model = model
        self.base = model.body("base_link").id
        self.chains = np.array([[model.body(name).id for name in names] for names in CHAINS])
        self.tips = self.chains[:, -1]
        self.offsets_m = np.stack([self.tip_offset(body, name) for body, name in zip(self.tips, PAD_GEOMS)])
        rest = mj.MjData(model)
        mj.mj_kinematics(model, rest)
        neutral_mcp = self.palm_local(rest, rest.xpos[self.chains[1:, 0]])
        self.origin_m = neutral_mcp.mean(axis=0)
        self.scale_m = float(np.linalg.norm(self.origin_m))

    def tip_offset(self, last_body: int, geom_name: str) -> np.ndarray:
        geom_id = self.model.geom(geom_name).id
        if self.model.geom_bodyid[geom_id] != last_body:
            raise ValueError(f"{geom_name} is not on the corresponding fingertip")
        mesh = self.model.geom_dataid[geom_id]
        start = self.model.mesh_vertadr[mesh]
        stop = start + self.model.mesh_vertnum[mesh]
        rotation = np.empty(9)
        mj.mju_quat2Mat(rotation, self.model.geom_quat[geom_id])
        points = self.model.mesh_vert[start:stop] @ rotation.reshape(3, 3).T + self.model.geom_pos[geom_id]
        bone = self.model.body_pos[last_body]
        projection = points @ (bone / np.linalg.norm(bone))
        return points[projection >= projection.max() - 0.003].mean(axis=0)

    def palm_local(self, data: mj.MjData, world: np.ndarray) -> np.ndarray:
        return ((world - data.xpos[self.base]) @ data.xmat[self.base].reshape(3, 3)) * BASE_TO_RIGHT_PALM

    def measure(self, data: mj.MjData) -> np.ndarray:
        tip_m = np.einsum("fij,fj->fi", data.xmat[self.tips].reshape(3, 3, 3), self.offsets_m)
        tip_world = data.xpos[self.tips] + tip_m
        world = np.concatenate([data.xpos[self.chains], tip_world[:, None, :]], axis=1)
        return self.palm_local(data, world) - self.origin_m

    def from_trace(self, poses: np.ndarray) -> np.ndarray:
        data = mj.MjData(self.model)
        result = np.empty((len(poses), 3, 4, 3))
        for k, pose in enumerate(poses):
            data.qpos[:] = pose
            mj.mj_kinematics(self.model, data)
            result[k] = self.measure(data)
        return result


def archived_trace(model: mj.MjModel, landmarks: Landmarks, source: Path, output: Path, case: int) -> dict:
    with np.load(source / "trace.npz") as data:
        selected = np.flatnonzero(data["active"][:, case])
        if not len(selected) or np.any(np.diff(selected) != 1):
            raise ValueError("Archived case must contain one contiguous nonempty episode")
        qpos = data["q"][selected, case].copy()
        time_s = data["time"][selected].copy()
        time_s -= time_s[0]
        goal_face = data["goal_face"][selected, case]
        limits = model.jnt_range[:12]
        excess = np.maximum(qpos[:, :12] - limits[:, 1], limits[:, 0] - qpos[:, :12])
    points_m = landmarks.from_trace(qpos)
    np.savez_compressed(output, time_s=time_s, points_m=points_m, palm_scale_m=landmarks.scale_m,
                        q_rad=qpos[:, :12], goal_face=goal_face)
    return {"case": case, "frames": len(time_s), "duration_s": float(time_s[-1]),
            "estimated_hz": float(1.0 / np.median(np.diff(time_s))),
            "max_limit_excess_deg": float(max(0.0, np.rad2deg(excess.max())))}


def gestures() -> tuple[np.ndarray, np.ndarray, list[str]]:
    # Each row follows the scene's joint order: thumb, middle, index.
    opened = np.array([-.18, .10, .06, .07, -.12, -.12, .08, .08, .12, -.12, .08, .08])
    curled = opened + [-.49, .17, .43, .32, -.01, -.49, .49, .40, .01, -.44, .49, .40]
    index = opened + [0, 0, 0, 0, 0, 0, 0, 0, -.04, -.43, .47, .38]
    middle = opened + [0, 0, 0, 0, .05, -.43, .47, .38, 0, 0, 0, 0]
    opp_index = index + [-.76, .30, .53, .37, 0, 0, 0, 0, 0, 0, 0, 0]
    opp_middle = middle + [-.88, .39, .56, .39, 0, 0, 0, 0, 0, 0, 0, 0]
    spread = opened + [.05, 0, 0, 0, -.20, -.05, 0, 0, .18, -.05, 0, 0]
    timeline = [
        (0.0, opened, "open start"),
        (1.0, opened, "open hold"),
        (2.2, curled, "close all"),
        (2.7, curled, "closed hold"),
        (4.0, opened, "open all"),
        (5.1, index, "index curl"),
        (5.6, index, "index hold"),
        (6.8, opened, "index release"),
        (7.7, middle, "middle curl"),
        (8.1, middle, "middle hold"),
        (9.2, opened, "middle release"),
        (10.8, opp_index, "thumb-index opposition attempt"),
        (11.2, opp_index, "opposition hold"),
        (12.4, opened, "opposition release"),
        (13.9, opp_middle, "thumb-middle opposition attempt"),
        (14.4, opp_middle, "opposition hold"),
        (15.6, opened, "opposition release"),
        (16.6, spread, "spread"),
        (17.4, curled, "close all"),
        (18.3, opened, "open all"),
        (19.1, index, "index curl"),
        (19.9, middle, "finger switch"),
        (20.8, opened, "return open"),
        (21.3, opened, "open hold"),
    ]
    return np.array([p[0] for p in timeline]), np.stack([p[1] for p in timeline]), [p[2] for p in timeline]


def empty_hand(model: mj.MjModel, landmarks: Landmarks, output: Path) -> dict:
    times, goals, labels = gestures()
    data = mj.MjData(model)
    cube_joint = model.joint("cube_freejoint").qposadr[0]
    for name in ("cube", "support"):
        mask = model.geom_bodyid == model.body(name).id
        model.geom_contype[mask] = 0
        model.geom_conaffinity[mask] = 0
    data.qpos[cube_joint + 2] = -2.0
    data.qpos[:12] = goals[0]
    mj.mj_forward(model, data)
    timestep = model.opt.timestep
    decimation = int(round(0.02 / timestep))
    if not np.isclose(decimation * timestep, 0.02):
        raise ValueError("Model timestep does not divide the desired 50 Hz sample period")

    limits = model.jnt_range[:12].copy()
    if np.any(goals < limits[:, 0]) or np.any(goals > limits[:, 1]):
        raise ValueError("A default gesture exceeds the model's joint limits")
    kp = np.tile([5.2, 5.8, 4.6, 4.0], 3)
    kd = np.tile([0.11, 0.13, 0.065, 0.038], 3)
    torque = np.max(np.abs(model.jnt_actfrcrange[:12]), axis=1)
    if np.any(torque <= 0):
        raise ValueError("Actuator torque bounds are not available")
    total_steps = int(round(times[-1] / timestep))
    samples = 1 + total_steps // decimation
    hand = np.empty((samples, 3, 4, 3))
    actual = np.empty((samples, 12))
    commands = np.empty((samples, 12))
    contact_count = np.empty(samples, dtype=int)
    effort = np.empty((samples, 12))
    steps = np.arange(samples) * decimation
    for step in range(total_steps + 1):
        t = step * timestep
        segment = np.searchsorted(times, t, side="right") - 1
        segment = min(segment, len(times) - 2)
        u = (t - times[segment]) / (times[segment + 1] - times[segment])
        smooth = u ** 3 * (10 - 15 * u + 6 * u * u)
        target = goals[segment] + smooth * (goals[segment + 1] - goals[segment])
        applied = np.clip(kp * (target - data.qpos[:12]) - kd * data.qvel[:12] + data.qfrc_bias[:12], -torque, torque)
        data.qfrc_applied[:12] = applied
        if step % decimation == 0:
            at = step // decimation
            mj.mj_kinematics(model, data)
            hand[at] = landmarks.measure(data)
            actual[at] = data.qpos[:12]
            commands[at] = target
            contact_count[at] = data.ncon
            effort[at] = applied
        if step < total_steps:
            mj.mj_step(model, data)

    time_s = steps * timestep
    if not np.isfinite(hand).all() or not np.isfinite(actual).all():
        raise ValueError("Simulation generated nonfinite values")
    np.savez_compressed(output, time_s=time_s, points_m=hand, palm_scale_m=landmarks.scale_m, q_rad=actual,
                        target_q_rad=commands, contacts=contact_count)
    error = np.abs(np.rad2deg(actual - commands))
    excess = np.maximum(actual - limits[:, 1], limits[:, 0] - actual)
    thumb_index = np.linalg.norm(hand[:, 0, -1] - hand[:, 1, -1], axis=1)
    thumb_middle = np.linalg.norm(hand[:, 0, -1] - hand[:, 2, -1], axis=1)
    return {
        "frames": samples, "duration_s": float(time_s[-1]), "estimated_hz": 50,
        "mean_abs_tracking_error_deg": float(error.mean()),
        "p95_abs_tracking_error_deg": float(np.percentile(error, 95)), "max_abs_tracking_error_deg": float(error.max()),
        "max_limit_excess_deg": float(max(0.0, np.rad2deg(excess.max()))),
        "frames_with_self_contact": int((contact_count > 0).sum()),
        "torque_clipped_fraction": float((np.abs(effort) >= torque - 1e-9).mean()),
        "tip_distance_min_mm": {
            "thumb_index": float(1000 * thumb_index.min()),
            "thumb_middle": float(1000 * thumb_middle.min()),
        },
        "motion_targets": [{"time_s": float(t), "pose": label} for t, label in zip(times, labels)],
    }


def review_coordinates(model: mj.MjModel, landmarks: Landmarks) -> dict:
    d = mj.MjData(model)
    mj.mj_kinematics(model, d)
    neutral = landmarks.measure(d)
    d.qpos[model.joint("finger2joint2").qposadr[0]] = -0.25
    mj.mj_kinematics(model, d)
    bent = landmarks.measure(d)
    if not (neutral[1, 0, 0] > neutral[2, 0, 0] and np.all(neutral[1:, -1, 1] > 0)
            and bent[1, -1, 2] > neutral[1, -1, 2]):
        raise ValueError("The neutral pose and finger flexion disagree with canonical palm axes")
    return {"neutral_points_mm": (neutral * 1000).round(3).tolist(),
            "index_flexion_test_tip_change_mm": ((bent[1, -1] - neutral[1, -1]) * 1000).round(3).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--case", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    model_path = args.source.resolve() / "cube-scene.xml"
    model = mj.MjModel.from_xml_path(str(model_path))
    landmarks = Landmarks(model)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "matching_scene": str(model_path),
        "finger_order": ["thumb", "index", "middle"],
        "landmark_bodies_base_to_tip": CHAINS,
        "most_distal_landmark": "Centroid of external pad vertices within 3 mm of far end along distal bone",
        "palm_frame": "Right hand. x toward thumb, y toward long fingers, z toward flexion",
        "palm_origin": "Fixed median-plane midpoint of index/middle MCP surrogates in neutral pose",
        "palm_scale": "Neutral MCP midpoint to the original base_link origin, used as virtual wrist surrogate",
        "palm_scale_m": landmarks.scale_m,
        "palm_origin_base_m": (landmarks.origin_m * BASE_TO_RIGHT_PALM).tolist(),
        "coordinate_check": review_coordinates(model, landmarks),
        "archived_source": str(args.source.resolve() / "trace.npz"),
        "archived": archived_trace(model, landmarks, args.source, args.output / f"jacobi_case{args.case}.npz", args.case),
        "empty_hand_controller": "Scripted targets tracked using bounded PD torque with gravity feedforward in MuJoCo; not a trained controller",
        "empty_hand": empty_hand(model, landmarks, args.output / "jacobi.npz"),
    }
    (args.output / "sim_samples.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
