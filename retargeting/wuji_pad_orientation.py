"""Joint position and fingertip-pad direction objectives for Wuji retargeting.

Human and robot vectors use the existing canonical right-palm frame. Only
spatial pose bounds and contact feasibility constrain the target; there is no
input low-pass, temporal regularizer, velocity cap, or per-frame step bound.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
PAD_PROFILE = ROOT / "configs" / "hand-profiles" / "v2-legacy-gpt-palm-link8pad-20261023.json"
TIP_BODIES = ("link4", "link8", "link12")  # Thumb, index, middle.
SDK_DISTAL_JOINTS = ("thumb_ip", "index_finger_dip", "middle_finger_dip")
BEND_IDS = np.array([2, 3, 10, 11, 6, 7])
DEFAULT_NORMAL_WEIGHT = 2.0
DEFAULT_POSITION_WEIGHT = 1.5


def sdk_pad_normals(joints, points):
    """Right-glove distal +Y is palmar; distal -Z points toward the tip.

    Cross-check -Z against the observed last bone. This rejects a mismatched
    skeleton/quaternion convention instead of silently controlling the back
    of a finger. SDK xyz -> canonical coordinates is (x, -z, y).
    """
    quats = np.asarray([[joints[name]["pose"]["orientation"][key] for key in "xyzw"]
                        for name in SDK_DISTAL_JOINTS], dtype=float)
    if quats.shape != (3, 4) or not np.isfinite(quats).all() or np.any(np.linalg.norm(quats, axis=1) < 1e-8):
        raise ValueError("Invalid distal-joint orientations.")
    rotation = Rotation.from_quat(quats)
    normals = rotation.apply(np.tile([0., 1., 0.], (3, 1)))[:, [0, 2, 1]] * [1, -1, 1]
    distal = rotation.apply(np.tile([0., 0., -1.], (3, 1)))[:, [0, 2, 1]] * [1, -1, 1]
    bones = points[:, -1] - points[:, -2]
    bones /= np.linalg.norm(bones, axis=1, keepdims=True)
    if np.any(np.sum(bones * distal, axis=1) < .95):
        raise ValueError("SDK distal orientation disagrees with fingertip geometry.")
    return normals


def angle_errors_deg(actual, desired):
    return np.rad2deg(np.arccos(np.clip(np.sum(actual * desired, axis=-1), -1., 1.)))


@dataclass(frozen=True)
class PoseGoal:
    directions: np.ndarray
    fingertips: np.ndarray
    pair: np.ndarray
    pair_gain: np.ndarray
    bend_cos: np.ndarray
    extension: np.ndarray
    bend_angles: np.ndarray
    pad_normals: np.ndarray


class PadOrientationSolver:
    """Extend the existing morphology objective with signed pad normals."""

    def __init__(self, geometry, normal_weight=DEFAULT_NORMAL_WEIGHT, position_weight=DEFAULT_POSITION_WEIGHT):
        if not np.isfinite(normal_weight) or normal_weight < 0:
            raise ValueError("Pad orientation weight must be finite and nonnegative.")
        if not np.isfinite(position_weight) or position_weight < 0:
            raise ValueError("Fingertip position weight must be finite and nonnegative.")
        self.geometry = geometry
        self.normal_weight = float(normal_weight)
        self.position_weight = float(position_weight)
        profile = json.loads(PAD_PROFILE.read_text(encoding="utf-8"))
        indices = [profile["tip_links"].index(name) for name in TIP_BODIES]
        # These are area-weighted outward front normals in the unchanged tip
        # link frames, not the normals of arbitrary mesh-local coordinate axes.
        self.normals_local = np.asarray(profile["pad_normals_link"])[indices]
        self.normals_local /= np.linalg.norm(self.normals_local, axis=1, keepdims=True)
        self.centers_local = np.asarray(profile["pad_offsets_link_m"])[indices]
        self.tip_ids = np.array([geometry.model.body(name).id for name in TIP_BODIES])
        self.affects = geometry.joint_affects[self.tip_ids]

    def pad_frames(self, data):
        rotations = data.xmat[self.tip_ids].reshape(3, 3, 3)
        world_normals = np.einsum("fij,fj->fi", rotations, self.normals_local)
        centers = data.xpos[self.tip_ids] + np.einsum("fij,fj->fi", rotations, self.centers_local)
        return centers, world_normals

    def normals_at(self, q):
        self.geometry.world_points(q)
        return self.pad_frames(self.geometry.data)[1] @ self.geometry.transform

    def normal_jacobian(self):
        g = self.geometry
        world = self.pad_frames(g.data)[1]
        jac = np.cross(g.data.xaxis[None, :12], world[:, None]) * self.affects[..., None]
        return (jac @ g.transform).swapaxes(-1, -2)

    def make_goal(self, human, pad_normals, thumb_ip):
        g = self.geometry
        bones = np.diff(human, axis=1)
        lengths = np.linalg.norm(bones, axis=-1)
        directions = bones / lengths[..., None]
        fingertips = g.bases + (directions * g.lengths[..., None]).sum(axis=1)
        pair = human[1:, -1] - human[0:1, -1]
        distance = np.linalg.norm(pair, axis=1)
        human_length, robot_length = lengths.sum(axis=1), g.lengths.sum(axis=1)
        ratio = (robot_length[1:] + robot_length[0]) / (human_length[1:] + human_length[0])
        near = pair * (1 + (1 - np.exp(-(distance / .45)**2)) * (ratio - 1))[:, None]
        blend = np.exp(-(distance / .60)**4)[:, None]
        desired_pair = blend * near + (1 - blend) * (fingertips[1:] - fingertips[0:1])
        target_length = np.linalg.norm(desired_pair, axis=1)
        gain = 1.8 * (1 + 4 * np.exp(-(distance / .35)**4)) * (.35 + .85 * np.exp(-(target_length / .55)**2))
        cosine = np.sum(directions[:, :-1] * directions[:, 1:], axis=-1)
        angles = np.arccos(np.clip(cosine, -1., 1.)).ravel()
        # The signed IP target participates in the joint optimization. A later
        # unconditional overwrite would undo the newly optimized pad direction.
        angles[1] = thumb_ip
        extension = np.linalg.norm(human[:, -1] - human[:, 0], axis=1) / human_length
        return PoseGoal(directions, fingertips, desired_pair, gain, cosine, extension, angles, pad_normals)

    def residual(self, q, goal):
        g = self.geometry
        world = g.world_points(q)
        points = (g.landmarks.palm_local(g.data, world) - g.landmarks.origin_m) / g.robot_scale
        jac = (g.jac_world(world.reshape(-1, 3), g.point_bodies) @ g.transform).reshape(3, 4, 12, 3).swapaxes(-1, -2) / g.robot_scale
        bones, dbones = np.diff(points, axis=1), np.diff(jac, axis=1)
        length = np.linalg.norm(bones, axis=-1)
        directions = bones / length[..., None]
        dlength = np.einsum("fbi,fbij->fbj", directions, dbones)
        du = (dbones - directions[..., None] * dlength[..., None, :]) / length[..., None, None]
        pair, dpair = points[1:, -1] - points[0:1, -1], jac[1:, -1] - jac[0:1, -1]
        distance = np.maximum(np.linalg.norm(pair, axis=1), 1e-8)
        ddistance = np.einsum("fi,fij->fj", pair / distance[:, None], dpair)
        cosine = np.sum(directions[:, :-1] * directions[:, 1:], axis=-1)
        dcosine = np.einsum("fbi,fbij->fbj", directions[:, :-1], du[:, 1:]) + np.einsum("fbi,fbij->fbj", directions[:, 1:], du[:, :-1])
        chord = points[:, -1] - points[:, 0]
        chord_length = np.maximum(np.linalg.norm(chord, axis=1), 1e-8)
        extension = chord_length / length.sum(axis=1)
        dextension = (np.einsum("fi,fij->fj", chord / chord_length[:, None], jac[:, -1] - jac[:, 0]) - extension[:, None] * dlength.sum(axis=1)) / length.sum(axis=1)[:, None]
        direction_weight = np.array([[.18, .29, .34], [.63, .47, .34], [.63, .47, .34]])
        extension_weight = np.array([3., .22, .22])
        normals = self.pad_frames(g.data)[1] @ g.transform
        terms = [
            ((directions - goal.directions) * direction_weight[..., None], du * direction_weight[..., None, None]),
            (self.position_weight * (points[:, -1] - goal.fingertips), self.position_weight * jac[:, -1]),
            ((pair - goal.pair) * goal.pair_gain[:, None], dpair * goal.pair_gain[:, None, None]),
            (.5 * goal.pair_gain * (distance - np.linalg.norm(goal.pair, axis=1)), .5 * goal.pair_gain[:, None] * ddistance),
            (.10 * (cosine - goal.bend_cos), .10 * dcosine),
            (extension_weight * (extension - goal.extension), extension_weight[:, None] * dextension),
            (.65 * (q[BEND_IDS] - goal.bend_angles), .65 * np.eye(12)[BEND_IDS]),
            (.012 * (q - g.default_seed), .012 * np.eye(12)),
            (self.normal_weight * (normals - goal.pad_normals), self.normal_weight * self.normal_jacobian()),
        ]
        return (np.concatenate([r.ravel() for r, _ in terms]),
                np.concatenate([j.reshape(-1, 12) for _, j in terms]))

    def solve(self, human, normals, previous, thumb_ip, nfev=6):
        goal = self.make_goal(human, normals, thumb_ip)
        cache = []

        def evaluate(q):
            if not cache or not np.array_equal(q, cache[0]):
                cache[:] = [q.copy(), *self.residual(q, goal)]
            return cache[1], cache[2]

        g = self.geometry
        return least_squares(lambda q: evaluate(q)[0], np.clip(previous, g.lo + 1e-8, g.hi - 1e-8),
                             jac=lambda q: evaluate(q)[1], bounds=(g.lo, g.hi), max_nfev=nfev,
                             ftol=.001, xtol=.0005, gtol=1e-5)
