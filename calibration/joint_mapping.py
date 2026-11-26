"""Explicit motor-to-joint mappings for the algorithm client."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from m16_frames import MOTOR_COUNT

NOMINAL_REDUCTION_RATIO = 50.0


@dataclass(frozen=True, slots=True)
class Joint:
    motor_id: int
    name: str
    direction: int
    zero_motor_deg: float
    reduction_ratio: float

    def __post_init__(self) -> None:
        if type(self.motor_id) is not int or not 1 <= self.motor_id <= MOTOR_COUNT:
            raise ValueError("motor_id must be an integer from 1 to 12")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("each joint needs a name")
        if type(self.direction) is not int or self.direction not in (-1, 1):
            raise ValueError("direction must be +1 or -1")
        if type(self.zero_motor_deg) not in (int, float) or not math.isfinite(self.zero_motor_deg):
            raise ValueError("zero_motor_deg must be finite")
        if (type(self.reduction_ratio) not in (int, float)
                or not math.isfinite(self.reduction_ratio) or self.reduction_ratio <= 0):
            raise ValueError("reduction_ratio must be finite and positive")

    def angle_rad(self, motor_deg: float) -> float:
        return self.direction * math.radians((motor_deg - self.zero_motor_deg) / self.reduction_ratio)

    def velocity_rad_s(self, motor_rpm: float) -> float:
        return self.direction * (motor_rpm / self.reduction_ratio) * (math.tau / 60)

    def motor_angle_deg(self, q_rad: float) -> float:
        return self.zero_motor_deg + self.direction * math.degrees(q_rad) * self.reduction_ratio

    def motor_velocity_rpm(self, dq_rad_s: float) -> float:
        return self.direction * dq_rad_s * (60 / math.tau) * self.reduction_ratio


@dataclass(frozen=True, slots=True)
class JointMap:
    """Tuple order is the joint vector order, independent of the CAN order."""

    joints: tuple[Joint, ...]
    calibrated: bool

    def __post_init__(self) -> None:
        joints = tuple(self.joints)
        if type(self.calibrated) is not bool:
            raise ValueError("calibrated must be a boolean")
        if len(joints) != MOTOR_COUNT or any(not isinstance(joint, Joint) for joint in joints):
            raise ValueError("a joint map needs exactly 12 Joint entries")
        if {joint.motor_id for joint in joints} != set(range(1, MOTOR_COUNT + 1)):
            raise ValueError("a joint map needs every motor ID exactly once")
        if len({joint.name for joint in joints}) != MOTOR_COUNT:
            raise ValueError("joint names must be unique")
        object.__setattr__(self, "joints", joints)

    @classmethod
    def from_json(cls, path: str | Path) -> JointMap:
        return cls.from_document(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_document(cls, doc) -> JointMap:
        if not isinstance(doc, dict) or doc.get("schema") != 1 or not isinstance(doc.get("joints"), list):
            raise ValueError("a joint-map file needs schema 1 and a joints array")
        try:
            return cls(tuple(Joint(**entry) for entry in doc["joints"]), calibrated=doc.get("calibrated"))
        except TypeError as exc:
            raise ValueError("invalid fields in joint map") from exc
