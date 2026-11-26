"""Resolve the taught stretch and the motor coordinates for this power cycle."""

import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from joint_mapping import Joint, JointMap
from origin_hold import START_SPEED_RPM
from wuji_hand import CAN_ORDER, NATIVE_ORDER

ROOT = Path(__file__).resolve().parent


def geometry():
    data = json.loads((ROOT / "profiles/demo-reference.json").read_text())
    by_name = {row["name"]: row["stretch_rad"] for row in data["joints"]}
    joints = tuple(Joint(row["motor_id"], row["name"], row["direction"], 0., row["reduction_ratio"])
                   for row in data["joints"])
    if tuple(j.name for j in joints) != CAN_ORDER or tuple(j.motor_id for j in joints) != tuple(range(1, 13)):
        raise ValueError("舒展示教参考的轴序不符")
    return joints, tuple(by_name[name] for name in NATIVE_ORDER)


def zero_receipt(logs):
    path = logs / "web-commission/register.json"
    zero = json.loads(path.read_text()).get("zero") if path.exists() else None
    digest = hashlib.sha256(json.dumps(zero, sort_keys=True).encode()).hexdigest()
    return digest, zero.get("utc") if zero else None


@dataclass(frozen=True)
class Calibration:
    observed_at: str
    motor_stretch_deg: tuple[float, ...]
    source: str
    zero_digest: str | None = None

    def __post_init__(self):
        if (datetime.fromisoformat(self.observed_at).tzinfo is None or len(self.motor_stretch_deg) != 12
                or any(type(v) not in (float, int) or not math.isfinite(v) for v in self.motor_stretch_deg)):
            raise ValueError("舒展标定记录不完整")

    def mapping(self):
        axes, q = geometry()
        return JointMap(tuple(replace(j, zero_motor_deg=m - j.motor_angle_deg(q[NATIVE_ORDER.index(j.name)]))
                              for j, m in zip(axes, self.motor_stretch_deg, strict=True)), True)

    def store(self, directory):
        path = directory / "demo-calibration.json"
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as stream:
            json.dump({"observed_at": self.observed_at, "motor_stretch_deg": self.motor_stretch_deg,
                       "source": self.source, "zero_digest": self.zero_digest}, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stream.name, path)


def last_calibration(directory):
    active = directory / "demo-calibration.json"
    latest = None
    if active.exists():
        data = json.loads(active.read_text())
        latest = Calibration(data["observed_at"], tuple(data["motor_stretch_deg"]), str(active), data["zero_digest"])
    # Import existing operator-matched captures using their acquisition times,
    # never a plan from an arbitrary run or a template with placeholder zeros.
    captures = []
    for path in directory.glob("wuji-reference-*/alignment.json"):
        doc = json.loads(path.read_text())
        if doc.get("read_only") is True and doc.get("all_online_stop_and_fault_free") is True:
            captures.append((datetime.fromisoformat(doc["last_utc"]), path, doc))
    if captures:
        stamp, path, doc = max(captures, key=lambda item: item[0])
        if latest is not None and datetime.fromisoformat(latest.observed_at) >= stamp:
            return latest
        axes = sorted(doc["axes"], key=lambda axis: axis["can_id"])
        candidate = Calibration(doc["last_utc"], tuple(axis["reference_motor_deg"] for axis in axes), str(path))
        measured_map = JointMap.from_json(path.parent / "joint-map.json")
        expected = candidate.mapping()
        recorded = sorted(measured_map.joints, key=lambda j: j.motor_id)
        _, stretch_q = geometry()
        for j, other, axis in zip(expected.joints, recorded, axes, strict=True):
            if (not measured_map.calibrated or j.name != other.name or j.direction != other.direction
                    or j.motor_id != other.motor_id or j.reduction_ratio != other.reduction_ratio
                    or axis["joint"] != j.name or axis["can_id"] != j.motor_id
                    or abs(stretch_q[NATIVE_ORDER.index(j.name)] - axis["reference_sim_rad"]) > 1e-9
                    or abs(j.zero_motor_deg - other.zero_motor_deg) > 1e-5):
                raise ValueError(f"舒展参考和软件映射不匹配，检查 {path}")
        return candidate
    return latest


def check_same_power(calibration, receipt):
    if calibration is None:
        raise ValueError("没有舒展标定记录，请摆到示教舒展位并使用 --at-stretch")
    digest, date = receipt
    if ((calibration.zero_digest is not None and digest != calibration.zero_digest)
            or (date and datetime.fromisoformat(date) > datetime.fromisoformat(calibration.observed_at))):
        raise ValueError("驱动器置零记录已变更，请在示教舒展位使用 --at-stretch")
    return calibration


def measured_pose(snapshot):
    rows = snapshot["motors"]
    now = snapshot["bridge_tick_ms"]
    if (not snapshot["adapter_open"] or snapshot["bus"]["health_flags"]
            or [r["id"] for r in rows] != list(range(1, 13))):
        raise ValueError("缺少健康的十二轴 CAN 反馈")
    for row in rows:
        stamp = row.get("status_tick_ms")
        if (not row["online"] or type(stamp) is not int or not 0 <= now - stamp < 80
                or row["state"] != 3 or row["fault"] != 0 or row.get("encoder_calibration") not in (0, 2)
                or not all(math.isfinite(row[k]) for k in ("speed_rpm", "current_a", "motor_deg"))
                or abs(row["speed_rpm"]) > START_SPEED_RPM):
            raise ValueError(f'ID {row["id"]} 缺少新鲜、静止且编码校准正常的 STOP 反馈')
    return tuple(row["motor_deg"] for row in rows)
