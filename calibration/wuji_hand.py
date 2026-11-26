"""Coordinate conversion and feedback checks for live MuJoCo hand targets."""

from dataclasses import asdict, dataclass
from collections import deque
import hashlib
import json
import math
import time

import can_codec as codec
from demo_motion import DemoStart, PERIOD_MS, ok_recording
from engine import sequence_is_newer
from joint_impedance import (AxisSetting, ImpedanceProfile, JointImpedanceTrial, WUJI_J3_ID, WUJI_J3_PROFILE,
                             WUJI_J3_RETURN_KP_CODE, load_document)
from joint_mapping import JointMap
from origin_hold import HOLD_KP, HOLD_KD, START_SPEED_RPM
from return_zero import RETURN_MAX_MS

NATIVE_ORDER = tuple(f"finger{finger}joint{joint}" for finger in (1, 3, 2) for joint in range(1, 5))
CAN_ORDER = tuple(f"finger{finger}joint{joint}" for finger in (2, 3, 1) for joint in range(1, 5))
CAN_INDICES = tuple(NATIVE_ORDER.index(name) for name in CAN_ORDER)
STALE_MS = 150
START_ALIGNMENT_DEG = 2.0  # Output degrees, checked only at startup.
STARTUP_BUFFER_MS = 2000
POSITION_MAX = codec.POSITION_CODE_MAX * 360.0 / codec.COUNTS_PER_REV


def sample_deadline_ms(device_us, now_ms):
    # Wall time is used once for source freshness. The resulting deadline
    # lives on the bridge's monotonic clock for all admission checks.
    wall_us = time.time_ns() // 1000
    age_us = ((wall_us - device_us + 2**31) & 0xFFFFFFFF) - 2**31
    if -25_000 <= age_us < STALE_MS * 1000:
        return now_ms + STALE_MS - (max(0, age_us) + 999) // 1000
    return None


@dataclass(frozen=True)
class WujiStartup:
    """A 2-second joint-space segment sampled and checked in MuJoCo."""

    from_q_rad: tuple[float, ...]
    to_q_rad: tuple[float, ...]

    def __post_init__(self):
        for name in ("from_q_rad", "to_q_rad"):
            q = tuple(getattr(self, name))
            if (len(q) != len(NATIVE_ORDER)
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in q)):
                raise ValueError("缓冲轨迹的每个端点都需要 12 个有限关节角")
            object.__setattr__(self, name, q)

    @classmethod
    def from_document(cls, data):
        if (type(data) is not dict or set(data) != {"schema", "from_q_rad", "to_q_rad", "duration_ms"}
                or data["schema"] != 1 or data["duration_ms"] != STARTUP_BUFFER_MS):
            raise ValueError("需要 MuJoCo 校验的 2 秒缓冲轨迹")
        return cls(data["from_q_rad"], data["to_q_rad"])

    def document(self):
        return {"schema": 1, "duration_ms": STARTUP_BUFFER_MS,
                "from_q_rad": self.from_q_rad, "to_q_rad": self.to_q_rad}


@dataclass(frozen=True)
class WujiHandPlan:
    joint_map: JointMap
    duration_ms: int
    startup: WujiStartup | None = None
    demo: DemoStart | None = None

    def __post_init__(self):
        if not self.joint_map.calibrated:
            raise ValueError("需要当次上电已标定的 12 轴仿真到真机关节映射")
        minimum = 0 if self.demo is not None else 1
        if type(self.duration_ms) is not int or not minimum <= self.duration_ms <= 0xFFFFFFFF:
            raise ValueError("跟随时长超出可表示范围")
        joints = tuple(sorted(self.joint_map.joints, key=lambda joint: joint.motor_id))
        if tuple(joint.name for joint in joints) != CAN_ORDER:
            raise ValueError("映射中的关节名与 JacobiHand 的 12 个 CAN ID 不对应")
        object.__setattr__(self, "joint_map", JointMap(joints, calibrated=True))
        if self.demo is not None:
            home = self.motor_targets(self.demo.stretch_q_rad)
            if self.demo.mode == "ok":
                if self.startup is not None:
                    raise ValueError("OK 回放从舒展帧开始")
                for offsets, zero in zip(zip(*ok_recording(), strict=True), home, strict=True):
                    if max(abs(zero + min(offsets)), abs(zero + max(offsets))) > POSITION_MAX:
                        raise ValueError("校准后的 OK 录制超出 CAN 位置范围")
            elif self.startup is None or self.startup.from_q_rad != self.demo.stretch_q_rad:
                raise ValueError("遥操需要从本次标定舒展位起步的 MuJoCo 缓冲段")
        if self.startup is not None:
            if self.demo is None and self.duration_ms <= STARTUP_BUFFER_MS:
                raise ValueError("启用缓冲后，试跑总时长必须超过 2 秒")
            self.motor_targets(self.startup.from_q_rad)
            self.motor_targets(self.startup.to_q_rad)

    @classmethod
    def load(cls, path):
        data = load_document(path)
        required = {"schema", "joint_map", "duration_ms"}
        if type(data) is not dict or not required <= set(data) <= required | {"startup", "demo"} or data["schema"] != 1:
            raise ValueError("无效的 Wuji 整手跟随配置")
        startup = WujiStartup.from_document(data["startup"]) if "startup" in data else None
        demo = None
        if "demo" in data:
            info = data["demo"]
            if type(info) is not dict or set(info) != {"mode", "stretch_q_rad"}:
                raise ValueError("无效的演示舒展参考")
            demo = DemoStart(info["mode"], info["stretch_q_rad"])
        return cls(JointMap.from_document(data["joint_map"]), data["duration_ms"], startup, demo)

    def document(self):
        document = {
            "schema": 1,
            "duration_ms": self.duration_ms,
            "joint_map": {
                "schema": 1,
                "calibrated": True,
                "joints": [asdict(joint) for joint in self.joint_map.joints],
            },
        }
        if self.startup is not None:
            document["startup"] = self.startup.document()
        if self.demo is not None:
            document["demo"] = asdict(self.demo)
        return document

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.document(), sort_keys=True, allow_nan=False).encode()).hexdigest()

    @property
    def impedance(self):
        setting = WUJI_J3_PROFILE.settings[WUJI_J3_ID - 1]
        if self.demo is not None and self.demo.mode == "ok":
            setting = AxisSetting(round(HOLD_KP / .00001), round(HOLD_KD / .01), 0.)
        return ImpedanceProfile(self.duration_ms, (setting,) * len(CAN_ORDER))

    def motor_targets(self, q_rad):
        if len(q_rad) != len(NATIVE_ORDER):
            raise ValueError("Wuji 目标必须正好有 12 轴")
        targets = tuple(joint.motor_angle_deg(q_rad[i])
                        for joint, i in zip(self.joint_map.joints, CAN_INDICES, strict=True))
        if not all(math.isfinite(target) and abs(target) <= POSITION_MAX for target in targets):
            raise ValueError("Wuji 目标超出 CAN 电机角表示范围")
        return targets

    def start_matches(self, targets, measured):
        return all(abs(target - actual) / joint.reduction_ratio <= START_ALIGNMENT_DEG
                   for target, actual, joint in zip(targets, measured, self.joint_map.joints, strict=True))

    def start_ready(self, targets, measured):
        if self.demo is not None:
            return all(math.isfinite(value) and abs(value) <= POSITION_MAX for value in measured)
        if self.startup is None:
            return self.start_matches(targets, measured)
        expected = self.motor_targets(self.startup.from_q_rad)
        return self.start_matches(expected, measured)


class WujiHandTrial(JointImpedanceTrial):
    def __init__(self, now_ms, anchors, plan, startup_allowance_ms, targets, device_us, sample_deadline):
        super().__init__(now_ms, anchors, plan.impedance, startup_allowance_ms)
        self.plan = plan
        self.live_targets = targets
        self.sample_device_us = device_us
        self.sample_deadline = sample_deadline
        self.tracking = False
        self.direct_following = False
        self.source_samples = 1
        if plan.startup is not None:
            self.buffer_end = plan.motor_targets(plan.startup.to_q_rad)

    def accept_targets(self, targets, device_us, sample_deadline):
        if not sequence_is_newer(device_us, self.sample_device_us):
            return False
        self.live_targets = targets
        self.sample_device_us = device_us
        self.sample_deadline = sample_deadline
        self.source_samples += 1
        return True

    def enter_tracking(self, motors, elapsed_ms):
        if not self.tracking:
            self.tracking = self.plan.start_ready(self.live_targets, (motor.position_deg for motor in motors))
        if self.tracking and not self.direct_following:
            if self.plan.startup is None or elapsed_ms >= STARTUP_BUFFER_MS:
                self.direct_following = True
        return self.tracking

    def advance(self, elapsed_ms, motors):
        self.elapsed_ms = elapsed_ms
        self.compute_angles = tuple(motor.position_deg for motor in motors)
        if self.tracking:
            if self.direct_following:
                self.targets = self.nominal_targets = self.live_targets
            else:
                fraction = min(elapsed_ms / STARTUP_BUFFER_MS, 1.0)
                self.targets = self.nominal_targets = tuple(start + fraction * (end - start)
                    for start, end in zip(self.anchors, self.buffer_end, strict=True))
        return elapsed_ms >= self.profile.duration_ms

    def axis_values_ok(self, i, position, speed, current, state):
        return all(math.isfinite(value) for value in (position, speed, current))

    def retained_position_ok(self, index, retained):
        # The position at zero velocity is exactly the commanded equilibrium.
        return abs(retained) <= POSITION_MAX

    def evidence(self):
        return {"wuji_hand_plan_sha256": self.plan.fingerprint,
                "source_samples": self.source_samples, "entered_tracking": self.tracking,
                "entered_direct_following": self.direct_following,
                "startup_buffer_ms": STARTUP_BUFFER_MS if self.plan.startup is not None else 0,
                "terminal_stop_echo_us": self.terminal_stop_echo_us,
                "terminal_stopped_us": self.terminal_stopped_us}


class HandDemoTrial(WujiHandTrial):
    """Latch the live anchors, home, then start the demo clock on measured arrival."""

    def __init__(self, now_ms, anchors, plan, startup_allowance_ms, *sample):
        super().__init__(now_ms, anchors, plan, startup_allowance_ms, *sample)
        self.home = plan.motor_targets(plan.demo.stretch_q_rad)
        self.recording = ok_recording() if plan.demo.mode == "ok" else None
        self.deadline = now_ms + startup_allowance_ms + RETURN_MAX_MS
        self.trace = deque(maxlen=2400)
        self.home_feedback_since = None
        self.homed_at = None
        self.home_sent = False
        self.cycles = 0
        self.mit_echo_ids = set()

    def enter_tracking(self, motors, elapsed_ms):
        self.tracking = True
        return True

    def observe_home(self, motors, periodic_angles, elapsed_ms):
        if not self.home_sent or self.homed_at is not None:
            return
        at_home = (self.plan.start_matches(self.home, (periodic_angles[m.controller_id] for m in motors))
                   and all(abs(m.speed_rpm) <= START_SPEED_RPM for m in motors))
        if not at_home:
            self.home_feedback_since = None
            return
        # Every axis must produce new periodic status packets during the dwell.
        stamp = min(m.periodic_status_ms for m in motors)
        if self.home_feedback_since is None:
            self.home_feedback_since = max(m.periodic_status_ms for m in motors)
        elif stamp - self.home_feedback_since >= 100:
            self.homed_at = elapsed_ms
            self.deadline = math.inf

    def advance(self, elapsed_ms, motors):
        self.elapsed_ms = elapsed_ms
        self.compute_angles = tuple(m.position_deg for m in motors)
        if not self.tracking:
            return False
        if self.homed_at is None:
            self.targets = self.nominal_targets = self.home
            self.codes = (WUJI_J3_RETURN_KP_CODE,) * len(CAN_ORDER)
            self.home_sent = True
            return False
        self.codes = tuple(s.kp_code if s else 0 for s in self.profile.settings)
        active_ms = elapsed_ms - self.homed_at
        if self.plan.startup is not None:
            if active_ms < STARTUP_BUFFER_MS:
                u = active_ms / STARTUP_BUFFER_MS
                self.targets = self.nominal_targets = tuple(
                    a + u * (b - a) for a, b in zip(self.home, self.buffer_end, strict=True))
                return False
            active_ms -= STARTUP_BUFFER_MS
        self.direct_following = True
        if self.recording is not None:
            self.cycles, index = divmod(active_ms // PERIOD_MS, len(self.recording))
            target = tuple(a + delta for a, delta in zip(self.home, self.recording[index], strict=True))
        else:
            target = self.live_targets
        self.targets = self.nominal_targets = target
        return bool(self.plan.duration_ms and active_ms >= self.plan.duration_ms)

    def evidence(self):
        return {**super().evidence(), "mode": self.plan.demo.mode,
                "home_motor_deg": self.home, "homed_at_tracking_ms": self.homed_at,
                "completed_cycles": self.cycles, "run_ids": sorted(self.run_sent),
                "mit_echo_ids": sorted(self.mit_echo_ids)}
