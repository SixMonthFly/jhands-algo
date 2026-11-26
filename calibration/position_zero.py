"""Set the stopped hand's current pose as the zero for this power cycle."""
from dataclasses import dataclass
import time

import can_codec as codec
from hand_commission import Survey, observe
from motion_guard import STATUS_LIMIT_MS

ALL_IDS = tuple(range(1, 13))
STATIONARY_SPAN_DEG = 2.0
STATIONARY_RPM = 5.0
ZERO_TOLERANCE_DEG = 0.25
ZERO_WINDOW_S = 0.80


@dataclass
class ZeroFrame:
    can_id: int
    data: bytes
    min_stamp: int
    echo_stamp: int | None = None


class ZeroSurvey(Survey):
    def __init__(self, expected, on_bus, clock=time.monotonic):
        super().__init__(expected, clock)
        self.on_bus = on_bus
        self.baseline = None
        self.pending: ZeroFrame | None = None
        self.attempt_count = 0

    def receive(self, device, event):
        if event.type != self.on_bus:
            super().receive(device, event)
            return
        with self.lock:
            frame, pending = event.frame, self.pending
            if (frame is None or pending is None or pending.echo_stamp is not None
                    or frame.channel != 0 or frame.ext or frame.rtr or not frame.canfd or not frame.brs
                    or frame.can_id != pending.can_id or frame.dlen != len(pending.data)
                    or bytes(frame.payload[:frame.dlen]) != pending.data
                    or frame.timestamp <= pending.min_stamp):
                self.fail("收到与本次置零不匹配的上总线回显")
                return
            pending.echo_stamp = frame.timestamp

    def check_counters(self, counters):
        if self.baseline is None:
            super().check_counters(counters)
            return
        admitted = counters[1] - self.baseline[1]
        completed = counters[2] - self.baseline[2]
        if (counters[:1] + counters[3:] != self.baseline[:1] + self.baseline[3:]
                or not 0 <= completed <= admitted <= self.attempt_count
                or any(counters[i] < self.counters[i] for i in (1, 2))):
            self.fail("置零过程中出现收发计数异常")

    def check_axes(self, before, confirmed, pending_id=None):
        if self.failure:
            raise RuntimeError(self.failure)
        now = self.clock()
        if self.rows.keys() != set(ALL_IDS):
            raise RuntimeError("置零要求 ID 1–12 齐全")
        for motor_id, row in self.rows.items():
            if now - row.last >= STATUS_LIMIT_MS / 1000 or abs(row.speed_rpm) > STATIONARY_RPM:
                raise RuntimeError(f"ID{motor_id} 的静止反馈不新鲜或转速超限")
            if motor_id == pending_id:
                continue
            target = 0 if motor_id in confirmed else before[motor_id]
            tolerance = ZERO_TOLERANCE_DEG if motor_id in confirmed else STATIONARY_SPAN_DEG
            if abs(row.position_deg - target) > tolerance:
                raise RuntimeError(f"ID{motor_id} 在置零期间偏离已锁存的姿态")


def execute_zero(survey, dev, checkpoint, wait=time.sleep):
    survey.poll(dev)
    with survey.lock:
        if survey.rows.keys() != set(ALL_IDS) or survey.counters is None:
            raise RuntimeError("置零要求 12 台反馈及适配器统计齐全")
        for motor_id, row in survey.rows.items():
            if row.max_deg - row.min_deg > STATIONARY_SPAN_DEG:
                raise RuntimeError(f"ID{motor_id} 的当前位置尚未稳定")
        before = {i: row.position_deg for i, row in survey.rows.items()}
        survey.check_axes(before, {})
        survey.baseline = survey.counters
    result = {"before_deg": before, "attempted": [], "confirmed_deg": {}, "echo_ids": []}
    checkpoint(result)
    for motor_id in ALL_IDS:
        survey.poll(dev)
        with survey.lock:
            survey.check_axes(before, result["confirmed_deg"])
            can_id, data = codec.build_set_position_zero_frame(motor_id)
            pending = ZeroFrame(can_id, data, survey.rows[motor_id].device_us)
            survey.pending = pending
            survey.attempt_count += 1
        result["attempted"].append(motor_id)
        checkpoint(result)
        dev.send_can(0, can_id, data, canfd=True, brs=True)
        deadline = survey.clock() + ZERO_WINDOW_S
        fresh_samples = 0
        previous_stamp = -1
        while survey.clock() < deadline:
            survey.poll(dev)
            with survey.lock:
                survey.check_axes(before, result["confirmed_deg"], pending_id=motor_id)
                row = survey.rows[motor_id]
                if (pending.echo_stamp is not None and row.device_us > pending.echo_stamp
                        and row.device_us > previous_stamp):
                    previous_stamp = row.device_us
                    fresh_samples = fresh_samples + 1 if abs(row.position_deg) <= ZERO_TOLERANCE_DEG else 0
                    if fresh_samples >= 2:
                        result["confirmed_deg"][motor_id] = row.position_deg
                        result["echo_ids"].append(motor_id)
                        survey.pending = None
                        break
            wait(.02)
        else:
            raise RuntimeError(f"ID{motor_id} 未在时限内同时确认置零回显和归零反馈")
        checkpoint(result)
    survey.poll(dev)
    with survey.lock:
        survey.check_axes(before, result["confirmed_deg"])
        result["final_deg"] = {i: survey.rows[i].position_deg for i in ALL_IDS}
    return result


def zero_all(checkpoint):
    with observe(ALL_IDS, 2, echo_survey=ZeroSurvey) as (survey, dev):
        return execute_zero(survey, dev, checkpoint)
