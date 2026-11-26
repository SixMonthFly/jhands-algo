"""Recorded JacobiHand OK sequence in motor displacements from taught stretch."""

import csv
import gzip
import math
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from time import time

import can_codec as codec
from motion_guard import TX_ECHO_LIMIT_MS

PERIOD_MS = 5
TELEOP_ECHO_START_S = datetime.fromisoformat("2026-11-24T00:00:00+08:00").timestamp()
TELEOP_ECHO_END_S = datetime.fromisoformat("2026-11-25T00:00:00+08:00").timestamp()


@lru_cache(maxsize=1)
def ok_recording():
    path = Path(__file__).parent / "profiles" / "ok-demo-200hz.csv.gz"
    with gzip.open(path, "rt", encoding="ascii", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader) != [f"motor_{i}_counts" for i in range(1, 13)]:
            raise ValueError("OK 录制轴序不符")
        rows = tuple(tuple(int(n) * 360 / codec.COUNTS_PER_REV for n in row) for row in reader)
    if (len(rows) < 2 or any(len(row) != 12 for row in rows)
            or rows[0] != (0.,) * 12 or rows[-1] != rows[0]):
        raise ValueError("OK 录制缺少完整的舒展首尾或十二轴目标")
    return rows


@dataclass(frozen=True)
class DemoStart:
    mode: str
    stretch_q_rad: tuple[float, ...]

    def __post_init__(self):
        if self.mode not in ("ok", "teleop"):
            raise ValueError("无效的整手演示模式")
        q = tuple(self.stretch_q_rad)
        if len(q) != 12 or any(type(v) not in (float, int) or not math.isfinite(v) for v in q):
            raise ValueError("舒展参考须包含十二轴有限角度")
        object.__setattr__(self, "stretch_q_rad", q)

    def drive_echo_limit_ms(self) -> int:
        if self.mode == "teleop":
            return 100
        return TX_ECHO_LIMIT_MS
