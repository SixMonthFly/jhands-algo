"""Single latest-target mailbox for a Windows Wuji/MuJoCo producer."""

import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from tools.wuji_hand_session import Target
from wuji_hand import WujiStartup


class GloveTargets:
    def __init__(self):
        self.startup = None
        self.latest = None
        self.error = None
        self.ready = threading.Condition()

    def receive(self, stream):
        try:
            for line in stream:
                row = json.loads(line)
                with self.ready:
                    if self.startup is None:
                        self.startup = WujiStartup.from_document(row["startup"])
                    else:
                        q, tick = row["q_target_rad"], row["device_timestamp_us"]
                        if (type(tick) is not int or len(q) != 12 or any(
                                type(v) not in (int, float) or not math.isfinite(v) for v in q)):
                            raise ValueError("MuJoCo 目标无效")
                        if self.latest is None or tick > self.latest.device_us:
                            self.latest = Target(tick, tuple(q))
                    self.ready.notify_all()
            raise RuntimeError("Wuji/MuJoCo 实时源已关闭")
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
            with self.ready:
                self.error = str(exc)
                self.ready.notify_all()

    def get(self):
        with self.ready:
            if self.error is not None:
                raise RuntimeError(self.error)
            target = self.latest
            if target and -25_000 <= time.time_ns() // 1000 - target.device_us < 100_000:
                return target
            return None


@contextmanager
def glove_targets(stretch, run_dir):
    python = os.environ.get("JACOBIHAND_WINDOWS_PYTHON") or shutil.which("python.exe")
    if not python:
        raise RuntimeError("找不到装有 MuJoCo 的 Windows Python，可设置 JACOBIHAND_WINDOWS_PYTHON")
    project = Path(os.environ.get("JACOBIHAND_PROJECT", "/mnt/d/JacobiHand/current"))

    def windows_path(path):
        return subprocess.check_output(["wslpath", "-w", str(path.resolve())], text=True).strip()

    args = [python, "-I", "-B", "-u", windows_path(Path(__file__).with_name("demo_glove_source.py")),
            windows_path(project), json.dumps(stretch)]
    targets = GloveTargets()
    with ((run_dir / "glove.log").open("x") as diagnostics,
          subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=diagnostics,
                           start_new_session=True) as proc):
        reader = threading.Thread(target=targets.receive, args=(proc.stdout,), daemon=True)
        reader.start()
        try:
            with targets.ready:
                if not targets.ready.wait_for(lambda: targets.error or targets.get(), timeout=25):
                    raise TimeoutError("未收到新鲜右手手套数据和已校验的起步段")
                if targets.error:
                    raise RuntimeError(f"{targets.error}，详见 {diagnostics.name}")
            yield targets
        finally:
            proc.stdin.close()
            try:
                proc.wait(timeout=6)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=3)
            reader.join(timeout=1)
            if proc.returncode and sys.exc_info()[0] is None:
                raise RuntimeError(f"MuJoCo 进程异常退出，详见 {diagnostics.name}")
