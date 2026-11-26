"""Run either whole-hand CLI demo through the dedicated twelve-axis bridge."""

import fcntl
import json
import signal
import sys
import tempfile
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path

import m16_frames as wire
from demo_calibration import (
    Calibration,
    check_same_power,
    geometry,
    last_calibration,
    measured_pose,
    zero_receipt,
)
from demo_motion import DemoStart
from engine import PoseState, StopSource
from jacobihand_client import BridgeError
from tools.demo_adapter import demo_adapter
from tools.demo_targets import glove_targets
from tools.impedance_session import owned_restricted_bridge
from tools.wuji_hand_session import ORIGIN, Target, WujiHandClient, fresh_stopped_ids
from verify_before_hardware import ROOT, require_pass
from wuji_hand import WujiHandPlan

IDS = list(range(1, 13))


def select_calibration(args, previous):
    if args.at_stretch:
        return "capture"
    if args.same_power:
        return "reuse"
    if not sys.stdin.isatty():
        raise ValueError("请选择 --same-power 沿用本次上电标定，或在示教舒展位使用 --at-stretch")
    if previous:
        stamp = datetime.fromisoformat(previous.observed_at).astimezone().isoformat(timespec="seconds")
        print(f"最近一次舒展标定 {stamp}。确认此后电机动力持续上电可输入 1 沿用。", file=sys.stderr)
    print("如已重新上电，请把真手摆到示教舒展位，输入 2 重新读取软件零点。回车取消。", file=sys.stderr)
    try:
        answer = input().strip()
    except EOFError:
        answer = ""
    if previous and answer == "1":
        return "reuse"
    if answer == "2":
        return "capture"
    raise ValueError("已取消演示")


def drive(client, source, home):
    deadline = time.monotonic() + 6
    while not client.ready().accepted:
        if time.monotonic() >= deadline:
            raise RuntimeError("十二轴静止反馈、编码校准或控制连接未就绪")
        time.sleep(.05)

    def target():
        return source.get() if source is not None else Target(time.time_ns() // 1000, home)

    first = target()
    if first is None:
        raise RuntimeError("没有新鲜的手套目标")
    last_tick = first.device_us
    sequence = 0
    stop_requested = False
    terminal = False
    source_error = None
    try:
        ack = client.start_following(first)
        if not ack.accepted:
            raise RuntimeError("专用通讯桥未接受本次演示")
        print("十二轴正在回舒展。到位后自动开始，Ctrl+C 或 jacobihand stop 可停止。", file=sys.stderr, flush=True)
        last_obs = time.monotonic()
        stop_deadline = None
        while True:
            try:
                obs = client.get_obs(wait=False)
                now = time.monotonic()
                if obs is not None:
                    last_obs = now
                    if obs.pose_state in (int(PoseState.COMPLETE), int(PoseState.FAULT)):
                        if fresh_stopped_ids(obs) == IDS:
                            terminal = True
                            return {"stop_source": obs.pose_stop_source, "stop_status": obs.pose_last_status,
                                    "fresh_stopped_ids": IDS, "source_error": source_error}
                        stop_deadline = stop_deadline or now + 3
                    if obs.pose_state == int(PoseState.STOPPING):
                        stop_deadline = stop_deadline or now + 3
                if now - last_obs >= 1 or (stop_deadline is not None and now >= stop_deadline):
                    raise TimeoutError("未收到新鲜的演示反馈或十二轴 STOP")
                if stop_deadline is None:
                    try:
                        current = target()
                    except RuntimeError as exc:
                        source_error = str(exc)
                        current = None
                        client.request_stop()
                        stop_requested = True
                        stop_deadline = now + 3
                    if current is not None and current.device_us > last_tick:
                        sequence = (sequence + 1) & 0xFFFFFFFF or 1
                        client.send_target(ack.token, sequence, current)
                        last_tick = current.device_us
                time.sleep(.002 if source is not None else .008)
            except KeyboardInterrupt:
                if not stop_requested:
                    client.request_stop()
                    stop_requested = True
                stop_deadline = stop_deadline or time.monotonic() + 3
    finally:
        if not terminal and not stop_requested:
            try:
                client.request_stop()
            except (BridgeError, OSError, TimeoutError):
                pass


def finish(result, directory, plan, sha):
    files = list(directory.glob("impedance-decisions-*.json"))
    if len(files) != 1:
        raise RuntimeError("没有取得桥端演示记录")
    data = json.loads(files[0].read_text())
    if data["wuji_hand_plan_sha256"] != plan.fingerprint or data["verified_release_sha256"] != sha:
        raise RuntimeError("演示配置或桥版本与运行回执不匹配")
    echoes, stopped = data["terminal_stop_echo_us"], data["terminal_stopped_us"]
    confirmed = [i for i in IDS if str(i) in echoes and stopped.get(str(i), -1) > echoes[str(i)]]
    result.update(homed=data["homed_at_tracking_ms"] is not None, entered_demo=data["entered_direct_following"],
                  cycles=data["completed_cycles"], confirmed_stop_ids=confirmed)
    if (confirmed != IDS or result["stop_status"] != wire.ACK_OK or result.get("source_error")
            or result["stop_source"] not in (int(StopSource.WEB_FRAME), int(StopSource.WUJI_HAND_COMPLETE))):
        raise RuntimeError(f'演示异常结束，STOP 来源 {result["stop_source"]}，已确认停止 {confirmed}。'
                           f'{result.get("source_error") or ""}')
    if result["stop_source"] == int(StopSource.WUJI_HAND_COMPLETE) and not (
            result["homed"] and result["entered_demo"] and data["run_ids"] == data["mit_echo_ids"] == IDS):
        raise RuntimeError("限时演示缺少舒展到位或十二轴运行证据")


@contextmanager
def demo_session(logs):
    # Keep two CLI launchers from capturing different offsets or racing for USB.
    with (logs / "demo.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("另一场整手演示正在运行") from exc
        def interrupt(_signum, _frame):
            raise KeyboardInterrupt
        previous = signal.signal(signal.SIGTERM, interrupt)
        try:
            yield
        finally:
            signal.signal(signal.SIGTERM, previous)


def execute(args):
    from hand_cli import scan
    release = require_pass()
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    duration = 0 if args.seconds is None else round(args.seconds * 1000)
    with demo_session(logs):
        previous = None if args.at_stretch else last_calibration(logs)
        choice = select_calibration(args, previous)
        receipt = zero_receipt(logs)
        if choice == "reuse":
            check_same_power(previous, receipt)
        with demo_adapter():
            snapshot, error = scan(2.)
            if error:
                raise RuntimeError(error["message"])
            measured = measured_pose(snapshot)
            if choice == "capture":
                calibration = Calibration(snapshot["observed_at"], measured, "operator-matched stretch", receipt[0])
                calibration.store(logs)
            else:
                calibration = previous
            directory = Path(tempfile.mkdtemp(prefix=f"demo-{args.mode}-", dir=logs))
            no_watchdog = bool(getattr(args, "no_device_watchdog", False))
            result = {"mode": args.mode, "duration_ms": duration or None, "log_dir": str(directory),
                      "calibration_source": calibration.source, "calibrated_at": calibration.observed_at,
                      "stretch_motor_deg": calibration.motor_stretch_deg, "start_motor_deg": measured,
                      "encoder_calibration": [r["encoder_calibration"] for r in snapshot["motors"]],
                      "device_watchdog_ms": 0 if no_watchdog else 300}
            print(f"已读取十二轴姿态与标定。运行记录在 {directory}", file=sys.stderr, flush=True)
            _, home = geometry()
            src = glove_targets(home, directory) if args.mode == "teleop" else nullcontext(None)
            try:
                with src as source:
                    plan = WujiHandPlan(calibration.mapping(), duration, source.startup if source else None,
                                        DemoStart(args.mode, home))
                    if args.mode == "teleop":
                        print(f"遥操启动时，CAN 驱动帧回执时限为 {plan.demo.drive_echo_limit_ms()} ms。",
                              file=sys.stderr, flush=True)
                    if no_watchdog:
                        print("本次运行已按请求屏蔽 USB2CANFD 300 ms 设备侧看门狗。", file=sys.stderr, flush=True)
                    path = directory / "plan.json"
                    path.write_text(json.dumps(plan.document(), indent=2, allow_nan=False))
                    sha = release["release_sha256"]
                    motion_args = ("--wuji-hand-follow", str(path)) + (
                        ("--device-watchdog-ms", "0") if no_watchdog else ())
                    with owned_restricted_bridge(motion_args, directory, ORIGIN, sha,
                                                 {"wuji_hand_plan_sha256": plan.fingerprint}) as port:
                        if zero_receipt(logs) != receipt:
                            raise RuntimeError("准备期间驱动器置零登记已改变，请读取新的舒展标定")
                        with WujiHandClient(port=port, origin=ORIGIN, timeout=.5) as client:
                            result.update(drive(client, source, home))
                    finish(result, directory, plan, sha)
                    return result
            except (Exception, KeyboardInterrupt) as exc:
                result["error"] = str(exc) or "已取消"
                raise
            finally:
                (directory / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
