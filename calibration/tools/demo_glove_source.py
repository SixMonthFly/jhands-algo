"""Windows producer for the CLI. stdout is a pipe of checked MuJoCo targets."""

import argparse
import hashlib
import json
import sys
import threading
from contextlib import redirect_stdout
from pathlib import Path
from xml.etree import ElementTree


def project_digest(project, spec):
    scene = project / spec["scene"]
    files = {scene, *(project / name for name in spec["python"])}
    for asset in ElementTree.parse(scene).findall("./asset/*"):
        for key, name in asset.attrib.items():
            if key.startswith("file"):
                local = scene.parent / name
                files.add(local if local.is_file() else project / name)
    digest = hashlib.sha256()
    for name in sorted(path.relative_to(project).as_posix() for path in files):
        digest.update(name.encode() + b"\0" + hashlib.sha256((project / name).read_bytes()).digest())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("stretch", help="MuJoCo native joint vector, in radians as JSON")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--replay", type=Path, help="offline producer test only")
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parents[1] / "profiles/demo-mujoco.json"
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    if project_digest(args.project, spec) != spec["sha256"]:
        parser.exit(1, "MuJoCo 映射代码或模型与已验证版本不匹配。\n")
    sys.path.insert(0, str(args.project / "scripts"))
    import live_wuji_glove as live
    import numpy as np
    from check_wuji_startup_buffer import check_segment

    home = np.asarray(json.loads(args.stretch), dtype=float)
    pipe = sys.stdout
    started = False
    stop = threading.Event()

    def publish(tick, mapper):
        nonlocal started
        q = mapper.target
        if not started:
            check_segment(mapper, home, q)
            print(json.dumps({"startup": {"schema": 1, "from_q_rad": home.tolist(),
                              "to_q_rad": q.tolist(), "duration_ms": 2000}}, allow_nan=False),
                  file=pipe, flush=True)
            started = True
        print(json.dumps({"device_timestamp_us": tick, "q_target_rad": q.tolist()}, allow_nan=False),
              file=pipe, flush=True)

    def wait_for_owner():
        # A closed parent pipe also ends the viewer and its Wuji subscription.
        import os
        os.read(sys.stdin.fileno(), 1)
        stop.set()

    threading.Thread(target=wait_for_owner, daemon=True).start()
    sys.argv = [live.__file__]
    if args.headless:
        sys.argv.append("--headless")
    if args.replay:
        sys.argv.extend(["--replay", str(args.replay)])
    existing = set(threading.enumerate())
    try:
        with redirect_stdout(sys.stderr):
            live.main(target_sink=publish, stop=stop)
    finally:
        # launch_passive closes its window asynchronously on a daemon thread.
        # Let the native renderer release its context before Python shuts down GLFW.
        if not args.headless:
            for thread in set(threading.enumerate()) - existing:
                thread.join()


if __name__ == "__main__":
    main()
