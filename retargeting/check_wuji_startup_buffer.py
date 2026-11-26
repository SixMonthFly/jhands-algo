"""Check a two-second Wuji startup segment against the live MuJoCo scene."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from live_wuji_glove import JOINT_NAMES, LiveMapping
from retarget_wuji_glove import LIVE_SCENE, load_solver

DURATION_MS = 2000


def latest_target(trace):
    with trace.open("rb") as stream:
        size = stream.seek(0, 2)
        stream.seek(max(0, size - 32768))
        rows = stream.read(size - stream.tell()).split(b"\n")
    for row in reversed(rows[1:-1]):
        try:
            frame = json.loads(row)
            tick = frame["device_timestamp_us"]
        except (ValueError, KeyError, TypeError):
            continue
        if (frame.get("updated") is True and frame.get("state") == "TRACKING RIGHT GLOVE"
                and type(tick) is int and -.025 <= time.time() - tick / 1e6 <= .100):
            return frame
    raise ValueError("MuJoCo trace is missing a fresh, complete target frame")


def check_segment(mapper, start, end):
    solver = mapper.solver
    # A measured STOP pose can already occupy part of the IK inset.
    # The transition remains inside the scene's actual joint limits.
    for pose, lo, hi in ((start, solver.native_lo, solver.native_hi), (end, solver.lo, solver.hi)):
        if (pose.shape != (len(JOINT_NAMES),) or not np.isfinite(pose).all()
                or np.any(pose < lo - 1e-9) or np.any(pose > hi + 1e-9)):
            raise ValueError("Startup endpoint is outside MuJoCo target bounds")
    deepest = 0.
    for fraction in np.linspace(0.0, 1.0, DURATION_MS + 1):
        penetration = mapper.contact_depth(start + fraction * (end - start))
        if penetration > solver.penetration_tol:
            raise ValueError(f"Startup trajectory has self-contact at {fraction * DURATION_MS:.0f} ms "
                             f"({penetration * 1000:.3f} mm of penetration)")
        deepest = max(deepest, penetration)
    return deepest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    if tuple(reference["joint_names_native_order"]) != JOINT_NAMES:
        parser.error("Reference axes do not match the live MuJoCo scene")
    mapper = LiveMapping(load_solver(LIVE_SCENE))
    try:
        first = latest_target(args.trace)
        start = np.asarray(reference["q_target_rad"], dtype=float)
        end = np.asarray(first["q_target_rad"], dtype=float)
        depth = check_segment(mapper, start, end)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    recipe = {"schema": 1, "duration_ms": DURATION_MS, "from_q_rad": start.tolist(), "to_q_rad": end.tolist()}
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(recipe, out, indent=2, allow_nan=False)
    print(json.dumps({"source_device_us": first["device_timestamp_us"], "poses_checked": DURATION_MS + 1,
                      "max_penetration_mm": depth * 1000,
                      "max_output_deg_per_s": float(np.rad2deg(abs(end - start)).max()) / (DURATION_MS / 1000),
                      "buffer": str(args.out)}))


if __name__ == "__main__":
    main()
