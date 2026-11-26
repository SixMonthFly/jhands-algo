"""Retarget a stored right Wuji skeleton recording to the local Jacobi simulation."""
import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "deliverables" / "hand-motion-prior-review-20261117" / "audit-20261118"
LIVE_SCENE = ROOT / "deliverables" / "palm-link-20261118" / "live-assets" / "scene.xml"
PALM_ROOT_BODIES = ("link1", "link5", "link9")  # Thumb, index, middle first links.
sys.path.insert(0, str(AUDIT))

from records import MARKERS as MARKERS  # noqa: E402
from records import PILOT, extract, sha  # noqa: E402
from replay_physics import integrate_clip  # noqa: E402
from signed_motion import projected_bend, read_quats, robot_offsets, unit  # noqa: E402
from signed_retarget import rt  # noqa: E402
from thumb_repair import repair  # noqa: E402


def load_solver(scene):
    # The live scene stores hand meshes relative to the project, and textures
    # relative to the scene. The archived scene also has scene-relative meshes.
    tree = ElementTree.parse(scene)
    # The Wuji viewer, retargeter, and startup check use this same model.
    # Ignore palm/root contact in both pose checks and simulated dynamics.
    contact = tree.find("./contact")
    if contact is None:
        contact = ElementTree.SubElement(tree.getroot(), "contact")
    excluded = {frozenset((pair.get("body1"), pair.get("body2")))
                for pair in contact.findall("exclude")}
    for body in PALM_ROOT_BODIES:
        if frozenset(("base_link", body)) not in excluded:
            ElementTree.SubElement(contact, "exclude", body1="base_link", body2=body)
    for asset in tree.findall("./asset/*"):
        for attribute, value in asset.attrib.items():
            if not attribute.startswith("file"):
                continue
            path = scene.parent / value
            if not path.is_file():
                path = ROOT / value
            if not path.is_file():
                raise FileNotFoundError(f"{scene}: missing asset {value}")
            asset.set(attribute, str(path.resolve()))
    with TemporaryDirectory() as directory:
        resolved = Path(directory) / scene.name
        tree.write(resolved, encoding="utf-8")
        previous = rt.prior.old.SCENE
        rt.prior.old.SCENE = resolved
        try:
            solver = rt.MorphologySolver(.60, False, True)
            solver.max_step_rad = np.inf
            return solver
        finally:
            rt.prior.old.SCENE = previous


def render_video(path, model, track):
    import imageio_ffmpeg
    import mujoco as mj
    from PIL import Image

    sys.path.insert(0, str(PILOT / "retarget-preview"))
    try:
        from render_retarget_preview import SIZE, camera_vectors, draw_human, f, make_video_frame
    finally:
        sys.path.pop(0)

    width, height, fps = 1120, 654, 25
    path.parent.mkdir(parents=True, exist_ok=True)
    sim = mj.MjData(model)
    sim.qpos[model.joint("cube_freejoint").qposadr[0] + 2] = -2
    for body in (0, model.body("cube").id, model.body("support").id):
        model.geom_group[model.geom_bodyid == body] = 0
    scene_option = mj.MjvOption()
    scene_option.geomgroup[:] = 0
    scene_option.geomgroup[1] = 1
    camera = mj.MjvCamera()
    camera.distance = .5
    fonts = f(26, True), f(16), f(19, True)
    timeline = track["time_s"]
    segments = [np.flatnonzero(track["segment_id"] == s)[::2] for s in dict.fromkeys(track["segment_id"])]
    duration = sum(len(indices) for indices in segments) / fps
    elapsed = 0.0
    with mj.Renderer(model, width=SIZE, height=SIZE) as renderer:
        writer = imageio_ffmpeg.write_frames(
            str(path), (width, height), fps=fps, codec="libx264", macro_block_size=2,
            output_params=["-movflags", "+faststart"],
        )
        writer.send(None)
        try:
            for indices in segments:
                for index in indices:
                    sim.qpos[:12] = track["q_rad"][index]
                    mj.mj_forward(model, sim)
                    renderer.update_scene(sim, camera, scene_option=scene_option)
                    eye, forward, up = camera_vectors(
                        sim, model, [.019, .087, .011], [.14, .105, .36], [0., 1., 0.]
                    )
                    for view in renderer.scene.camera:
                        view.pos[:], view.forward[:], view.up[:] = eye, forward, up
                    renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = 0
                    pixels = renderer.render()
                    pixels[np.all(pixels == 0, axis=-1)] = (16, 25, 37)
                    robot = Image.fromarray(pixels)
                    human_points = track["human_points_m"][index]
                    robot_points = track["points_m"][index]
                    human = draw_human(human_points, track["human_palm_scale_m"], SIZE)
                    human_gaps = 1000 * np.linalg.norm(human_points[1:, -1] - human_points[0, -1], axis=1)
                    robot_gaps = 1000 * np.linalg.norm(robot_points[1:, -1] - robot_points[0, -1], axis=1)
                    frame = make_video_frame(
                        human, robot, "Wuji 手套映射", timeline[index], elapsed, duration,
                        human_gaps, robot_gaps, "MuJoCo 物理回放", fonts,
                    )
                    writer.send(np.asarray(frame).tobytes())
                    elapsed += 1 / fps
        finally:
            writer.close()


def interpolate_landmarks(grid, source, flat):
    return np.column_stack([np.interp(grid, source, column) for column in flat.T]).reshape(-1, 3, 4, 3)


def map_recording(paths, start, end, solver):
    raw = [extract(path) for path in paths]
    ticks = np.concatenate([chunk["ticks"] for chunk in raw])
    time_s = (ticks - ticks[0]) / 1e6
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("Original timestamps must increase across the supplied chunks.")
    scales = np.array([chunk["scale_m"] for chunk in raw])
    scale = scales[0]
    if not np.isfinite(scales).all() or scale < .01 or np.ptp(scales) > 1e-5:
        raise ValueError("Supply one recording with a valid, consistent SDK palm scale.")

    points = np.concatenate([chunk["p"] for chunk in raw])
    axes = read_quats(paths)
    bones = unit(np.diff(points, axis=2))
    signed = projected_bend(bones[:, :, :-1], bones[:, :, 1:], axes[:, :, None])
    selected = (time_s >= start) & (time_s < end)
    valid = np.concatenate([chunk["valid"] for chunk in raw]) & selected
    if valid.any() and np.quantile(abs((bones * axes[:, :, None]).sum(axis=-1))[valid], .99) > .03:
        raise ValueError("The declared SDK hinge axes disagree with the observed bone plane.")
    edges = np.r_[0, np.flatnonzero(~valid[1:] | ~valid[:-1] | (np.diff(time_s) > .025)) + 1, len(time_s)]
    thumb_offset = robot_offsets(solver.model)[0][0, 1]
    spans, stats = [], []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        if not valid[a] or time_s[b-1] - time_s[a] < .5:
            continue
        source = time_s[a:b]
        grid = np.arange(np.ceil(source[0]*50-1e-8), np.floor(source[-1]*50+1e-8)+1) / 50
        flat = points[a:b].reshape(b-a, -1)
        sigma = .018 / np.median(np.diff(source))

        human = interpolate_landmarks(grid, source, flat)
        goal = interpolate_landmarks(grid, source, gaussian_filter1d(flat, sigma, axis=0, mode="nearest"))
        thumb = np.interp(grid, source, gaussian_filter1d(signed[a:b, 0, 1], sigma, mode="nearest"))
        _, native_target, _, _ = rt.run_ik(solver, grid, goal / scale, "reference")
        gap = np.linalg.norm(human[:, 0:1, -1] - human[:, 1:, -1], axis=-1).min(axis=1) / scale
        target, reference, depth = repair(solver, native_target, thumb, thumb_offset, np.exp(-(gap/.35)**4))
        actual, robot, velocity, physical_depth, limit_excess, telemetry = integrate_clip(
            solver.model, solver.landmarks, grid, target
        )
        spans.append(dict(
            time_s=grid, segment_id=np.repeat(str(len(spans)), len(grid)),
            human_points_m=human, reference_points_m=reference, points_m=robot,
            q_target_rad=target, q_rad=actual, qd_rad_s=velocity, valid=depth <= .0005,
            self_penetration_m=depth, physical_depth_mm=physical_depth, physical_limit_excess_deg=limit_excess,
        ))
        stats.append(telemetry)

    if not spans:
        raise ValueError("No valid continuous interval longer than half a second.")
    result = {key: np.concatenate([span[key] for span in spans]) for key in spans[0]}
    result.update(palm_scale_m=solver.robot_scale, human_palm_scale_m=scale)
    result["admitted"] = rt.prior.gates(result)[0]
    result["physical_feasible"] = (result["physical_depth_mm"] <= .25) & (result["physical_limit_excess_deg"] <= .05)
    return result, stats, int(selected.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="One recording in chronological JSONL chunks.")
    parser.add_argument("--output", type=Path, required=True, help="NPZ with native joint angles and simulated landmark tracks.")
    parser.add_argument("--start", type=float, default=0.0, help="Seconds since the first frame of the first supplied chunk.")
    parser.add_argument("--end", type=float, default=float("inf"))
    parser.add_argument("--scene", type=Path, default=LIVE_SCENE, help="Compatible Jacobi v1028 MuJoCo scene.")
    parser.add_argument("--video", type=Path, help="Optional synchronized glove and robot MP4.")
    args = parser.parse_args()
    if not np.isfinite(args.start) or args.start < 0 or not args.end > args.start:
        parser.error("Use a nonnegative, increasing capture interval.")
    outputs = [args.output, args.output.with_suffix(".json")]
    if args.video is not None:
        outputs.append(args.video)
    if args.output.suffix.lower() != ".npz" or (args.video is not None and args.video.suffix.lower() != ".mp4"):
        parser.error("Use .npz for --output and .mp4 for --video.")
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Output exists: {path}")

    scene = args.scene.resolve()
    solver = load_solver(scene)
    result, stats, source_frames = map_recording(args.input, args.start, args.end, solver)
    metadata = {
        "inputs": [{"name": path.name, "sha256": sha(path)} for path in args.input],
        "reference": "V5 m060c plus signed thumb-IP repair with near-pinch MCP compensation",
        "scene": str(scene), "scene_sha256": sha(scene),
        "palm_root_contact_exclusions": [["base_link", body] for body in PALM_ROOT_BODIES],
        "input_interval_s": [args.start, args.end if np.isfinite(args.end) else None],
        "source_frames_in_interval": source_frames,
        "mapped_50hz_frames": len(result["time_s"]),
        "spans": len(stats), "torque_bounded_mujoco": stats,
        "native_q_order": "thumb, middle, index, four axes each; radians",
        "joint_names": [solver.model.joint(i).name for i in range(12)],
        "tracking_error_p95_deg": float(np.rad2deg(np.quantile(abs(result["q_target_rad"] - result["q_rad"]), .95))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    if args.video is not None:
        render_video(args.video, solver.model, result)
    print(metadata["mapped_50hz_frames"], "frames", args.output)


if __name__ == "__main__":
    main()
