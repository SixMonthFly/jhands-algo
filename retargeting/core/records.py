"""Re-read the source SDK records independently of the production extractors."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DELIVERABLES = HERE.parents[1]
def capture_bundle(name):
    original = DELIVERABLES / name
    return original if original.is_dir() else DELIVERABLES / "_archive_20261123" / name


PILOT = capture_bundle("hand-motion-likeness-pilot-20261116")
S2 = capture_bundle("hand-motion-likeness-session2-20261116")
S3 = capture_bundle("hand-motion-likeness-session3-20261117") / "capture-141552"
V5 = capture_bundle("hand-motion-likeness-v5-20261117")
V4 = S2 / "scorer-v4"
MARKERS = (
    ("thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"),
    ("index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip"),
    ("middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip"),
)


def read_json(path):
    return Path(path).read_text(encoding="utf-8-sig")


def load(path):
    return json.loads(read_json(path))


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract(path, trimmed_start=0):
    times, seqs, points, confidences, wrists = [], [], [], [], []
    with path.open("rb") as stream:
        for line in stream:
            record = json.loads(line)
            header = record["header"]
            assert header["frame_id"] == "r_wrist"
            joints = {joint["name"]: joint for joint in record["joints"]}
            assert len(joints) == len(record["joints"]) == 21
            times.append(header["timestamp_us"])
            seqs.append(header["seq"])
            points.append([[joints[k]["pose"]["position"] for k in chain] for chain in MARKERS])
            confidences.append([min(joints[k]["confidence"] for k in chain) for chain in MARKERS])
            wrists.append(joints["wrist"]["pose"]["position"])
    times = np.asarray(times, dtype=np.int64)
    seqs = np.asarray(seqs)
    selected = times - times[0] >= trimmed_start * 1e6
    ticks = times[selected]
    q = np.asarray(points)[selected]
    wrist = np.asarray(wrists)[selected]
    conf = np.asarray(confidences)[selected]
    roots = q[:, 1:, 0].mean(axis=1)
    origin = np.median(roots, axis=0)
    scale = np.linalg.norm(origin - np.median(wrist, axis=0))
    length = np.linalg.norm(np.diff(q, axis=2), axis=-1)
    valid = np.isfinite(q).all(axis=(1, 2, 3)) & (conf >= .35).all(axis=1) & (length > .003).all(axis=(1, 2))
    canonical = (q-origin)[..., [0, 2, 1]] * [1, -1, 1]
    return {
        "raw_count": len(times), "ticks": ticks, "seq": seqs[selected], "p": canonical, "conf": conf,
        "roots_range_m": float(np.ptp(roots, axis=0).max()), "wrist_range_m": float(np.ptp(wrist, axis=0).max()),
        "scale_m": float(scale), "valid": valid,
    }


def check_archive(raw, path, zero):
    with np.load(path, allow_pickle=False) as arc:
        np.testing.assert_allclose(raw["p"], arc["points_m"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(raw["conf"], arc["confidence"], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(raw["seq"], arc["sequence"])
        np.testing.assert_array_equal(raw["valid"], arc["valid"])
        np.testing.assert_allclose((raw["ticks"]-zero)/1e6, arc["time_s"], rtol=0, atol=1e-9)
        assert abs(float(arc["palm_scale_m"])-raw["scale_m"]) < 1e-12


def main():
    todo = [(PILOT/"right-glove-take01.jsonl", PILOT/"human.npz", 39)]
    todo += [(S2/(k+".jsonl"), S2/"qc"/(k+".npz"), 0) for k in ("take01", "take02", "take03")]
    todo += [(f, S3/"qc"/(f.stem+".npz"), 0) for f in sorted(S3.glob("part*.jsonl"))]
    parts, rows = [], []
    for file, arc, trim in todo:
        raw = extract(file, trim)
        if file.parent == S3:
            parts.append(raw)
        zero = parts[0]["ticks"][0] if file.parent == S3 else raw["ticks"][0]
        check_archive(raw, arc, zero)
        rows.append({
            "file": file.name, "batch": file.parent.name, "raw": raw["raw_count"], "formal": len(raw["ticks"]),
            "formal_valid": int(raw["valid"].sum()), "formal_seconds": (raw["ticks"][-1]-raw["ticks"][0]).item()/1e6,
            "sdk_palm_origin_range_m": raw["roots_range_m"], "wrist_range_m": raw["wrist_range_m"],
        })
    clock = np.concatenate([x["ticks"] for x in parts])
    times = (clock-clock[0])/1e6
    assert len(np.unique(clock)) == len(clock) and np.all(np.diff(clock)>0)
    source = S3/"qc/all-motion.npz"
    m = load(V5/"data/manifest.json")
    assert sha(source) == m["source_sha256"]
    split_stats = {}
    memberships = np.zeros(len(times), dtype=int)
    with np.load(source, allow_pickle=False) as z:
        np.testing.assert_allclose(times, z["time_s"], rtol=0, atol=1e-9)
        for origin_key, archive_key in (("p","points_m"),("conf","confidence"),("valid","valid"),("seq","sequence")):
            np.testing.assert_array_equal(np.concatenate([x[origin_key] for x in parts]), z[archive_key])
        for role, entry in m["splits"].items():
            mask = (times >= entry["start_s"]) & ((times <= entry["end_s"]+1e-9) if role=="test" else (times < entry["end_s"]))
            memberships += mask
            file = V5/"data"/(role+".npz")
            assert sha(file)==entry["sha256"]
            with np.load(file, allow_pickle=False) as split:
                np.testing.assert_allclose(split["time_s"], times[mask], atol=1e-9, rtol=0)
                for key in ("points_m","sequence","valid","confidence"):
                    np.testing.assert_array_equal(z[key][mask], split[key])
                split_stats[role] = {"start": float(times[mask][0]),"end": float(times[mask][-1]),"n": int(mask.sum()),"valid": int(split["valid"].sum())}
    frozen = load(V5/"checkpoint/frozen.json")["files"]
    assert all(sha(V5/"checkpoint"/name)==digest for name,digest in frozen.items())
    old_frozen = load(V4/"evaluation.json")["frozen_checkpoint_sha256"]
    assert all(sha(V4/"checkpoint"/name)==digest for name,digest in old_frozen.items())
    assert sha(V5/"prior_v5.py") == load(V5/"checkpoint/config.json")["fit_script_sha256"]
    case = PILOT/"jacobi_case10.npz"
    run = DELIVERABLES.parent/"runs/mjwarp-wuji-cube-face-sm-cycle-g2u25-dev-20261116"
    with np.load(run/"trace.npz", allow_pickle=False) as trace, np.load(case, allow_pickle=False) as c:
        active = trace["active"][:,10].astype(bool)
        np.testing.assert_array_equal(np.flatnonzero(active), np.arange(len(c["time_s"])))
        np.testing.assert_array_equal(c["q_rad"],trace["q"][active,10,:12])
        np.testing.assert_allclose(trace["time"][active], c["time_s"]+.02, rtol=0, atol=1e-8)
        case_frames = int(active.sum())
    inventory = load(HERE.parent/"data_inventory.json")
    formal_seconds = sum(v["formal_seconds"] for v in rows[:4]) + times[-1]
    assert sum(v["formal"] for v in rows)==inventory["formal_motion_frames"]
    assert sum(v["formal_valid"] for v in rows)==inventory["formal_basic_valid_frames"]
    assert abs(formal_seconds - inventory["formal_motion_seconds"]) < 1e-6
    assert max(memberships) == 1
    save("ledger.json",{
        "files":rows,"formal_seconds":float(formal_seconds),"formal_frames":sum(v["formal"] for v in rows),
        "formal_valid":sum(v["formal_valid"] for v in rows),"s3_first_to_last_s":float(times[-1]),
        "splits":split_stats,"overlap_raw_frames":int((memberships>1).sum()),"guard_raw_frames":int((memberships==0).sum()),
        "sdk_palm_origin_all_chunks_static":all(v["sdk_palm_origin_range_m"]<1e-12 for v in rows),
        "case10_equal_archived_q_and_poststep_clock":True,"case10_frames":case_frames,"hash_verified_files_v5":len(frozen),"hash_verified_files_v4":len(old_frozen),
    })
    print("raw-to-archive checks passed", formal_seconds, split_stats, flush=True)


if __name__=="__main__":
    main()
