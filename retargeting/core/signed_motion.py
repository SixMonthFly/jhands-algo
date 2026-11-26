"""Signed anatomical marker measurements on SDK samples and the frozen Jacobi scene."""
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from records import HERE, S2, S3, V5, PILOT, DELIVERABLES, MARKERS, save

CACHE = HERE/"local-cache"
SCENE = DELIVERABLES.parent/"runs/mjwarp-wuji-cube-face-sm-cycle-g2u25-dev-20261116/cube-scene.xml"
ORDER = np.array([0,1,2,3,8,9,10,11,4,5,6,7])
DIGITS = ("thumb","index","middle")


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def projected_bend(first, second, axis):
    # Signed rotation between two bone vectors about their actual hinge axis.
    sine = np.sum(axis*np.cross(first,second),axis=-1)
    cosine = np.sum(first*second,axis=-1) - np.sum(first*axis,axis=-1)*np.sum(second*axis,axis=-1)
    return np.arctan2(sine, cosine)


def read_quats(paths):
    axes = []
    for path in paths:
        with path.open("rb") as stream:
            for line in stream:
                frame = json.loads(line)
                entries = {j["name"]:j["pose"]["orientation"] for j in frame["joints"]}
                axes.append([[entries[chain[2]][k] for k in "xyzw"] for chain in MARKERS])
    q = np.asarray(axes, dtype=np.float64)
    world = Rotation.from_quat(q.reshape(-1,4)).apply([1.,0.,0.]).reshape(len(q),3,3)
    # Same proper rotation as the positions: +x thumbward, +y distal, +z palmar.
    return world[..., [0,2,1]] * [1,-1,1]


def extract(name, files, archive):
    axes = read_quats(files)
    with np.load(archive, allow_pickle=False) as z:
        t, xyz, valid, scale = z["time_s"], z["points_m"], z["valid"], float(z["palm_scale_m"])
    assert len(t)==len(axes)
    bone = unit(np.diff(xyz, axis=2))
    hinges = projected_bend(bone[:,:,:-1],bone[:,:,1:],axes[:,:,None])
    unsigned = np.arctan2(np.linalg.norm(np.cross(bone[:,:,:-1],bone[:,:,1:]),axis=-1),np.sum(bone[:,:,:-1]*bone[:,:,1:],axis=-1))
    plane_residual = np.abs(np.sum(bone*axes[:,:,None],axis=-1))
    gaps = np.linalg.norm(xyz[:,0:1,-1]-xyz[:,1:,-1],axis=-1)/scale
    np.savez_compressed(CACHE/(name+".npz"),t=t,valid=valid,proximal=bone[:,:,0],hinges=hinges,gaps=gaps)
    return {"frames":len(t),"p99_hinge_axis_dot_bone":float(np.quantile(plane_residual[valid], .99)),
            "max_hinge_axis_dot_bone":float(plane_residual[valid].max()),
            "p99_abs_signed_magnitude_vs_3d_angle_deg":float(np.rad2deg(np.quantile(abs(abs(hinges[valid])-unsigned[valid]),.99))),
            "signed_1st_and_2nd_min_deg_per_finger":np.rad2deg(hinges[valid].min(0)).tolist(),
            "fraction_bend_below_minus_3deg_per_finger_joint":(np.rad2deg(hinges[valid])<-3).mean(axis=0).tolist()}


def robot_offsets(model=None):
    import sys
    sys.path.insert(0,str(PILOT))
    from export_jacobi import BASE_TO_RIGHT_PALM, Landmarks
    if model is None:
        model = mj.MjModel.from_xml_path(str(SCENE))
    data = mj.MjData(model)
    markers = Landmarks(model)
    idx=ORDER.reshape(3,4)[:,2:]
    def angle(q):
        data.qpos[:12]=q
        mj.mj_kinematics(model,data)
        points=markers.measure(data)
        bone=unit(np.diff(points,axis=1))
        axes=(data.xaxis[idx]@data.xmat[markers.base].reshape(3,3))*BASE_TO_RIGHT_PALM
        return projected_bend(bone[:,:-1],bone[:,1:],axes)
    offsets=angle(np.zeros(12))
    rng=np.random.default_rng(881);error=0
    for q in rng.uniform(model.jnt_range[:12,0],model.jnt_range[:12,1],(100,12)):
        exact=angle(q)
        error=max(error,float(np.max(np.abs(exact-(q[idx]+offsets)))))
    assert error<1e-9,error
    return offsets, error


def main():
    CACHE.mkdir(exist_ok=True)
    data={}
    for name in ("take01","take02","take03"):
        data[name]=extract(name,[S2/(name+".jsonl")],S2/"qc"/(name+".npz"))
    data["session3"]=extract("session3", sorted(S3.glob("part*.jsonl")), S3/"qc/all-motion.npz")
    offsets,err=robot_offsets()
    save("signed-geometry-check.json",{"glove":data,"robot_rest_signed_offset_deg":np.rad2deg(offsets).tolist(),
           "max_angle_offset_identity_error_rad_100_random_native_poses":err})
    print(json.dumps(data["session3"]),np.rad2deg(offsets))


if __name__=="__main__":
    main()
