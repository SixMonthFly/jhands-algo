"""Offline, constrained retargeting onto the archived Jacobi v1028 hand."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

OUT = Path(__file__).resolve().parent
PILOT = OUT.parent
sys.path.insert(0, str(PILOT))
from export_jacobi import BASE_TO_RIGHT_PALM, DEFAULT_SOURCE, Landmarks, gestures

SCENE = DEFAULT_SOURCE / "cube-scene.xml"
PREVIEW = [
    ("open", 41.64, 42.78, 0),
    ("index", 7.70, 8.65, 1),
    ("middle", 8.64, 9.65, 2),
]
ALTERNATE = [
    ("open", 56.75, 57.90, 0),
    ("index", 11.10, 12.05, 1),
    ("middle", 10.44, 11.18, 2),
]
FINGER_NAMES = ["thumb", "index", "middle"]


def unit(v):
    return v / np.maximum(1e-7, np.linalg.norm(v, axis=-1, keepdims=True))


def angle_corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-6 and np.std(y) > 1e-6 else None


def human_samples(a, b, stages):
    with np.load(PILOT / "human.npz", allow_pickle=False) as source:
        tsh = source["time_s"]
        p = source["points_m"]
        source_scale = float(source["palm_scale_m"])
        usable = source["valid"].astype(bool)
    time = np.arange(a, min(b, tsh[-1]) + 1e-6, 1.0 / stages)
    shape = p.shape[1:]
    polished = gaussian_filter1d(p.reshape(len(p), -1), sigma=.018 / np.median(np.diff(tsh)), axis=0)
    raw = np.column_stack([np.interp(time, tsh, col) for col in p.reshape(len(p), -1).T]).reshape((-1,) + shape)
    filtered = np.column_stack([np.interp(time, tsh, col) for col in polished.T]).reshape((-1,) + shape)
    near = np.abs(tsh[np.searchsorted(tsh, time, side="left").clip(0, len(tsh)-1)] - time)
    supporting = np.interp(time, tsh, usable.astype(float)) >= .999
    return time, raw, filtered, source_scale, supporting & (near < .02)


class Solver:
    def __init__(self, pair_weight, smooth, penetration_tol, thumb_extension, max_step_rad=None):
        self.model = mj.MjModel.from_xml_path(str(SCENE))
        self.data = mj.MjData(self.model)
        self.landmarks = Landmarks(self.model)
        self.robot_scale = self.landmarks.scale_m
        self.lo, self.hi = self.model.jnt_range[:12].T.copy()
        self.qpos0 = self.data.qpos.copy()
        cube = self.model.joint("cube_freejoint").qposadr[0]
        self.qpos0[cube + 2] = -2.
        for name in ("cube", "support"):
            mask = self.model.geom_bodyid == self.model.body(name).id
            self.model.geom_contype[mask] = 0
            self.model.geom_conaffinity[mask] = 0
        self.data.qpos[:] = self.qpos0
        mj.mj_kinematics(self.model, self.data)
        self.transform = self.data.xmat[self.landmarks.base].reshape(3, 3) * BASE_TO_RIGHT_PALM
        self.geom_bodies = self.model.geom_bodyid.copy()
        self.joint_affects = np.zeros((self.model.nbody, 12), dtype=bool)
        for body in range(1, self.model.nbody):
            lineage = [body]
            while lineage[-1] > 0:
                lineage.append(self.model.body_parentid[lineage[-1]])
            self.joint_affects[body] = np.isin(self.model.jnt_bodyid[:12], lineage)
        self.point_bodies = np.column_stack((self.landmarks.chains, self.landmarks.tips)).reshape(-1)
        self.default_seed = gestures()[1][0]
        self.pair_weight = pair_weight
        self.smooth = smooth
        self.penetration_tol = penetration_tol
        self.thumb_extension = thumb_extension
        self.max_step_rad = max_step_rad

    def world_points(self, q):
        d = self.data
        d.qpos[:12] = q
        mj.mj_kinematics(self.model, d)
        offsets = np.einsum("fij,fj->fi", d.xmat[self.landmarks.tips].reshape(3,3,3),self.landmarks.offsets_m)
        tips = d.xpos[self.landmarks.tips] + offsets
        return np.concatenate([d.xpos[self.landmarks.chains], tips[:,None]],axis=1)

    def jac_world(self, points, bodies):
        d = self.data
        return np.cross(d.xaxis[None,:12], points[:,None] - d.xanchor[None,:12]) * self.joint_affects[bodies,:,None]

    def solve(self, human, previous, seed, emphasis, smooth=.12, nfev=15, penalties=(), local=True):
        h = human
        b = np.diff(h, axis=1)
        hv = unit(b)
        hev = np.linalg.norm(h[:,-1]-h[:,0],axis=-1) / np.linalg.norm(b,axis=-1).sum(axis=-1)
        hpair = h[1:,-1] - h[0:1,-1]
        hdist = np.linalg.norm(hpair, axis=-1)
        # The thumb base and link lengths differ substantially. Match the distal
        # direction and inter-finger displacement; soften proximal thumb orientation.
        bone_weight = np.array([[.18,.29,.34],[.63,.47,.34],[.63,.47,.34]])
        tip_weight = np.array([[.16],[.46],[.46]])
        pair_gain = self.pair_weight * (.40 + .75*np.exp(-np.square(hdist/.50)))
        if emphasis:
            pair_gain[emphasis-1] += .37*np.exp(-np.square(hdist[emphasis-1]/.62))
        dist_gain = pair_gain*.65
        ext_gain = np.array([self.thumb_extension]+[.22 if self.thumb_extension else 0.] * 2)
        cache = [None, None, None]

        def call(q):
            if cache[0] is not None and np.array_equal(cache[0], q):
                return cache[1], cache[2]
            world = self.world_points(q)
            p = self.landmarks.palm_local(self.data, world) - self.landmarks.origin_m
            p /= self.robot_scale
            j = (self.jac_world(world.reshape(-1,3),self.point_bodies) @ self.transform).reshape(3,4,12,3)
            j = j.swapaxes(-1,-2) / self.robot_scale
            bv = np.diff(p,axis=1)
            db = np.diff(j,axis=1)
            bnorm = np.maximum(1e-7,np.linalg.norm(bv,axis=-1))
            u = bv/bnorm[...,None]
            dlength = np.einsum("fbi,fbij->fbj",u,db)
            du = (db-u[...,None]*dlength[...,None,:])/bnorm[...,None,None]
            chord = p[:,-1]-p[:,0]
            chord_norm = np.linalg.norm(chord,axis=-1)
            length_total = bnorm.sum(axis=1)
            extension = chord_norm / length_total
            dchord_norm = np.einsum("fi,fij->fj",unit(chord),j[:,-1]-j[:,0])
            dext = (dchord_norm-extension[:,None]*dlength.sum(axis=1))/length_total[:,None]
            dv = p[1:,-1]-p[0:1,-1]
            jdv = j[1:,-1]-j[0:1,-1]
            dn = np.maximum(1e-7,np.linalg.norm(dv,axis=-1))
            jdn = np.einsum("fi,fij->fj",dv,jdv)/dn[:,None]
            r = [
                ((u-hv)*bone_weight[...,None]).reshape(-1),
                ((p[:,-1]-h[:,-1])*tip_weight).reshape(-1),
                ((dv-hpair)*pair_gain[:,None]).reshape(-1),
                (dn-hdist)*dist_gain,
                (extension-hev)*ext_gain,
                smooth*(q-previous),
                .028*(q-self.default_seed),
            ]
            jac = [
                (du*bone_weight[...,None,None]).reshape(-1,12),
                (j[:,-1]*tip_weight[...,None]).reshape(-1,12),
                (jdv*pair_gain[:,None,None]).reshape(-1,12),
                jdn*dist_gain[:,None],
                dext*ext_gain[:,None],
                smooth*np.eye(12), .028*np.eye(12),
            ]
            if penalties:
                linear = np.concatenate([d0 + grad @ (q-q0) for q0,d0,grad in penalties])
                grad = np.concatenate([grad for _,_,grad in penalties])
                r.append(90*np.maximum(0, .00045-linear))
                jac.append(-90*(linear<.00045)[:,None]*grad)
            cache[:] = [q.copy(), np.concatenate(r), np.concatenate(jac)]
            return cache[1], cache[2]

        lo, hi = self.lo, self.hi
        if local and self.max_step_rad is not None:
            lo = np.maximum(lo,previous-self.max_step_rad)
            hi = np.minimum(hi,previous+self.max_step_rad)
        return least_squares(lambda q:call(q)[0],np.clip(seed,lo+1e-8,hi-1e-8),
                             jac=lambda q:call(q)[1],bounds=(lo,hi),max_nfev=nfev,
                             ftol=.003,xtol=.001,gtol=1e-4)

    def contacts(self, q):
        self.data.qpos[:12] = q
        mj.mj_forward(self.model, self.data)
        con = self.data.contact
        if self.data.ncon == 0:
            return np.zeros(0), np.empty((0,12))
        gids = con.geom[:self.data.ncon]
        b = self.geom_bodies[np.maximum(gids,0)]
        select = (b.min(axis=1) >= self.landmarks.base) & (b.max(axis=1) <= self.landmarks.tips.max())
        depths = con.dist[:self.data.ncon][select].copy()
        pts = con.pos[:self.data.ncon][select].copy()
        direction = con.frame[:self.data.ncon][select,:3].copy()
        v1 = self.jac_world(pts,b[select,0]); v2 = self.jac_world(pts,b[select,1])
        return depths,np.einsum("fji,fi->fj",v2-v1,direction)

    def safe_solve(self, human, previous, seed, emphasis, smooth=.12, nfev=13, local=True):
        sol = self.solve(human, previous, seed, emphasis, smooth, nfev, local=local)
        q = sol.x
        contacts, grad = self.contacts(q)
        if not len(contacts) or contacts.min() >= -self.penetration_tol:
            return sol, float(max(0.,-contacts.min(initial=0.)))
        penalties = []
        best, best_depth = sol, -contacts.min()
        for _ in range(4):
            near = contacts < .0005
            if not near.any():
                break
            penalties.append((q.copy(),contacts[near],grad[near]))
            sol = self.solve(human, previous, q, emphasis, smooth, nfev, penalties, local=local)
            q = sol.x
            contacts, grad = self.contacts(q)
            depth = float(max(0.,-contacts.min(initial=0.)))
            if depth < best_depth:
                best,best_depth = sol,depth
            if depth <= self.penetration_tol:
                break
        return best,best_depth

    def seed(self,human,emphasis):
        rng = np.random.default_rng(38890)
        starts = np.vstack((self.default_seed, gestures()[1][[2,4,7,11,14,16]],
                            rng.uniform(self.lo,self.hi,(10,12))))
        sol = [self.safe_solve(human,self.default_seed,s,emphasis,smooth=.025,nfev=30,local=False) for s in starts]
        return min(sol,key=lambda entry:entry[0].cost+1000*entry[1]**2)[0].x

    def run(self,time,human,heading):
        actual = np.zeros((len(time),3,4,3))
        result = np.zeros((len(time),12))
        depth = np.zeros(len(time))
        q = self.seed(human[0],heading)
        for i,h in enumerate(human):
            qnew,depth[i] = self.safe_solve(h,q,q,heading,smooth=self.smooth,nfev=15)
            q = qnew.x
            result[i] = q
            world = self.world_points(q)
            actual[i] = self.landmarks.palm_local(self.data,world)-self.landmarks.origin_m
        return actual,result,depth


def metrics(times, raw, robot, q, depth, native_scale, robot_scale, lo, hi):
    norm_h, norm_r = raw/native_scale, robot/robot_scale
    hunit, runit = unit(np.diff(norm_h,axis=2)), unit(np.diff(norm_r,axis=2))
    dir_deg = np.rad2deg(np.arccos(np.clip(np.sum(hunit*runit,axis=-1),-1,1)))
    hg = np.linalg.norm(raw[:,1:,-1]-raw[:,0:1,-1],axis=-1)
    rg = np.linalg.norm(robot[:,1:,-1]-robot[:,0:1,-1],axis=-1)
    relative_error = np.abs(hg/native_scale-rg/robot_scale) * robot_scale
    hex = np.linalg.norm(norm_h[:,:,-1]-norm_h[:,:,0],axis=-1)/np.linalg.norm(np.diff(norm_h,axis=2),axis=-1).sum(axis=-1)
    rex = np.linalg.norm(norm_r[:,:,-1]-norm_r[:,:,0],axis=-1)/np.linalg.norm(np.diff(norm_r,axis=2),axis=-1).sum(axis=-1)
    return {
        "n_frames": len(times),
        "body_direction_p50_deg_by_digit": {k:float(v) for k,v in zip(FINGER_NAMES,np.median(dir_deg,axis=(0,2)))},
        "extension_correlation_by_digit": {k:angle_corr(hex[:,i],rex[:,i]) for i,k in enumerate(FINGER_NAMES)},
        "human_tip_min_mm": (1000*hg.min(axis=0)).tolist(),
        "jacobi_tip_min_mm": (1000*rg.min(axis=0)).tolist(),
        "jacobi_tip_mm_at_human_min": [float(1000*rg[int(np.argmin(hg[:,j])),j]) for j in range(2)],
        "aperture_error_robot_mm_p50": (1000*np.median(relative_error,axis=0)).tolist(),
        "aperture_error_robot_mm_p90": (1000*np.quantile(relative_error,.9,axis=0)).tolist(),
        "aperture_min_time_human_s": times[np.argmin(hg,axis=0)].tolist(),
        "aperture_min_time_jacobi_s": times[np.argmin(rg,axis=0)].tolist(),
        "aperture_correlation": [angle_corr(hg[:,j],rg[:,j]) for j in range(2)],
        "self_penetration_max_mm": float(1000*depth.max()),
        "frames_self_penetration_over_0p5mm": int((depth>.0005).sum()),
        "max_outside_joint_limits_deg": float(np.rad2deg(max(0.,max((lo-q).max(),(q-hi).max())))),
        "max_target_speed_deg_per_s": float(np.rad2deg(np.max(np.abs(np.diff(q,axis=0)/np.diff(times)[:,None])))),
    }


def write(args):
    stages = args.hz
    rate_step = None if args.max_target_speed is None else np.deg2rad(args.max_target_speed)/stages
    solver = Solver(args.pair_weight,args.smooth,args.penetration_tol_mm/1000,args.thumb_extension,rate_step)
    intervals = (ALTERNATE if args.alternate else PREVIEW) if args.mode=="preview" else [("all",0,71.0,0)]
    real, source_times, normalized_time, points, poses, ids, valid, depths = [],[],[],[],[],[],[],[]
    descriptions = []
    for name,a,b,heading in intervals:
        time, human, smooth, source_scale, human_valid = human_samples(a,b,stages)
        robot, q, depth = solver.run(time,smooth/source_scale,heading)
        acceptable = human_valid & (depth <= .0005) & np.all(np.isfinite(q),axis=1)
        real.append(human)
        source_times.append(time)
        normalized_time.append(time-time[0] if args.mode=="preview" else time)
        points.append(robot)
        poses.append(q)
        ids.append(np.full(len(time),name))
        valid.append(acceptable)
        depths.append(depth)
        descriptions.append({"id":name,"start_s":a,"end_s":float(time[-1]),"target":heading,
                             "valid_frames":int(acceptable.sum()),
                             **metrics(time,human,robot,q,depth,source_scale,solver.robot_scale,solver.lo,solver.hi)})
        print(json.dumps(descriptions[-1],ensure_ascii=False),flush=True)
    dst = OUT/(("retarget_clips" if args.mode=="preview" else "retarget_all")+args.suffix+".npz")
    np.savez_compressed(dst,time_s=np.concatenate(normalized_time),source_time_s=np.concatenate(source_times),
                        segment_id=np.concatenate(ids),points_m=np.concatenate(points),q_rad=np.concatenate(poses),
                        palm_scale_m=solver.robot_scale,human_points_m=np.concatenate(real),human_palm_scale_m=source_scale,
                        valid=np.concatenate(valid),self_penetration_m=np.concatenate(depths))
    diag = OUT/(("retarget_metrics" if args.mode=="preview" else "retarget_all_metrics")+args.suffix+".json")
    diag.write_text(json.dumps({"scene":str(SCENE),"fps":stages,"method":"constrained offline IK, original model joint range",
                               "marker_gap_is_not_surface_contact":True,"clips":descriptions,
                               "pair_weight":args.pair_weight,"smooth":args.smooth,
                               "thumb_extension":args.thumb_extension,
                               "kinematic_speed_budget_deg_per_s":args.max_target_speed,
                               "penetration_tol_mm":args.penetration_tol_mm},indent=2),
                    encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("preview","all"),default="preview")
    p.add_argument("--hz",type=int,default=50)
    p.add_argument("--alternate",action="store_true")
    p.add_argument("--pair-weight",type=float,default=1.)
    p.add_argument("--smooth",type=float,default=.12)
    p.add_argument("--penetration-tol-mm",type=float,default=.25)
    p.add_argument("--thumb-extension",type=float,default=0.)
    p.add_argument("--max-target-speed",type=float)
    p.add_argument("--suffix",default="")
    write(p.parse_args())
