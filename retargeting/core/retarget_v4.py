"""Morphology-aware bounded Jacobi retargeting. Offline; no hardware access."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import mujoco as mj

HERE=Path(__file__).resolve().parent
SESSION=HERE.parent
PILOT=SESSION.parent/'hand-motion-likeness-pilot-20261116'
sys.path[:0]=[str(PILOT/'retarget-preview'),str(SESSION/'scorer-v3'/'retarget'),str(PILOT)]
import retarget as old
from retarget_session2_take import resample_input
from validate_retarget_physics import integrate_clip

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

class Solver(old.Solver):
    def __init__(self,pair=1.5,smooth=.20,speed=300):
        super().__init__(pair,smooth,.00005,0,np.deg2rad(speed)/50)
        p=self.landmarks.measure(self.data)/self.robot_scale
        self.lengths=np.linalg.norm(np.diff(p,axis=1),axis=-1)
        self.bases=p[:,0].copy()
        self.older=None
        # Stay 1 degree inside model stops: this is not a change to native limits.
        self.native_lo,self.native_hi=self.lo.copy(),self.hi.copy()
        self.lo+=np.deg2rad(3); self.hi-=np.deg2rad(3)

    def solve(self,human,previous,seed,emphasis,smooth=.14,nfev=18,penalties=(),local=True):
        hv=old.unit(np.diff(human,axis=1))
        # Each human bone direction is placed on the actual robot bone length/base.
        # Absolute human fingertip positions cannot be scaled by one palm ratio.
        target=self.bases[:,None]+np.concatenate([np.zeros((3,1,3)),np.cumsum(hv*self.lengths[...,None],axis=1)],axis=1)
        hb=np.diff(human,axis=1)
        hlen=np.linalg.norm(hb,axis=-1).sum(axis=1)
        rlen=self.lengths.sum(axis=1)
        hp=human[1:,-1]-human[0:1,-1]
        # Preserve inter-tip displacement scaled by pair mean finger reach.
        # Close pinches get more weight, far-open poses prefer anatomical direction.
        ratio=(rlen[1:]+rlen[0])/(hlen[1:]+hlen[0])
        # At pinch distances avoid expanding gaps simply because the long fingers
        # are larger; smoothly use the reach ratio only away from opposition.
        blend=1-np.exp(-(np.linalg.norm(hp,axis=-1)/.45)**2)
        hp=hp*(1+blend*(ratio-1))[:,None]
        hd=np.linalg.norm(hp,axis=-1)
        gain=self.pair_weight*(.35+.85*np.exp(-(hd/.55)**2))
        bw=np.array([[.18,.29,.34],[.63,.47,.34],[.63,.47,.34]])
        tw=np.array([.25,.25,.25])
        hdot=np.sum(hv[:,:-1]*hv[:,1:],axis=-1)
        hext=np.linalg.norm(human[:,-1]-human[:,0],axis=-1)/hlen
        # Preserve each interphalangeal bend, not only the end-to-end extension:
        # identical extension can hide very different distributions of joint bend.
        bend_target=np.arccos(np.clip(hdot,-1.,1.)).ravel()
        bend_ids=np.array([2,3,10,11,6,7])
        older=self.older if local and self.older is not None else previous
        cache=[None,None,None]
        def call(q):
            if cache[0] is not None and np.array_equal(cache[0],q):return cache[1],cache[2]
            world=self.world_points(q)
            p=(self.landmarks.palm_local(self.data,world)-self.landmarks.origin_m)/self.robot_scale
            j=(self.jac_world(world.reshape(-1,3),self.point_bodies)@self.transform).reshape(3,4,12,3).swapaxes(-1,-2)/self.robot_scale
            b=np.diff(p,axis=1); db=np.diff(j,axis=1)
            n=np.maximum(np.linalg.norm(b,axis=-1),1e-7);u=b/n[...,None]
            du=(db-u[...,None]*np.einsum('fbi,fbij->fbj',u,db)[...,None,:])/n[...,None,None]
            dv=p[1:,-1]-p[0:1,-1]; jdv=j[1:,-1]-j[0:1,-1]
            dn=np.maximum(np.linalg.norm(dv,axis=-1),1e-7)
            jdn=np.einsum('fi,fij->fj',dv,jdv)/dn[:,None]
            dot=np.sum(u[:,:-1]*u[:,1:],axis=-1)
            jdot=np.einsum('fbi,fbij->fbj',u[:,:-1],du[:,1:])+np.einsum('fbi,fbij->fbj',u[:,1:],du[:,:-1])
            chord=p[:,-1]-p[:,0];total=n.sum(axis=1);ext=np.linalg.norm(chord,axis=-1)/total
            jext=(np.einsum('fi,fij->fj',old.unit(chord),j[:,-1]-j[:,0])-ext[:,None]*np.einsum('fbi,fbij->fbj',u,db).sum(axis=1))/total[:,None]
            eg=np.array([3.,.22,.22])
            r=[((u-hv)*bw[...,None]).ravel(),((p[:,-1]-target[:,-1])*tw[:,None]).ravel(),
               ((dv-hp)*gain[:,None]).ravel(),(dn-hd)*gain*.5,
               ((dot-hdot)*.10).ravel(),(ext-hext)*eg,.65*(q[bend_ids]-bend_target),smooth*(q-previous),.06*(q-2*previous+older),.012*(q-self.default_seed)]
            jac=[(du*bw[...,None,None]).reshape(-1,12),(j[:,-1]*tw[:,None,None]).reshape(-1,12),
                 (jdv*gain[:,None,None]).reshape(-1,12),jdn*gain[:,None]*.5,(jdot*.10).reshape(-1,12),jext*eg[:,None],
                 .65*np.eye(12)[bend_ids],smooth*np.eye(12),.06*np.eye(12),.012*np.eye(12)]
            if penalties:
                linear=np.concatenate([d+g@(q-q0) for q0,d,g in penalties]);grad=np.concatenate([g for _,_,g in penalties])
                r.append(500*np.maximum(0,.0006-linear));jac.append(-500*(linear<.0006)[:,None]*grad)
            cache[:]=[q.copy(),np.concatenate(r),np.concatenate(jac)]
            return cache[1],cache[2]
        lo,hi=self.lo,self.hi
        if local:
            lo=np.maximum(lo,previous-self.max_step_rad);hi=np.minimum(hi,previous+self.max_step_rad)
        return least_squares(lambda q:call(q)[0],np.clip(seed,lo+1e-8,hi-1e-8),jac=lambda q:call(q)[1],bounds=(lo,hi),
                             max_nfev=nfev,ftol=.001,xtol=.0005,gtol=1e-5)

    def run(self,t,human,heading=0):
        q=self.seed(human[0],0);self.older=q.copy()
        ds,_=self.contacts(q)
        if ds.min(initial=0)<-.00005:
            # A known open geometry is a fallback for an infeasible initial IK seed.
            q=np.clip(np.zeros(12),self.lo,self.hi);self.older=q.copy()
        result=[];points=[];depth=[]
        for h in human:
            sol,dep=self.safe_solve(h,q,q,0,self.smooth,nfev=18)
            proposed=sol.x
            if dep>.00005:
                lo,hi=0.,1.
                for _ in range(11):
                    frac=(lo+hi)/2;d,_=self.contacts(q+frac*(proposed-q))
                    if d.min(initial=0)>=-.00005:lo=frac
                    else:hi=frac
                proposed=q+lo*(proposed-q)
                d,_=self.contacts(proposed);dep=max(0.,-d.min(initial=0))
            self.older=q.copy();q=proposed
            result.append(q.copy());depth.append(dep)
            points.append((self.landmarks.palm_local(self.data,self.world_points(q))-self.landmarks.origin_m).copy())
        return np.array(points),np.array(result),np.array(depth)

def gates(z):
    qerr=np.abs(np.rad2deg(z['q_rad']-z['q_target_rad']))
    tiperr=1000*np.linalg.norm(z['points_m'][:,:,-1]-z['reference_points_m'][:,:,-1],axis=-1)
    h=z['human_points_m']/float(z['human_palm_scale_m']);r=z['points_m']/float(z['palm_scale_m'])
    gap=np.abs(np.linalg.norm(h[:,1:,-1]-h[:,0:1,-1],axis=-1)-np.linalg.norm(r[:,1:,-1]-r[:,0:1,-1],axis=-1)).max(axis=1)
    g=dict(input=z['valid'].astype(bool),collision=z['physical_depth_mm']<=.25,limits=z['physical_limit_excess_deg']<=.05,
           tip_tracking=tiperr.max(axis=1)<=25,angle_tracking=np.percentile(qerr,95,axis=1)<=10,gap=gap<=.35)
    return np.logical_and.reduce(list(g.values())),g

def run(take,tag,pair=1.5,speed=400,interval=None):
    start=time.perf_counter();t,raw,h,hs,valid,stats=resample_input(SESSION/'qc'/f'{take}.npz',50)
    if interval:
        choose=(t>=interval[0])&(t<=interval[1]);t,raw,h,valid=t[choose],raw[choose],h[choose],valid[choose]
    solver=Solver(pair=pair,speed=speed)
    rp,q,depth=solver.run(t,h/hs)
    actual,points,qd,pdepth,limit,info=integrate_clip(solver.model,solver.landmarks,t,q)
    z=dict(time_s=t,source_time_s=t,segment_id=np.full(len(t),take),points_m=points,q_rad=actual,qd_rad_s=qd,
           q_target_rad=q,reference_points_m=rp,palm_scale_m=solver.robot_scale,human_points_m=raw,human_palm_scale_m=hs,
           valid=valid&(depth<=.0005),self_penetration_m=depth,physical_depth_mm=pdepth,physical_limit_excess_deg=limit)
    good,g=gates(z);z['admitted']=good
    dest=HERE/'retarget'/tag;dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/f'{take}.npz',**z)
    metric=old.metrics(t,raw,points,actual,pdepth/1000,hs,solver.robot_scale,solver.native_lo,solver.native_hi)
    result=dict(take=take,tag=tag,pair=pair,speed=speed,frames=len(t),accepted=int(good.sum()),gate_counts={k:int(v.sum()) for k,v in g.items()},
                physical=info,metrics=metric,wall_seconds=time.perf_counter()-start,scene_sha256=sha(old.SCENE),source_sha256=sha(SESSION/'qc'/f'{take}.npz'),solver_sha256=sha(__file__))
    (dest/f'{take}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--take',required=True);a.add_argument('--tag',required=True)
    a.add_argument('--pair',type=float,default=1.5);a.add_argument('--speed',type=float,default=400);a.add_argument('--interval',nargs=2,type=float)
    x=a.parse_args();run(x.take,x.tag,x.pair,x.speed,x.interval)
