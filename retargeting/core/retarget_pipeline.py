"""Split-safe retargeting and unchanged PD replay; original sensor data retained."""
from pathlib import Path
import argparse,json,sys,time,hashlib
import numpy as np
from scipy.ndimage import gaussian_filter1d
HERE=Path(__file__).resolve().parent
V4=HERE.parent/'hand-motion-likeness-session2-20261116/scorer-v4'
sys.path.insert(0,str(V4));import retarget_v4 as prior

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

class MorphologySolver(prior.Solver):
    def __init__(self,radius,opposition_priority=False,close_boost=False):
        super().__init__(pair=1.8,smooth=.20,speed=400);self.radius=radius;self.opposition_priority=opposition_priority;self.close_boost=close_boost

    def solve(self,human,*args,**kwargs):
        # Preserve all within-finger geometry. Only the inter-tip target changes:
        # far-open poses use robot-length forward directions, close opposition
        # smoothly approaches the human relative-tip objective used by V4.
        hv=prior.old.unit(np.diff(human,axis=1));hlen=np.linalg.norm(np.diff(human,axis=1),axis=-1).sum(axis=1)
        rlen=self.lengths.sum(axis=1);ratio=(rlen[1:]+rlen[0])/(hlen[1:]+hlen[0])
        target=self.bases+np.sum(hv*self.lengths[...,None],axis=1)
        hpair=human[1:,-1]-human[0:1,-1];hd=np.linalg.norm(hpair,axis=-1)
        close=hpair*(1+(1-np.exp(-(hd/.45)**2))*(ratio-1))[:,None]
        weight=np.exp(-(hd/self.radius)**4)
        desired=weight[:,None]*close+(1-weight[:,None])*(target[1:]-target[0:1])
        # Invert V4's scalar pair-size transform, so its unchanged analytic IK
        # objective receives exactly the desired target, not a double scaling.
        raw=desired.copy()
        for _ in range(12):
            distance=np.linalg.norm(raw,axis=-1)
            raw=desired/(1+(1-np.exp(-(distance/.45)**2))*(ratio-1))[:,None]
        adjusted=human.copy();adjusted[1:]+= (raw-hpair)[:,None,:]
        saved=self.pair_weight
        if self.opposition_priority:
            contact=np.exp(-(hd.min()/.30)**4)
            near=np.exp(-(hd/.35)**4)
            self.pair_weight=1.8*(1-.95*contact*(1-near))
        if self.close_boost:
            self.pair_weight=1.8*(1+4*np.exp(-(hd/.35)**4))
        try:return super().solve(adjusted,*args,**kwargs)
        finally:self.pair_weight=saved

def segments(role):
    cfg=json.loads((HERE/'protocol.json').read_text());sources=['train','validation'] if role=='development' else [role]
    result=[];count=0;valid_count=0
    for source in sources:
        z=np.load(HERE/'data'/f'{source}.npz');t=z['time_s'];p=z['points_m'];valid=z['valid'].copy();hs=float(z['palm_scale_m'])
        chosen=np.ones(len(t),bool)
        if role=='development':
            chosen=np.logical_or.reduce([(t>=a)&(t<b) for a,b in cfg['development_retarget_clips_seconds']])
        count+=int(chosen.sum());valid_count+=int((chosen&valid).sum());valid&=chosen
        cuts=np.r_[0,np.flatnonzero(~valid[1:]|~valid[:-1]|(np.diff(t)>.025))+1,len(t)]
        for a,b in zip(cuts[:-1],cuts[1:]):
            if not valid[a] or t[b-1]-t[a]<.5:continue
            grid=np.arange(np.ceil(t[a]*50-1e-8),np.floor(t[b-1]*50+1e-8)+1)/50
            packed=p[a:b].reshape(b-a,-1);smooth=gaussian_filter1d(packed,.018/np.median(np.diff(t[a:b])),axis=0,mode='nearest')
            interp=lambda values:np.column_stack([np.interp(grid,t[a:b],col) for col in values.T]).reshape(-1,3,4,3)
            result.append((source,grid,interp(packed),interp(smooth),hs))
    return result,{'source_frames':count,'sensor_valid_source_frames':valid_count,'valid_contiguous_segments':len(result)}

def run_ik(solver,t,h,label):
    q=solver.seed(h[0],0);solver.older=q.copy();depth,_=solver.contacts(q)
    if depth.min(initial=0)<-.00005:q=np.clip(np.zeros(12),solver.lo,solver.hi);solver.older=q.copy()
    qs=[];ps=[];depths=[];guards=0
    for i,hp in enumerate(h):
        sol,dep=solver.safe_solve(hp,q,q,0,solver.smooth,nfev=18);proposed=sol.x
        if dep>.00005:
            guards+=1;lo,hi=0.,1.
            for _ in range(11):
                f=(lo+hi)/2;dd,_=solver.contacts(q+f*(proposed-q))
                if dd.min(initial=0)>=-.00005:lo=f
                else:hi=f
            proposed=q+lo*(proposed-q);dd,_=solver.contacts(proposed);dep=max(0.,-dd.min(initial=0))
        solver.older=q.copy();q=proposed
        qs.append(q.copy());ps.append((solver.landmarks.palm_local(solver.data,solver.world_points(q))-solver.landmarks.origin_m).copy());depths.append(dep)
        if i and i%2500==0:print(label,f'IK {i}/{len(h)}',flush=True)
    return np.array(ps),np.array(qs),np.array(depths),guards

def fidelity(h,p,hs,rs):
    def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
    hu=unit(np.diff(h,axis=2));pu=unit(np.diff(p,axis=2))
    direction=np.rad2deg(np.arccos(np.clip(np.sum(hu*pu,axis=-1),-1,1)))
    hb=np.arccos(np.clip(np.sum(hu[:,:,:-1]*hu[:,:,1:],axis=-1),-1,1));pb=np.arccos(np.clip(np.sum(pu[:,:,:-1]*pu[:,:,1:],axis=-1),-1,1))
    bend=np.rad2deg(abs(hb-pb));hg=np.linalg.norm(h[:,1:,-1]-h[:,0:1,-1],axis=-1)/hs;pg=np.linalg.norm(p[:,1:,-1]-p[:,0:1,-1],axis=-1)/rs
    close=hg<.25
    return {'direction_median_deg':float(np.median(direction)),'direction_p90_deg':float(np.quantile(direction,.9)),
       'direction_median_deg_per_digit':np.median(direction,axis=(0,2)).tolist(),'bend_mae_deg':float(bend.mean()),
       'bend_p90_abs_deg':float(np.quantile(bend,.9)),'gap_error_median_robot_mm':(np.median(abs(hg-pg),axis=0)*rs*1000).tolist(),
       'close_opposition_samples':int(close.sum()),'close_gap_mae_robot_mm':float(abs(hg-pg)[close].mean()*rs*1000) if close.any() else None}

def run(role,method):
    if role=='test' and not (HERE/'checkpoint'/'frozen.json').exists():raise RuntimeError('Freeze models before accessing held-out test mapping')
    started=time.perf_counter();parts,meta=segments(role);arrays=[];physical=[];guards=0
    radius={'v4':None,'m035':.35,'m060':.60,'m060p':.60,'m060c':.60}[method]
    solver=prior.Solver(pair=1.8,smooth=.20,speed=400) if radius is None else MorphologySolver(radius,method=='m060p',method=='m060c')
    for idx,(source,t,raw,h,hs) in enumerate(parts):
        label=f'{role}/{method}/{idx}';print(label,'start',len(t),'frames',flush=True)
        p,q,depth,n_guard=run_ik(solver,t,h/hs,label);guards+=n_guard
        actual,points,qd,coll,limit,info=prior.integrate_clip(solver.model,solver.landmarks,t,q);physical.append(info)
        a={'time_s':t,'source_time_s':t,'segment_id':np.repeat(f'{role}-{idx}',len(t)),
           'human_points_m':raw,'points_m':points,'reference_points_m':p,'q_target_rad':q,'q_rad':actual,'qd_rad_s':qd,
           'self_penetration_m':depth,'physical_depth_mm':coll,'physical_limit_excess_deg':limit,'valid':depth<=.0005}
        arrays.append(a)
    z={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]};z.update(palm_scale_m=solver.robot_scale,human_palm_scale_m=hs)
    admitted,gates=prior.gates(z);z['admitted']=admitted
    # Temporal branches use safe IK even when old palm-scaled aperture fidelity
    # rejects it; the original human motion is never removed for this reason.
    z['physical_feasible']=(z['physical_depth_mm']<=.25)&(z['physical_limit_excess_deg']<=.05)
    dest=HERE/'retarget'/method;dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/f'{role}.npz',**z)
    metrics=fidelity(z['human_points_m'],z['points_m'],hs,solver.robot_scale)
    errors=abs(z['q_rad']-z['q_target_rad'])*180/np.pi
    result={'role':role,'method':method,**meta,'mapped_frames':len(z['time_s']),'mapped_span_seconds':sum(len(a['time_s'])/50 for a in arrays),
      'old_six_gate_accepted_frames':int(admitted.sum()),'gate_pass_counts':{k:int(v.sum()) for k,v in gates.items()},
      'physical_feasible_frames':int(z['physical_feasible'].sum()),'fidelity':metrics,'ik_guard_interventions':guards,
      'max_ik_self_penetration_mm':float(z['self_penetration_m'].max()*1000),'max_physical_self_penetration_mm':float(z['physical_depth_mm'].max()),
      'max_physical_limit_excess_deg':float(z['physical_limit_excess_deg'].max()),'physical_tracking_p95_deg':float(np.quantile(errors,.95)),
      'near_native_limit_2deg_fraction':float((np.minimum(z['q_rad']-solver.native_lo,solver.native_hi-z['q_rad'])<np.deg2rad(2)).any(axis=1).mean()),
      'max_joint_speed_rad_s':float(abs(z['qd_rad_s']).max()),'physical_segments':physical,'elapsed_seconds':time.perf_counter()-started,
      'scene_sha256':sha(prior.old.SCENE),'v4_solver_sha256':sha(V4/'retarget_v4.py'),'this_script_sha256':sha(__file__)}
    (dest/f'{role}.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print('DONE',json.dumps({k:v for k,v in result.items() if k!='physical_segments'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['development','train','validation','calibration','test'],required=True);p.add_argument('--method',choices=['v4','m035','m060','m060p','m060c'],required=True);a=p.parse_args();run(a.role,a.method)
