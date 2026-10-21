"""Numerical IK + phase trajectory expert. Uses privileged cube pose only for demonstrations."""
import numpy as np
import mujoco
from scipy.optimize import least_squares

class IKExpert:
    def __init__(self,env):
        self.env=env
        self.ikdata=mujoco.MjData(env.model)
        self.last=np.array([0.,-.8,.8,1.5,0.])

    def solve(self,position):
        e=self.env
        def residual(q):
            self.ikdata.qpos[e.qadr[:5]]=q
            mujoco.mj_forward(e.model,self.ikdata)
            p=self.ikdata.site_xpos[e.tcp_id]
            axis=self.ikdata.site_xmat[e.tcp_id].reshape(3,3)[:,0]
            return np.r_[p-position,.08*(axis-[0,0,-1]),.002*q[4]]
        result=least_squares(residual,np.clip(self.last,e.limits[:5,0]+1e-5,e.limits[:5,1]-1e-5),
            bounds=(e.limits[:5,0]+1e-5,e.limits[:5,1]-1e-5),
            max_nfev=100,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        self.last=result.x
        err=np.linalg.norm(residual(result.x)[:3])
        if err>.008:
            raise RuntimeError(f"IK position error {err:.4f} at {position}")
        return result.x.copy()

    def trajectory(self):
        e=self.env
        c=e.cube_position.copy()
        # The TCP is at the fixed inner jaw, not at the center between fingers.
        direction=np.r_[c[:2]-[.008,0],0.]
        direction/=np.linalg.norm(direction)
        grasp=c-.012*direction
        grasp[2]=.014
        target=e.target-.012*np.r_[e.target[:2]/np.linalg.norm(e.target[:2]),0]
        target[2]=.014
        waypoints=[
            (grasp+[0,0,.095],1.0,20,"approach"),
            (grasp,1.0,20,"descend"),
            (grasp,-.174,18,"close"),
            (grasp+[0,0,.09],-.174,20,"lift"),
            (target+[0,0,.09],-.174,22,"carry"),
            (target,-.174,20,"lower"),
            (target,1.0,14,"release"),
            (target+[0,0,.095],1.0,18,"retreat"),
            (target+[0,0,.095],1.0,24,"settle"),
        ]
        previous=e.state
        for pos,grip,count,phase in waypoints:
            goal=np.r_[self.solve(pos),grip]
            for t in range(1,count+1):
                alpha=t/count
                blend=alpha*alpha*(3-2*alpha)
                yield previous+(goal-previous)*blend,phase
            previous=goal

def run_expert(env,seed=0,callback=None):
    obs=env.reset(seed)
    expert=IKExpert(env)
    phases=[]
    for action,phase in expert.trajectory():
        if callback:
            callback(obs,action,phase,env)
        obs,success,done,info=env.step(action)
        phases.append(phase)
        if success or done:
            break
    return {"seed":seed,"success":env.success,"steps":env.steps,
            "max_cube_height":env.max_cube_height,"final_cube":env.cube_position.tolist(),
            "contact_steps":env.grasp_contact_steps,"last_phase":phases[-1]}
