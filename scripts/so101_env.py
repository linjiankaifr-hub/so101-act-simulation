"""Contact-based SO101 pick-place task; observations exclude object ground truth."""
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"official/so101-ros2-mujoco/src"
JOINTS = ("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper")
TASK_VERSION = "verified-pinch-lift-release-v2"
IMAGE_KEYS = ("observation.images.front","observation.images.wrist")

class SO101Env:
    def __init__(self,image_size=128,render=True):
        scene=SOURCE/"mujoco_sim/mujoco/tasks/red_cube_to_target.xml"
        self.model=mujoco.MjModel.from_xml_path(str(scene))
        self.data=mujoco.MjData(self.model)
        self.image_size=image_size
        self.render_enabled=render
        self.renderer=None
        self.epoch=0
        self.qadr=np.array([self.model.joint(n).qposadr[0] for n in JOINTS])
        self.dadr=np.array([self.model.joint(n).dofadr[0] for n in JOINTS])
        self.aids=np.array([self.model.actuator(n).id for n in JOINTS])
        self.limits=np.array([self.model.joint(n).range for n in JOINTS])
        self.home=np.array([0.,-1.45,1.35,1.20,0.,1.0])
        self.cube_body=self.model.body("red_cube").id
        self.cube_geom=self.model.geom("red_cube_geom").id
        self.cube_qadr=self.model.joint("red_cube_free").qposadr[0]
        self.target=np.array([.225,-.095,.012])
        self.model.geom_pos[self.model.geom("red_place_target").id,:2]=self.target[:2]
        self.tcp_id=self.model.site("gripperframe").id
        self.steps=0
        self.reset(0)
        self._configure_front_camera()

    def _configure_front_camera(self):
        # Local task viewpoint; original camera-mount meshes remain visual only.
        cam=self.model.camera("global_d435_camera").id
        parent=self.model.cam_bodyid[cam]
        pos=np.array([.43,-.38,.38])
        z=pos-np.array([.20,-.035,.025]); z/=np.linalg.norm(z)
        x=np.cross([0,0,1],z); x/=np.linalg.norm(x)
        y=np.cross(z,x)
        rotation=np.column_stack([x,y,z])
        body_R=self.data.xmat[parent].reshape(3,3)
        self.model.cam_pos[cam]=body_R.T@(pos-self.data.xpos[parent])
        mujoco.mju_mat2Quat(self.model.cam_quat[cam],(body_R.T@rotation).flatten())
        self.model.cam_fovy[cam]=42
        mujoco.mj_forward(self.model,self.data)

    @property
    def state(self):
        return self.data.qpos[self.qadr].copy()

    @property
    def cube_position(self):
        return self.data.xpos[self.cube_body].copy()

    @property
    def tcp(self):
        return self.data.site_xpos[self.tcp_id].copy()

    def reset(self,seed=0,variation="standard"):
        self.epoch+=1
        rng=np.random.default_rng(seed)
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[self.qadr]=self.home
        self.data.ctrl[self.aids]=self.home
        span=.025 if variation!="position" else .04
        self.data.qpos[self.cube_qadr:self.cube_qadr+3]=[rng.uniform(.205,.25),rng.uniform(-span,span),.013]
        self.data.qpos[self.cube_qadr+3:self.cube_qadr+7]=[1,0,0,0]
        self.model.light_diffuse[:]=.7 if variation!="lighting" else .3
        self.model.geom_rgba[self.model.geom("red_place_target").id]=[1,.1,.1,.65]
        if variation=="background_test":
            from background_tools import TEST_PALETTE
            table_id=self.model.body("work_table").id
            self.model.geom_rgba[self.model.geom_bodyid==table_id,:3]=TEST_PALETTE[seed%len(TEST_PALETTE)]
        elif variation=="background":
            table_id=self.model.body("work_table").id
            self.model.geom_rgba[self.model.geom_bodyid==table_id,:3]=[.25,.35,.4]
        else:
            table_id=self.model.body("work_table").id
            self.model.geom_rgba[self.model.geom_bodyid==table_id,:3]=[1,1,1]
        for _ in range(50):
            mujoco.mj_step(self.model,self.data)
        self.steps=0
        self.max_cube_height=float(self.cube_position[2])
        self.stable_steps=0
        self.success=False
        self.grasp_contact_steps=0
        self.verified_grasp_lift=False
        return self.observe()

    def observe(self):
        obs={"observation.state":self.state.astype(np.float32)}
        if self.render_enabled:
            if self.renderer is None:
                self.renderer=mujoco.Renderer(self.model,self.image_size,self.image_size)
            for key,cam in zip(IMAGE_KEYS,("global_d435_camera","wrist_cam")):
                self.renderer.update_scene(self.data,camera=cam)
                self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
                obs[key]=self.renderer.render().copy()
        return obs

    def step(self,action,epoch=None,render_obs=True):
        action=np.asarray(action,dtype=float)
        if action.shape!=(6,) or not np.isfinite(action).all():
            raise ValueError("action must contain six finite joint targets in radians")
        if epoch is not None and epoch!=self.epoch:
            raise ValueError("stale action from an earlier reset epoch")
        action=np.clip(action,self.limits[:,0],self.limits[:,1])
        self.data.ctrl[self.aids]=action
        for _ in range(25):
            mujoco.mj_step(self.model,self.data)
        self.steps+=1
        self.max_cube_height=max(self.max_cube_height,float(self.cube_position[2]))
        qv=self.model.joint("red_cube_free").dofadr[0]
        xy_ok=np.all(np.abs(self.cube_position[:2]-self.target[:2])<.03)
        released=self.state[-1]>.5 and np.linalg.norm(self.tcp-self.cube_position)>.03
        stopped=np.linalg.norm(self.data.qvel[qv:qv+3])<.025
        on_table=.009<self.cube_position[2]<.018
        valid=xy_ok and released and stopped and on_table and self.verified_grasp_lift
        self.stable_steps=self.stable_steps+1 if valid else 0
        self.success=self.stable_steps>=20
        contact_geoms=[]
        contact_bodies=set()
        for c in self.data.contact[:self.data.ncon]:
            if self.cube_geom in (c.geom1,c.geom2):
                other=c.geom2 if c.geom1==self.cube_geom else c.geom1
                contact_bodies.add(int(self.model.geom_bodyid[other]))
                name=self.model.geom(other).name or ""
                if "jaw" in name:
                    contact_geoms.append(name)
        if self.cube_position[2]>.045 and {self.model.body("gripper").id,self.model.body("moving_jaw_so101_v1").id}<=contact_bodies:
            self.verified_grasp_lift=True
        if contact_geoms:
            self.grasp_contact_steps+=1
        return (self.observe() if render_obs else {"observation.state":self.state.astype(np.float32)}),self.success,self.steps>=300,{"cube":self.cube_position.tolist(),"max_cube_height":self.max_cube_height,"contact_geoms":contact_geoms}

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer=None
