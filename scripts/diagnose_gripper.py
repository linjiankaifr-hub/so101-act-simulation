from so101_env import SO101Env
import mujoco,numpy as np
from scipy.optimize import least_squares
e=SO101Env(render=False)
q=np.array([-.063,-1.058,1.059,1.477,-.025])
for angle in [1.,.4,.1,-.174]:
    e.data.qpos[e.qadr]=np.r_[q,angle]
    mujoco.mj_forward(e.model,e.data)
    print("gripper",angle,"tcp",e.tcp,"R",e.data.site_xmat[e.tcp_id].reshape(3,3).round(3))
    for name in ("fixed_jaw_box4","moving_jaw_box2"):
        print(name,e.data.geom_xpos[e.model.geom(name).id])
e.close()
