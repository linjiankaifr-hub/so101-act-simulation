from so101_env import SO101Env
import mujoco, numpy as np, json
from pathlib import Path
from PIL import Image
e=SO101Env(image_size=256)
o=e.reset(0)
print("state",e.state,"tcp",e.tcp,"cube",e.cube_position,"target",e.target)
for name in ("gripperframe",):
    print("site",name,"matrix",e.data.site_xmat[e.model.site(name).id].reshape(3,3))
for name in ("fixed_jaw_box4","moving_jaw_box2","fixed_jaw_sph_tip1","moving_jaw_sph_tip1"):
    g=e.model.geom(name)
    print(name, "local",g.pos, "world",e.data.geom_xpos[g.id])
for k in ("front","wrist"):
    Image.fromarray(o["observation.images."+k]).save(Path(__file__).resolve().parents[1]/"results"/("initial_"+k+".png"))
e.close()
