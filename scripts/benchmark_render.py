import os,time,json
from so101_env import SO101Env
start=time.time()
e=SO101Env(image_size=96)
print("initialized",time.time()-start,flush=True)
from OpenGL.GL import glGetString,GL_RENDERER
print("renderer",glGetString(GL_RENDERER),flush=True)
o=e.reset(0)
start=time.time()
for _ in range(10): e.step(e.state)
print(json.dumps({"backend":os.environ.get("MUJOCO_GL"),"threads":os.environ.get("LP_NUM_THREADS"),"seconds_per_step":(time.time()-start)/10}),flush=True)
e.close()
