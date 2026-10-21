import json
from so101_env import SO101Env
from expert import run_expert
e=SO101Env(render=False)
def callback(obs,action,phase,env):
    if env.steps%10==0:
        print(phase,env.steps,"cube",env.cube_position.round(4),"tcp",env.tcp.round(4),"grip",round(env.state[-1],3),flush=True)
print(json.dumps(run_expert(e,0,callback)))
e.close()
