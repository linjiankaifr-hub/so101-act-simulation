"""Autonomous closed-loop ACT evaluation. This module never imports or invokes the expert."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch,imageio.v2 as imageio
from PIL import Image,ImageDraw
from so101_env import SO101Env,ROOT,IMAGE_KEYS,TASK_VERSION
from act_tools import Runtime
from experiment import partitions
from evaluation_identity import identity

def panel(obs,name,seed,step,result=None,execution_steps=30):
    p=Image.new("RGB",(640,368),"#182430")
    p.paste(Image.fromarray(obs[IMAGE_KEYS[0]]).resize((352,352)),(0,16))
    p.paste(Image.fromarray(obs[IMAGE_KEYS[1]]).resize((256,256)),(368,56))
    d=ImageDraw.Draw(p)
    d.text((8,2),f"SO101 autonomous ACT | {name} | seed {seed}",fill="white")
    d.text((375,32),f"Live wrist RGB | query every {execution_steps} steps",fill="white")
    d.text((375,330),f"step {step} | {result or 'executing'} | {'QUERY' if step%execution_steps==0 else 'queued'}",fill="white")
    return np.asarray(p)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--name",default="base")
    p.add_argument("--checkpoint",default=None)
    p.add_argument("--split",choices=["development","test","perturbation","brightness_development","brightness_test","background_development","background_test"],default="development")
    p.add_argument("--episodes",type=int,default=60)
    p.add_argument("--variation",choices=["standard","position","lighting","background","background_test"],default="standard")
    p.add_argument("--offset",type=int,default=0)
    p.add_argument("--action-steps",type=int,default=30)
    p.add_argument("--dense",action="store_true")
    p.add_argument("--output-tag",default="")
    a=p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(123)
    if a.output_tag and not all(c.isalnum() or c in "-_" for c in a.output_tag):raise ValueError("invalid output tag")
    directory=ROOT/"results"/a.name/(a.split+"-"+a.variation+("-dense" if a.dense else "")+("-"+a.output_tag if a.output_tag else ""))
    directory.mkdir(parents=True,exist_ok=True)
    deployed=ROOT/"checkpoints"/a.name/"deploy"
    ckpt=Path(a.checkpoint) if a.checkpoint else (deployed if deployed.exists() else Path((ROOT/"checkpoints"/a.name/"best.txt").read_text().strip()))
    runtime=Runtime(ckpt,execution_steps=a.action_steps)
    record_interval=1 if a.dense else a.action_steps
    env=SO101Env(image_size=96)
    seeds=partitions()[a.split][a.offset:a.offset+a.episodes]
    rows=[];start=time.time();saved=set()
    if not seeds:raise ValueError("empty evaluation partition")
    for seed in seeds:
        obs=env.reset(seed,variation=a.variation)
        runtime.reset()
        frames=[];trace={"actions":[],"states":[],"cube":[]}
        for step in range(300):
            # The policy consumes fresh images only when its action queue is empty.
            if step%record_interval==0:
                frames.append((step,{k:v.copy() for k,v in obs.items()}))
            action=runtime.action(obs)
            trace["actions"].append(action.copy());trace["states"].append(env.state);trace["cube"].append(env.cube_position)
            obs,success,done,info=env.step(action,render_obs=(step+1)%record_interval==0)
            if success or done: break
        status="success" if success else "failure"
        xy=np.linalg.norm(env.cube_position[:2]-env.target[:2])
        reason="success" if success else ("missed_grasp" if not env.verified_grasp_lift else
                ("outside_target_or_drop" if np.any(np.abs(env.cube_position[:2]-env.target[:2])>=.03) else "release_or_stability"))
        row={"seed":seed,"success":success,"steps":env.steps,"simulation_seconds":env.steps/20,
            "max_cube_height":env.max_cube_height,"final_xy_error_m":float(xy),
            "verified_grasp_lift":env.verified_grasp_lift,"dropped_outside_target":bool(env.verified_grasp_lift and env.cube_position[2]<.02 and np.any(np.abs(env.cube_position[:2]-env.target[:2])>=.03)),
            "timed_out":bool(not success and env.steps>=300),"contact_steps":env.grasp_contact_steps,"policy_queries":runtime.calls,"reason":reason}
        rows.append(row)
        if status not in saved:
            last_obs=env.observe()
            video=directory/(status+".mp4")
            with imageio.get_writer(video,fps=20/record_interval) as w:
                for st,ob in frames:w.append_data(panel(ob,a.name,seed,st,execution_steps=a.action_steps))
                final=panel(last_obs,a.name,seed,env.steps,status,execution_steps=a.action_steps)
                for _ in range(max(1,round(20/record_interval))):w.append_data(final)
            Image.fromarray(final).save(directory/(status+".png"))
            np.savez_compressed(directory/(status+"-trace.npz"),**{k:np.array(v) for k,v in trace.items()})
            saved.add(status)
        (directory/"episodes.json").write_text(json.dumps(rows,indent=2))
        print(json.dumps({**row,"completed":len(rows),"elapsed":round(time.time()-start,1)}),flush=True)
    report={"evaluation_identity":identity(ckpt,seeds,a.variation,a.action_steps,TASK_VERSION),"task_version":TASK_VERSION,"name":a.name,"split":a.split,"variation":a.variation,"checkpoint":str(ckpt),
        "episodes":len(rows),"successes":sum(r["success"] for r in rows),
        "success_rate":sum(r["success"] for r in rows)/len(rows),"elapsed_seconds":time.time()-start,
        "mean_control_steps":float(np.mean([r["steps"] for r in rows])),
        "policy_inputs":list(runtime.policy.config.input_features),"expert_used":False,
        "control_hz":20,"policy_query_hz":20/a.action_steps,"execution_steps":a.action_steps,
        "success_definition":"verified opposing-jaw contact above 4.5cm; inside target; released; stationary 1 second"}
    (directory/"summary.json").write_text(json.dumps(report,indent=2))
    env.close()
    print(json.dumps(report),flush=True)

if __name__=="__main__":main()
