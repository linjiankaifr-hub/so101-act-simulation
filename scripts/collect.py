"""Generate contact-based IK expert demonstrations, with explicit provenance and audit."""
import argparse,json,time
from pathlib import Path
import h5py,numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from so101_env import SO101Env,ROOT
from expert import run_expert
from dataset_tools import audit_episode,split_episodes
from stage_artifacts import recover_incomplete

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--name",default="base")
    p.add_argument("--count",type=int,default=100)
    p.add_argument("--start",type=int,default=10000)
    p.add_argument("--seeds",default=None)
    p.add_argument("--image-size",type=int,default=96)
    a=p.parse_args()
    dest=ROOT/"data"/a.name
    recover_incomplete(dest,"manifest.json",ROOT/"data")
    dest.mkdir(parents=True,exist_ok=True)
    path=dest/"episodes.h5"
    if path.exists():
        raise FileExistsError("Refusing to overwrite existing demonstrations: "+str(path))
    env=SO101Env(image_size=a.image_size)
    seed_list=json.loads(Path(a.seeds).read_text()) if a.seeds else list(range(a.start,a.start+1000))
    manifest=[]; attempts=[]; start=time.time()
    video=None
    with h5py.File(path,"w") as f:
        f.attrs["provenance"]="local scripted numerical-IK expert; physical contact; no human teleoperation"
        f.attrs["fps"]=20
        for seed in seed_list:
            records={k:[] for k in ("state","action","front","wrist","timestamp","cube","phase")}
            def callback(obs,action,phase,e):
                records["state"].append(obs["observation.state"].copy())
                records["action"].append(action.astype(np.float32))
                records["front"].append(obs["observation.images.front"].copy())
                records["wrist"].append(obs["observation.images.wrist"].copy())
                records["timestamp"].append(e.steps/20)
                records["cube"].append(e.cube_position)
                records["phase"].append(phase)
            result=run_expert(env,seed,callback)
            attempts.append(result)
            if result["success"]:
                ep={k:np.asarray(v) for k,v in records.items() if k!="phase"}
                n=audit_episode(ep)
                g=f.create_group(str(seed))
                for k,v in ep.items():
                    g.create_dataset(k,data=v,compression="lzf")
                g.create_dataset("phase",data=np.array(records["phase"],dtype="S16"))
                g.attrs["success"]=True; g.attrs["seed"]=seed
                manifest.append(result)
                if a.name=="base" and len(manifest)==1:
                    writer=imageio.get_writer(ROOT/"results/expert_demo.mp4",fps=20)
                    for i,phase in enumerate(records["phase"]):
                        panel=Image.new("RGB",(640,368),"#182430")
                        panel.paste(Image.fromarray(ep["front"][i]).resize((352,352)),(0,16))
                        panel.paste(Image.fromarray(ep["wrist"][i]).resize((256,256)),(368,56))
                        draw=ImageDraw.Draw(panel)
                        draw.text((8,2),"SO101 expert demonstrations | physical contact",fill="white")
                        draw.text((375,32),"Wrist RGB | actual policy input",fill="white")
                        draw.text((375,330),f"{phase} | step {i}",fill="white")
                        writer.append_data(np.asarray(panel))
                    writer.close()
                print(json.dumps({"accepted":len(manifest),"seed":seed,"frames":n,"elapsed":round(time.time()-start,1)}),flush=True)
            else:
                print(json.dumps({"rejected":result}),flush=True)
            (dest/"attempts.json").write_text(json.dumps(attempts,indent=2))
            if len(manifest)>=a.count:
                break
        if len(manifest)!=a.count:
            raise RuntimeError(f"Only {len(manifest)} successful demonstrations")
    ids=[x["seed"] for x in manifest]
    train,val=split_episodes(ids) if a.name=="base" else (ids,[])
    meta={"name":a.name,"fps":20,"image_size":a.image_size,"episodes":len(ids),
          "total_frames":sum(r["steps"] for r in manifest),"train":train,"validation":val,
          "expert_attempts":len(attempts),"elapsed_seconds":time.time()-start,
          "provenance":"scripted IK demonstrations, not human demonstrations","episodes_detail":manifest}
    (dest/"manifest.json").write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:v for k,v in meta.items() if k!="episodes_detail"}),flush=True)
    env.close()

if __name__=="__main__": main()
