"""Replay original IK trajectories, checking every label before saving table segmentation."""
import json,time
import h5py,numpy as np
from pathlib import Path
from so101_env import ROOT,SO101Env,IMAGE_KEYS
from expert import run_expert
from background_tools import table_masks
from brightness_experiment import digest

def build(name):
    output=ROOT/"data/background_masks";output.mkdir(parents=True,exist_ok=True)
    source=ROOT/"data"/name
    manifest=json.loads((source/"manifest.json").read_text())
    expected={"dataset":name,"source_sha256":digest(source/"episodes.h5"),
              "manifest_sha256":digest(source/"manifest.json"),"seeds":manifest["train"],
              "code_sha256":{k:digest(ROOT/"scripts"/k) for k in
                  ("so101_env.py","expert.py","background_tools.py","build_background_masks.py")}}
    path=output/(name+".h5");audit=output/(name+".json")
    if audit.exists():
        recorded=json.loads(audit.read_text())
        if recorded["inputs"]!=expected or recorded["mask_sha256"]!=digest(path):
            raise ValueError("Existing mask artifact inputs differ: "+name)
        return
    temporary=path.with_suffix(".partial.h5")
    if path.exists():raise ValueError("Unverified complete mask file; preserve for investigation")
    env=SO101Env(image_size=96,render=False)
    start=time.time();count=0;frames=0;max_errors={"state":0.,"action":0.,"cube":0.}
    try:
        with h5py.File(source/"episodes.h5","r") as original,h5py.File(temporary,"w") as target:
            target.attrs["inputs"]=json.dumps(expected)
            for seed in expected["seeds"]:
                old=original[str(seed)]
                rows={k:[] for k in ("front","wrist")};frame=0
                def callback(obs,action,phase,e):
                    nonlocal frame
                    for key,current in (("state",obs["observation.state"]),("action",action),("cube",e.cube_position)):
                        difference=float(np.max(np.abs(old[key][frame]-current)))
                        max_errors[key]=max(max_errors[key],difference)
                        if difference>1e-6:raise ValueError(f"Replay mismatch {name}/{seed}/{frame}/{key}: {difference}")
                    if old["phase"][frame].decode()!=phase:raise ValueError("phase mismatch")
                    masks=table_masks(e)
                    for camera in rows:rows[camera].append(masks[camera])
                    frame+=1
                result=run_expert(env,seed,callback)
                if not result["success"] or frame!=len(old["state"]):raise ValueError("Replay outcome mismatch")
                group=target.create_group(str(seed))
                for camera,values in rows.items():group.create_dataset(camera,data=np.asarray(values,dtype=bool),compression="lzf")
                frames+=frame;count+=1
                print(json.dumps({"dataset":name,"episodes":count,"total":len(expected["seeds"]),"frames":frames,
                                  "elapsed":round(time.time()-start,1)}),flush=True)
        temporary.replace(path)
        audit.write_text(json.dumps({"inputs":expected,"mask_sha256":digest(path),"episodes":count,
            "frames":frames,"maximum_replay_error":max_errors,"elapsed_seconds":time.time()-start,
            "privileged_use":"training-only table segmentation; absent from policy inputs and autonomous inference"},indent=2))
    finally:env.close()

if __name__=="__main__":
    for name in ("base","random"):build(name)
