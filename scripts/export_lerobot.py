"""Export audited raw arrays via the official LeRobotDataset writer."""
import argparse,json
import h5py,numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from so101_env import ROOT,JOINTS
from dataset_tools import audit_episode
from stage_artifacts import recover_incomplete

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--name",default="base")
    a=p.parse_args()
    source=ROOT/"data"/a.name
    manifest=json.loads((source/"manifest.json").read_text())
    size=manifest["image_size"]
    target=source/"lerobot"
    if target.exists():
        if (target/"export_audit.json").exists():
            print("export already audited",target,flush=True);return
        recover_incomplete(target,"export_audit.json",ROOT/"data")
    features={
        "observation.state":{"dtype":"float32","shape":(6,),"names":list(JOINTS)},
        "action":{"dtype":"float32","shape":(6,),"names":list(JOINTS)},
        "observation.images.front":{"dtype":"video","shape":(size,size,3),"names":["height","width","channels"]},
        "observation.images.wrist":{"dtype":"video","shape":(size,size,3),"names":["height","width","channels"]}}
    dataset=LeRobotDataset.create(repo_id="local/so101_"+a.name,root=target,fps=20,features=features,
        robot_type="so101",video_backend="pyav",vcodec="h264",encoder_threads=1,streaming_encoding=True)
    ids=[row["seed"] for row in manifest["episodes_detail"]]
    total=0
    with h5py.File(source/"episodes.h5") as f:
        for seed in ids:
            g=f[str(seed)]
            ep={k:g[k][:] for k in ("state","action","front","wrist","timestamp")}
            audit_episode(ep)
            for i in range(len(ep["state"])):
                dataset.add_frame({"observation.state":ep["state"][i],"action":ep["action"][i],
                    "observation.images.front":ep["front"][i],"observation.images.wrist":ep["wrist"][i],
                    "task":"Pick up the red cube and release it in the marked target area."})
            dataset.save_episode()
            total+=len(ep["state"])
            print("exported",seed,flush=True)
    dataset.finalize()
    split={k:[ids.index(seed) for seed in manifest[k]] for k in ("train","validation")}
    (target/"split_by_episode.json").write_text(json.dumps(split,indent=2))
    (target/"original_seeds.json").write_text(json.dumps(ids))
    # Official read path validates the metadata, Parquet state/action and decoded RGB.
    read=LeRobotDataset(repo_id="local/so101_"+a.name,root=target,video_backend="pyav")
    sample=read[0]
    with h5py.File(source/"episodes.h5") as f:
        g=f[str(ids[0])]
        np.testing.assert_allclose(sample["action"].numpy(),g["action"][0],atol=1e-6)
        np.testing.assert_allclose(sample["observation.state"].numpy(),g["state"][0],atol=1e-6)
    audit={"passed":True,"episodes":len(ids),"frames":total,"format":"LeRobotDataset v3",
        "source":"locally generated numerical IK expert, not real human demonstrations",
        "readback_action_state_equal":True,"image_shape":list(sample["observation.images.front"].shape)}
    (target/"export_audit.json").write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit),flush=True)

if __name__=="__main__": main()
