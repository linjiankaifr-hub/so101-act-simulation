"""Reproducible local workflow. Never overwrites raw demonstrations or uploads artifacts."""
import json,subprocess,sys
from pathlib import Path
from so101_env import ROOT,TASK_VERSION
from experiment import failed_development_seeds
from freeze_model import freeze
from evaluation_identity import identity
from experiment import partitions
STEPS=6000

def run(script,*args,log):
    print("START",script,*args,flush=True)
    with (ROOT/"logs"/log).open("w") as output:
        subprocess.run([sys.executable,str(ROOT/"scripts"/script),*map(str,args)],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True)
    print("DONE",script,flush=True)

def collect(name,count=20,seeds=None,start=None):
    manifest=ROOT/"data"/name/"manifest.json"
    if manifest.exists():
        completed=json.loads(manifest.read_text())
        assert completed["episodes"]==count
        if seeds and completed["train"]!=json.loads(Path(seeds).read_text()):
            raise ValueError("Completed targeted data belongs to different development seeds; preserve this experiment and use a new directory.")
        return
    args=["--name",name,"--count",count]
    if seeds:args+=["--seeds",seeds]
    if start is not None:args+=["--start",start]
    run("collect.py",*args,log="collect-"+name+".log")

def export(name):
    if not (ROOT/"data"/name/"lerobot/export_audit.json").exists():
        run("export_lerobot.py","--name",name,log="export-"+name+".log")

def train(name,datasets):
    done=ROOT/"checkpoints"/name/"training_summary.json"
    if done.exists():
        assert json.loads(done.read_text())["steps"]==STEPS
        return
    run("train_act.py","--name",name,"--datasets",*datasets,"--steps",STEPS,"--batch-size",8,log="train-"+name+".log")

def evaluate(name,split="test",variation="standard",episodes=100):
    out=ROOT/"results"/name/(split+"-"+variation)/"summary.json"
    deployed=ROOT/"checkpoints"/name/"deploy"
    checkpoint=deployed if split!="development" and deployed.exists() else Path((ROOT/"checkpoints"/name/"best.txt").read_text().strip())
    expected=identity(checkpoint,partitions()[split][:episodes],variation,30,TASK_VERSION)
    if out.exists():
        cached=json.loads(out.read_text())
        if cached["episodes"]==episodes and cached.get("evaluation_identity")==expected:return
    run("evaluate_act.py","--name",name,"--checkpoint",checkpoint,"--split",split,"--episodes",episodes,"--variation",variation,
        log="eval-"+name+"-"+split+"-"+variation+".log")

def main():
    collect("base",100,start=10000);export("base");train("base",["base"])
    evaluate("base","development",episodes=60)
    dev_path=ROOT/"results/base/development-standard/episodes.json"
    rows=json.loads(dev_path.read_text())
    if sum(not r["success"] for r in rows)<20:
        evaluate("base","development",episodes=200);rows=json.loads(dev_path.read_text())
    seeds=failed_development_seeds(rows,20)
    selected=ROOT/"results/targeted_seeds.json";selected.write_text(json.dumps(seeds,indent=2))
    collect("targeted",seeds=selected);collect("random",start=11000)
    export("targeted");export("random")
    train("targeted",["base","targeted"]);train("random",["base","random"])
    frozen={name:str(freeze((ROOT/"checkpoints"/name/"best.txt").read_text().strip(),ROOT/"checkpoints"/name/"deploy")) for name in ("base","targeted","random")}
    (ROOT/"results/frozen_models.json").write_text(json.dumps(frozen,indent=2))
    for name in frozen:
        evaluate(name)
        rows=json.loads((ROOT/"results"/name/"test-standard/episodes.json").read_text())
        for status in (True,False):
            index=next((i for i,row in enumerate(rows) if row["success"]==status),None)
            if index is not None:
                run("evaluate_act.py","--name",name,"--split","test","--episodes",1,"--offset",index,"--dense",
                    log="dense-"+name+"-"+str(status)+".log")
        for variation in ("position","lighting","background"):
            evaluate(name,"perturbation",variation,10)
    run("report.py",log="report.log")
    print("Pipeline complete. Read results/实验报告.md",flush=True)

if __name__=="__main__":main()
