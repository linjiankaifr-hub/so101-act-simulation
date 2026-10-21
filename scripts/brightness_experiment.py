"""Run the approved brightness-only control experiment, retaining all v1 artifacts."""
import hashlib,json,subprocess
from pathlib import Path
from so101_env import ROOT,TASK_VERSION
from experiment import partitions
from pipeline import run,evaluate
from freeze_model import freeze

NAMES={"brightness_control":None,"brightness_aug":[.4,1.4]}
OUT=ROOT/"results/brightness"

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""):h.update(block)
    return h.hexdigest()

def protocol():
    return {"experiment":"brightness_only_v1","task_version":TASK_VERSION,
        "datasets":["base","random"],"steps":6000,"batch_size":8,"seed":42,
        "training_episodes":100,"validation_episodes":20,"variants":NAMES,
        "development_seeds":partitions()["brightness_development"],"test_seeds":partitions()["brightness_test"],
        "conditions":["standard","lighting"],"execution_steps":30,
        "snapshot_selection":"minimum unaugmented validation action L1; no final-test model selection",
        "data_sha256":{n:digest(ROOT/"data"/n/"episodes.h5") for n in ("base","random")},
        "manifest_sha256":{n:digest(ROOT/"data"/n/"manifest.json") for n in ("base","random")},
        "ordered_splits":{n:{split:json.loads((ROOT/"data"/n/"manifest.json").read_text())[split]
            for split in ("train","validation")} for n in ("base","random")},
        "code_sha256":{n:digest(ROOT/"scripts"/n) for n in
            ("act_tools.py","train_act.py","so101_env.py","evaluate_act.py","experiment.py")}}

def require_run_binding(name,expected,directory=OUT/"run_bindings",create=False):
    directory=Path(directory);path=directory/(name+".json")
    inputs={"name":name,"protocol_sha256":hashlib.sha256(json.dumps(expected,sort_keys=True,separators=(",",":")).encode()).hexdigest()}
    if path.exists():
        recorded=json.loads(path.read_text())
        if recorded["inputs"]!=inputs:raise ValueError("Training input binding differs: "+name)
    elif create:
        directory.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"inputs":inputs,"capture":"before_training_launch"},indent=2))
        temporary.replace(path)
    else:raise ValueError("Cannot reuse a run without verifiable input binding: "+name)
    return inputs

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    expected=protocol()
    path=OUT/"protocol.json"
    if path.exists():
        if json.loads(path.read_text())!=expected:
            raise ValueError("Existing brightness experiment protocol differs; preserve it and create a new experiment.")
    else:
        temporary=path.with_suffix(".tmp")
        temporary.write_text(json.dumps(expected,indent=2));temporary.replace(path)
    for name,bounds in NAMES.items():
        destination=ROOT/"checkpoints"/name
        if protocol()!=expected:raise ValueError("Inputs changed before training: "+name)
        require_run_binding(name,expected,create=not (destination/"training_summary.json").exists())
        if (destination/"training_summary.json").exists():
            recorded=json.loads((destination/"run.json").read_text())["run"]
            if any(recorded[k]!=v for k,v in {"datasets":["base","random"],"steps":6000,
                "batch_size":8,"seed":42,"brightness_range":bounds}.items()):
                raise ValueError("Existing training metadata differs: "+name)
        else:
            args=["--name",name,"--datasets","base","random","--steps",6000,"--batch-size",8,"--seed",42]
            if bounds is not None:args+=["--brightness-range",*bounds]
            run("train_act.py",*args,log="train-"+name+".log")
    initialization=[json.loads((ROOT/"checkpoints"/n/"run.json").read_text())["initial_parameter_sha256"] for n in NAMES]
    if len(set(initialization))!=1:raise ValueError("Control and augmented models did not share the same initialization")
    frozen={n:str(freeze((ROOT/"checkpoints"/n/"best.txt").read_text().strip(),
        ROOT/"checkpoints"/n/"deploy",selection="minimum unaugmented validation L1; 30-step window fixed before this experiment")) for n in NAMES}
    (OUT/"frozen_models.json").write_text(json.dumps(frozen,indent=2))
    for name in NAMES:
        for condition in expected["conditions"]:
            evaluate(name,"brightness_development",condition,20)
    # The development stage is diagnostic; the predefined recipe is never tuned on final seeds.
    for name in NAMES:
        for condition in expected["conditions"]:
            evaluate(name,"brightness_test",condition,100)
            rows=json.loads((ROOT/"results"/name/("brightness_test-"+condition)/"episodes.json").read_text())
            for status in (True,False):
                index=next((i for i,row in enumerate(rows) if row["success"]==status),None)
                if index is not None:
                    run("evaluate_act.py","--name",name,"--split","brightness_test","--variation",condition,
                        "--episodes",1,"--offset",index,"--dense",
                        log="dense-"+name+"-"+condition+"-"+str(status)+".log")
    run("brightness_report.py",log="brightness-report.log")
    print("Brightness experiment complete: results/brightness/实验报告.md",flush=True)

if __name__=="__main__":main()
