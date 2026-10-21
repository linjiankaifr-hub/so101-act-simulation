"""Fixed-budget table-background experiment; no final-test-driven tuning."""
import argparse,json
from pathlib import Path
from so101_env import ROOT,TASK_VERSION
from experiment import partitions
from background_tools import TRAIN_PALETTE,TEST_PALETTE
from brightness_experiment import digest,require_run_binding
from pipeline import run,evaluate
from freeze_model import freeze
from evaluation_identity import checkpoint_hashes

OUT=ROOT/"results/background"
NAMES={"background_control":False,"background_aug":True}
CORE=("act_tools.py","train_act.py","so101_env.py","evaluate_act.py","experiment.py",
      "background_tools.py","build_background_masks.py","background_experiment.py")

def protocol():
    return {"experiment":"table_background_v1","task_version":TASK_VERSION,
        "datasets":["base","random"],"training_episodes":100,"validation_episodes":20,
        "steps":6000,"batch_size":8,"seed":42,"brightness_range":None,"variants":NAMES,
        "training_palette":TRAIN_PALETTE,"test_palette":TEST_PALETTE,
        "development_seeds":partitions()["background_development"],"test_seeds":partitions()["background_test"],
        "conditions":["standard","background_test"],"execution_steps":30,
        "selection":"minimum unaugmented validation L1; fixed 200 frames from 20 validation episodes",
        "data_sha256":{n:digest(ROOT/"data"/n/"episodes.h5") for n in ("base","random")},
        "manifest_sha256":{n:digest(ROOT/"data"/n/"manifest.json") for n in ("base","random")},
        "ordered_splits":{n:{s:json.loads((ROOT/"data"/n/"manifest.json").read_text())[s]
             for s in ("train","validation")} for n in ("base","random")},
        "mask_audits":{n:json.loads((ROOT/"data/background_masks"/(n+".json")).read_text()) for n in ("base","random")},
        "mask_sha256":{n:digest(ROOT/"data/background_masks"/(n+".h5")) for n in ("base","random")},
        "code_sha256":{n:digest(ROOT/"scripts"/n) for n in CORE},
        "previous_models":{n:checkpoint_hashes(ROOT/"checkpoints"/n/"deploy")
             for n in ("base","targeted","random","brightness_control","brightness_aug")}}

def canonical(value):return json.loads(json.dumps(value))

def run_experiment():
    OUT.mkdir(parents=True,exist_ok=True)
    # This records all data, mask, code and prior-model identities BEFORE either new training run.
    expected=canonical(protocol());path=OUT/"protocol.json"
    if path.exists():
        if json.loads(path.read_text())!=expected:raise ValueError("Background inputs changed; preserve this experiment")
    else:path.write_text(json.dumps(expected,indent=2))
    for name,augmentation in NAMES.items():
        destination=ROOT/"checkpoints"/name
        if canonical(protocol())!=expected:raise ValueError("Training inputs changed")
        done=(destination/"training_summary.json").exists()
        require_run_binding(name,expected,directory=OUT/"run_bindings",create=not done)
        if done:
            actual=json.loads((destination/"run.json").read_text())["run"]
            for k,v in {"datasets":["base","random"],"steps":6000,"batch_size":8,"seed":42,
                        "brightness_range":None,"table_augmentation":augmentation}.items():
                if actual[k]!=v:raise ValueError("Training config differs")
        else:
            args=["--name",name,"--datasets","base","random","--steps",6000,"--batch-size",8,"--seed",42]
            if augmentation:args+=["--table-augmentation"]
            run("train_act.py",*args,log="train-"+name+".log")
    fingerprints=[json.loads((ROOT/"checkpoints"/n/"run.json").read_text())["initial_parameter_sha256"] for n in NAMES]
    if len(set(fingerprints))!=1:raise ValueError("Initialization mismatch")
    frozen={n:str(freeze((ROOT/"checkpoints"/n/"best.txt").read_text().strip(),
        ROOT/"checkpoints"/n/"deploy",selection=expected["selection"])) for n in NAMES}
    (OUT/"frozen_models.json").write_text(json.dumps(frozen,indent=2))
    for name in NAMES:
        for variation in ("standard","background"):
            evaluate(name,"background_development",variation,20)
    # Recipe is fixed. Development results are diagnostic, never used to inspect or tune on final seeds.
    for name in NAMES:
        for variation in expected["conditions"]:
            evaluate(name,"background_test",variation,100)
            rows=json.loads((ROOT/"results"/name/("background_test-"+variation)/"episodes.json").read_text())
            for success in (True,False):
                index=next((i for i,r in enumerate(rows) if r["success"]==success),None)
                if index is not None:
                    run("evaluate_act.py","--name",name,"--split","background_test","--variation",variation,
                        "--episodes",1,"--offset",index,"--dense",log="dense-"+name+"-"+variation+"-"+str(success)+".log")
    run("background_report.py",log="background-report.log")

if __name__=="__main__":run_experiment()
