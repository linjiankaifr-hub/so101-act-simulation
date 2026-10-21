import json
from so101_env import ROOT
print("SO101 ACT lab:",ROOT)
for name in ("base","targeted","random"):
    manifest=ROOT/"data"/name/"manifest.json"
    training=ROOT/"checkpoints"/name/"training_summary.json"
    evaluation=ROOT/"results"/name/"test-standard/summary.json"
    row={"variant":name}
    if manifest.exists():
        m=json.loads(manifest.read_text());row["expert_episodes"]=m["episodes"];row["training_episodes"]=len(m["train"])
    if training.exists():row["training_steps"]=json.loads(training.read_text())["steps"]
    if evaluation.exists():
        e=json.loads(evaluation.read_text());row["test_successes"]=e["successes"];row["test_episodes"]=e["episodes"]
    print(json.dumps(row,ensure_ascii=False))
