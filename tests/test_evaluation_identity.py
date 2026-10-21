import json,hashlib
import pytest
import pipeline
from so101_env import TASK_VERSION

@pytest.mark.parametrize("changed",["model.safetensors","normalization.json","config.json","execution_steps","seeds"])
def test_cached_scores_require_exact_model_and_protocol(tmp_path,monkeypatch,changed):
    monkeypatch.setattr(pipeline,"ROOT",tmp_path)
    checkpoint=tmp_path/"checkpoints/base/deploy"
    checkpoint.mkdir(parents=True)
    for name in ("model.safetensors","normalization.json","config.json"):
        (checkpoint/name).write_text("original")
    identity={"files":{n:hashlib.sha256((checkpoint/n).read_bytes()).hexdigest() for n in
        ("model.safetensors","normalization.json","config.json")},"task_version":TASK_VERSION,
        "seeds":list(range(30000,30100)),"variation":"standard","execution_steps":30}
    out=tmp_path/"results/base/test-standard"
    out.mkdir(parents=True)
    summary={"episodes":100,"task_version":TASK_VERSION,"evaluation_identity":identity}
    (out/"summary.json").write_text(json.dumps(summary))
    called=[]
    monkeypatch.setattr(pipeline,"run",lambda *args,**kwargs:called.append(args))
    pipeline.evaluate("base")
    assert not called
    if changed=="execution_steps": identity[changed]=5
    elif changed=="seeds":identity[changed][0]=99999
    else:(checkpoint/changed).write_text("changed")
    (out/"summary.json").write_text(json.dumps(summary))
    pipeline.evaluate("base")
    assert len(called)==1, "cached metrics must never be reused after checkpoint/protocol changes"
