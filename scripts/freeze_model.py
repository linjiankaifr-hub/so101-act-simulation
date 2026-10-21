"""Freeze deployment-only settings without changing a trained snapshot's weights."""
from pathlib import Path
import hashlib,json,shutil

def freeze(source,destination,execution_steps=30,selection="chosen on development-only action-window comparison before final tests"):
    source=Path(source);destination=Path(destination)
    config=json.loads((source/"config.json").read_text())
    if not 1<=execution_steps<=config["chunk_size"]:raise ValueError("invalid execution window")
    config["n_action_steps"]=execution_steps
    destination.mkdir(parents=True,exist_ok=True)
    for name in ("model.safetensors","normalization.json","experiment.json"):
        shutil.copy2(source/name,destination/name)
    (destination/"config.json").write_text(json.dumps(config,indent=2))
    (destination/"deployment.json").write_text(json.dumps({"source_checkpoint":str(source),
        "execution_steps":execution_steps,"control_hz":20,
        "selection":selection},indent=2))
    return destination
