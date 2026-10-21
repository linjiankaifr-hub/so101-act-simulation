"""Bind scores to immutable model contents and the exact evaluation protocol."""
import hashlib
from pathlib import Path

def checkpoint_hashes(checkpoint):
    checkpoint=Path(checkpoint)
    result={}
    for name in ("model.safetensors","normalization.json","config.json"):
        h=hashlib.sha256()
        with (checkpoint/name).open("rb") as stream:
            for block in iter(lambda:stream.read(2**20),b""):h.update(block)
        result[name]=h.hexdigest()
    return result

def identity(checkpoint,seeds,variation,execution_steps,task_version):
    return {"files":checkpoint_hashes(checkpoint),"seeds":list(seeds),"variation":variation,
        "execution_steps":execution_steps,"task_version":task_version}
