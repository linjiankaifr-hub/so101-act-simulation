"""Recover interrupted stages by retaining their complete original directory."""
import json,uuid
from pathlib import Path
from datetime import datetime,timezone

def recover_incomplete(directory,completion_marker,root):
    directory=Path(directory).resolve();root=Path(root).resolve()
    directory.relative_to(root)
    if directory==root:raise ValueError("refusing to move the stage root")
    if not directory.exists():return
    if (directory/completion_marker).exists():
        raise FileExistsError("Completed stage is immutable: "+str(directory))
    archive=root/"_interrupted"/(directory.name+"-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid.uuid4().hex[:8])
    archive.parent.mkdir(parents=True,exist_ok=True)
    directory.rename(archive)
    print(json.dumps({"preserved_interrupted_stage":str(archive)}),flush=True)
