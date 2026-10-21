import pytest
from pathlib import Path
from stage_artifacts import recover_incomplete

def test_interrupted_outputs_are_preserved_and_completed_stages_are_immutable(tmp_path):
    for name,marker in (("raw","manifest.json"),("lerobot","export_audit.json"),("train","training_summary.json")):
        stage=tmp_path/name;stage.mkdir()
        (stage/"partial.bin").write_bytes(b"important demonstration")
        recover_incomplete(stage,marker,tmp_path)
        assert not stage.exists()
        copies=list((tmp_path/"_interrupted").glob(name+"-*/partial.bin"))
        assert len(copies)==1 and copies[0].read_bytes()==b"important demonstration"
        stage.mkdir();(stage/marker).write_text("{}")
        with pytest.raises(FileExistsError):recover_incomplete(stage,marker,tmp_path)
        assert (stage/marker).exists()

def test_recovery_cannot_move_paths_outside_declared_root(tmp_path):
    with pytest.raises(ValueError):recover_incomplete(tmp_path.parent,"done.json",tmp_path)
