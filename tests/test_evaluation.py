import importlib.util
import pytest

def module():
    assert importlib.util.find_spec("experiment") is not None, "evaluation partitions are missing"
    import experiment
    return experiment

def test_final_test_is_separate_and_failed_dev_selection_is_explicit():
    m=module()
    p=m.partitions()
    sets=[set(v) for v in p.values()]
    for i,left in enumerate(sets):
        for right in sets[i+1:]: assert not left&right
    rows=[{"seed":s,"success":i%3==0} for i,s in enumerate(p["development"][:40])]
    chosen=m.failed_development_seeds(rows,20)
    assert len(chosen)==20 and set(chosen)<=set(p["development"])
    assert all(not next(r for r in rows if r["seed"]==s)["success"] for s in chosen)
    with pytest.raises(ValueError):
        m.failed_development_seeds([{"seed":30000,"success":False}],1)
