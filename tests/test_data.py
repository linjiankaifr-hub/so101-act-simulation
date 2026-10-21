import importlib.util
import numpy as np
import pytest

def functions():
    assert importlib.util.find_spec("dataset_tools") is not None, "demonstration audit is missing"
    from dataset_tools import audit_episode, split_episodes
    return audit_episode,split_episodes

def episode():
    return {"state":np.zeros((5,6),np.float32),"action":np.ones((5,6),np.float32),
        "front":np.zeros((5,96,96,3),np.uint8),"wrist":np.ones((5,96,96,3),np.uint8),
        "timestamp":np.arange(5)/20}

def test_synced_finite_arrays_and_missing_camera_rejected():
    audit,_=functions()
    assert audit(episode())==5
    wrong=episode(); wrong["wrist"]=wrong["wrist"][:-1]
    with pytest.raises(ValueError): audit(wrong)
    wrong=episode(); wrong["action"][2,1]=np.nan
    with pytest.raises(ValueError): audit(wrong)
    wrong=episode(); wrong["timestamp"][3]=0
    with pytest.raises(ValueError): audit(wrong)

def test_split_by_episode_is_disjoint_reproducible():
    _,split=functions()
    a,b=split(list(range(100)),seed=42)
    assert len(a)==80 and len(b)==20
    assert not set(a)&set(b)
    assert set(a)|set(b)==set(range(100))
    assert (a,b)==split(list(range(100)),seed=42)
