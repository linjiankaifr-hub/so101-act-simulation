import importlib.util
import numpy as np
import pytest

def module():
    assert importlib.util.find_spec("act_tools") is not None, "ACT data and normalization adapter is missing"
    import act_tools
    return act_tools

def test_action_chunks_stay_in_episode_and_mask_padding():
    m=module()
    actions=np.arange(24,dtype=np.float32).reshape(4,6)
    chunk,pad=m.action_chunk(actions,2,5)
    assert chunk.shape==(5,6)
    np.testing.assert_array_equal(chunk[:2],actions[2:])
    np.testing.assert_array_equal(pad,[False,False,True,True,True])
    np.testing.assert_array_equal(chunk[-1],actions[-1])

def test_normalization_round_trip_with_constant_joint():
    m=module()
    x=np.array([[1.,2.,3.,4.,5.,6.],[1.,3.,4.,5.,6.,7.]],dtype=np.float32)
    mean,std=m.vector_stats(x)
    assert (std>0).all()
    np.testing.assert_allclose(((x-mean)/std)*std+mean,x,atol=1e-6)
