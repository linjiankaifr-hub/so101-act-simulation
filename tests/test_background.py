import numpy as np
import pytest
import torch
from experiment import partitions

def test_background_partitions_cannot_leak_into_old_experiments():
    groups=partitions()
    assert "background_development" in groups and "background_test" in groups, "missing background partitions"
    old=set().union(*(set(v) for k,v in groups.items() if not k.startswith("background_")))
    dev=groups["background_development"]; final=groups["background_test"]
    assert len(set(dev))==20 and len(set(final))==100
    assert not set(dev)&set(final) and not (set(dev)|set(final))&old

def sample():
    ep={"state":np.zeros((3,6),np.float32),"action":np.ones((3,6),np.float32),
        "front":np.full((3,4,4,3),200,np.uint8),"wrist":np.full((3,4,4,3),100,np.uint8)}
    for camera in ("front","wrist"):
        ep[camera+"_table_mask"]=np.zeros((3,4,4),bool)
        ep[camera+"_table_mask"][:,1:3,1:3]=True
    stats={k:np.full(6,v,np.float32) for k,v in
        (("state_mean",0),("state_std",1),("action_mean",0),("action_std",1))}
    return ep,stats

def test_table_tint_preserves_object_pixels_labels_raw_data_and_rng():
    import importlib.util
    assert importlib.util.find_spec("background_tools"), "table-only augmentation is missing"
    from background_tools import TableChunkDataset
    from act_tools import ChunkDataset
    ep,stats=sample()
    original=ChunkDataset([ep],stats)[0]
    dataset=TableChunkDataset([ep],stats,palette=[(.5,1.,.25)],augmentation_seed=42)
    torch.manual_seed(77);np.random.seed(77)
    expected_t=torch.rand(3);expected_n=np.random.random(3)
    torch.manual_seed(77);np.random.seed(77)
    augmented=dataset[0]
    mean=torch.tensor([.485,.456,.406])[:,None,None]
    std=torch.tensor([.229,.224,.225])[:,None,None]
    for camera,pixel in (("front",200),("wrist",100)):
        rgb=augmented["observation.images."+camera]*std+mean
        torch.testing.assert_close(rgb[:,0,0],torch.full((3,),pixel/255.),atol=1e-6,rtol=0)
        torch.testing.assert_close(rgb[:,1,1],torch.tensor([pixel/510.,pixel/255.,pixel/1020.]),atol=1e-6,rtol=0)
        assert (ep[camera]==pixel).all()
    for key in ("observation.state","action","action_is_pad"):
        assert torch.equal(original[key],augmented[key])
    assert torch.equal(torch.rand(3),expected_t)
    np.testing.assert_array_equal(np.random.random(3),expected_n)

def test_mask_misalignment_and_invalid_palette_cannot_silently_train():
    import importlib.util
    assert importlib.util.find_spec("background_tools"), "table-only augmentation is missing"
    from background_tools import TableChunkDataset
    ep,stats=sample()
    with pytest.raises(ValueError):TableChunkDataset([ep],stats,palette=[(-1,1,1)])
    ep["front_table_mask"]=np.zeros((3,3,3),bool)
    with pytest.raises(ValueError):TableChunkDataset([ep],stats)

def test_segmentation_selects_only_table_geometries_and_preserves_physics():
    import importlib.util
    assert importlib.util.find_spec("background_tools"), "table segmentation is missing"
    from background_tools import table_masks
    from so101_env import SO101Env
    env=SO101Env(image_size=96)
    try:
        before=env.data.qpos.copy()
        masks=table_masks(env)
        assert set(masks)=={"front","wrist"}
        for mask in masks.values():
            assert mask.shape==(96,96) and mask.dtype==np.bool_
        assert 100<masks["front"].sum()<9000
        np.testing.assert_array_equal(env.data.qpos,before)
    finally:env.close()

def test_unseen_background_reset_is_deterministic_visual_only_and_restores_standard():
    from so101_env import SO101Env,IMAGE_KEYS
    env=SO101Env(image_size=96)
    try:
        standard=env.reset(60000);qpos=env.data.qpos.copy()
        modified=env.reset(60000,variation="background_test")
        np.testing.assert_array_equal(env.data.qpos,qpos)
        delta=np.abs(modified[IMAGE_KEYS[0]].astype(float)-standard[IMAGE_KEYS[0]]).mean()
        assert delta>1, "novel background not rendered"
        repeat=env.reset(60000,variation="background_test")
        np.testing.assert_array_equal(repeat[IMAGE_KEYS[0]],modified[IMAGE_KEYS[0]])
        restored=env.reset(60000)
        np.testing.assert_array_equal(restored[IMAGE_KEYS[0]],standard[IMAGE_KEYS[0]])
    finally:env.close()
