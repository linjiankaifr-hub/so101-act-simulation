import importlib.util
import numpy as np
import pytest
import torch
from act_tools import ChunkDataset

def fixture():
    ep={"state":np.zeros((3,6),np.float32),"action":np.ones((3,6),np.float32),
        "front":np.full((3,4,4,3),128,np.uint8),"wrist":np.full((3,4,4,3),64,np.uint8)}
    stats={key:np.full(6,value,np.float32) for key,value in
        (("state_mean",0),("state_std",1),("action_mean",0),("action_std",1))}
    return ep,stats

def test_brightness_changes_only_pixels_and_uses_same_factor_for_both_cameras():
    ep,stats=fixture()
    base=ChunkDataset([ep],stats,chunk=4)[1]
    aug=ChunkDataset([ep],stats,chunk=4,brightness_range=(.5,.5),augmentation_seed=42)[1]
    mean=torch.tensor([.485,.456,.406])[:,None,None]
    std=torch.tensor([.229,.224,.225])[:,None,None]
    for key,pixel in (("front",128),("wrist",64)):
        recovered=aug["observation.images."+key]*std+mean
        torch.testing.assert_close(recovered,torch.full((3,4,4),pixel/255*.5),atol=1e-6,rtol=0)
        np.testing.assert_array_equal(ep[key],np.full((3,4,4,3),pixel,np.uint8))
    for key in ("observation.state","action","action_is_pad"):
        assert torch.equal(aug[key],base[key])

def test_disabled_or_identity_brightness_preserves_original_validation_images():
    ep,stats=fixture()
    original=ChunkDataset([ep],stats)[0]
    identity=ChunkDataset([ep],stats,brightness_range=(1.,1.))[0]
    for key in original:assert torch.equal(original[key],identity[key])

def test_augmentation_rng_is_reproducible_without_consuming_model_or_sampler_rng():
    ep,stats=fixture()
    a=ChunkDataset([ep],stats,brightness_range=(.4,1.4),augmentation_seed=17)
    b=ChunkDataset([ep],stats,brightness_range=(.4,1.4),augmentation_seed=17)
    torch.manual_seed(77);np.random.seed(77)
    expected_torch=torch.rand(3);expected_numpy=np.random.random(3)
    torch.manual_seed(77);np.random.seed(77)
    images_a=[a[0]["observation.images.front"] for _ in range(3)]
    images_b=[b[0]["observation.images.front"] for _ in range(3)]
    for x,y in zip(images_a,images_b):assert torch.equal(x,y)
    assert not torch.equal(images_a[0],images_a[1])
    assert torch.equal(torch.rand(3),expected_torch)
    np.testing.assert_array_equal(np.random.random(3),expected_numpy)

@pytest.mark.parametrize("bounds",[(0,1),(-1,1),(2,1),(.5,float("inf")),(.5,)])
def test_invalid_brightness_ranges_are_rejected(bounds):
    ep,stats=fixture()
    with pytest.raises(ValueError):ChunkDataset([ep],stats,brightness_range=bounds)

def test_initial_parameter_fingerprint_detects_changed_values_or_shapes():
    import act_tools
    assert hasattr(act_tools,"parameter_fingerprint"),"initialization fingerprint is missing"
    fingerprint=act_tools.parameter_fingerprint
    a={"a":torch.tensor([1.,2.]),"b":torch.tensor([3.])}
    same={"b":a["b"].clone(),"a":a["a"].clone()}
    assert fingerprint(a)==fingerprint(same)
    same["a"][0]=9
    assert fingerprint(a)!=fingerprint(same)
    same={"a":a["a"].reshape(1,2),"b":a["b"]}
    assert fingerprint(a)!=fingerprint(same)

def test_new_brightness_evaluation_seeds_are_separate_from_existing_experiments():
    from experiment import partitions
    groups=partitions()
    assert "brightness_development" in groups and "brightness_test" in groups
    existing=set().union(*(set(v) for k,v in groups.items() if not k.startswith("brightness_")))
    dev=set(groups["brightness_development"]);test=set(groups["brightness_test"])
    assert len(dev)==20 and len(test)==100
    assert not dev&test and not (dev|test)&existing

def test_paired_comparison_counts_gains_and_losses_and_rejects_different_seeds():
    assert importlib.util.find_spec("brightness_report"),"paired experiment report is missing"
    from brightness_report import paired_counts
    control=[{"seed":i,"success":s} for i,s in zip([1,2,3,4],[True,False,False,True])]
    augmented=[{"seed":i,"success":s} for i,s in zip([1,2,3,4],[True,True,True,False])]
    assert paired_counts(control,augmented)=={"both_success":1,"gained":2,"lost":1,"both_failure":0}
    augmented[-1]["seed"]=99
    with pytest.raises(ValueError):paired_counts(control,augmented)

def test_video_loading_does_not_preallocate_using_unbounded_ffmpeg_length(tmp_path):
    import imageio.v2 as imageio
    import brightness_report
    assert hasattr(brightness_report,"video_frames"), "safe video frame loader is missing"
    path=tmp_path/"tiny.mp4"
    with imageio.get_writer(path,fps=20) as writer:
        writer.append_data(np.zeros((32,32,3),np.uint8))
        writer.append_data(np.full((32,32,3),255,np.uint8))
    frames,fps=brightness_report.video_frames(path)
    assert len(frames)==2 and fps==20
    assert frames[0].mean()<10 and frames[1].mean()>240

@pytest.mark.parametrize("change",["validation_order","source_hash","training_data"])
def test_run_binding_rejects_changed_inputs_despite_same_training_cli(tmp_path,change):
    import copy
    import brightness_experiment
    assert hasattr(brightness_experiment,"require_run_binding"), "run-level provenance binding is missing"
    original={"ordered_splits":{"base":{"train":[1,2],"validation":[3,4]}},
              "code_sha256":{"train_act.py":"original"},"data_sha256":{"base":"original"}}
    brightness_experiment.require_run_binding("brightness_control",original,directory=tmp_path,create=True)
    changed=copy.deepcopy(original)
    if change=="validation_order":changed["ordered_splits"]["base"]["validation"].reverse()
    elif change=="source_hash":changed["code_sha256"]["train_act.py"]="changed"
    else:changed["data_sha256"]["base"]="changed"
    with pytest.raises(ValueError):brightness_experiment.require_run_binding("brightness_control",changed,directory=tmp_path)

def test_completed_training_without_run_binding_cannot_be_reused(tmp_path):
    import brightness_experiment
    assert hasattr(brightness_experiment,"require_run_binding"), "run-level provenance binding is missing"
    with pytest.raises(ValueError):brightness_experiment.require_run_binding("brightness_control",{},directory=tmp_path)
