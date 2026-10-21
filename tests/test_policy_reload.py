import numpy as np
import torch
from act_tools import ACTPolicy,make_config,Runtime,save_stats

def test_reloaded_act_action_and_reset_queue_are_reproducible(tmp_path):
    torch.manual_seed(4)
    cfg=make_config(); cfg.pretrained_backbone_weights=None
    policy=ACTPolicy(cfg).eval()
    policy.save_pretrained(tmp_path)
    stats={"state_mean":np.zeros(6),"state_std":np.ones(6),"action_mean":np.zeros(6),"action_std":np.ones(6)}
    save_stats(stats,tmp_path/"normalization.json")
    obs={"observation.state":np.zeros(6,np.float32),
         "observation.images.front":np.ones((96,96,3),np.uint8)*100,
         "observation.images.wrist":np.ones((96,96,3),np.uint8)*150}
    runtime=Runtime(tmp_path,device="cpu",execution_steps=5)
    first=runtime.action(obs)
    assert first.shape==(6,) and np.isfinite(first).all()
    for _ in range(2):runtime.action(obs)
    assert len(runtime.policy._action_queue)==2
    runtime.reset()
    assert len(runtime.policy._action_queue)==0
    np.testing.assert_allclose(runtime.action(obs),first,atol=1e-6)
