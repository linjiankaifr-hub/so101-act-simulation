import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

def load_class():
    assert importlib.util.find_spec("so101_env") is not None, "SO101 environment is not implemented"
    from so101_env import SO101Env
    return SO101Env

def test_images_states_reset_and_no_privileged_inputs():
    env = load_class()(image_size=96)
    first = env.reset(seed=123)
    assert set(first) == {"observation.images.front", "observation.images.wrist", "observation.state"}
    assert first["observation.state"].shape == (6,)
    for k in ("front", "wrist"):
        image = first["observation.images."+k]
        assert image.shape == (96,96,3) and image.dtype == np.uint8
        assert image.std() > 5
    pose = env.cube_position.copy()
    env.step(env.state)
    again = env.reset(seed=123)
    np.testing.assert_allclose(pose,env.cube_position)
    np.testing.assert_allclose(first["observation.state"],again["observation.state"],atol=1e-5)
    assert not env.success
    env.close()

def test_invalid_actions_and_epoch_are_rejected():
    env = load_class()(render=False)
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(np.zeros(7))
    with pytest.raises(ValueError):
        env.step(np.full(6,np.nan))
    old = env.epoch
    env.reset(seed=2)
    with pytest.raises(ValueError):
        env.step(env.state,epoch=old)
    assert not env.success
    env.close()

def test_cube_falls_under_gravity_without_attachment():
    env = load_class()(render=False)
    env.reset(seed=3)
    assert env.model.neq == 0
    cube_joint = env.model.joint("red_cube_free").qposadr[0]
    env.data.qpos[cube_joint+2] = 0.12
    import mujoco
    mujoco.mj_forward(env.model,env.data)
    for _ in range(20):
        env.step(env.state)
    assert 0.009 < env.cube_position[2] < 0.018
    assert not env.success
    env.close()
