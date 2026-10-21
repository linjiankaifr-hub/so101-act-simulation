import numpy as np
from so101_env import SO101Env,IMAGE_KEYS

def test_visual_perturbations_change_pixels_without_moving_initial_object():
    env=SO101Env(image_size=96)
    try:
        original=env.reset(40000)
        pose=env.cube_position.copy()
        for variation in ("lighting","background"):
            changed=env.reset(40000,variation=variation)
            np.testing.assert_allclose(env.cube_position,pose,atol=1e-10)
            delta=np.abs(changed[IMAGE_KEYS[0]].astype(float)-original[IMAGE_KEYS[0]].astype(float)).mean()
            assert delta>1, (variation,delta)
        env.reset(40000,variation="position")
        assert not np.allclose(env.cube_position,pose)
    finally:env.close()
