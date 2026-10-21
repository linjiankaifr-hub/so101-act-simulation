from test_environment import load_class
import mujoco

def test_cube_dropped_into_target_without_grasp_is_not_success():
    e=load_class()(render=False)
    e.reset(7)
    q=e.cube_qadr
    e.data.qpos[q:q+3]=[*e.target[:2],.12]
    mujoco.mj_forward(e.model,e.data)
    for _ in range(50):e.step(e.state)
    assert not e.success, "falling or throwing a cube into target is not a verified grasp-and-place"
    e.close()
