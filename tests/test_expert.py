import importlib.util
from test_environment import load_class

def test_expert_physically_lifts_releases_and_places():
    assert importlib.util.find_spec("expert") is not None, "IK demonstration expert is not implemented"
    from expert import run_expert
    env=load_class()(render=False)
    for seed in (0,1,2):
        result=run_expert(env,seed)
        assert result["success"], result
        assert result["max_cube_height"]>.045
        assert result["contact_steps"]>0
        assert env.state[-1]>.5
    env.close()
