"""Catch contaminated splits, persistent forces and misleading paired statistics."""
import importlib.util
import numpy as np
import pytest


def tools():
    assert importlib.util.find_spec("closed_loop_tools"), "closed-loop experiment tools are missing"
    import closed_loop_tools
    return closed_loop_tools


def test_new_evaluation_scenes_are_disjoint_from_all_previous_data():
    from experiment import partitions
    m = tools()
    dev, final = m.DEVELOPMENT_SEEDS, m.TEST_SEEDS
    previous = set().union(*(set(v) for v in partitions().values()))
    assert len(dev) == len(set(dev)) == 20
    assert len(final) == len(set(final)) == 100
    assert not set(dev) & set(final)
    assert not (set(dev) | set(final)) & previous


def test_pulse_moves_cube_by_physics_and_clears_without_teleporting():
    from so101_env import SO101Env
    m = tools()
    env = SO101Env(render=False)
    try:
        outcomes = []
        for condition in ("standard", "push"):
            env.reset(70000)
            start = env.cube_position.copy()
            action = env.state.copy()
            for step in range(60):
                before = env.data.qpos.copy()
                force = m.apply_force(env, condition, 70000, step)
                np.testing.assert_array_equal(env.data.qpos, before)
                expected = [0., .12, 0.] if condition == "push" and step in (37, 38) else [0., 0., 0.]
                np.testing.assert_allclose(force, expected)
                env.step(action, render_obs=False)
            assert not env.data.xfrc_applied[env.cube_body].any()
            outcomes.append(env.cube_position - start)
        assert outcomes[1][1] - outcomes[0][1] > .003
        m.apply_force(env, "push", 70001, 37)
        assert env.data.xfrc_applied[env.cube_body, 1] == -.12
        env.reset(70001)
        assert not env.data.xfrc_applied.any(), "reset must clear prior forces"
        with pytest.raises(ValueError):
            m.apply_force(env, "unknown", 70000, 0)
    finally:
        env.close()


def test_pairing_matches_seed_instead_of_row_order_and_rejects_bad_inputs():
    m = tools()
    baseline = [{"seed": 1, "success": True}, {"seed": 2, "success": False},
                {"seed": 3, "success": True}, {"seed": 4, "success": False}]
    candidate = [{"seed": 4, "success": False}, {"seed": 3, "success": False},
                 {"seed": 2, "success": True}, {"seed": 1, "success": True}]
    actual = m.paired_counts(baseline, candidate)
    assert actual == {"both_success": 1, "gained": 1, "lost": 1, "both_failure": 1}
    with pytest.raises(ValueError):
        m.paired_counts(baseline, candidate[:-1])
    with pytest.raises(ValueError):
        m.paired_counts(baseline, candidate + candidate[:1])
    assert m.wilson_interval(0, 10)[1] == pytest.approx(.2775328, abs=1e-6)
    assert m.wilson_interval(10, 10)[0] == pytest.approx(.7224672, abs=1e-6)


def test_cache_requires_complete_rows_and_exact_bound_identity(tmp_path):
    import json
    m = tools()
    expected = {"seeds": [11, 12], "execution_steps": 5, "force": {"newtons": .12}}
    (tmp_path / "summary.json").write_text(json.dumps({"identity": expected, "episodes": 2, "successes": 1}))
    rows = [{"seed": 11, "success": True}, {"seed": 12, "success": False}]
    (tmp_path / "episodes.json").write_text(json.dumps(rows))
    assert m.cache_complete(tmp_path, expected)
    assert not m.cache_complete(tmp_path, {**expected, "execution_steps": 10})
    (tmp_path / "episodes.json").write_text(json.dumps(rows[:1]))
    assert not m.cache_complete(tmp_path, expected)


@pytest.mark.parametrize("execution_steps", [5, 10, 30])
def test_actual_policy_requeries_on_execution_window_boundaries(execution_steps):
    from act_tools import Runtime
    from so101_env import ROOT, IMAGE_KEYS
    runtime = Runtime(ROOT / "checkpoints/background_control/deploy", device="cpu", execution_steps=execution_steps)
    obs = {"observation.state": np.zeros(6, np.float32)}
    obs.update({k: np.full((96, 96, 3), 120, np.uint8) for k in IMAGE_KEYS})
    for _ in range(execution_steps):
        assert np.isfinite(runtime.action(obs)).all()
    assert runtime.calls == 1
    runtime.action(obs)
    assert runtime.calls == 2
    runtime.reset()
    assert runtime.calls == 0 and len(runtime.policy._action_queue) == 0
