import importlib.util
import numpy as np
import pytest
from so101_env import JOINTS

def test_ros_command_names_and_epoch_contract():
    assert importlib.util.find_spec("ros_contract") is not None, "ROS command validation is missing"
    from ros_contract import ordered_targets
    q=np.arange(6)/10
    np.testing.assert_allclose(ordered_targets(list(JOINTS)[::-1],q[::-1]),q)
    with pytest.raises(ValueError):ordered_targets(list(JOINTS)[:-1],q[:-1])
    with pytest.raises(ValueError):ordered_targets(list(JOINTS)[:-1]+[JOINTS[0]],q)
    with pytest.raises(ValueError):ordered_targets(list(JOINTS),np.full(6,np.inf))
