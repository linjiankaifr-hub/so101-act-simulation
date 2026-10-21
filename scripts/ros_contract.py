import numpy as np
from so101_env import JOINTS

def ordered_targets(names,positions):
    if len(names)!=6 or len(set(names))!=6 or set(names)!=set(JOINTS) or len(positions)!=6:
        raise ValueError("commands require the six unique SO101 joint names")
    values=dict(zip(names,positions))
    q=np.array([values[n] for n in JOINTS],dtype=float)
    if not np.isfinite(q).all():
        raise ValueError("joint targets must be finite radians")
    return q
