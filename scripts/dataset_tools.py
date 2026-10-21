"""Validation of raw demonstrations and episode-level split; no policy inputs from truth."""
import numpy as np

def audit_episode(ep):
    n=len(ep["state"])
    if n<2 or ep["state"].shape!=(n,6) or ep["action"].shape!=(n,6):
        raise ValueError("state/action must be synchronized six-dimensional arrays")
    for key in ("state","action","timestamp"):
        if not np.isfinite(ep[key]).all():
            raise ValueError("non-finite "+key)
    for key in ("front","wrist"):
        a=ep[key]
        if len(a)!=n or a.ndim!=4 or a.shape[-1]!=3 or a.dtype!=np.uint8:
            raise ValueError("invalid or unsynchronized camera "+key)
    if ep["timestamp"].shape!=(n,) or not np.allclose(np.diff(ep["timestamp"]),.05,atol=1e-6):
        raise ValueError("timestamps must be synchronized at 20Hz")
    return n

def split_episodes(ids,seed=42):
    ids=np.random.default_rng(seed).permutation(ids).tolist()
    n=int(.8*len(ids))
    return sorted(ids[:n]), sorted(ids[n:])
