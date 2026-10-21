"""Fixed, isolated action-window study; no training or expert actions."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

DEVELOPMENT_SEEDS = list(range(23000, 23020))
TEST_SEEDS = list(range(70000, 70100))
WINDOWS = (5, 10, 30)
FORCE = {"start_control_step": 37, "duration_control_steps": 2,
         "world_y_newtons": .12, "direction": "even seed +y; odd seed -y"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def apply_force(env, condition, seed, step):
    if condition not in ("standard", "push"):
        raise ValueError("unknown disturbance condition")
    env.data.xfrc_applied[env.cube_body] = 0.
    if condition == "push" and FORCE["start_control_step"] <= step < FORCE["start_control_step"] + FORCE["duration_control_steps"]:
        env.data.xfrc_applied[env.cube_body, 1] = FORCE["world_y_newtons"] * (1 if seed % 2 == 0 else -1)
    return env.data.xfrc_applied[env.cube_body, :3].copy()


def paired_counts(baseline, candidate):
    left = {r["seed"]: bool(r["success"]) for r in baseline}
    right = {r["seed"]: bool(r["success"]) for r in candidate}
    if len(left) != len(baseline) or len(right) != len(candidate) or left.keys() != right.keys():
        raise ValueError("paired evaluation requires the same unique seeds")
    return {"both_success": sum(left[s] and right[s] for s in left),
            "gained": sum(not left[s] and right[s] for s in left),
            "lost": sum(left[s] and not right[s] for s in left),
            "both_failure": sum(not left[s] and not right[s] for s in left)}


def wilson_interval(successes, n):
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("invalid binomial count")
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0., center - radius), min(1., center + radius)


def cache_complete(directory, expected):
    directory = Path(directory)
    try:
        summary = json.loads((directory / "summary.json").read_text())
        rows = json.loads((directory / "episodes.json").read_text())
        return (summary["identity"] == expected and len(rows) == summary["episodes"] == len(expected["seeds"])
                and [r["seed"] for r in rows] == expected["seeds"]
                and summary["successes"] == sum(r["success"] for r in rows))
    except (OSError, ValueError, KeyError, TypeError):
        return False
