"""Paired ACT execution-window experiment, preserving all historical protocols."""
import argparse
from collections import Counter
import json
import platform
from pathlib import Path
import time
import numpy as np
import torch
from act_tools import Runtime
from so101_env import SO101Env, ROOT, SOURCE, IMAGE_KEYS, TASK_VERSION
from experiment import partitions
from evaluation_identity import checkpoint_hashes
from closed_loop_tools import DEVELOPMENT_SEEDS, TEST_SEEDS, WINDOWS, FORCE, digest, cache_complete, apply_force

OUT = ROOT / "results/closed_loop"
CHECKPOINT = ROOT / "checkpoints/background_control/deploy"
CORE = ("closed_loop_tools.py", "closed_loop_experiment.py", "act_tools.py", "so101_env.py", "evaluation_identity.py", "experiment.py")


def protocol():
    old_seeds = set().union(*(set(v) for v in partitions().values()))
    if old_seeds & set(DEVELOPMENT_SEEDS + TEST_SEEDS):
        raise ValueError("new scenes overlap a previous partition")
    return {"experiment": "act_execution_window_v1", "task_version": TASK_VERSION,
            "checkpoint": str(CHECKPOINT), "checkpoint_hashes": checkpoint_hashes(CHECKPOINT),
            "checkpoint_selection": "background_control fixed before any new scores; no augmentation confound",
            "predicted_chunk_steps": 30, "execution_windows": list(WINDOWS),
            "control_hz_simulation": 20, "maximum_control_steps": 300,
            "conditions": ["standard", "push"], "force": FORCE,
            "development_seeds": DEVELOPMENT_SEEDS, "test_seeds": TEST_SEEDS,
            "code_sha256": {n: digest(ROOT / "scripts" / n) for n in CORE},
            "xml_sha256": {str(p.relative_to(SOURCE)): digest(p) for p in sorted(SOURCE.rglob("*.xml"))},
            "previous_models": {p.name: checkpoint_hashes(p / "deploy") for p in sorted((ROOT / "checkpoints").iterdir()) if (p / "deploy").exists()},
            "selection_rule": "all three windows reported; no final-test tuning or retraining",
            "latency": "synchronized preprocessing+inference+CPU action transfer on query steps; simulation does not wait for wall latency"}


def dump(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def evaluation_identity(frozen, seeds, condition, window):
    return {"protocol_sha256": __import__("hashlib").sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest(),
            "seeds": seeds, "condition": condition, "execution_steps": window}


def run_condition(env, runtime, frozen, split, seeds, condition, window):
    directory = OUT / split / f"{condition}-h{window}"
    expected = evaluation_identity(frozen, seeds, condition, window)
    if cache_complete(directory, expected):
        print("CACHED", directory, flush=True)
        return
    if directory.exists() and any(directory.iterdir()):
        retained = directory.with_name(directory.name + "-interrupted-" + str(time.time_ns()))
        directory.rename(retained)
    directory.mkdir(parents=True, exist_ok=True)
    rows, latencies, saved = [], [], set()
    started = time.perf_counter()
    for seed in seeds:
        obs = env.reset(seed)
        runtime.reset()
        images, trace, local_latency = [], {k: [] for k in ("actions", "states", "cube", "force", "tcp")}, []
        pulse_start = pulse_end = None
        for step in range(300):
            query = len(runtime.policy._action_queue) == 0
            if query:
                images.append((step, obs[IMAGE_KEYS[0]].copy(), obs[IMAGE_KEYS[1]].copy()))
                torch.cuda.synchronize()
                tick = time.perf_counter()
            action = runtime.action(obs)
            if query:
                torch.cuda.synchronize()
                local_latency.append((time.perf_counter() - tick) * 1000)
            if step == FORCE["start_control_step"]:
                pulse_start = env.cube_position.copy()
            force = apply_force(env, condition, seed, step)
            for k, value in (("actions", action), ("states", env.state), ("cube", env.cube_position), ("force", force), ("tcp", env.tcp)):
                trace[k].append(np.asarray(value).copy())
            obs, success, done, info = env.step(action, render_obs=(step + 1) % window == 0)
            if step + 1 == FORCE["start_control_step"] + FORCE["duration_control_steps"]:
                pulse_end = env.cube_position.copy()
            if success or done:
                break
        env.data.xfrc_applied[env.cube_body] = 0.
        reason = "success" if success else ("missed_grasp" if not env.verified_grasp_lift else
                 ("outside_target_or_drop" if np.any(np.abs(env.cube_position[:2] - env.target[:2]) >= .03) else "release_or_stability"))
        row = {"seed": seed, "success": bool(success), "steps": env.steps, "simulation_seconds": env.steps / 20,
               "verified_grasp_lift": bool(env.verified_grasp_lift), "reason": reason,
               "final_xy_error_m": float(np.linalg.norm(env.cube_position[:2] - env.target[:2])),
               "policy_queries": runtime.calls, "query_latency_ms": local_latency,
               "pulse_applied": bool(condition == "push" and pulse_end is not None),
               "pulse_delta_xyz_m": None if pulse_end is None else (pulse_end - pulse_start).tolist(),
               "cube_at_pulse_start": None if pulse_start is None else pulse_start.tolist()}
        rows.append(row)
        latencies.extend(local_latency)
        # Save genuine, query-spaced frames and per-control-step traces for the first success/failure.
        status = "success" if success else "failure"
        if status not in saved:
            np.savez_compressed(directory / (status + "-trace.npz"), **{k: np.asarray(v) for k, v in trace.items()})
            final = env.observe()
            images.append((env.steps, final[IMAGE_KEYS[0]], final[IMAGE_KEYS[1]]))
            np.savez_compressed(directory / (status + "-frames.npz"), steps=np.array([f[0] for f in images]),
                                front=np.array([f[1] for f in images]), wrist=np.array([f[2] for f in images]), seed=seed)
            saved.add(status)
        dump(directory / "episodes.json", rows)
        if len(rows) % 10 == 0:
            print(split, condition, window, len(rows), "successes", sum(r["success"] for r in rows),
                  "seconds", round(time.perf_counter() - started, 1), flush=True)
    summary = {"identity": expected, "episodes": len(rows), "successes": sum(r["success"] for r in rows),
               "elapsed_seconds": time.perf_counter() - started,
               "mean_control_steps": float(np.mean([r["steps"] for r in rows])),
               "mean_policy_queries": float(np.mean([r["policy_queries"] for r in rows])),
               "query_latency_ms_p50": float(np.percentile(latencies, 50)), "query_latency_ms_p95": float(np.percentile(latencies, 95)),
               "failures": dict(Counter(r["reason"] for r in rows if not r["success"])),
               "policy_inputs": list(runtime.policy.config.input_features), "expert_used": False}
    dump(directory / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["development", "test", "all"], default="all")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(123)
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = protocol()
    path = OUT / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != frozen:
            raise ValueError("Inputs changed; preserve this experiment and use a new protocol")
    else:
        dump(path, frozen)
    dump(OUT / "environment.json", {"python": platform.python_version(), "torch": torch.__version__,
         "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda})
    env = SO101Env(image_size=96)
    try:
        for split, seeds in (("development", DEVELOPMENT_SEEDS), ("test", TEST_SEEDS)):
            if args.stage not in (split, "all"):
                continue
            for window in WINDOWS:
                runtime = Runtime(CHECKPOINT, execution_steps=window)
                # Untimed warmup so first-use CUDA initialization does not dominate measured latency.
                runtime.action(env.reset(23000))
                torch.cuda.synchronize()
                for condition in frozen["conditions"]:
                    if protocol() != frozen:
                        raise ValueError("Inputs changed during evaluation")
                    run_condition(env, runtime, frozen, split, seeds, condition, window)
                del runtime
        if protocol() != frozen:
            raise ValueError("Inputs changed after evaluation")
    finally:
        env.data.xfrc_applied[env.cube_body] = 0.
        env.close()
    print("Execution-window experiment complete", flush=True)


if __name__ == "__main__":
    main()
