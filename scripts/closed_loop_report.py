"""Audit complete paired evaluations and produce evidence-linked interview material."""
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio
from so101_env import ROOT, IMAGE_KEYS
from evaluation_identity import checkpoint_hashes
from closed_loop_experiment import OUT, CHECKPOINT, protocol, evaluation_identity, dump
from closed_loop_tools import DEVELOPMENT_SEEDS, TEST_SEEDS, WINDOWS, FORCE, cache_complete, paired_counts, wilson_interval


def read(path):
    return json.loads(Path(path).read_text())


def success_error_bars(successes, n):
    intervals = np.asarray([wilson_interval(k, n) for k in successes]) * 100
    percent = np.asarray(successes) * 100 / n
    # Floating-point cancellation near 0/100 must not create a negative error length.
    return np.maximum(0., np.vstack((percent - intervals[:, 0], intervals[:, 1] - percent)))


def validate_metrics(summary, rows):
    latencies = [v for row in rows for v in row["query_latency_ms"]]
    measured = {"mean_control_steps": float(np.mean([r["steps"] for r in rows])),
                "mean_policy_queries": float(np.mean([r["policy_queries"] for r in rows])),
                "query_latency_ms_p50": float(np.percentile(latencies, 50)),
                "query_latency_ms_p95": float(np.percentile(latencies, 95))}
    for key, value in measured.items():
        if not np.isclose(summary[key], value, rtol=1e-10, atol=1e-10):
            raise ValueError("Summary measurement differs from episodes: " + key)


def audit():
    frozen = read(OUT / "protocol.json")
    if protocol() != frozen:
        raise ValueError("Inputs differ from the preregistered protocol")
    for name, hashes in frozen["previous_models"].items():
        if checkpoint_hashes(ROOT / "checkpoints" / name / "deploy") != hashes:
            raise ValueError("A previous model changed: " + name)
    summaries, episodes = {}, {}
    for split, seeds in (("development", DEVELOPMENT_SEEDS), ("test", TEST_SEEDS)):
        for condition in ("standard", "push"):
            for window in WINDOWS:
                directory = OUT / split / f"{condition}-h{window}"
                expected = evaluation_identity(frozen, seeds, condition, window)
                if not cache_complete(directory, expected):
                    raise ValueError("Incomplete or mismatched evaluation: " + str(directory))
                summary, rows = read(directory / "summary.json"), read(directory / "episodes.json")
                validate_metrics(summary, rows)
                if summary["expert_used"] or set(summary["policy_inputs"]) != set(IMAGE_KEYS) | {"observation.state"}:
                    raise ValueError("Unexpected policy inputs")
                for row in rows:
                    if row["policy_queries"] != math.ceil(row["steps"] / window):
                        raise ValueError("Wrong number of policy queries")
                    if len(row["query_latency_ms"]) != row["policy_queries"] or not all(v > 0 for v in row["query_latency_ms"]):
                        raise ValueError("Missing synchronized query measurements")
                    if row["success"] and not row["verified_grasp_lift"]:
                        raise ValueError("Success without a verified grasp")
                    if condition == "push" and not row["pulse_applied"]:
                        raise ValueError("Pulse was not delivered")
                if summary["failures"] != dict(Counter(r["reason"] for r in rows if not r["success"])):
                    raise ValueError("Failure counts differ")
                for path in directory.glob("*-trace.npz"):
                    with np.load(path) as trace:
                        if not all(np.isfinite(trace[k]).all() for k in trace.files):
                            raise ValueError("Nonfinite trajectory")
                        applied = np.flatnonzero(np.linalg.norm(trace["force"], axis=1) > 0).tolist()
                        if applied != ([37, 38] if condition == "push" else []):
                            raise ValueError("Wrong pulse duration in the actual trajectory")
                summaries[split, condition, window], episodes[split, condition, window] = summary, rows
    return frozen, summaries, episodes


def first_scene_frames(condition, window):
    directory = OUT / "test" / f"{condition}-h{window}"
    for path in sorted(directory.glob("*-frames.npz")):
        with np.load(path) as data:
            if int(data["seed"]) == TEST_SEEDS[0]:
                return {k: data[k].copy() for k in data.files}, path.stem.replace("-frames", "")
    raise ValueError("Missing the preselected first paired scene")


def comparison_video(condition, episodes):
    samples = {window: first_scene_frames(condition, window) for window in WINDOWS}
    length = max(episodes["test", condition, window][0]["steps"] for window in WINDOWS)
    destination = OUT / (condition + "-comparison.mp4")
    preview = None
    with imageio.get_writer(destination, fps=20) as writer:
        for step in range(length + 20):
            canvas = Image.new("RGB", (960, 384), "#182430")
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 8), f"ACT closed-loop | SAME seed 70000 | {condition} | simulation t={min(step, length)/20:.2f}s", fill="white")
            for i, window in enumerate(WINDOWS):
                data, status = samples[window]
                frame = min(max(0, int(np.searchsorted(data["steps"], step, side="right") - 1)), len(data["steps"]) - 1)
                x = i * 320
                canvas.paste(Image.fromarray(data["front"][frame]).resize((240, 240)), (x + 8, 56))
                canvas.paste(Image.fromarray(data["wrist"][frame]).resize((96, 96)), (x + 216, 200))
                draw.text((x + 8, 35), f"Execute {window} steps | query {20/window:.2f} Hz", fill="white")
                draw.text((x + 8, 309), f"Observed step {int(data['steps'][frame])} | final={status}", fill="white")
                if condition == "push":
                    draw.text((x + 8, 332), "Force: step37-38, +/-0.12N world y", fill="#ffbe77")
            draw.text((12, 360), "Real query-spaced observations held until next sample; not dense video. Simulation only.", fill="#ccd9e3")
            pixels = np.asarray(canvas)
            writer.append_data(pixels)
            if step == 60:
                preview = pixels.copy()
    if preview is not None:
        Image.fromarray(preview).save(OUT / (condition + "-comparison.png"))


def main():
    frozen, summaries, episodes = audit()
    original_environment = read(OUT / "original_environment.json")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(WINDOWS))
    for index, condition in enumerate(("standard", "push")):
        successes = [summaries["test", condition, w]["successes"] for w in WINDOWS]
        axes[0].bar(x + (index - .5) * .36, successes, width=.36, label=condition,
                    yerr=success_error_bars(successes, 100), capsize=3)
        axes[1].plot(WINDOWS, [summaries["test", condition, w]["mean_policy_queries"] for w in WINDOWS], "o-", label=condition)
    axes[0].set(xticks=x, xticklabels=[str(w) for w in WINDOWS], xlabel="Executed actions before observing", ylabel="Successes / 100", ylim=(0, 100), title="Paired scenes; 95% Wilson intervals")
    axes[1].set(xlabel="Executed actions before observing", ylabel="Mean model queries per episode", title="Inference workload (simulation)")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    first_scene_diagnostics = {}
    for column, condition in enumerate(("standard", "push")):
        first_scene_diagnostics[condition] = {}
        for window in WINDOWS:
            _, status = first_scene_frames(condition, window)
            with np.load(OUT / "test" / f"{condition}-h{window}" / (status + "-trace.npz")) as trace:
                t = np.arange(len(trace["states"])) / 20
                axes[0, column].plot(t, trace["states"][:, -1], label=f"H={window}")
                axes[1, column].plot(t, trace["cube"][:, 2] * 100, label=f"H={window}")
                first_scene_diagnostics[condition][str(window)] = {
                    "seed": 70000, "outcome": status,
                    "minimum_gripper_rad": float(trace["states"][:, -1].min()),
                    "maximum_cube_height_cm": float(trace["cube"][:, 2].max() * 100)}
        axes[0, column].set(title=f"Same seed 70000: {condition}", ylabel="Actual gripper joint (rad)")
        axes[1, column].set(xlabel="Simulation seconds", ylabel="Cube height (cm)")
        axes[1, column].axhline(4.5, color="gray", linestyle="--", linewidth=1)
        for row in range(2):
            if condition == "push":
                axes[row, column].axvspan(1.85, 1.95, alpha=.2, color="red")
            axes[row, column].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "first-scene-trajectories.png", dpi=150)
    plt.close(fig)
    for condition in ("standard", "push"):
        comparison_video(condition, episodes)
    pairs = {c: {str(w): paired_counts(episodes["test", c, 30], episodes["test", c, w]) for w in (5, 10)} for c in ("standard", "push")}
    disturbance_pairs = {str(w): paired_counts(episodes["test", "standard", w], episodes["test", "push", w]) for w in WINDOWS}
    verification = {"passed": True, "evaluation_episodes": 720, "unique_final_initial_conditions": 100,
                    "weights_unchanged": True, "previous_models_unchanged": True, "expert_used": False,
                    "original_environment": original_environment,
                    "checkpoint": str(CHECKPOINT), "window_pairs_against_30": pairs,
                    "push_pairs_against_standard": disturbance_pairs,
                    "preselected_first_scene_diagnostics": first_scene_diagnostics}
    dump(OUT / "verification.json", verification)
    table = "| 执行步数 | 条件 | 成功 | 95%区间 | 平均查询次数 | 查询P50/P95毫秒 |\n| --- | --- | --- | --- | --- | --- |\n"
    for window in WINDOWS:
        for condition in ("standard", "push"):
            s = summaries["test", condition, window]
            low, high = wilson_interval(s["successes"], 100)
            table += f"| {window} | {'正常' if condition == 'standard' else '外力'} | {s['successes']}/100 | {low:.1%}—{high:.1%} | {s['mean_policy_queries']:.1f} | {s['query_latency_ms_p50']:.2f}/{s['query_latency_ms_p95']:.2f} |\n"
    report = "# ACT重新观察频率：真实实验报告\n\n日期：2026-10-20。使用同一个已训练模型，不更新任何参数。一次预测30步，分别执行5、10、30步后重新看图。\n\n"
    report += "## 最终100个新初始条件\n\n" + table
    report += (f"\n本轮正常任务中，30步窗口成功{summaries['test','standard',30]['successes']}/100，"
               f"缩短到5步为{summaries['test','standard',5]['successes']}/100；"
               f"外力条件下分别为{summaries['test','push',30]['successes']}/100与{summaries['test','push',5]['successes']}/100。"
               "同一模型的执行方式存在任务推进与扰动响应的取舍，更频繁地看图并未带来所有条件下的稳定提升。具体机制还需轨迹或新实验验证。\n")
    report += "\n六组共600次执行，配对初始条件只有100组；另有120次开发检查。区间只描述固定模型在本测试初始条件分布下的抽样不确定性，不覆盖训练随机性。\n\n"
    report += "平均查询次数同时受窗口长度和回合结束时间影响；成功较早的回合会提前停止。它衡量实际推理工作量，不能直接当作模型速度的提升倍数。\n\n"
    report += "## 同题得失：以30步为对照\n\n| 条件 | 新窗口 | 都成功 | 原失败新成功 | 原成功新失败 | 都失败 |\n| --- | --- | --- | --- | --- | --- |\n"
    for condition, by_window in pairs.items():
        for window, counts in by_window.items():
            report += f"| {condition} | {window} | {counts['both_success']} | {counts['gained']} | {counts['lost']} | {counts['both_failure']} |\n"
    report += "\n## 外力影响：同窗口正常→外力\n\n| 窗口 | 两条件都成功 | 正常失败外力成功 | 正常成功外力失败 | 都失败 |\n| --- | --- | --- | --- | --- |\n"
    for window, counts in disturbance_pairs.items():
        report += f"| {window} | {counts['both_success']} | {counts['gained']} | {counts['lost']} | {counts['both_failure']} |\n"
    report += "\n这张表不等于失败恢复率。脉冲在固定时刻施加，没有限定当时正在抓取，也没有先判定失败再启动恢复控制器。外力偶尔可能把方块推到更容易的位置，应同时报告进步和退步。\n\n"
    report += "## 失败分类\n\n| 窗口 | 条件 | 未有效夹起 | 掉落或放偏 | 释放/稳定未达标 |\n| --- | --- | --- | --- | --- |\n"
    for window in WINDOWS:
        for condition in ("standard", "push"):
            reasons = summaries["test", condition, window]["failures"]
            report += f"| {window} | {condition} | {reasons.get('missed_grasp', 0)} | {reasons.get('outside_target_or_drop', 0)} | {reasons.get('release_or_stability', 0)} |\n"
    report += "\n这些是规则统计的失败现象，不能仅靠标签断言感知错误或动作预测错误。代表回合保留动作、关节、方块位置、TCP和外力轨迹，应结合它们定位原因。\n\n"
    report += "## 开发检查（不用于调参）\n\n| 窗口 | 正常 | 外力 |\n| --- | --- | --- |\n"
    for window in WINDOWS:
        report += f"| {window} | {summaries['development','standard',window]['successes']}/20 | {summaries['development','push',window]['successes']}/20 |\n"
    report += "\n## 方法与边界\n\n"
    report += "模型固定为background_control：100条训练示范、20条独立回合验证、seed42训练6000次更新。ResNet18使用ImageNet预训练，ACT动作模块本地训练；本轮仅更改执行窗口，不训练新参数。\n\n"
    report += "脉冲为世界y方向±0.12N、0.1秒，在第37—38控制步施加，方向由种子奇偶确定。MuJoCo计算实际位移，没有传送或绑定方块。所有窗口使用同一个模型、初始条件、成功判据和15秒仿真超时。\n\n"
    report += "名义查询频率分别4、2、0.67Hz，均为仿真时间。延迟是同步CUDA查询的预处理到动作传回CPU耗时；仿真没有等待这段墙钟时间，不代表真实机器人实时闭环。窗口之间延迟差别也可能受GPU频率和运行顺序影响。\n\n"
    report += "运行期间本机有其他WSL项目任务，延迟是该并发环境下的实测日志，不据此给不同窗口做算法速度排名。仿真任务按固定控制步推进，不按墙钟等待时间改变成功规则。\n\n"
    report += f"原始运行环境单独保存在original_environment.json：Python {original_environment['python']}、Torch {original_environment['torch']}、{original_environment['gpu']}。低层评测入口重跑会更新environment.json；它仅表示最近调用环境，不用于替换原始环境或为旧延迟提供新硬件背书。\n\n"
    report += "短窗口反馈更及时，但也反复丢弃动作块后半段；是否导致阶段停滞应看轨迹，不能把所有下降都归因于它。没有使用最终测试来改变窗口、外力或模型。一个模型种子、单一方块、固定脉冲和受限位置分布不能证明普遍收益或实物迁移。\n\n"
    report += "对旧结果的精确复现仍使用旧提交和协议；本轮没有改动旧核心源码。所有旧部署模型指纹检查通过。\n\n"
    report += "![成功率与查询次数](comparison.png)\n\n"
    report += "![同一初始条件的夹爪和物体轨迹](first-scene-trajectories.png)\n\n上图来自预先选定的第一题seed70000，展示实际夹爪关节与方块高度。4.5cm虚线仅是有效抓起的高度要求，还需双侧接触；单条轨迹不能代替整体统计。脉冲时间窗标为红色。\n\n"
    report += "## 看同一场景的实际执行\n\n固定第一条正式场景seed70000，避免根据成败挑选展示。standard-comparison.mp4与push-comparison.mp4各横向对比三种窗口。画面来自实际查询时刻，查询间隔保持上一观测，已经标明不是逐帧录像；final标签是整回合最终结局，不表示画面当前时刻已成功或失败。每个条件另有首个成功／失败的原始帧和轨迹。\n\n"
    report += "## 复现与证据\n\n`./run_closed_loop.ps1 -Status`读取成绩；`./run_closed_loop.ps1`执行指纹匹配的评测缓存与报告。正式成绩位于test/{standard,push}-h{5,10,30}；development是另一分区。protocol.json锁定源码、XML、权重、种子和外力，verification.json记录完整审计。单元与整体验证见docs/closed-loop-progress.md。\n"
    (OUT / "实验报告.md").write_text(report, encoding="utf-8")
    vault = ROOT.parents[1] / "obsdian知识库"
    notes = vault / "知识点/SO101完整闭环零基础"
    for filename, title in (("comparison.png", "SO101-ACT-动作窗口对照.png"),
                             ("first-scene-trajectories.png", "SO101-ACT-同场景动作轨迹.png"),
                             ("standard-comparison.mp4", "SO101-ACT-同场景正常对照.mp4"),
                             ("push-comparison.mp4", "SO101-ACT-同场景外力对照.mp4")):
        shutil.copy2(OUT / filename, vault / "附件" / title)
    note = report.replace("![成功率与查询次数](comparison.png)", "![[SO101-ACT-动作窗口对照.png]]")
    note = note.replace("![同一初始条件的夹爪和物体轨迹](first-scene-trajectories.png)", "![[SO101-ACT-同场景动作轨迹.png]]")
    note += "\n\n![[SO101-ACT-同场景正常对照.mp4]]\n\n![[SO101-ACT-同场景外力对照.mp4]]\n\n继续读：[[SO101-21-重新观察与动作分块用大白话讲]] · [[SO101-22-主项目讲述与动手验收]]。\n"
    (notes / "SO101-20-动作窗口与外力实验结果.md").write_text(note, encoding="utf-8")
    scores = {condition: [summaries["test", condition, window]["successes"] for window in WINDOWS]
              for condition in ("standard", "push")}
    entry = "\n\n第四阶段实际实验：[[SO101-20-动作窗口与外力实验结果]]；通俗解释：[[SO101-21-重新观察与动作分块用大白话讲]]；主项目讲述与练习：[[SO101-22-主项目讲述与动手验收]]。\n"
    for relative in ("知识点/SO101完整闭环零基础/SO101-00-从这里开始.md", "实战/07-SO101-ACT完整闭环.md"):
        target = vault / relative
        text = target.read_text(encoding="utf-8")
        if "[[SO101-20-动作窗口与外力实验结果]]" not in text:
            target.write_text(text + entry, encoding="utf-8")
    for relative in ("README.md", "docs/interview.md"):
        target = ROOT / relative
        text = target.read_text(encoding="utf-8")
        if "## 第四阶段：动作窗口与外力干扰" not in text:
            addition = "\n\n## 第四阶段：动作窗口与外力干扰\n\n"
            addition += f"固定同一个background_control模型，预测30步，执行5/10/30步。100个新初始条件下，正常条件分别{scores['standard']}，外力分别{scores['push']}，均为成功次数/100。另有120次开发检查，正式600次执行只有100个不同初始条件。\n\n"
            addition += "保留全部窗口、配对得失、失败标签、95%区间和推理开销。本轮只改变部署执行方式，没有重训或实现专门失败恢复控制器。固定时刻脉冲可能帮助或损害抓取，不能把外力成功数称为恢复率。\n\n"
            addition += "完整结果见[动作窗口实验报告](" + ("../" if relative.startswith("docs/") else "") + "results/closed_loop/实验报告.md)，协议见" + ("[closed-loop-protocol](closed-loop-protocol.md)" if relative.startswith("docs/") else "[closed-loop-protocol](docs/closed-loop-protocol.md)") + "。入口run_closed_loop.ps1，状态参数-Status；独立审计scripts/audit_closed_loop.ps1。Obsidian新增SO101-20/21/22。\n"
            target.write_text(text + addition, encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)
    print(table, flush=True)


if __name__ == "__main__":
    main()
