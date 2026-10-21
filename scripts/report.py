"""Audit the full experiment and produce figures, videos and beginner-facing results."""
import ast,hashlib,json,shutil,subprocess
from pathlib import Path
import h5py,numpy as np,torch,imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from so101_env import ROOT,IMAGE_KEYS,TASK_VERSION
from evaluation_identity import identity
from dataset_tools import audit_episode
from act_tools import Runtime
from experiment import partitions

def read(path):return json.loads(Path(path).read_text())
def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""):h.update(block)
    return h.hexdigest()

def main():
    names=["base","targeted","random"]
    frozen=read(ROOT/"results/frozen_models.json")
    summaries={n:read(ROOT/"results"/n/"test-standard/summary.json") for n in names}
    training={n:read(ROOT/"checkpoints"/n/"training_summary.json") for n in names}
    assert all(t["steps"]==6000 for t in training.values())
    runs={n:read(ROOT/"checkpoints"/n/"run.json") for n in names}
    assert all(r["run"]["batch_size"]==8 and r["run"]["seed"]==42 for r in runs.values())
    assert all(t["parameter_change_l2"]>0 for t in training.values())
    expected=partitions()["test"]
    provenance={}
    for n in names:
        summary=summaries[n]
        rows=read(ROOT/"results"/n/"test-standard/episodes.json")
        assert [r["seed"] for r in rows]==expected
        assert summary["episodes"]==100 and not summary["expert_used"]
        assert summary["checkpoint"]==frozen[n]
        assert summary["evaluation_identity"]==identity(frozen[n],expected,"standard",30,TASK_VERSION)
        assert summary["successes"]==sum(r["success"] for r in rows)
        assert set(summary["policy_inputs"])==set(IMAGE_KEYS)|{"observation.state"}
        provenance[n]={"checkpoint_sha256":digest(Path(frozen[n])/"model.safetensors")}
        assert read(ROOT/"data"/n/"lerobot/export_audit.json")["passed"]
        manifest=read(ROOT/"data"/n/"manifest.json")
        assert manifest["episodes"]==(100 if n=="base" else 20)
        assert not set(manifest["train"])&set(expected)
        with h5py.File(ROOT/"data"/n/"episodes.h5") as f:
            assert len(f)==manifest["episodes"]
            for g in f.values():
                audit_episode({k:g[k] for k in ("state","action","front","wrist","timestamp")})
    targeted_seeds=read(ROOT/"results/targeted_seeds.json")
    assert read(ROOT/"data/targeted/manifest.json")["train"]==targeted_seeds
    dev_rows=read(ROOT/"results/base/development-standard/episodes.json")
    dev_failures={r["seed"] for r in dev_rows if not r["success"]}
    assert set(targeted_seeds)<=dev_failures
    assert set(targeted_seeds)<=set(partitions()["development"])
    base_manifest=read(ROOT/"data/base/manifest.json")
    assert len(base_manifest["train"])==80 and len(base_manifest["validation"])==20
    assert not set(base_manifest["train"])&set(base_manifest["validation"])
    tree=ast.parse((ROOT/"scripts/evaluate_act.py").read_text())
    imports=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    assert "expert" not in imports
    # Controlled observation sensitivity, resetting memory each time.
    runtime=Runtime(frozen["base"])
    with h5py.File(ROOT/"data/base/episodes.h5") as f:
        g=f[str(base_manifest["validation"][0])]
        obs={"observation.state":g["state"][40],"observation.images.front":g["front"][40],"observation.images.wrist":g["wrist"][40]}
    actions=[]
    for blank in (False,False,True):
        runtime.reset()
        probe={k:(np.zeros_like(v) if blank and k in IMAGE_KEYS else v) for k,v in obs.items()}
        actions.append(runtime.action(probe))
    audit={"same_observation_action_l2":float(np.linalg.norm(actions[0]-actions[1])),
           "black_images_action_l2":float(np.linalg.norm(actions[0]-actions[2])),
           "input_shapes":{k:list(v.shape) for k,v in obs.items()},
           "meaning":"sensitivity check only, not a complete causal performance test"}
    assert audit["same_observation_action_l2"]<1e-5
    (ROOT/"results/input_audit.json").write_text(json.dumps(audit,indent=2))
    provenance["upstream"]={repo:subprocess.check_output(["git","-C",str(ROOT/"official"/repo),"rev-parse","HEAD"],text=True).strip()
        for repo in ("lerobot","so101-ros2-mujoco")}
    provenance["local_source_commit"]=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    provenance["script_sha256"]={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/"scripts").glob("*.py"))}
    provenance["torch"]=torch.__version__
    provenance["gpu"]=torch.cuda.get_device_name()
    (ROOT/"results/provenance.json").write_text(json.dumps(provenance,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    labels=["Base","Failure-targeted +20","Random +20"]
    axes[0].bar(labels,[summaries[n]["successes"] for n in names],color=["#3887be","#269b77","#e9a33b"])
    axes[0].set(ylim=(0,100),ylabel="Successful episodes / 100",title="Held-out pick-and-place success")
    axes[0].tick_params(axis="x",rotation=15)
    for n,label in zip(names,labels):
        metrics=[json.loads(line) for line in (ROOT/"logs"/("train-"+n+".jsonl")).read_text().splitlines() if line.strip()]
        axes[1].plot([m["step"] for m in metrics],[m["l1_loss"] for m in metrics],label=label,alpha=.8)
    axes[1].set(xlabel="Optimizer updates",ylabel="Normalized action L1",title="Actual ACT training")
    axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/"results/comparison.png",dpi=150);plt.close(fig)
    best=max(names,key=lambda n:summaries[n]["successes"])
    status="success" if summaries[best]["successes"] else "failure"
    dense=ROOT/"results"/best/"test-standard-dense"/(status+".mp4")
    video=dense if dense.exists() else ROOT/"results"/best/"test-standard"/(status+".mp4")
    shutil.copy2(video,ROOT/"results/demo.mp4")
    reader=imageio.get_reader(video)
    fps=reader.get_meta_data()["fps"]
    frames=[frame for frame in reader];reader.close()
    imageio.imwrite(ROOT/"results/demo_preview.png",frames[len(frames)//2])
    stride=max(1,len(frames)//80)
    imageio.mimsave(ROOT/"results/demo.gif",frames[::stride],duration=1000*stride/fps,loop=0)
    report="# SO101 ACT本机完整实验报告\n\n"
    report+="示范是本地脚本IK专家；ACT动作模型进行了真实训练，视觉骨干使用ImageNet预训练。以下均为本机自主评测，不调用专家兜底。\n\n"
    report+="| 版本 | 训练示范 | 训练更新预算 | 部署快照步数 | 最终独立成功率 | 平均控制步 |\n| --- | --- | --- | --- | --- | --- |\n"
    for n in names:
        s=summaries[n];t=training[n]
        selected=read(Path(frozen[n])/"experiment.json")["training_step"]
        report+=f"| {n} | {t['training_episodes']} | {t['steps']} | {selected} | {s['successes']}/100 | {s['mean_control_steps']:.1f} |\n"
    report+="\n| 版本 | 平均仿真耗时/秒 | 结束时落在目标区外 | 超时 |\n| --- | --- | --- | --- |\n"
    for n in names:
        rows=read(ROOT/"results"/n/"test-standard/episodes.json")
        report+=f"| {n} | {summaries[n]['mean_control_steps']/20:.2f} | {sum(r['dropped_outside_target'] for r in rows)} | {sum(r['timed_out'] for r in rows)} |\n"
    report+="\n落在目标区外：确认夹持抬起后，结束时方块已回到桌面且在目标区外；它包含放偏，不能当成提前滑落事件的逐次计数。该结果与超时可同时发生。\n"
    report+="\n三组使用相同100个初始种子30000—30099；训练种子相同，只有一次训练重复，结论不能视作全面统计证明。验证集与开发集、最终测试用途分开。\n\n"
    report+="## 数据回流结论\n\n"
    baseline=summaries["base"]["successes"]
    for n in ("targeted","random"):
        delta=summaries[n]["successes"]-baseline
        report+=f"- {n}相对基础模型：{delta:+d}个成功回合，即{delta:+d}个百分点。\n"
    report+="\n## 独立扰动测试\n\n| 版本 | 位置范围扩大 | 变暗光照 | 背景变化 |\n| --- | --- | --- | --- |\n"
    for n in names:
        values=[]
        for v in ("position","lighting","background"):
            s=read(ROOT/"results"/n/("perturbation-"+v)/"summary.json")
            seeds=partitions()["perturbation"][:10]
            assert s["evaluation_identity"]==identity(frozen[n],seeds,v,30,TASK_VERSION)
            perturb_rows=read(ROOT/"results"/n/("perturbation-"+v)/"episodes.json")
            assert [r["seed"] for r in perturb_rows]==seeds
            assert s["successes"]==sum(r["success"] for r in perturb_rows)
            values.append(f"{s['successes']}/{s['episodes']}")
        report+=f"| {n} | "+" | ".join(values)+" |\n"
    report+="\n三种扰动共用10个新种子，每种各测10局，与训练和标准测试种子隔离。样本量较少，分别报告；不能与标准100次混合后宣称同一成功率。\n\n"
    report+="## 完成标准与限制\n\n"
    report+="成功要求两侧夹爪在抬高时真实接触、到位、松开、物体静止并连续保持一秒。控制20Hz，策略每30步看新图，约0.67Hz仿真时间；墙上时钟耗时受硬件影响。\n\n"
    report+=f"最佳实测为{best}：{summaries[best]['successes']}/100。80/100展示目标"+("已达到。" if summaries[best]["successes"]>=80 else "尚未达到；需要继续分析失败，不能虚报完成率。")+"\n\n"
    report+="标准初始位置：x=20.5—25.0cm，y=−2.5—2.5cm；24mm方块，固定目标(22.5,−9.5)cm。位置扰动仅将y扩大到±4cm。不是任意位置或任意物体抓取。\n\n"
    report+="本次仅仿真；没有真实机械臂部署、语音、触觉、SLAM和学习世界模型。数据为脚本专家，可能存在覆盖偏差。\n\n"
    report+=f"图像检查：同输入重复差异{audit['same_observation_action_l2']:.8f}；两图涂黑后动作差异L2={audit['black_images_action_l2']:.6f}。只能说明图像敏感性。\n\n"
    report+="![实测曲线与成功率](comparison.png)\n\n演示来源："+str(video.relative_to(ROOT))+"。完整逐局结果与失败视频保存在各版本目录。\n"
    (ROOT/"results/实验报告.md").write_text(report,encoding="utf-8")
    verification={"passed":True,"datasets":[100,20,20],"training_steps":[training[n]["steps"] for n in names],
        "final_episodes":300,"no_test_seed_in_training":True,"official_lerobot_readback":True,"input_audit":audit,
        "best_variant":best,"best_successes":summaries[best]["successes"]}
    (ROOT/"results/verification.json").write_text(json.dumps(verification,indent=2))
    print(report,flush=True)

if __name__=="__main__":main()
