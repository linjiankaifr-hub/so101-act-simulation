"""Audit and report the controlled brightness-only training experiment."""
import hashlib,json,math,shutil,subprocess
from pathlib import Path
import numpy as np
from so101_env import ROOT,TASK_VERSION,IMAGE_KEYS
from evaluation_identity import identity,checkpoint_hashes
from brightness_experiment import protocol,NAMES,OUT,require_run_binding

def paired_counts(control,augmented):
    left=[r["seed"] for r in control];right=[r["seed"] for r in augmented]
    if left!=right or len(set(left))!=len(left):raise ValueError("paired evaluations require identical unique ordered seeds")
    counts={"both_success":0,"gained":0,"lost":0,"both_failure":0}
    for a,b in zip(control,augmented):
        key=("both_success" if a["success"] and b["success"] else
             "gained" if b["success"] else "lost" if a["success"] else "both_failure")
        counts[key]+=1
    return counts

def video_frames(path):
    import imageio.v2 as imageio
    reader=imageio.get_reader(path)
    try:
        fps=reader.get_meta_data()["fps"]
        frames=[frame for frame in reader]
        return frames,fps
    finally:reader.close()

def read(path):return json.loads(Path(path).read_text())

def main():
    expected=read(OUT/"protocol.json")
    assert expected==protocol(),"Code or raw datasets changed after preregistration"
    frozen=read(OUT/"frozen_models.json")
    runs={n:read(ROOT/"checkpoints"/n/"run.json") for n in NAMES}
    training={n:read(ROOT/"checkpoints"/n/"training_summary.json") for n in NAMES}
    initial={r["initial_parameter_sha256"] for r in runs.values()}
    assert len(initial)==1
    summaries={};rows={};development={}
    for name,bounds in NAMES.items():
        require_run_binding(name,expected)
        run=runs[name]["run"]
        assert run["datasets"]==["base","random"] and run["steps"]==6000 and run["batch_size"]==8 and run["seed"]==42
        assert run["brightness_range"]==bounds and runs[name]["validation_augmentation"] is None
        assert runs[name]["training_episodes"]==100 and runs[name]["validation_episodes"]==20
        assert training[name]["steps"]==6000 and training[name]["parameter_change_l2"]>0
        assert training[name]["initial_parameter_sha256"] in initial
        deployment_path=Path(frozen[name])/"deployment.json"
        deployment=read(deployment_path)
        chosen=Path((ROOT/"checkpoints"/name/"best.txt").read_text().strip())
        assert Path(deployment["source_checkpoint"])==chosen
        assert checkpoint_hashes(chosen)["model.safetensors"]==checkpoint_hashes(frozen[name])["model.safetensors"]
        selection="minimum unaugmented validation L1; 30-step window fixed before this experiment"
        if deployment["selection"]!=selection:
            assert deployment["selection"]=="chosen on development-only action-window comparison before final tests"
            deployment["selection_metadata_corrected_from"]=deployment["selection"]
            deployment["selection"]=selection
            deployment_path.write_text(json.dumps(deployment,indent=2))
        for condition in expected["conditions"]:
            for split,seeds,count in (("brightness_development",expected["development_seeds"],20),
                                      ("brightness_test",expected["test_seeds"],100)):
                directory=ROOT/"results"/name/(split+"-"+condition)
                s=read(directory/"summary.json");r=read(directory/"episodes.json")
                assert s["evaluation_identity"]==identity(frozen[name],seeds,condition,30,TASK_VERSION)
                assert [e["seed"] for e in r]==seeds and len(r)==count==s["episodes"]
                assert s["successes"]==sum(e["success"] for e in r) and not s["expert_used"]
                assert set(s["policy_inputs"])==set(IMAGE_KEYS)|{"observation.state"}
                assert all(e["verified_grasp_lift"] for e in r if e["success"])
                if split=="brightness_test":summaries[name,condition]=s;rows[name,condition]=r
                else:development[name,condition]=s
    control,augmented=list(NAMES)
    assert read(Path(frozen[control])/"normalization.json")==read(Path(frozen[augmented])/"normalization.json")
    assert read(Path(frozen[control])/"config.json")==read(Path(frozen[augmented])/"config.json")
    train_ids=set()
    for dataset in expected["datasets"]:
        manifest=read(ROOT/"data"/dataset/"manifest.json")
        train_ids.update(manifest["train"])
    validation=set(read(ROOT/"data/base/manifest.json")["validation"])
    assert len(train_ids)==100 and len(validation)==20 and not train_ids&validation
    assert not (train_ids|validation)&set(expected["development_seeds"]+expected["test_seeds"])
    # Verify the original three model files are still paired with their original scores.
    for old in ("base","targeted","random"):
        recorded=read(ROOT/"results"/old/"test-standard/summary.json")["evaluation_identity"]["files"]
        assert checkpoint_hashes(ROOT/"checkpoints"/old/"deploy")==recorded
    paired={c:paired_counts(rows[control,c],rows[augmented,c]) for c in expected["conditions"]}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors={"brightness_control":"#3887be","brightness_aug":"#e9a33b"}
    labels={"brightness_control":"Control","brightness_aug":"Brightness augmentation"}
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(2)
    for i,name in enumerate(NAMES):
        axes[0].bar(x+(i-.5)*.34,[summaries[name,c]["successes"] for c in expected["conditions"]],
            width=.34,color=colors[name],label=labels[name])
        metrics=[json.loads(line) for line in (ROOT/"logs"/("train-"+name+".jsonl")).read_text().splitlines() if line.strip()]
        assert metrics[-1]["step"]==6000
        axes[1].plot([m["step"] for m in metrics],[m["l1_loss"] for m in metrics],color=colors[name],label=labels[name])
    axes[0].set(xticks=x,xticklabels=["Standard lighting","Dim lighting"],ylim=(0,100),ylabel="Successful episodes / 100",title="New held-out paired scenes")
    axes[1].set(xlabel="Optimizer updates",ylabel="Normalized action L1",title="Actual ACT training")
    for ax in axes:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/"comparison.png",dpi=150);plt.close(fig)
    import imageio.v2 as imageio
    name=augmented;condition="lighting"
    status="success" if summaries[name,condition]["successes"] else "failure"
    source=ROOT/"results"/name/("brightness_test-"+condition+"-dense")/(status+".mp4")
    assert source.exists()
    shutil.copy2(source,OUT/"demo.mp4")
    frames,fps=video_frames(source)
    stride=max(1,math.ceil(len(frames)/90))
    imageio.mimsave(OUT/"demo.gif",frames[::stride],duration=1000*stride/fps,loop=0)
    imageio.imwrite(OUT/"demo_preview.png",frames[len(frames)//2])
    standard_delta=summaries[augmented,"standard"]["successes"]-summaries[control,"standard"]["successes"]
    lighting_delta=summaries[augmented,"lighting"]["successes"]-summaries[control,"lighting"]["successes"]
    report="# 亮度增强：真实训练与新场景对照实验\n\n"
    report+="我代你完成了本机代码、实际训练和评测。这个实验只改变训练图片亮度，不改变动作答案、验证图片或仿真评测画面。\n\n"
    report+="## 先看结论\n\n"
    report+=f"变暗光照下，亮度增强版相对对照版变化{lighting_delta:+d}个百分点；普通光照变化{standard_delta:+d}个百分点。"
    report+=("本次观察到光照场景的成功率提高。" if lighting_delta>0 else "本次没有观察到光照场景成功率提高。")
    report+="只有一个训练种子，结论限定于本次条件；不能据此声称普遍有效。\n\n"
    report+="## 最终独立测试\n\n| 版本 | 训练示范 | 更新预算 | 部署快照步数 | 普通光照 | 变暗光照 |\n| --- | --- | --- | --- | --- | --- |\n"
    for n in NAMES:
        snapshot=read(Path(frozen[n])/"experiment.json")["training_step"]
        report+=f"| {labels[n]} | 100 | 6000 | {snapshot} | {summaries[n,'standard']['successes']}/100 | {summaries[n,'lighting']['successes']}/100 |\n"
    report+="\n每组使用相同的新种子50000—50099，普通与变暗条件也配对。共400回合，不是400个互不相关的初始位置。旧61/100使用不同测试种子，不能直接与本表比较来认定改进。\n\n"
    report+="## 同一个场景，哪些变好、哪些变坏\n\n| 条件 | 两版都成功 | 原失败、新成功 | 原成功、新失败 | 两版都失败 |\n| --- | --- | --- | --- | --- |\n"
    for c in expected["conditions"]:
        q=paired[c]
        report+=f"| {c} | {q['both_success']} | {q['gained']} | {q['lost']} | {q['both_failure']} |\n"
    report+="\n这样可以看到模型具体在哪些场景获得收益、又在哪些场景退步。\n\n"
    report+="## 开发集诊断\n\n| 版本 | 普通光照 | 变暗光照 |\n| --- | --- | --- |\n"
    for n in NAMES:
        report+=f"| {labels[n]} | {development[n,'standard']['successes']}/20 | {development[n,'lighting']['successes']}/20 |\n"
    report+="\n开发种子21000—21019与最终测试、训练和验证均隔离。本轮预先固定增强范围，没有根据这些分数调整后再挑选结果。\n\n"
    report+="## 到底改了什么\n\n"
    report+="对同一训练样本的两张RGB图，乘同一个随机亮度因子0.4—1.4，限制到合法像素范围，再做ImageNet归一化。可以理解成给老师的照片调亮、调暗，老师的动作答案保持原样。\n\n"
    report+="这只是光照变化的近似训练方法。真实评测在MuJoCo中改变灯光，物理渲染不等价于简单乘像素，因此必须实测。背景颜色和纹理没有在本轮改变，也没有证明背景泛化得到改善。\n\n"
    report+="## 公平性和审计\n\n"
    report+="- 数据相同：base的80条＋random的20条训练示范；独立20条验证。\n"
    report+="- 两组从相同的ACT动作模型随机初始化重新训练，视觉骨干沿用ImageNet预训练。\n"
    report+="- 初始参数SHA256相同；网络、归一化、seed42、batch8与6000次预算相同。\n"
    report+="- 增强使用独立随机数，避免改变模型初始化与数据抽样随机数。\n"
    report+="- 验证与推理不做亮度增强；训练与曲线均为真实记录。每次快照从20个验证回合固定等距抽取200帧动作块，按误差选择。\n"
    report+="- 模型加载后自主执行，输入仅两路RGB和6维关节状态，不调用专家。\n"
    report+="- 仍要求双侧夹爪真实接触抬起、到位、释放并稳定一秒。\n"
    report+="- 原始数据、第一轮三个模型与旧成绩保留；文件指纹校验通过。\n\n"
    report+="## 当前边界\n\n单物体、固定目标与受限初始范围的仿真实验，尚未部署真实机械臂。没有保证80/100，也没有实现强化学习或完整DAgger。下一轮应先选一个开发问题，再设置新的独立考试。\n\n"
    report+="![训练曲线与成功率](comparison.png)\n\n"
    report+=f"演示来源：{source.relative_to(ROOT)}，为亮度增强模型在变暗光照下的{status}回合。两组每种条件的成功与失败视频保存在对应brightness_test目录。\n"
    (OUT/"实验报告.md").write_text(report,encoding="utf-8")
    audit={"passed":True,"training_updates":[training[n]["steps"] for n in NAMES],
        "same_initial_parameter_sha256":next(iter(initial)),"same_data_normalization_and_model":True,
        "development_episodes":80,"final_episodes":400,"old_models_unchanged":True,
        "standard_delta_percentage_points":standard_delta,"lighting_delta_percentage_points":lighting_delta,
        "scores":{n:{c:summaries[n,c]["successes"] for c in expected["conditions"]} for n in NAMES},
        "paired_counts":paired,"demo_status":status,"demo_source":str(source.relative_to(ROOT)),
        "run_bindings":{n:read(OUT/"run_bindings"/(n+".json")) for n in NAMES},
        "source_commit":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()}
    (OUT/"verification.json").write_text(json.dumps(audit,indent=2))
    vault=ROOT.parents[1]/"obsdian知识库"
    for file,title in (("comparison.png","SO101-ACT-亮度增强对照.png"),("demo.mp4","SO101-ACT-亮度变化自主.mp4"),("demo.gif","SO101-ACT-亮度变化演示.gif")):
        shutil.copy2(OUT/file,vault/"附件"/title)
    note=report.replace("![训练曲线与成功率](comparison.png)","![[SO101-ACT-亮度增强对照.png]]")
    note+="\n\n## 看实际执行\n\n![[SO101-ACT-亮度变化自主.mp4]]\n\n返回：[[SO101-11-亮度增强到底改了什么]] · [[SO101-09-现在先做什么]] · [[SO101-00-从这里开始]]。\n"
    (vault/"知识点/SO101完整闭环零基础/SO101-10-亮度增强实验结果.md").write_text(note,encoding="utf-8")
    for relative in ("实战/07-SO101-ACT完整闭环.md","知识点/SO101完整闭环零基础/SO101-00-从这里开始.md"):
        target=vault/relative;text=target.read_text(encoding="utf-8")
        if "[[SO101-10-亮度增强实验结果]]" not in text:
            target.write_text(text+"\n\n第二阶段实际结果：[[SO101-10-亮度增强实验结果]]。它使用新测试种子，应按该页的新对照比较，保留第一轮成绩。\n",encoding="utf-8")
    print(report,flush=True)

if __name__=="__main__":main()
