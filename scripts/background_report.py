"""Audit real background runs and publish an evidence-linked beginner report."""
import json,math,shutil,subprocess
from collections import Counter
from pathlib import Path
import numpy as np
from so101_env import ROOT,TASK_VERSION,IMAGE_KEYS
from evaluation_identity import identity,checkpoint_hashes
from brightness_experiment import require_run_binding,digest
from brightness_report import paired_counts,video_frames
from background_experiment import OUT,NAMES,protocol,canonical
from background_tools import TRAIN_PALETTE,TEST_PALETTE

def read(path):return json.loads(Path(path).read_text())

def main():
    expected=read(OUT/"protocol.json")
    if expected!=canonical(protocol()):raise ValueError("Background inputs changed after preregistration")
    frozen=read(OUT/"frozen_models.json")
    runs={n:read(ROOT/"checkpoints"/n/"run.json") for n in NAMES}
    train={n:read(ROOT/"checkpoints"/n/"training_summary.json") for n in NAMES}
    initial={v["initial_parameter_sha256"] for v in runs.values()}
    assert len(initial)==1
    train_ids=[];val_ids=[]
    for dataset in expected["datasets"]:
        splits=expected["ordered_splits"][dataset]
        train_ids+=splits["train"];val_ids+=splits["validation"]
        mask=expected["mask_audits"][dataset]
        assert mask["episodes"]==len(splits["train"])
        assert all(error<=1e-6 for error in mask["maximum_replay_error"].values())
        assert mask["mask_sha256"]==expected["mask_sha256"][dataset]
    assert len(train_ids)==100 and len(set(train_ids))==100 and len(val_ids)==20
    assert not set(train_ids)&set(val_ids)
    assert not set(train_ids+val_ids)&set(expected["development_seeds"]+expected["test_seeds"])
    for name,old_hashes in expected["previous_models"].items():
        assert checkpoint_hashes(ROOT/"checkpoints"/name/"deploy")==old_hashes
    scores={};episodes={};development={};curves={}
    for name,augmentation in NAMES.items():
        require_run_binding(name,expected,directory=OUT/"run_bindings")
        run=runs[name]["run"]
        for key,value in {"datasets":["base","random"],"steps":6000,"batch_size":8,"seed":42,
                          "brightness_range":None,"table_augmentation":augmentation}.items():
            assert run[key]==value,(name,key)
        assert runs[name]["validation_augmentation"] is None
        assert runs[name]["training_episodes"]==100 and runs[name]["validation_episodes"]==20
        assert train[name]["steps"]==6000 and train[name]["parameter_change_l2"]>0
        assert train[name]["initial_parameter_sha256"] in initial
        chosen=Path((ROOT/"checkpoints"/name/"best.txt").read_text().strip())
        deployment=read(Path(frozen[name])/"deployment.json")
        assert Path(deployment["source_checkpoint"])==chosen
        assert deployment["selection"]==expected["selection"]
        assert checkpoint_hashes(chosen)["model.safetensors"]==checkpoint_hashes(frozen[name])["model.safetensors"]
        curves[name]=[json.loads(line) for line in (ROOT/"logs"/("train-"+name+".jsonl")).read_text().splitlines() if line.strip()]
        assert [r["step"] for r in curves[name]]==list(range(100,6001,100))
        for split,conditions,seeds in (
            ("background_development",("standard","background"),expected["development_seeds"]),
            ("background_test",expected["conditions"],expected["test_seeds"])):
            for condition in conditions:
                directory=ROOT/"results"/name/(split+"-"+condition)
                summary=read(directory/"summary.json");rows=read(directory/"episodes.json")
                assert summary["evaluation_identity"]==identity(frozen[name],seeds,condition,30,TASK_VERSION)
                assert [r["seed"] for r in rows]==seeds
                assert len(rows)==summary["episodes"]==len(seeds)
                assert summary["successes"]==sum(r["success"] for r in rows)
                assert not summary["expert_used"]
                assert set(summary["policy_inputs"])==set(IMAGE_KEYS)|{"observation.state"}
                assert all(r["verified_grasp_lift"] for r in rows if r["success"])
                if split=="background_test":scores[name,condition]=summary;episodes[name,condition]=rows
                else:development[name,condition]=summary
    control,augmented=list(NAMES)
    for filename in ("config.json","normalization.json"):
        assert read(Path(frozen[control])/filename)==read(Path(frozen[augmented])/filename)
    # Old models are diagnostic references, not a substitute for the freshly trained control.
    diagnostics={}
    for old in ("brightness_control","brightness_aug"):
        directory=ROOT/"results"/old/"background_development-background"
        summary=read(directory/"summary.json");rows=read(directory/"episodes.json")
        assert summary["evaluation_identity"]==identity(ROOT/"checkpoints"/old/"deploy",expected["development_seeds"],"background",30,TASK_VERSION)
        assert len(rows)==20 and [r["seed"] for r in rows]==expected["development_seeds"]
        assert summary["successes"]==sum(r["success"] for r in rows)
        diagnostics[old]=summary["successes"]
    paired={c:paired_counts(episodes[control,c],episodes[augmented,c]) for c in expected["conditions"]}
    per_color={n:{str(i):{"rgb":color,"episodes":sum(r["seed"]%4==i for r in episodes[n,"background_test"]),
                    "successes":sum(r["seed"]%4==i and r["success"] for r in episodes[n,"background_test"])}
                    for i,color in enumerate(TEST_PALETTE)} for n in NAMES}
    failures={n:{c:dict(Counter(r["reason"] for r in episodes[n,c] if not r["success"])) for c in expected["conditions"]} for n in NAMES}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors={control:"#3887be",augmented:"#e9a33b"}
    labels={control:"Control",augmented:"Table color augmentation"}
    md_labels={control:"普通对照",augmented:"桌色增强"}
    md_conditions={"standard":"原桌面","background_test":"四种新桌色"}
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(2)
    for i,name in enumerate(NAMES):
        axes[0].bar(x+(i-.5)*.34,[scores[name,c]["successes"] for c in expected["conditions"]],
                    width=.34,color=colors[name],label=labels[name])
        axes[1].plot([r["step"] for r in curves[name]],[r["l1_loss"] for r in curves[name]],
                     color=colors[name],label=labels[name])
    axes[0].set(xticks=x,xticklabels=["Original table","4 novel table colors"],ylim=(0,100),
                ylabel="Successful episodes / 100",title="Paired held-out scenes")
    axes[1].set(xlabel="Optimizer updates",ylabel="Normalized action L1",title="Actual ACT training")
    for ax in axes:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/"comparison.png",dpi=150);plt.close(fig)
    # A real demonstration frame with true segmentation masks illustrates the training-only edit.
    import h5py
    seed=expected["ordered_splits"]["base"]["train"][0];frame=40
    with h5py.File(ROOT/"data/base/episodes.h5") as raw,h5py.File(ROOT/"data/background_masks/base.h5") as masks:
        fig,axes=plt.subplots(2,3,figsize=(9,6))
        for row,camera in enumerate(("front","wrist")):
            rgb=raw[str(seed)][camera][frame];mask=masks[str(seed)][camera][frame].astype(bool)
            changed=rgb.copy()
            changed[mask]=(rgb[mask]*np.array(TRAIN_PALETTE[1])).astype(np.uint8)
            for ax,img,title in zip(axes[row],(rgb,mask,changed),("Source expert RGB","Table-only training mask","Augmented training RGB")):
                ax.imshow(img,cmap="gray" if img.ndim==2 else None);ax.set_title(camera+": "+title,fontsize=9);ax.axis("off")
        fig.suptitle(f"Scripted IK training data, seed {seed}, frame {frame} (not autonomous inference)",fontsize=10)
        fig.tight_layout();fig.savefig(OUT/"augmentation_example.png",dpi=160);plt.close(fig)
    status="success" if scores[augmented,"background_test"]["successes"] else "failure"
    source=ROOT/"results"/augmented/"background_test-background_test-dense"/(status+".mp4")
    shutil.copy2(source,OUT/"demo.mp4")
    frames,fps=video_frames(source)
    import imageio.v2 as imageio
    stride=max(1,math.ceil(len(frames)/90))
    imageio.mimsave(OUT/"demo.gif",frames[::stride],duration=1000*stride/fps,loop=0)
    imageio.imwrite(OUT/"demo_preview.png",frames[len(frames)//2])
    delta={c:scores[augmented,c]["successes"]-scores[control,c]["successes"] for c in expected["conditions"]}
    audit={"passed":True,"scores":{n:{c:scores[n,c]["successes"] for c in expected["conditions"]} for n in NAMES},
        "deltas_percentage_points":delta,"same_initial_parameter_sha256":next(iter(initial)),
        "training_updates":{n:train[n]["steps"] for n in NAMES},"old_models_unchanged":True,
        "diagnostic_episodes":40,"new_development_episodes":80,"final_episodes":400,
        "unique_final_initial_conditions":100,"per_color":per_color,"paired_counts":paired,
        "failure_reasons":failures,"diagnostics":diagnostics,
        "artifacts_sha256":{f:digest(OUT/f) for f in ("comparison.png","augmentation_example.png","demo.mp4","demo.gif")},
        "source_commit":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()}
    (OUT/"verification.json").write_text(json.dumps(audit,indent=2))
    report="# 背景泛化：换一张桌子，还会抓方块吗？\n\n"
    report+=f"本机已实际训练、加载保存权重并完成自主评测。新背景变化 {delta['background_test']:+d} 个百分点，原背景变化 {delta['standard']:+d} 个百分点。只报告本次单一训练种子的观察。\n\n"
    report+="## 同题对照成绩\n\n| 版本 | 部署快照步数 | 原桌面 | 训练没见过的四种桌色（合计） |\n| --- | --- | --- | --- |\n"
    for name in NAMES:
        report+=f"| {md_labels[name]} | {read(Path(frozen[name])/'experiment.json')['training_step']} | {scores[name,'standard']['successes']}/100 | {scores[name,'background_test']['successes']}/100 |\n"
    report+="\n最终场景固定为种子60000—60099，两模型、两条件使用同一批100个初始条件，合计400回合。新背景按种子分成四组，每种25回合，不是每种100回合。不能把本表与旧测试集61或76分直接相减来认定进步。\n\n"
    report+="## 每种新桌色\n\n| MuJoCo桌面RGB因子 | 对照 | 背景增强 |\n| --- | --- | --- |\n"
    for i,color in enumerate(TEST_PALETTE):
        report+=f"| {tuple(color)} | {per_color[control][str(i)]['successes']}/25 | {per_color[augmented][str(i)]['successes']}/25 |\n"
    report+="\n## 为什么要做这一步\n\n"
    report+=f"先在单独开发场景22000—22019上换成固定深色桌面：旧对照模型成功{diagnostics['brightness_control']}/20，旧亮度增强模型成功{diagnostics['brightness_aug']}/20。它们提示了背景弱点；本轮正式结论使用重新训练的同预算对照。\n\n"
    report+="## 新模型开发诊断\n\n| 版本 | 原桌面 | 固定深色桌面 |\n| --- | --- | --- |\n"
    for name in NAMES:report+=f"| {md_labels[name]} | {development[name,'standard']['successes']}/20 | {development[name,'background']['successes']}/20 |\n"
    report+="\n配方预先固定为8种训练桌色，没有根据最终成绩重选范围或训练次数。\n\n"
    report+="## 同场景的进退\n\n| 条件 | 两版都成功 | 原失败新成功 | 原成功新失败 | 两版都失败 |\n| --- | --- | --- | --- | --- |\n"
    for c,q in paired.items():report+=f"| {md_conditions[c]} | {q['both_success']} | {q['gained']} | {q['lost']} | {q['both_failure']} |\n"
    report+="\n## 具体改变了什么\n\n老师的动作答案不变，只把老师照片里的桌面换颜色。使用MuJoCo几何分割识别桌面，给两台相机的同一训练样本使用同一个颜色因子。物体、机械臂与目标标记的非桌面像素保持不变；边缘抗锯齿处的图像变换属于近似。\n\n"
    report+="不是凭颜色猜哪里是桌面：重放已有100条训练示范，逐帧核对关节状态、动作、物体位置和阶段之后才保存桌面掩码。原始图片、状态、动作与验证集没有改写。验证和自主推理均不使用掩码，也不读取物体真值来生成动作。\n\n"
    report+="训练桌色是8个预定因子（包括原色），每次抽到样本时随机选一个；测试在MuJoCo中真实改变桌面外观，使用4个不同的因子。物理参数、相机、位置范围和成功规则保持一致。\n\n"
    report+="## 训练与公平性\n\n"
    report+="- 两版同为base的80条加random的20条，共100条训练示范；验证为base独立20条。\n- ACT各从相同初始参数训练6000次，batch8、seed42；ResNet18视觉骨干使用ImageNet预训练。没有叠加亮度增强。\n- 增强独立随机数不会消耗Torch模型初始化和采样随机数；网络、关节归一化相同。\n- 每1000步保存快照，按原图验证动作误差选择；验证固定等距200帧动作块，来自20个验证回合。\n- 两模型均加载部署权重，输入两路RGB和6维关节状态，每30步重新观察；没有调用专家。\n- 成功仍须双侧夹爪接触并抬起、移入目标、释放和稳定1秒。\n\n"
    report+="## 失败类型\n\n"
    report+="| 版本 | 条件 | 没夹起来 | 掉落或放偏 | 释放或稳定未达标 |\n| --- | --- | --- | --- | --- |\n"
    for name in NAMES:
        for c in expected["conditions"]:
            reasons=failures[name][c]
            report+=f"| {md_labels[name]} | {md_conditions[c]} | {reasons.get('missed_grasp',0)} | {reasons.get('outside_target_or_drop',0)} | {reasons.get('release_or_stability',0)} |\n"
    report+="\nmissed_grasp=没夹起来；outside_target_or_drop=搬运掉落或放偏；release_or_stability=释放或稳定条件未满足。\n\n"
    report+="## 结果的边界\n\n只有一个训练种子与四种桌色，不能证明对所有房间、纹理、相机和真实机器人都可靠。它是桌面颜色泛化实验，尚未完成新纹理、真实硬件或sim-to-real验证。80/100是展示目标，实际分数以本表为准。\n\n"
    report+="![对照成绩和训练曲线](comparison.png)\n\n![真实训练图片的桌面增强](augmentation_example.png)\n\n"
    report+=f"自主视频：demo.mp4，来源 {source.relative_to(ROOT)}，类型为{status}，不是专家录像。对应结果目录也保存各模型、各条件的失败视频和轨迹。\n\n"
    report+="复现入口：run_background.ps1；仅看成绩：run_background.ps1 -Status。旧亮度实验按提交cc7f803复现，旧protocol和权重未改。\n"
    (OUT/"实验报告.md").write_text(report,encoding="utf-8")
    vault=ROOT.parents[1]/"obsdian知识库"
    assets=(("comparison.png","SO101-ACT-背景泛化对照.png"),("augmentation_example.png","SO101-ACT-桌面增强实例.png"),
            ("demo.mp4","SO101-ACT-新背景自主.mp4"),("demo.gif","SO101-ACT-新背景演示.gif"))
    for filename,title in assets:shutil.copy2(OUT/filename,vault/"附件"/title)
    note=report.replace("![对照成绩和训练曲线](comparison.png)","![[SO101-ACT-背景泛化对照.png]]").replace(
        "![真实训练图片的桌面增强](augmentation_example.png)","![[SO101-ACT-桌面增强实例.png]]")
    note+="\n\n## 看实际执行\n\n![[SO101-ACT-新背景自主.mp4]]\n\n继续读：[[SO101-14-域随机化用大白话讲]] · [[SO101-00-从这里开始]]。\n"
    (vault/"知识点/SO101完整闭环零基础/SO101-13-背景泛化实验结果.md").write_text(note,encoding="utf-8")
    for relative in ("实战/07-SO101-ACT完整闭环.md","知识点/SO101完整闭环零基础/SO101-00-从这里开始.md"):
        target=vault/relative;text=target.read_text(encoding="utf-8")
        if "[[SO101-13-背景泛化实验结果]]" not in text:
            target.write_text(text+"\n\n第三阶段真实实验：[[SO101-13-背景泛化实验结果]]；通俗解释：[[SO101-14-域随机化用大白话讲]]。\n",encoding="utf-8")
    print(json.dumps(audit),flush=True)

if __name__=="__main__":main()
