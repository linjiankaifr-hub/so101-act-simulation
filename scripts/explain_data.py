"""Create a beginner explanation using real recorded expert frames and joint values."""
import json
import h5py,numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from so101_env import ROOT,JOINTS

def main():
    manifest=json.loads((ROOT/"data/base/manifest.json").read_text())
    seed=manifest["train"][0]
    with h5py.File(ROOT/"data/base/episodes.h5") as h:
        g=h[str(seed)]
        indexes=[0,35,65,90,120,150]
        fig,axes=plt.subplots(2,3,figsize=(10,6))
        for ax,index in zip(axes.flat,indexes):
            ax.imshow(g["front"][index]);ax.axis("off")
            phase=g["phase"][index].decode()
            ax.set_title(f"t={g['timestamp'][index]:.2f}s | {phase}",fontsize=10)
        fig.suptitle(f"Recorded IK expert demonstration | seed {seed} | NOT learned policy")
        fig.tight_layout();fig.savefig(ROOT/"results/data_example.png",dpi=150);plt.close(fig)
        frame=40
        sample={"seed":seed,"frame":frame,"time_seconds":float(g["timestamp"][frame]),
            "state_rad":g["state"][frame].tolist(),"action_rad":g["action"][frame].tolist(),"joints":list(JOINTS)}
    (ROOT/"results/data_example.json").write_text(json.dumps(sample,indent=2))
    vault=ROOT.parents[1]/"obsdian知识库"
    import shutil
    shutil.copy2(ROOT/"results/data_example.png",vault/"附件/SO101-ACT-真实示范数据.png")
    text="# 用真实一帧看懂数据\n\n[[SO101-00-从这里开始]] · [[SO101-02-示范数据究竟是什么]]\n\n"
    text+="下面是本机实际记录的**脚本专家示范**，不是训练后模型的成功录像。模型的独立考试成绩见[[07-SO101-ACT完整闭环]]。\n\n![[SO101-ACT-真实示范数据.png]]\n\n"
    text+=f"以种子{seed}的第{frame}帧为例，时间是{sample['time_seconds']:.2f}秒。还有前视和手腕两张照片。照片和下面的状态共同作为模型输入，动作列是希望模型学会输出的答案。\n\n"
    text+="| 关节 | 当前状态/rad | 下一个目标/rad |\n| --- | --- | --- |\n"
    for name,state,action in zip(JOINTS,sample["state_rad"],sample["action_rad"]):
        text+=f"| {name} | {state:.4f} | {action:.4f} |\n"
    text+="\nrad是弧度：约1.5708rad等于90度。不要把rad数字直接当度数交给别的控制接口。\n\n"
    text+="一帧是一个时刻，不是一整个动作。一个回合通常约160帧，20帧代表1秒。ACT取某一帧的照片和关节状态，学习从这一时刻往后30步的动作答案。训练时可以随机抽帧，动作块仍然只能来自同一回合。\n\n"
    text+="读取关节状态等于问仿真器“各个关节现在在哪里”；真实机械臂通常从电机编码器等接口读取，接口单位需要另外确认。状态不是动作，动作不是方块坐标。\n\n"
    text+="这张示范图只能说明数据长什么样。老师会做，不等于学生已学会；必须加载学生自己的权重重新考试。\n"
    (vault/"知识点/SO101完整闭环零基础/SO101-12-用真实一帧看懂数据.md").write_text(text,encoding="utf-8")
    print(json.dumps(sample),flush=True)

if __name__=="__main__":main()
