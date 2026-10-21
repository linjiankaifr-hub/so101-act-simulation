"""Analyze saved ACT evaluations; does not run or modify policies."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--root',required=True)
args=parser.parse_args()
ROOT=Path(args.root)
SOURCE=ROOT/'results/closed_loop'
OUT=SOURCE/'diagnosis-20261021'
OUT.mkdir(exist_ok=True)
hashes={}
def read(path):
    path=Path(path)
    hashes[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf-8'))
def load_npz(path):
    path=Path(path)
    hashes[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}
def stats(values):
    return {'mean':float(np.mean(values)),'p95':float(np.percentile(values,95)),'n':len(values)}

groups={}
for condition in ['standard','push']:
    for h in [5,10,30]:
        rows=read(SOURCE/'test'/f'{condition}-h{h}'/'episodes.json')
        assert len(rows)==100 and [r['seed'] for r in rows]==list(range(70000,70100))
        groups[condition,h]=rows
paired=[]
for c in ['standard','push']:
    for h in [5,10]:
        a=groups[c,30];b=groups[c,h]
        paired.append({'condition':c,'window':h,'both_success':sum(x['success'] and y['success'] for x,y in zip(a,b)),
                       'new_success':sum(not x['success'] and y['success'] for x,y in zip(a,b)),
                       'lost_success':sum(x['success'] and not y['success'] for x,y in zip(a,b)),
                       'both_failure':sum(not x['success'] and not y['success'] for x,y in zip(a,b))})

representatives=[]
traces={}
for h in [5,10,30]:
    directory=SOURCE/'test'/f'standard-h{h}'
    candidates=[]
    for f in directory.glob('*-frames.npz'):
        frames=load_npz(f)
        if int(frames['seed'])==70000:candidates.append(f.stem.replace('-frames',''))
    assert len(candidates)==1
    status=candidates[0]
    t=load_npz(directory/f'{status}-trace.npz')
    traces[h]=t
    assert t['actions'].shape==t['states'].shape and t['actions'].shape[1]==6
    arm_change=np.linalg.norm(np.diff(t['actions'][:,:5],axis=0),axis=1)
    grip_change=np.abs(np.diff(t['actions'][:,-1]))
    boundary=np.arange(1,len(t['actions']))%h==0
    representatives.append({'seed':70000,'condition':'standard','window':h,'success':groups['standard',h][0]['success'],
                            'steps':len(t['actions']),'max_cube_height_m':float(t['cube'][:,2].max()),
                            'minimum_tcp_cube_distance_m':float(np.linalg.norm(t['tcp']-t['cube'],axis=1).min()),
                            'arm_target_change_rad_boundary':stats(arm_change[boundary]),
                            'arm_target_change_rad_within_chunk':stats(arm_change[~boundary]),
                            'gripper_target_change_rad_boundary':stats(grip_change[boundary]),
                            'gripper_target_change_rad_within_chunk':stats(grip_change[~boundary]),
                            'actual_gripper_min_rad':float(t['states'][:,-1].min())})
out={'purpose':'Descriptive analysis of existing results; no new policy or recovery claims',
     'paired_counts':paired,'representatives':representatives,'source_sha256':hashes,
     'limitations':['All trace comparisons use only the preselected seed70000.',
                    'Window changes the trajectory, phase and number of boundary samples; boundary jump averages are descriptive, not causal.',
                    'Cube height does not prove verified two-jaw contact or sustained success.',
                    'Ground-truth object state is offline diagnostic data only; it was not a policy input.']}
(OUT/'diagnosis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(4,1,figsize=(10,10),sharex=True)
colors={5:'#dc784f',10:'#3c8c9d',30:'#7466b0'}
for h,t in traces.items():
    x=np.arange(len(t['actions']))/20
    ax[0].plot(x,t['cube'][:,2]*100,label=f'Execute {h} steps',color=colors[h])
    ax[1].plot(x,t['states'][:,-1],color=colors[h])
    ax[2].plot(x,t['actions'][:,-1],color=colors[h],alpha=.85)
    change=np.r_[np.nan,np.linalg.norm(np.diff(t['actions'][:,:5],axis=0),axis=1)]
    ax[3].plot(x,change,color=colors[h],alpha=.85)
ax[0].axhline(4.5,color='gray',linestyle='--',label='Height criterion only (not full success)')
ax[0].set_ylabel('Cube center height (cm)');ax[0].legend(fontsize=8)
ax[1].set_ylabel('Actual gripper angle (rad)')
ax[2].set_ylabel('Gripper target angle (rad)')
ax[3].set_ylabel('Arm target change norm (rad)');ax[3].set_xlabel('Simulation time (s)')
for a in ax:a.grid(alpha=.2)
fig.suptitle('ACT saved trajectories | same initial seed 70000 | normal condition\nSingle-case diagnostics; no new policy and no causal proof')
fig.tight_layout();fig.savefig(OUT/'saved-trajectory-diagnosis.png',dpi=160);plt.close(fig)

table='\n'.join(f"| {r['window']} | {r['steps']} | {'成功' if r['success'] else '失败'} | {r['max_cube_height_m']*100:.2f} | {r['arm_target_change_rad_boundary']['mean']:.4f} | {r['arm_target_change_rad_within_chunk']['mean']:.4f} |" for r in representatives)
pairtable='\n'.join(f"| {r['condition']} | {r['window']} | {r['both_success']} | {r['new_success']} | {r['lost_success']} | {r['both_failure']} |" for r in paired)
griptable='\n'.join(f"| {r['window']} | {r['gripper_target_change_rad_boundary']['mean']:.4f} | {r['gripper_target_change_rad_within_chunk']['mean']:.4f} | {r['actual_gripper_min_rad']:.4f} |" for r in representatives)
report=f'''# ACT 动作窗口失败分析

分析日期：2026-10-21。本轮只读取既有100组配对测试及预先指定的第一题seed70000轨迹，没有重训、改模型或增加评测成绩。

## 全体配对结果

以30步执行窗口为基线。新成功与退步必须同时报告，不能只展示被短窗口救回的回合。

| 条件 | 新窗口 | 都成功 | 原失败新成功 | 原成功新失败 | 都失败 |
|---|---:|---:|---:|---:|---:|
{pairtable}

正常条件缩短窗口的代价大于收益；外力条件则出现部分新成功。这支持“执行方式存在取舍”，尚不支持“短窗口实现失败恢复”。

## 同一初始条件的轨迹

以下是所有窗口共同的第一题seed70000，不是挑选最佳回合。动作变化量仅使用前五个关节目标的相邻步差范数，单位rad；块边界是重新查询模型的控制步，块内为同一预测块的其他相邻步。

| 执行窗口 | 步数 | 结果 | 方块最大高度cm | 块边界平均目标变化rad | 块内平均目标变化rad |
|---|---:|---|---:|---:|---:|
{table}

![已保存轨迹诊断](saved-trajectory-diagnosis.png)

5步和10步在这一个初始条件下没有将方块抬到高度阈值；30步成功完成。短窗口存在块边界目标变化，具体数值见表；但不能据此归因所有失败。窗口本身改变了机器人走过的路径、阶段和采样数，不能把不同轨迹的均值差直接当成因果效应。夹爪闭合角也不能单独证明正确夹住物体。

夹爪目标单独统计，避免前五关节变化范数掩盖夹爪指令的特点：

| 执行窗口 | 块边界夹爪目标平均变化rad | 块内平均变化rad | 实际夹爪最小角度rad |
|---|---:|---:|---:|
{griptable}

本回合5步窗口在重新查询边界有更明显的夹爪目标变化，但实际夹爪状态与目标不同，不能把目标震荡直接等同于物体掉落。5步和30步的最小实际夹爪角相近，物体高度却不同，也说明“夹爪闭合”不是“成功夹起”的充分证据。

## 可以讲的结论

- 更频繁观察不是所有条件下都更好：完整配对实验显示正常任务退步与外力条件收益并存。
- 失败分类主要是未达到有效夹起；这属于观测到的任务结果，不能直接等同于视觉错误。
- 保存的轨迹允许检查动作衔接、夹爪状态与物体高度，比仅看成功率更容易提出可检验假设。

## 下一项可检验的改进

优先在独立开发场景比较原动作块执行与一种动作衔接方案，固定权重、输入和任务判据。先验证是否减少块边界目标跳变、同时维持抓取推进，再锁定方案后评测新的测试场景。不能使用本次70000—70099反复调参并当作新的独立成绩。

本轮未实现时序聚合、自适应窗口或恢复控制器。仿真物体真值只用于离线分析，不加入策略观测；单个诊断回合也不代表全体机制。原始文件SHA256、全体配对次数及详细轨迹指标见diagnosis.json。
'''
(OUT/'分析报告.md').write_text(report,encoding='utf-8')
print(json.dumps({'output':str(OUT),'paired':paired,'representatives':representatives},ensure_ascii=False),flush=True)
