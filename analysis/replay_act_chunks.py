"""CPU-only action-chunk replay on three fixed, previously saved observations."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
root=Path(a.root);sys.path.insert(0,str(root/'scripts'))
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy

torch.set_num_threads(2)
checkpoint=root/'checkpoints/background_control/deploy'
protocol=json.loads((root/'results/closed_loop/protocol.json').read_text())
for name,digest in protocol['checkpoint_hashes'].items():
    assert hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()==digest,name
config=PreTrainedConfig.from_pretrained(checkpoint,local_files_only=True)
config.device='cpu'
policy=ACTPolicy.from_pretrained(checkpoint,config=config,local_files_only=True,strict=True).eval()
stats={k:np.array(v,dtype=np.float32) for k,v in json.loads((checkpoint/'normalization.json').read_text()).items()}
folder=root/'results/closed_loop/test/standard-h5'
with np.load(folder/'failure-frames.npz') as f:frames={k:f[k].copy() for k in f.files}
with np.load(folder/'failure-trace.npz') as f:trace={k:f[k].copy() for k in f.files}
assert int(frames['seed'])==70000
mean=torch.tensor([.485,.456,.406])[None,:,None,None]
std=torch.tensor([.229,.224,.225])[None,:,None,None]
rows=[];chunks={}
for step in [60,150,250]:
    indexes=np.flatnonzero(frames['steps']==step);assert len(indexes)==1
    index=int(indexes[0])
    # Match SO101Env.observe(): saved physical state is float64, policy observation is float32.
    observed_state=trace['states'][step].astype(np.float32)
    batch={'observation.state':torch.from_numpy((observed_state-stats['state_mean'])/stats['state_std'])[None]}
    assert batch['observation.state'].dtype==torch.float32
    for key,short in [('observation.images.front','front'),('observation.images.wrist','wrist')]:
        image=torch.from_numpy(frames[short][index].copy()).permute(2,0,1).float()[None]/255.
        batch[key]=(image-mean)/std
    with torch.inference_mode():normalized=policy.predict_action_chunk(batch)[0].numpy()
    chunk=normalized*stats['action_std']+stats['action_mean']
    assert chunk.shape==(30,6) and np.isfinite(chunk).all()
    head_error=float(np.abs(chunk[:5]-trace['actions'][step:step+5]).max())
    # Different CPU/GPU kernels can introduce small numeric differences; verify the replay explicitly.
    assert head_error<1e-3,f'CPU replay does not reproduce saved executed prefix at {step}: {head_error}'
    chunks[str(step)]=chunk.tolist()
    rows.append({'saved_control_step':step,'cpu_vs_saved_gpu_prefix_max_abs_rad':head_error,
                 'arm_change_from_prediction_start_at_offset4_rad':float(np.linalg.norm(chunk[4,:5]-chunk[0,:5])),
                 'arm_change_from_prediction_start_at_offset29_rad':float(np.linalg.norm(chunk[29,:5]-chunk[0,:5])),
                 'gripper_target_offset0_rad':float(chunk[0,-1]),'gripper_target_offset4_rad':float(chunk[4,-1]),
                 'gripper_target_offset29_rad':float(chunk[29,-1])})
out=root/'results/closed_loop/diagnosis-20261021'
(out/'saved-observation-chunks.json').write_text(json.dumps({'device':'cpu','seed':70000,'fixed_saved_steps':[60,150,250],
            'checkpoint_sha256':protocol['checkpoint_hashes'],'rows':rows,'raw_predicted_chunks_rad':chunks,
            'interpretation':'Tail is an unexecuted model prediction from a frozen historical observation, not an actual successful rollout.'},indent=2),encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,2,figsize=(10,8),sharex=True)
for i,r in enumerate(rows):
    chunk=np.array(chunks[str(r['saved_control_step'])])
    axes[i,0].plot(np.arange(30),np.linalg.norm(chunk[:,:5]-chunk[0,:5],axis=1),color='#3d7c9e')
    axes[i,1].plot(np.arange(30),chunk[:,-1],color='#bc754f')
    axes[i,0].set_ylabel(f"Saved step {r['saved_control_step']}\nTarget change (rad)")
    for ax in axes[i]:
        ax.axvspan(-.5,4.5,color='#a4cfb6',alpha=.35,label='Executed 5-step prefix')
        ax.grid(alpha=.2)
axes[0,0].set_title('Arm target displacement from chunk start')
axes[0,1].set_title('Predicted gripper target angle (rad)')
for ax in axes[-1]:ax.set_xlabel('Predicted action offset (0..29)')
fig.suptitle('CPU replay of frozen historical inputs | same seed70000 model\nGreen = executed prefix; remainder = unexecuted prediction, not a rollout')
fig.tight_layout();fig.savefig(out/'saved-observation-chunks.png',dpi=160);plt.close(fig)
table='\n'.join(f"| {r['saved_control_step']} | {r['cpu_vs_saved_gpu_prefix_max_abs_rad']:.2e} | {r['arm_change_from_prediction_start_at_offset4_rad']:.4f} | {r['arm_change_from_prediction_start_at_offset29_rad']:.4f} |" for r in rows)
text=f'''# 固定历史观测下的ACT动作块重放

只在CPU上加载原background_control权重，读取5步窗口失败回合seed70000的第60、150、250控制步已保存图像与本体状态，重新预测完整30步。没有运行环境、施加新动作、训练或计算新的成功率。

## 数值核对

| 保存的控制步 | 重放前5步与原实际指令的最大绝对差rad | 预测offset4相对offset0的前五关节目标变化rad | offset29相对offset0变化rad |
|---|---:|---:|---:|
{table}

前5步与历史实际指令的差小于1e-3rad，核对了本次CPU重放与历史GPU执行前缀的数值一致性。尾部变化描述模型在同一张历史观测下继续预测了什么，并不表示执行尾部一定能抓起，也不是已验证的纠错方案。

![历史观测动作块](saved-observation-chunks.png)

绿色区域是历史5步窗口真正执行的前缀；后续预测当时被丢弃。该诊断可以提出“短窗口反复重启相似动作前段，未充分推进”的假设，但它只覆盖一个失败初始条件。验证机制及改进收益仍需独立开发和新测试，不能由这三次重放直接作因果结论。

完整预测、模型指纹与数值误差见saved-observation-chunks.json。图中动作是预测目标，不是实际机器人状态或物体轨迹。
'''
(out/'历史观测重放.md').write_text(text,encoding='utf-8')
print(json.dumps({'device':'cpu','rows':rows,'no_new_rollout':True},ensure_ascii=False),flush=True)
