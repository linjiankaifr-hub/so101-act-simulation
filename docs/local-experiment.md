# SO-101视觉抓取与ACT学习闭环

任务：通过两路相机与自身关节状态，将红色方块从随机位置抓起，放到目标区域，松开并稳定一秒。包含脚本专家数据、官方LeRobot ACT实际训练、独立测试、失败示范回流与随机补数据对照。

本机已完成140条成功专家示范、三版各6000次实际更新、300回合标准测试与90回合扰动测试。基础/针对性补数据/随机补数据分别58/100、56/100、61/100，80/100目标尚未达到。详见[实际实验报告](results/实验报告.md)、[失败分析](docs/failure-analysis.md)和[系统关系图](docs/system.md)。

## 入门先看

Obsidian：SO101-00-从这里开始。代码、模型与原始数据都在此项目目录；第一版为仿真，没有真实硬件部署。

在当前目录PowerShell：

~~~powershell
.\run.ps1 -Task status
.\run.ps1 -Task test
.\run.ps1 -Task demo -Variant base
.\run.ps1 -Task evaluate -Variant base -Episodes 10
~~~

demo优先打开模型自主成功视频；如果模型视频还未生成，会显示并打开文件名明确的expert_demo.mp4（专家示范，不是模型成绩）。evaluate会另存到带manual和时间的目录，保留正式100回合成绩。

## 任务与模型约定

- SO101六维状态/动作：shoulder_pan、shoulder_lift、elbow_flex、wrist_flex、wrist_roll、gripper，均为rad关节位置/目标。
- 输入：front和wrist两路96×96 RGB、6维关节位置。策略不读取方块真值。
- 动作：官方ACT一次预测30步，先执行30步再重新观察。控制20Hz、模型查询约0.67Hz仿真时间。
- 范围：24mm红方块，初始中心x=20.5—25.0cm、y=−2.5—2.5cm；固定目标(x,y)=(22.5,−9.5)cm，均为仿真世界坐标。位置扰动将y扩大到±4cm，其他条件另记。
- 物理：MuJoCo3.3.7，dt=.002，25个物理步组成一次控制步。原模型接触，无绑定或传送方块。
- 成功：两侧夹爪真实接触并抬高到4.5cm以上、方块中心在目标中心每轴3cm内、夹爪张开且TCP离开、速度足够小、回到桌面并保持至少1秒。
- 专家用真值和数值IK制作示范；自主评测不导入专家模块。
- ACT小模型约1555万参数，Transformer维度256、两层encoder；ResNet18 ImageNet预训练，ACT动作模型随机初始化训练。
- RGB按ImageNet统计归一化，状态/动作仅用训练集统计；normalization.json随权重保存。

## 数据与公平对照

base首批100条成功示范，80训练20验证；targeted和random各补20条进入训练。按完整回合划分，开发/最终测试分开。三版本同结构、更新预算、batch和训练种子，在相同100个测试种子下比较。

原始无损数组：data/<name>/episodes.h5。官方格式：data/<name>/lerobot。训练使用HDF5镜像提高本机I/O速度，官方导出读回审计确保状态与动作一致。训练没有额外下载抓取策略。

## 目录

| 目录 | 内容 |
| --- | --- |
| official | 未修改的上游源码和模型许可证 |
| scripts | 环境、专家、采集、训练、评测、ROS2桥和报告 |
| tests | 物理、数据、动作块、重载、测试隔离与ROS契约 |
| data | 示范数组、LeRobot导出、回合划分 |
| checkpoints | 本机ACT权重、归一化、训练配置和日志 |
| results | 逐次测试、真实视频、曲线和结果 |
| docs | 用户批准计划、执行台账、面试材料 |
| logs | 安装、采集、训练与验证输出 |

## 本机环境与重跑

Windows+WSL Ubuntu24.04；专用环境 /home/buntu/.venvs/codex-so101-act；Python3.12，Torch2.7.1+cu126，LeRobot0.5.1。完整版本见requirements-lock.txt，不修改此前六个实战环境。

本机渲染用WSLg的GLFW+D3D12，实测识别RTX4060。Linux原生系统需要按自己的OpenGL驱动修改common.sh；必要时设置MUJOCO_GL=osmesa回退软件渲染。common.sh中的D3D12设置属于此机器，不能当通用Linux安装步骤。

新环境安装执行scripts/setup.sh；该脚本复用已有uv和系统OSMesa/GLFW库。完整流程入口是run.ps1 -Task full，过程耗时较长。已完成阶段保留；中断的数据、导出和训练目录移到对应的_interrupted中再重做，不删除原数据。训练中断后按原种子从头重训，该版本没有不完整的断点续训。评测按权重、配置、归一化、精确种子和动作窗口共同校验缓存。

ROS2演示：run.ps1 -Task ros -Variant base。域ID47，发布/so101/front/image_raw、/so101/wrist/image_raw、/so101/joint_states和/so101/applied_action；提供/so101/reset与/so101/status。没有实物串口驱动被启用。

## 贡献与边界

见THIRD_PARTY.md。基于SO101 ROS2 MuJoCo模型资产与官方LeRobot ACT，本地实现采集、训练适配、评测和学习闭环。结果只对应记录的仿真分布；不声称覆盖语音、SLAM、触觉或学习世界模型。真实机械臂仍需标定、动作单位适配、真实示范和独立实物测试。

## 第二阶段：训练图片亮度增强

已完成两组各6000步实际训练、80回合开发测试和400回合新测试。对照/增强在普通光照分别65/100、76/100，变暗光照64/100、68/100。80/100目标尚未达到；一个训练种子不能证明普遍收益。

新旧测试种子不同，比较应使用本轮配对结果，不能直接拿76与旧61认定提升。原始数据和旧模型已通过指纹校验保留。

查看历史成绩：run_brightness.ps1 -Status。该轮精确重跑使用源码提交cc7f803与run_brightness.ps1；当前背景实验扩展了核心源码，旧亮度协议会按设计拒绝不同源码指纹，不能覆盖旧protocol来绕过。完整协议见[brightness-protocol](docs/brightness-protocol.md)，实测见[亮度增强报告](results/brightness/实验报告.md)，实现取舍与代码复查见[执行记录](docs/brightness-progress.md)。原入口run.ps1继续用于第一轮模型。

Obsidian新增SO101-10实验结果、SO101-11增强原理、SO101-12真实一帧数据。对照图与模型在变暗光照下的实际执行视频均已保存。

## 第三阶段：桌面背景泛化

已完成100条原训练示范的16000帧对齐重放，仅补充桌面几何掩码；两组同初始化、同数据各训练6000次。正式100个新初始条件配对评测：原桌面普通对照52/100、桌色增强58/100；四种新桌色合计33/100、45/100，每种25次。共400个正式回合，另有40次旧模型诊断、80次新开发测试。80/100目标尚未达到，一个训练种子不代表普遍收益。

查看成绩：run_background.ps1 -Status；完整匹配缓存流程：run_background.ps1；Windows独立审计：scripts/audit_background.ps1。具体方法见[背景协议](docs/background-protocol.md)，实际成绩、曲线、失败统计见[背景实验报告](results/background/实验报告.md)，验证记录见[执行进度](docs/background-progress.md)与[代码审查](docs/background-review.md)。

Obsidian新增SO101-13背景泛化结果与真实自主录像、SO101-14域随机化通俗解释和流程图。当前源码扩展不改变旧模型与原始示范；旧亮度实验的精确源码是cc7f803。

## 第四阶段：动作窗口与外力干扰

固定同一个background_control模型，预测30步，执行5/10/30步。100个新初始条件下，正常条件分别[22, 27, 58]，外力分别[17, 12, 0]，均为成功次数/100。另有120次开发检查，正式600次执行只有100个不同初始条件。

保留全部窗口、配对得失、失败标签、95%区间和推理开销。本轮只改变部署执行方式，没有重训或实现专门失败恢复控制器。固定时刻脉冲可能帮助或损害抓取，不能把外力成功数称为恢复率。

完整结果见[动作窗口实验报告](results/closed_loop/实验报告.md)，协议见[closed-loop-protocol](docs/closed-loop-protocol.md)。入口run_closed_loop.ps1，状态参数-Status；独立审计scripts/audit_closed_loop.ps1。Obsidian新增SO101-20/21/22。
