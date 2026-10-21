# 上游来源与贡献边界

- SO101 ROS2 MuJoCo：https://github.com/MuRain37/so101-ros2-mujoco ，Apache-2.0，源码保存在 official/so101-ros2-mujoco，SHA 6f63e78ccaac928c55e2dbe515866c3a013e599d。
- LeRobot：https://github.com/huggingface/lerobot ，Apache-2.0，v0.5.1，SHA 1396b9fab7aecddd10006c33c47a487ffdcb54b4。实际ACT模型/损失/权重读写使用该版本官方安装包。
- SO101模型来源链包括 TheRobotStudio/SO-ARM100：https://github.com/TheRobotStudio/SO-ARM100 ，保留上游模型说明和许可证；接触几何还引用 MuJoCo Menagerie。未重新许可这些资产。
- MuJoCo 3.3.7：https://github.com/google-deepmind/mujoco ，Apache-2.0。
- ACT论文与原实现：https://tonyzhaozh.github.io/aloha/ ，Action Chunking with Transformers。
- ResNet18视觉骨干初始权重为 torchvision ImageNet1K_V1，来源 https://download.pytorch.org/models/resnet18-f37072fd.pth 。ACT动作模型不是预训练的抓取策略。

本地实现贡献：单任务评测封装、严格放置判定、数值IK示范生成、数据审计与分割、官方ACT的轻量配置及训练适配、失败开发集回流和对照实验、可重跑入口与学习说明。没有声称发明ACT或独立训练视觉基础模型。

official目录保存完整上游LICENSE。上游代码未修改；任务位置、相机视角和渲染质量在本地封装中配置。批量训练使用HDF5镜像，LeRobot导出是同一原始数组，经官方读回校验。数据为脚本IK专家而非人类遥操作。
