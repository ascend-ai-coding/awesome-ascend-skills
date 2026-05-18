# Training

面向训练链路、通信、强化学习与 MindSpeed 系列训练流程的开发入口目录。

当前 `skills/training/` 已承载真实 skill 目录；请按下面链接进入对应 skill 开发：

- [`hccl-test/`](hccl-test/)：HCCL 通信基准与带宽测试
- [`torch-npu-comm-test/`](torch-npu-comm-test/)：torch.distributed 通信性能测试
- [`mindspeed-llm/`](mindspeed-llm/)：MindSpeed-LLM 环境、数据、权重与训练全流程
- [`mindspeed-mm/`](mindspeed-mm/)：MindSpeed-MM 多模态训练环境、权重、VLM、生成模型与流水线
- [`verl-quickstart/`](verl-quickstart/)：VERL 强化学习训练 quickstart
- [`../profiling/training-mfu-calculator/`](../profiling/training-mfu-calculator/)：训练 MFU 计算与分析

推荐场景：

- 分布式训练准备
- 通信链路验证
- MindSpeed-LLM / MindSpeed-MM / VERL 训练流程开发

对应 bundle：`ascend-training`
