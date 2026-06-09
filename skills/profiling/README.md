# Profiling

面向 Profiling 采集与性能分析的开发入口目录。

当前 `skills/profiling/` 已承载真实 skill 目录；请按下面链接进入对应 skill 开发：

- [`profiling-analysis/`](profiling-analysis/)：hostbound / computing / communication 分析
- [`mindspeed-llm-train-profiler/`](mindspeed-llm-train-profiler/)：MindSpeed-LLM 训练 Profiling 采集
- [`mindspeed-mm-train-profiler/`](mindspeed-mm-train-profiler/)：MindSpeed-MM 多模态训练 Profiling 采集
- [`pytorch-profiling-collection/`](pytorch-profiling-collection/)：基于 torch_npu.profiler 的通用训练/推理脚本 Profiling 采集（非 MindSpeed 场景）
- [`../ops/npu-op-benchmark/`](../ops/npu-op-benchmark/)：算子基准测试
- [`training-mfu-calculator/`](training-mfu-calculator/)：训练侧 FLOPs / MFU 分析

推荐场景：

- 采集性能数据
- 分析训练或推理瓶颈
- 定位 hostbound / communication / computing 问题

对应 bundle：`ascend-profiling`
