# Ops

面向算子开发、迁移与调优的开发入口目录。

当前 `skills/ops/` 已承载真实 skill 目录；请按下面链接进入对应 skill 开发：

- [`ascendc/`](ascendc/)：AscendC 算子端到端开发（ascend-kernel 自定义算子：初始化→设计→用例→代码生成→编译调试→文档→精度→性能→优化→检视）
- [`ascend-opplugin/`](ascend-opplugin/)：op-plugin 接入与自定义算子
- [`triton-ascend-migration/`](triton-ascend-migration/)：Triton / PyTorch 算子迁移
- [`npu-op-benchmark/`](npu-op-benchmark/)：算子基准测试与调优辅助

推荐场景：

- 自定义算子开发
- CUDA / Triton 到昇腾迁移
- 算子级性能调优

对应 bundle：`ascend-ops`
