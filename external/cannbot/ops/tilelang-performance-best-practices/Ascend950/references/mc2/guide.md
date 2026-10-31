# 当前仓库通算组合优化

## 目标实现

当前 PTO DSL 没有在当前仓库中建立可复用的集合通信 kernel API，因此计算使用 examples/ascend/example_gemm.py，通信由当前运行时的正式 distributed API 承担。Python host wrapper 编排 local GEMM、异步 collective、remote GEMM 与 stream/event；不得在 T.prim_func 中伪造通信或跨核同步。

## 流程

1. 单独验证 local GEMM 精度与 latency。
2. 单独验证 collective 的 tensor layout、rank offset、stream 与完成语义。
3. 建立不融合的通信→计算或计算→通信 reference。
4. 按 chunk 启动异步通信并在独立 stream 执行对应 GEMM；只使用运行时公开 API。
5. 多 rank 同步验证后，再搜索 chunk size 与 local-first 排布。

精度覆盖全部 rank、非整 chunk、不同 world size 和 fp32 GEMM accumulator。性能同时报告通信、计算、端到端 latency、重叠率和膨胀；缺少运行时 API 时明确拒绝该优化。
