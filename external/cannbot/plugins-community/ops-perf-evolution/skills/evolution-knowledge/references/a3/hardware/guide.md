# 硬件知识快速参考

## 硬件参数获取

具体硬件参数（UB 大小、核数、带宽、向量宽度等）由 `npu-arch` skill 在 Init 阶段动态检测（数据源：CANN 包芯片配置文件 `platform_config/*.ini` 与 `PlatformAscendC` 运行时接口）并写入 `world_model.json` 的 `hw_params` 字段。

**不要使用本文件中的硬编码数字**，始终以 `hw_params` 中的实际值为准。

## 快速判断算子瓶颈类型

以下启发式规则适用于所有昇腾芯片（阈值为经验值，不依赖具体型号）：

```
算术强度 < 10 FLOPs/byte (elementwise/reduction/softmax)
  → 内存密集型 → 优先: 双缓冲(P1) + 大tile(P2) + 向量化搬运(P10)

算术强度 > 100 FLOPs/byte (matmul/attention)
  → 计算密集型 → 优先: 算法级优化 + 混合精度(D1)

MTE2 ≈ VECTOR (within 10%)
  → 平衡态 → 指令级优化收益递减，需算法级突破
```

## Tiling 上限公式

```
max_tile_elements = ub_size / (buf_num × pipe_num × sizeof(T))
```

其中 `ub_size` 从 `hw_params.ub_size_bytes` 获取，`buf_num` 为缓冲区数（双缓冲=2），`pipe_num` 为流水级数（通常=3: CopyIn/Compute/CopyOut），`sizeof(T)` 为数据类型字节数。结果需按 32B 对齐向下取整。

## 深入阅读

需要详细流水线模型、UB 规划公式、DMA 效率指标 → 读 [ascend910b_arch.md](ascend910b_arch.md)
