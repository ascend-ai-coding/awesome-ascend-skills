# Attention 族策略选型指南

> 算子族：`attention` / `flash_attention`
> 适用于 attention 系列变体：`SoftmaxAttention`, `FlashAttention`, `GQA`, `MQA`, `SparseAttention`, `ChunkPrefill` 等
> 关联设计：[knowledge-strategy-architecture-v3.2](../../../../../docs/design/knowledge-strategy-architecture-v3.2.md) §3.2

## 选型决策树

```
是否使用 FlashAttention 结构（Online Softmax）？
├── 是
│   ├── 长序列 (S2 > 4K) → 必备：P14 CV 预发射 + P18 L1 7-Buffer + P38 Softmax 状态常驻
│   ├── 稀疏 Attention → P45 Sparse Block / P47 对角线调度
│   └── decode 模式 (S1=1) + KV cache → P60 KV NZ 格式
└── 否 (传统 Softmax + matmul)
    ├── Softmax stability → A5 数值安全 (max-trick)
    └── Softmax + matmul 流水 → P5 + P28 显式 HardEvent
```

## 族特定策略表（10 张）

| 策略 | Tier | Playbook | 触发场景 | 核心手段 |
|---|---|---|---|---|
| **P14** CV 流水预发射 | L2 | ✅ | FA + matmul 与 vector 并存 | Kernel 主循环三段结构，提前发射对侧 buffer |
| **P18** L1 7-Buffer Resident | L2 | — | FA 长序列 + Q/K/V/mask/out 并存 | L1 划 7 个分区，Q/K/V/dQ/dK/dV/mask 各占独立区 |
| **P38** Softmax 状态常驻 | L1 | — | FA 跨 S2 循环维护 m/l 状态 | 把 max/sum 状态留 UB，避免每轮 reload |
| **P55** BlockReduceMax 替代 DataCopy slice | L1 | ✅ | attention/normalization + max 归约 | 用 `BlockReduceMax<half>()` 一条指令替代 stride DataCopy |
| **P5** 流水线同步优化 | L0 | ✅ | 所有 attention（基础） | HardEvent 升级，避免过度 PipeBarrier |
| **P68** 低延迟归约组合 | L1 | — | softmax 的 reduce_max + reduce_sum | 树状归约 + Vector Counter 组合 |
| **P81** Resident UB/TBuf 复用 | L1 | — | FA 反向，多个临时 buffer | UB 划分常驻区，dQ/dK/dV 中间结果复用 |
| **A3** Rounding Mode | L0 | — | FA 精度对齐基线 | 显式控制 fp16→fp32 舍入 |
| **A4** SetFlag/WaitFlag 事件同步 | L0 | — | FA 流水节点间精度 | 用 event 替代 barrier，保数据依赖 |
| **A5** 数值安全 | L0 | — | Softmax 数值稳定（max-trick） | 减去 max 防 exp 上溢 |

## 推荐组合（cannbot 风格 golden combo）

| 组合 | 适用 | 预期收益 |
|---|---|---|
| **P1 + P5 + P14 + P18** | FA 长序列前向（FlashAttention-2 标准实现） | 接近理论极限 |
| **P1 + P5 + P55 + P67** | Softmax + scalar_compute 瓶颈 | 已在 sparse_flash_attention_gqa 验证 0.989x（P55+P67） |
| **P14 + P79** | FA backward, d=64/128 | Cube2+Cube3 融合显著缩减 |
| **P38 + P81** | FA decode + KV cache | UB 占用降 30%+ |

## 叠加正交性

| 优化 | P14 CV 预发射 | P18 L1 7-buffer | P38 状态常驻 |
|---|---|---|---|
| **P14** | — | 正交 | 正交 |
| **P18** | 正交 | — | 协同（L1 + UB 都常驻）|
| **P38** | 正交 | 协同 | — |

## 已 Playbook 化的策略

- ✅ P1, P5, P14, P55, P67 已有完整 6 步 SOP（含 grep 自检）
- ❌ P18, P38, P81 等 7 张待 Phase C 补 Playbook
