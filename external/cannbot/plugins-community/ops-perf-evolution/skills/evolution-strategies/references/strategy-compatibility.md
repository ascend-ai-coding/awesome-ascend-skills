# Strategy Compatibility Guide

本文档定义策略之间的互斥关系、依赖关系、推荐组合、UB 内存预算指南、不适用场景，以及 Agent 快速决策流程。

## §1 互斥表

以下策略对不可同时出现在 `strategy_combination` 中：

| Strategy A | Strategy B | 冲突原因 |
|---|---|---|
| P1 (Double Buffer) | P22 (TQueBind) | P1 要求 queue depth=2，P22 绑定固定 queue 布局，depth 不兼容 |
| P1 (Double Buffer) | P19 (Custom PingPong) | P19 用手动事件管理替代 TQue 自动调度，与 P1 的 TQue 双缓冲冲突 |
| P1 (Double Buffer) | P20 (Triple Buffer) | P20 是 P1 的超集（depth=3 vs depth=2），不可叠加 |
| P52 (L2 Cache Hint) | P74 (Selective L2 Disable) | 两者都控制 L2 cache 行为但机制冲突：P52 用 hint 引导缓存，P74 直接禁用 |
| P72 (Split-K) | P9 (Deterministic) | P72 使用 AtomicAdd 跨核累加，结果非确定性，与 P9 的确定性输出要求冲突 |

### 条件性约束（非硬互斥，需注意协调）

| Strategy A | Strategy B | 约束条件 |
|---|---|---|
| P1 (Double Buffer) | P14 (CV Preload) | P14 管理 L1→L0 (Cube侧)，P1 管理 GM→UB (Vector侧)，在 CV 融合算子中可共存，但需协调 DMA 通道分配 |
| P3 (Small-D) | P4 (Load Balance) | P3 合并多行产生 tile，P4 分配 tile 到核。当合并后总 tile 数 < 核数时 P4 无意义；否则可协同工作 |
| P53 (L1 Resident) | P83 (L1 Resident Partition) | P83 是 P53 的多缓冲扩展版；若同时使用需确保 L1 分区方案一致，避免地址重叠 |
| P67 (Counter Mode) | P84 (Vector Efficiency) | P84 包含 Counter 模式作为子模式；若同时选中，以 P84 为准，P67 的独立 counter 逻辑不再单独应用 |
| P81 (Resident UB/TBuf) | P85 (Buffer Zone Reuse) | P81 管理参数级常驻，P85 管理区域级复用；可共存但需协调 UB 地址分配避免冲突 |
| P71 (IBShare L1) | P78 (On-chip Cache Matmul) | 两者都优化 Matmul 的 L1 使用；P71 侧重 B 矩阵共享，P78 侧重全局缓存加速，需确认 L1 容量足够 |

## §2 依赖关系

箭头表示"需要先包含"：左侧策略依赖右侧策略。

| 策略 | 依赖 | 原因 |
|---|---|---|
| P1 | P5, P7 | 双缓冲需要流水同步(P5)和对齐保证(P7) |
| P4 | P11 | 多核负载均衡需要尾块处理 |
| P14 | P5, P28 | CV 预加载需要流水同步和 HardEvent 生产者消费者协调 |
| P19 | P5, P28 | 自定义 PingPong 需要流水同步和 HardEvent 管理 |
| P20 | P5, P28 | 三缓冲需要流水同步和 HardEvent 管理 |
| P53 | P18 | L1 常驻复用需要 L1 分区方案(P18)作为基础 |
| P57 | P58 | FlashDecode G 分区归约依赖 TND 负载均衡分区 |
| P72 | P4 | Split-K 多核并行需要负载均衡(P4)分配 K 块 |
| P73 | P16 | AIC:AIV 核配比依赖代价模型分核(P16) |
| P79 | P46 | Grad Cube2+Cube3 融合依赖 MatmulImpl API(P46) |
| P83 | P53 | L1 多缓冲分区是 L1 常驻复用(P53)的扩展 |
| P86 | P46 | Matmul 数据通路调优依赖 MatmulImpl API(P46) |
| P87 | P46 | 手动 Mmad 时序控制依赖 MatmulImpl API(P46) |

### 推荐搭配（非硬依赖，但组合效果显著优于单独使用）

| 策略 | 推荐搭配 | 原因 |
|---|---|---|
| A1 | P1 | FP32 中间计算增加每 tile 计算量，双缓冲可掩盖增加的延迟；A1 单缓冲也能工作，但性能不如搭配 P1 |
| P55 | P56 | 块级 ReduceMax 优化搭配批量 DataCopy 输出，减少归约后的搬运开销 |
| P64 | P66 | GM 地址冲突规避搭配 512B 对齐，双重保证多核访存效率 |
| P67 | P69 | Counter 模式搭配 UB 融合链，Scalar 控制流+中间结果保留 UB 协同优化 |
| P68 | P84 | 低延迟归约搭配 Vector 效率模式，归约指令+计算链联合优化 |
| P71 | P83 | IBShare L1 共享搭配 L1 多缓冲分区，最大化 L1 利用率 |
| P81 | P85 | 参数常驻搭配 buffer 分区复用，协同管理 UB 空间 |

简化依赖图：

```
P14 ──→ P5 ←── P1 ──→ P7
  └────→ P28 ←── P19
         ↑ └──── P20
         │
    P19 ──→ P5
    P20 ──→ P5
P4 ──→ P11
P53 ──→ P18
P57 ──→ P58
P72 ──→ P4 ──→ P11
P73 ──→ P16
P79 ──→ P46
P83 ──→ P53 ──→ P18
P86 ──→ P46
P87 ──→ P46
```

## §3 推荐组合

经验证的高效策略组合：

| 组合名称 | 策略 | 适用场景 |
|---|---|---|
| 基础三件套 | P1 + P5 + P7 | 所有 memory-bound 算子的起点 |
| 自适应分核 | P2 + P4 + P11 | 变长维度 + 不均匀数据分布（P4 依赖 P11 处理尾块） |
| CV 代价优化 | P16 + P51 | Cube+Vector 融合算子的代价模型调优 |
| Matmul 缓存 | P47 + P52 | 矩阵乘法的 L1/L0 缓存策略 |
| 混合精度链 | D1 + A1 | FP16/BF16 输入需要 FP32 中间精度 |
| 量化全链路 | P48 + P49 + P70 | 量化算子的完整数据通路优化（含 FixPipe 内联量化） |
| 双缓冲全套 | P1 + P5 + P7 + P8 | 多 UB tensor 的完整双缓冲方案 |
| FlashDecode 分区 | P57 + P58 + P60 | FlashDecode G 分区 + TND 均衡 + KV NZ 格式 |
| FA L1 常驻链 | P53 + P18 + P61 | Flash Attention L1 常驻复用 + 七缓冲分区 + L2 缓存 |
| Vector 效率链 | P67 + P68 + P69 + P84 | Counter 模式 + 低延迟归约 + UB 融合链 + 效率模式 |
| Matmul 深度调优 | P46 + P71 + P83 + P86 | MatmulImpl API + IBShare + L1 分区 + 数据通路调优 |
| 多核访存优化 | P64 + P65 + P66 | GM 冲突规避 + UB bank 冲突规避 + 512B 对齐 |
| CV 并行全套 | P73 + P75 + P16 + P51 | AIC:AIV 配比 + 双 AIV 拆分 + 代价模型 + 动态核配比 |

## §4 UB 内存压力指南

### 压力分级

| 级别 | UB 占用比 | 典型策略 | 说明 |
|---|---|---|---|
| Light | < 50% | P7, P10, P12 | 单缓冲 + 少量临时 tensor |
| Medium | 50-75% | P1, P5, P8 | 双缓冲 + 中等 tensor 数量 |
| Heavy | > 75% | P1+P8+A1, P20 | 三缓冲或双缓冲+FP32中间+多tensor |

### 组合预算规则

1. 基础预算：`tileLength = (ubSize / BUFFER_NUM / pipeCount / sizeof(T)) / 32 * 32`
2. 每增加一个 TBuf tensor，可用 tile 大小按比例缩减
3. A1 (FP32 intermediate) 使 sizeof(T) 翻倍，tile 大小减半
4. P20 (Triple Buffer) 使 BUFFER_NUM=3，比 P1 的 BUFFER_NUM=2 多占 50% UB

### 危险组合（UB 溢出风险）

- P1 + P8 + A1 + 多个 TBuf：FP32 中间 + 双缓冲 + 多 tensor，极易超出 UB
- P20 + A1：三缓冲 + FP32 中间，几乎必然需要极小 tile
- P81 + P85 + A1：参数常驻 + 分区复用 + FP32 中间，UB 空间极度紧张
- P69 + P81 + P1：UB 融合链保留中间结果 + 参数常驻 + 双缓冲，需严格计算 UB 预算

## §5 不适用场景

### 按策略维度

| 策略 | 不适用条件 |
|---|---|
| P1 (Double Buffer) | 单次处理全部数据（无需分 tile）；UB 已满无法分配第二份缓冲 |
| P2 (Adaptive Tiling) | 固定 shape 算子（编译期已知所有维度） |
| P3 (Small-D) | D > 640 或行数极少（合并无收益） |
| P4 (Load Balance) | 单核算子；数据天然均匀分布 |
| P9 (Deterministic) | 推理场景（不需要确定性输出） |
| P11 (Tail Block) | 数据量恰好整除核数和 tile 大小 |
| P14 (CV Preload) | 纯 Vector 算子（无 Cube 计算） |
| P20 (Triple Buffer) | UB 空间不足以容纳 3 份缓冲 |
| P53 (L1 Resident) | 纯 Vector 算子（无 L1 使用）；单次迭代无复用机会 |
| P57 (FlashDecode G-Part) | 非 FlashDecode 算子；无 G 维分区需求 |
| P67 (Counter Mode) | 无 Scalar 控制流瓶颈；纯 Cube 算子 |
| P72 (Split-K) | K 维较小无需切分；需要确定性输出(P9) |
| P73 (AIC:AIV Ratio) | 纯 Vector 或纯 Cube 算子（无 CV 混合） |
| P76 (KV Pre-Merge) | 非 Sparse Attention；无 KV 预合并需求 |
| P88 (Multiphase Decomp) | 单阶段计算算子；无资源竞争需要拆分 |
| A1 (FP32 Intermediate) | 输入已是 FP32；精度要求不高的场景 |
| A2 (Welford) | 非归一化算子（无 mean/var 计算） |

### 按算子类型维度

| 算子类型 | 应跳过的策略 | 原因 |
|---|---|---|
| Element-wise | P3, P11, P14, P53, P57, P72, P73, P76, P88 | 无 reduction、无 Cube、无 L1/FlashDecode/Sparse 需求 |
| Normalization | P3(大D时), P14, P57, P72, P73, P76 | 通常 D 较大；纯 Vector；无 FlashDecode/Matmul/Sparse 需求 |
| Pooling | P3, P14, P57, P72, P73, P76 | 窗口滑动模式不适合行合并；无 Cube/FlashDecode/Sparse |
| Optimizer | P3, P14, A2, P53, P57, P72, P73, P76, P88 | 无 Cube、无 mean/var、无 L1/FlashDecode/Sparse 需求 |
| Matmul | P3, P11, P57, P67, P76 | D 通常很大；Cube 自动处理尾块；非 FlashDecode/Sparse |
| FlashAttention | P3, P11, P67 | 大规模矩阵运算，不适合小 D 优化；无 Scalar 控制流瓶颈 |
| Quantization | P3, P9, A2, P53, P57, P73, P76 | 量化有专用策略(P48/P49/P70/A8)；无 L1/FlashDecode/Sparse 需求 |

## §6 Agent 快速决策流程图

```
                        ┌─────────────────┐
                        │  算子类型判断     │
                        └────────┬────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
               ┌────▼────┐             ┌──────▼──────┐
               │ CV 融合  │             │  纯 Vector   │
               │(有Cube)  │             │  (无Cube)    │
               └────┬────┘             └──────┬──────┘
                    │                         │
         ┌──────────┴──────────┐    ┌─────────┴─────────┐
         │                     │    │                    │
    ┌────▼────┐          ┌─────▼──┐ │              ┌────▼────┐
    │ Matmul  │          │ 其他CV │ │              │ 有reduction│
    │         │          │        │ │              │ (Norm等)  │
    └────┬────┘          └────┬───┘ │              └────┬────┘
         │                    │     │                   │
    P47+P52+P14          P14+P16   ├── Element-wise    A1+A2+P1+P5
    +P5+P28              +P51+P5       P1+P5+P7+P2    +P7+P2
    +P71+P83+P86         +P73+P75      +P67+P69+P84   +P55+P68+P81
         │                    │     │
         │               ┌────┘     ├── Optimizer
         │               │         │   P1+P5+P7+P9
    ┌────▼────┐          │         │   +P81+P84
    │ Flash   │          │         │
    │Attention│          │         ├── Pooling
    └────┬────┘          │         │   P1+P5+P7+P11
         │               │         │   +P56+P59
    P53+P57+P58          │         │
    +P60+P61+P80         │         └── Quantization
    +P76+P77             │             D4+P48+P49+A8
         │               │             +P70
         ▼               ▼
    ┌─────────────────────────┐
    │  通用检查               │
    │  1. 互斥？→ §1 排除    │
    │  2. 依赖？→ §2 补充    │
    │  3. UB？ → §4 验证     │
    │  4. 精度？→ D1/A1-A8   │
    │  5. 访存？→ P64+P65+P66│
    └─────────────────────────┘
```

### 决策步骤

1. **判断算子类型**：CV 融合 vs 纯 Vector
2. **选择基础组合**：从 §3 推荐组合中选择最匹配的
3. **检查互斥**：对照 §1 表格，确认无冲突
4. **补充依赖**：对照 §2 表格，确认所有前置策略已包含
5. **验证 UB 预算**：对照 §4，确认不超出 Heavy 级别（除非有充分理由）
6. **添加精度策略**：根据数据类型和精度需求，按需添加 D/A 系列策略



