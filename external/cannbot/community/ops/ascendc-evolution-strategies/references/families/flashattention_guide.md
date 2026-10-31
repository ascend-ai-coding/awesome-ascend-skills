# FlashAttention 族策略选型指南

> 算子族：`attention` / `flash_attention`
> 适用算子：`FlashAttention`, `FlashAttention-2`, `GQA`, `MHA`, `MLA`, `SparseAttention`, `ChunkedPrefill` 等
> 关联设计：[knowledge-strategy-architecture-v3.2](../../../../../docs/design/knowledge-strategy-architecture-v3.2.md) §3.2
> **路径约定**：本文档中 `subfamilies/`、`foundation/`、`design/`、`specialization/`、`patterns.md` 等相对路径均基于插件本地 `community/ops/ascendc-evolution-strategies/references/tiling-design/flashattention/`（例：`subfamilies/base_design.md` = `.../tiling-design/flashattention/subfamilies/base_design.md`）

## 选型决策树

```
是否 FlashAttention 结构（Online Softmax）？
├─ 是
│   ├─ 长序列 (S2 > 4K) → P14 CV 预发射 + P18 L1 7-Buffer + P38 Softmax 状态常驻
│   ├─ 稀疏 Attention → 稀疏 pattern + KV chunk 跳过
│   └─ decode (S1=1) + KV cache → P60 KV NZ 格式
└─ 否（传统 Softmax + matmul）
    ├─ Softmax stability → A5 max-trick
    └─ Softmax + matmul 流水 → P5 + P28 显式 HardEvent
```

## 族特定策略

| 策略 | Tier | Playbook | 触发场景 | 核心手段 |
|---|---|---|---|---|
| **P115** FA Tiling 路由 | L1 | ✅ | 任意 FA 变体 | 四层文档架构 + 设计节点 G→N12 |
| **P14** CV 流水预发射 | L2 | ✅ | FA 长序列 | 三段式 Kernel，提前发射对侧 buffer |
| **P18** L1 七缓冲常驻 | L2 | — | FA 长序列 | L1 分区给 Q/K/V/dQ/dK/dV/mask |
| **P38** Softmax 状态常驻 | L1 | — | FA 跨 S2 循环 | max/sum 状态留 UB，避免 reload |
| **P55** BlockReduceMax | L1 | ✅ | attention + max 归约 | 单指令替代 stride DataCopy |
| **P60** KV NZ 格式 | L1 | — | FA decode + KV cache | ND 转 NZ，搬运对齐 |
| **P137** 稀疏 top-k 紧凑化走 Cube | L2 | ✅ | 稀疏 FA（top-k/sparseIndices） | workspace 紧凑 tile（有效前置+零填充）+ QK^T/PV 上 AIC；先消 gather 标量再 M 融合（17.28x） |
| **P138** host tiling L0 容量守卫 | L1 | ✅ | Cube tiling 跨芯片移植 | `GetCoreMemSize(L0_A/L0_C)` 运行时查询 + groupTile 自适应减半，防随机 SIGSEGV |
| **P139** 多槽 workspace 全槽零填充 | L1 | ✅ | P14 多槽流水 FA | 零初始化遍历全部 kPreloadNum 槽（含 prob/score padded），防 NaN 偶发崩溃 |
| **P140** MLA 大 M-tile | L1 | ✅ | kvHeads=1（MLA） | M-tile 放大到 qHeads（UB/L0 自适应），QK^T/PV 一次 mmad 吃满 Cube |

---

## Ascend C FlashAttention Tiling 系统化设计

> 来源：[`tiling-design/flashattention/patterns.md`](../tiling-design/flashattention/patterns.md)
>
> 本章节沉淀 FlashAttention 类算子在 Ascend C 上的 Tiling 设计方法论。

### 1. 四层文档架构

| 层级 | 目录 | 作用 |
|---|---|---|
| L1 路由 | `patterns.md`（本文件） | 变体选型表、设计节点注册表、加载指南 |
| L2 地基 | `foundation/` | 通用数学模型、Roofline、扩展点、组合规则 |
| L3 设计流 | `design/` | 节点化设计流程 G→N12 |
| L4 变体 | `subfamilies/` + `specialization/` | 变体特定设计 + 实现参考 |

### 2. 基础 FA 形态（三选一）

| 形态 | 数学差异 | 触发条件 | 落地章节 |
|---|---|---|---|
| **GQA**（默认） | `Hq = G × Hkv` | 默认；`Sq ≥ 1 + Hq % Hkv == 0` | `subfamilies/base_design.md` §1-§10 |
| **MHA**（GQA 退化） | `G = 1` | `gMaxPerTask=1` | `subfamilies/base_design.md` §10.3 |
| **MLA**（latent） | KV 经 latent compression，`kvHeadNum=1` | 数学公式含 latent absorption | `subfamilies/base_design.md` §11 |

### 3. 正交叠加 trait

定完基础形态后，按需叠加：

| trait | 触发条件 | 落地契约 |
|---|---|---|
| **量化** | `dtype ∈ {MXFP8, INT8, AntiQuant}` | `subfamilies/quantization_design.md` |
| **稀疏** | KV 维度有 sparse pattern | `specialization/sparse_design.md` |
| **Feature flags** | PSE / RoPE / Sink / PostQuant / Prefix / ChunkedPrefill | `specialization/feature_flags.md` |

> FlashDecoding（split-KV reduce）是并行实现技术，可叠加在任意组合之上。

### 4. 设计节点注册表（G→N12）

| # | 节点名 | 落地文件 |
|---|---|---|
| G | 设计治理 | `design/_governance.md` |
| N0 | 选型与分析确认 | `foundation/fundamentals.md` §4 + `patterns.md` §2 |
| N1 | 宏观形态 | `design/shape.md` §1 |
| N2 | 并行策略 | `design/shape.md` §2 |
| N3 | 基本块与 Roofline | `design/shape.md` §3 + `foundation/roofline.md` |
| N4 | Softmax 算法模式 | `design/execution.md` §1 |
| N5 | 平台依赖识别 | `design/shape.md` §4 |
| N6 | 多核切分 + 负载均衡 | `design/shape.md` §5 |
| N7 | 内存划分 | `design/resources.md` §1 |
| N8 | 编译期特化 | `design/execution.md` §2 |
| N9 | Host Tiling 汇总 | `design/resources.md` §2 |
| N10 | 流水编排与跨核同步 | `design/execution.md` §3 |
| N11 | 平台基线 + 开发起点 | `design/execution.md` §4 |
| N12 | 优化策略产出 | `design/execution.md` §5 |

### 5. 组合空间成熟度

```
                 ┌──── 稀疏轴 ────┐
                 │ 非稀疏 │  稀疏 │
       ┌─────────┼────────┼───────┤
量化轴 │ 非量化  │   ✓    │  📋   │
       │ MXFP8   │   ✓    │   ?   │
       │ INT8    │   ✓    │   ?   │
       │ AntiQ   │   ✓    │   ?   │
       └─────────┴────────┴───────┘
```

✓ = 已验证 / 有契约；📋 = 占位骨架；? = 未探索。

### 6. 架构约束

1. **一个算子只属一个基础子族**。
2. **量化 / 稀疏与基础子族正交**，但须在 `foundation/composition.md` 校核交互。
3. **常量语义分离**：跨核握手槽、L1 rotation 槽、V2 自读自写槽必须用不同常量。
4. **变体文件不修改地基/设计流**。
5. **新变体落地后回写**：若同类型算子出现稳定共性 → 升级契约。

### 7. 加载协议

Architect 设计阶段按以下顺序推进：

1. 读取 `foundation/fundamentals.md` 完成 N0 分析产出。
2. 用 `patterns.md` §2 选型表确定基础形态与正交 trait。
3. 加载 `foundation/extension_points.md` 明确 5 扩展点特化范围。
4. 沿 G→N12 读取 `design/` 对应节点文件。
5. 按需加载 `subfamilies/` / `specialization/` 变体文件。

### 8. 设计检查清单

- [ ] 是否已确认基础 FA 形态（GQA / MHA / MLA）？
- [ ] 是否识别了正交 trait（量化 / 稀疏 / feature flags）？
- [ ] 是否按 G→N12 顺序推进设计节点？
- [ ] 是否在 `foundation/composition.md` 校核 trait 组合可行性？
- [ ] 常量语义是否分离（跨核槽 / L1 rotation 槽 / V2 自读自写槽）？
- [ ] 是否根据 Roofline 选择基本块大小与并行策略？
- [ ] 是否按变体文件补充 dtype 路径与平台差异？
