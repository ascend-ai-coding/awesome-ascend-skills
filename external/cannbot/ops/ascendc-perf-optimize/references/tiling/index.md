# 算子 Tiling 理论建模

> Tiling 建模按算子语义、shape 和 dtype 展开；资源参数取自目标平台。只有代际相关的 API、模板与阈值需要按 [适用范围](../architecture-routing.md) 筛选，通用模型不按代际拆分。


> 按算子 pattern 路由到对应的 Tiling 建模参考目录。每个 pattern 有独立的建模文件夹，包含该族算子的 Tiling 理论模型、参数推导和门禁决策逻辑。
>
> 输出统一格式的理想 tiling data（多核切分、Buffer 规划、分支覆盖）。

---

## Pattern 路由表

| 算子类别 | 典型算子 | 建模目录 |
|---------|---------|---------|
| MatMul 矩阵乘类 | MatMul, BatchMatMul, MatMul_MXFP4, MatMul_A16W16 | [matmul/](matmul/) |
| MC² 通算融合类 | matmul_all_reduce, allgather_matmul, matmul_reducescatter, alltoall_matmul | [mc2/](mc2/) |
| Reduction 归约类 | ReduceSum, Softmax, LayerNorm, ArgMax | [reduction/](reduction/) |
| Scan / Cum 前缀扫描类 | Cumsum, Cummin, Cummax, Cumprod, LogCumsumExp | [scan/](scan/) |
| Gather 索引类 | GatherV2, Gather, torch.gather, Embedding | [gather/](gather/) |
| Sort / Select 排序选择类 | TopK, Sort, ArgSort, KthValue | [sort/](sort/) |
| Elementwise 逐元素类 | Sin, Cos, Abs, Exp | [elewise/](elewise/) |
| Broadcast 广播类 | Add, Mul, Sub | [broadcast/](broadcast/) |
| Scatter 按轴散射类 | Scatter, ScatterAdd, ScatterReduce（ScatterElementsV2） | [scatter/](scatter/) |
| Conversion 数据转换类 | Transpose, Concat, Split | [conversion/](conversion/) |
| Convolution 卷积类 | Conv2D, DepthwiseConv | [convolution/](convolution/) |
| FA FlashAttention 类 | FlashAttention, FusedInferAttentionScore, QSFA, SparseFlashAttention, SCFA | [fa/](fa/) |

## 路由流程

```
给定：算子名 + Shape + dtype

Step 0 — 算子分类：
  ├─ MatMul 族 → matmul/
  ├─ MC² 通算融合族 → mc2/
  ├─ Reduction 族 → reduction/
  ├─ Scan/Cum 族 → scan/
  ├─ Gather 族 → gather/
  ├─ Sort / Select 族 → sort/
  ├─ Elementwise 族 → elewise/
  ├─ Broadcast 族 → broadcast/
  ├─ Scatter 族 → scatter/
  ├─ Conversion 族 → conversion/
  ├─ Convolution 族 → convolution/
  ├─ FA 族 → fa/
  └─ 未匹配 → 返回「该算子 pattern 暂未收录，请提供更多信息」
```

---

## 算法注册表

每个算子族目录下 `index.md` 是该族的**算法路由**，负责将具体 shape/dtype 条件路由到正确算法。各算子族有一个**默认参考算法**（由 `index.md` 兜底），社区可贡献替代算法（放入子目录），通过路由条件匹配选择。

**两级路由**：
1. `tiling/index.md`（本文件）→ 按算子范式路由到 `tiling/<范式>/index.md`
2. `tiling/<范式>/index.md` → 按 shape/dtype 条件路由到具体算法

**选择逻辑**：
1. 遍历该算子族的所有贡献算法，按优先级从高到低匹配条件
2. 首个条件匹配的贡献算法被选中
3. 无匹配时仅回退到满足目标架构/能力条件的默认参考算法；无目标实现时返回参考缺口

### 当前注册

| 算子族 | 算法路由 | 兜底算法 | 扩展算法 | 选择条件 |
|--------|---------|---------|---------|---------|
| MatMul | [matmul/](matmul/) | arch22 [A2/A3 指南](../../../ascendc-performance-best-practices/references/matmul/arch22/guide.md) / arch35 [fallback/](matmul/fallback/)（SWAT / FullLoad / StreamK） | — | 先按目标代际 |
| MC² 通算融合 | [mc2/](mc2/) | [fallback/](mc2/fallback/)（matmul tiling + 通算切分参数） | — | — |
| Convolution | [convolution/](convolution/) | [fallback/](convolution/fallback/)（FORMULAS: Mmode + HWmode） | — | — |
| Reduction | [reduction/](reduction/) | [fallback/](reduction/fallback/)（五模板通用算法） | — | — |
| Scan / Cum | [scan/](scan/) | [fallback/](scan/fallback/)（流式 + oneway·twoway·ub·core 三层组合） | — | — |
| Gather | [gather/](gather/) | [fallback/](gather/fallback/)（六模板通用算法） | — | — |
| Sort / Select | [sort/](sort/) | [topk.md](sort/topk.md)（A5 TopK 多模板）/ [fallback/](sort/fallback/)（通用 Sort） | — | — |
| Elementwise | [elewise/](elewise/) | [fallback/](elewise/fallback/)（占位） | — | — |
| Broadcast | [broadcast/](broadcast/) | [fallback/](broadcast/fallback/)（显式分代广播模型） | — | — |
| Scatter | [scatter/](scatter/) | [fallback/](scatter/fallback/)（六路径通用算法：并行划分 + 冲突消解） | — | — |
| Conversion | [conversion/](conversion/) | [fallback/](conversion/fallback/)（占位） | — | — |
| FA | [fa/](fa/) | arch22 [A2/A3 FIA 指南](../../../ascendc-performance-best-practices/references/flash_attention/arch22/guide.md) / arch35 [fallback/](fa/fallback/)（占位） | — | 先按目标代际与 FA 变体 |

### 贡献新算法

在 `<算子族>/` 下创建新目录，包含 `index.md`（算法描述、适用条件、tiling 推导）。同时在 `<算子族>/index.md` 的算法路由表中注册，注明选择条件和优先级；条件包含 dtype、shape 范围、核数等可判定字段，依赖特定代际的能力时另行标注。

---

## Buffer 层级约定

不同算子类型的 on-chip buffer 层级不同，Tiling 输出需区分：

| 算子类型 | 主要 Buffer | 说明 |
|---------|------------|------|
| **Vec 类**（Elementwise, Broadcast, Conversion, Reduction 部分） | **UB** | 数据从 GM → UB，计算在 UB 上完成，无 L1 参与 |
| **Cube 类**（MatMul, Convolution） | **L1 + UB** | 数据 GM → L1 → L0 → Cube，L1 做全局复用缓冲，UB 做 weight/fmap DMA 暂存 |
| **融合类**（FA 等） | **L1 + UB** | 多层流水，L1 承载大块数据驻留，UB 承载计算窗口 |
| **MC² 通算融合类** | **L1 + UB + GM（通信）** | 复用 matmul 的 L1/L0 层级，通信数据经 GM 侧 buffer 收发 |

> **关键区分**：Cube/融合类算子的切分粒度在 **L1**（不是 UB），L1 split 决定了单次搬运的数据量；L0/UB 是 L1 的进一步子切分。

---

## 理想 Tiling Data 输出格式

### Cube / 融合类（L1 + UB 两级）

```yaml
tiling_data:
  multicore:
    split_dim: "M,N"
    single_core_m: 128
    single_core_n: 256
    core_num: 8
  l1_split:                    # L1 是主切分层
    base_m: 64
    base_n: 128
    base_k: 32
    need_chunk: false
    chunk_formula: ""
    double_buffer: "A1_B1"     # L1 ping-pong 策略
  l0_split:                    # L0 是 L1 的子切分
    mL0: 64
    nL0: 16
    kL0: 16
  ub_split:                    # UB 是 weight/fmap DMA 暂存
    weight_trans: false
    fmap_dma: false
  buffers:
    - name: L1_A
      size_formula: "kAL1 * baseM * sizeof(AType) * (pbAL1 ? 2 : 1)"
    - name: L1_B
      size_formula: "kBL1 * baseN * sizeof(BType) * (pbBL1 ? 2 : 1)"
    - name: L0_C
      size_formula: "mL0 * nL0 * sizeof(float32) * (pbCL0 ? 2 : 1)"
  branches:
    - dim: dtype
      conditions: ["fp16", "bf16", "fp32"]
    - dim: shape_size
      conditions: ["large", "small"]
```

### Vec 类（仅 UB）

```yaml
tiling_data:
  multicore:
    split_dim: "batch"
    single_core_size: 4096
    core_num: 4
  ub_split:                    # UB 是唯一切分层
    block_size: 256
    repeat: 16
  buffers:
    - name: UB_in
      size_formula: "block_size * sizeof(dtype)"
    - name: UB_out
      size_formula: "block_size * sizeof(dtype)"
  branches:
    - dim: dtype
      conditions: ["fp16", "fp32"]
```

---
