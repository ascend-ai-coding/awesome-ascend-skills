# Step 2: 设计前核对（轻量检查清单）

> **定位**：设计前的源码事实核对清单。**默认将核对事实内联记录于 DESIGN.md §0.3**，**不要求独立调查报告文件**；仅在设计委托方明确要求时才产出独立 `apace-investigation-report.md`。不做方案推荐、不匹配场景、不选择路线、不编译或运行设备。
>
> plugin 场景下本步骤在 plugin Step 2（设计）内由 Architect 完成，不产生独立交付物。

## 1. 需求四要素（设计三拷问的事实化）

动手核对源码前，先把需求落成可对照的四要素（对应方法论原则 2「设计三拷问」）：

| 要素 | field id | 说明 | 示例 |
|------|------|------|------|
| golden 语义 | `golden_semantics` | 每卡输入/输出分布与切分轴（**输入分布轴与输出分布轴分别冻结**） | 输入：每卡完整 A（K 不切分）；输出：按 M 轴聚合 → [M/R, N] |
| 通信方向 | `communication_direction` | GET / PUT / compute-first | PUT（通信→计算） |
| 编排形态 | `orchestration` | 严格分离 / 时分复用 | 严格分离（后 R 核通信 / 前核归约） |
| dtype/shape/rankSize | `dtype` / `shape` / `rank_size` | 输入/累加/输出 dtype、逻辑 shape、rank 数 | FP8_E4M3 × FP8_E4M3 → BF16，M=2048，R=4 |

> **四要素提取不到时必须澄清**：golden 语义的"每卡输入分布"无法从用户需求唯一确定时（典型：用户说"每卡有 M/rankSize"但未说明是输入切分还是输出分布），**必须回 [`requirement-analysis/grill-protocol.md`](../../../requirement-analysis/grill-protocol.md) 维度 9 向用户追问**，禁止默认假设后直接进入源码核对。

## 2. 源码核对范围（只读）

只读核对 CANN 内置 apace 框架路径（Step 1 实测登记值；两种已验证形态：`opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/apace/`、`vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/apace/`），禁止 clone、更新、切换源码：

```
apace/
├── basic/          # 基础数据结构与抽象（fragment_tensor/：FragmentTensor/MakeFragmentTensor/FragmentSliceCopy）
├── block/          # 接口层
│   ├── aiv_comm/
│   │   ├── collective_comm_api.h    # 四段式通信 API 契约 + CommCollectiveOp/CommMode 分发
│   │   ├── all_to_all/              # AllToAll GET/PUT 钩子
│   │   ├── all_gather/              # AllGather PUT 钩子
│   │   └── barrier/                 # TeamBarrier（barrier_ubmem.h）
│   ├── blaze_ext/                   # Blaze 扩展（gemm/block/qmm_mx_block_mmad_fragment.h——FragmentTensor 路径的 MMAD 组件，AG 类算子必查）
│   └── aiv_compute/                 # AIV 计算接口（⚠️ 空占位——量化/反量化/规约组件规划中，勿假定存在）
├── kernel/         # 官方算子实现（可直接调用或参考）
│   ├── all_to_all_quant_matmul/     # PUT AllToAll + QuantMatmul（UDMA 版 + hcomm/CCU 注册形态版）
│   └── all_gather_quant_matmul/     # PUT AllGather + QuantMatmul（FragmentTensor 三区）
├── tiling/         # tiling 算法（quant_matmul_tiling_{base,common,data,swat}.h + comm_tiling_data.h）
└── utils/          # 通用工具与常量（constant.h + host 侧 comm_channel_builder.h）
```

可选：经 `scripts/fetch_apace.sh` 获取官网 master 最新代码核对契约，使用前必须与内置版本 diff 校验。

## 3. 必须核对的事实（内联记录于 DESIGN.md §0.3）

| 事实类别 | 内容 | 记录位置 |
|---------|------|---------|
| matmul 链路事实 | M/N/K、dtype 组合、scale 编码、Blaze 组件（`BlockMmad` + `BlockScheduler` + `MatmulWithScaleMx` 或 FragmentTensor 路径）、localMatmul 模式候选、tiling 基线 | DESIGN.md §0.3 |
| 通信接口事实 | `CollectiveComm` 四段式契约、GET/PUT 钩子职责、CommContext 字段、Win 区布局 | 同上 |
| 入口 ABI 事实 | 入口函数签名、CommContext 传递方式、dtype 变体入口数 | 同上 |
| 官方覆盖性 | 官方 kernel 可直接调用/复用的部分、未覆盖的需求项 | DESIGN.md §2.2 |
| 注册形态原型事实（需求存在既有实现时） | 原型语义/算法骨架/验证基准三要素 + 可借鉴/必须重写边界判定（按 [`paradigm-mapping.md`](../operator-design/paradigm-mapping.md) §3 清单） | DESIGN.md §0.3 |

每条事实必须带 `文件:行号` 证据引用；声明读取边界内未找到 ≠ 不支持，不得据此虚构接口或文件名（教训：AllGather GET 钩子在源码中不存在，仅 `all_to_all_udma_get.h` 有记录）。

## 4. 调用链核对要点

从官方 kernel 入口追踪完整调用链，记录每个环节的逻辑对象、设备物理表示、资源生命周期、同步 owner：

```
__global__ 入口 → Impl::Init → Impl::Run
  → AIV 侧：CollectiveComm::Init → Commit → Wait → Finalize
  → AIC 侧：BlockMmad::Init → Run → Epilogue
  → 同步：CrossCore SetFlag/WaitFlag / SyncAll
```

接口组合以官方 kernel 实例为参照（实例化的模板参数、钩子注册、调度器选择），不做组件名的笛卡尔积式拼接；不得把参考算子名称当作能力证据。

## 5. 独立报告模式（可选）

仅当设计委托方明确要求时，将 §1-§4 事实写入 `operators/{op}/docs/apace-investigation-report.md`：

```text
investigation_id
source_root                     # CANN 内置 apace 路径
requirement_facts               # §1 需求四要素
matmul_chain_facts              # §3 matmul 链路事实
comm_interface_facts: []        # 通信接口事实
entry_abi_facts: []             # 入口 ABI 事实
official_coverage: full | partial | none
uncovered_requirements: []      # 官方未覆盖的需求项
read_boundary: []               # 实际读取的文件清单
```

## 6. 边界声明

- 不做方案推荐、不匹配场景、不选择路线
- 不编译或运行设备
- 事实不足时在 DESIGN.md 标记 `blocking` 并停止等待澄清，最多一次补充核对
