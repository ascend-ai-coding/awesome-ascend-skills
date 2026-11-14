# apace 算子开发指南

> 本文档定义 apace 算子开发的工程执行规范：事实源、布局、步骤、构建与验收。设计原理与编排契约以 [`fusion.md`](../fundamentals/fusion.md) 为唯一事实源，本文不重复展开。

## 目录

1. [开发起点：事实源、布局与地图](#1-开发起点事实源布局与-modify-地图)
2. [开发步骤](#2-开发步骤)
3. [改造场景食谱](#3-改造场景食谱)
4. [工程构建](#4-工程构建)
5. [验收清单](#5-验收清单)

---

## 1. 开发起点：事实源、布局与 [MODIFY] 地图

### 1.1 事实源与获取方式

**默认（直调独立工程）：CMake 直引 CANN 内置 apace，零复制**。CANN 已内置完整 apace 框架（含 `basic/` `block/` `kernel/` `tiling/` `utils/` 与参考算子如 AllToAll/AllGather）。内置路径随 CANN 打包形态不同，**必须 `test -d` 实测定位后登记**（Step 1 路径实测纪律）。两种已验证形态：

```
① $ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/apace/        （cann-9.2.0 实测）
② $ASCEND_HOME_PATH/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/apace/  （cann-9.1.0 实测；亦见于 opp/vendors/ 下同构镜像）
```

> ⚠️ **禁止整包复制 apace 到算子目录**（生产教训）：复制的官网 master 代码与当前 CANN 工具链脱钩（目录布局会重构漂移），实测引发通信/布局类"幽灵 bug"，漂移后排查成本极高。

> 📌 **子路径为逻辑引用，物理路径以实测为准**：本文档及 `fundamentals/`、`scenarios/` 中出现的 `block/aiv_comm`、`block/blaze_ext` 等子路径是**逻辑分层引用**，其在 CANN 内置树中的物理位置随版本漂移（如 CANN 9.2.0 实测通信层在 `core/aiv_comm/` 而非 `block/aiv_comm/`）。**禁止按文档写死路径**——Step 1 环境登记时用 `test -d` 实测定位并记录到 environment.md；算子源码 include 统一用 `apace/...` 前缀，由 CMake `OP_KERNEL_ROOT` include 根解析（§4.1），与物理子路径位置解耦。

**可选（跟踪 master 新特性 / 核对最新契约）**：运行 `scripts/fetch_apace.sh` 获取官网最新代码，仅作设计参考；使用前必须与内置版本 `diff -rq` 校验，存在差异时以 CANN 内置版本为编译事实源。

### 1.2 推荐布局与命名规范

```
operators/{OpName}/            ← camelCase（对齐 CANN 算子注册名）
├── CMakeLists.txt            ← [MODIFY] 独立工程构建（§4.1，APACE_ROOT 指向 CANN 内置 apace）
├── kernel/                   ← [MODIFY] 本算子 kernel 头文件（snake_case 文件名）
│   ├── {op_name}_tiling_data.h   ← tiling 结构体（host/device 契约）
│   ├── {op_name}_impl.h          ← Impl：Init/Run/编排
│   ├── {op_name}_frag_kernel.h   ← FragmentTensor mm 内核（命名遵循 apace 惯例：qmm_mx_kernel_{rs/ag/a2a}_frag.h；compute-first 默认，fusion.md §6.2.2）
│   └── reduce_sum_ref.h          ← 归约文件（compute-first，手动 UB，fusion.md §6.2.6）
├── src/                      ← [MODIFY] host 侧
│   ├── kernel_launcher.h     ← __global__ 入口（KERNEL_TYPE_MIX_AIC_1_1）
│   ├── main.cpp              ← Tiling + fork 多 rank + HCCL 建链 + launch（perf 模式内嵌 L2 flush）
│   ├── root_info_exchanger.h ← [REUSE] TCP rootInfo 交换（从参考算子复制）
│   └── utils.h               ← [REUSE]（同上）
├── scripts/                  ← [MODIFY] gen_data.py / verify_result.py / parse_perf.py
├── cases.csv                 ← [MODIFY] 用例矩阵（含 tile 粒度参数列，§4.5）
├── run.sh                    ← [MODIFY] 一键脚本（--skip-build/--gen-only/--verify-only/--cli/--perf）
└── profiling/                ← 性能证据链（结构唯一事实源：host-and-testing.md §6.1）
```

> 共享层 `block/` `tiling/` `basic/` `utils/` **不复制、不出现在算子目录**——经 CMake `-I` 直引 CANN 内置路径（§4.1）；参考 kernel（`quant_matmul_mx_kernel.h`、`comm_channel_builder.h` 等）同样直引。
>
> 命名规范：算子目录 camelCase（对齐 CANN 算子注册名，如 `allToAllQuantMatmul/`；snake_case 目录名为遗留形态，新算子禁止沿用）；kernel/src 文件 snake_case；kernel 入口 `{OpName}Kernel{DtypeA}{DtypeB}`；Impl 类 `{OpName}Impl`；tiling 结构体 `{OpName}TilingData`；全局无旧算子名残留（类名/文件名/namespace/ctxTag）。
>
> **namespace 规范**：每个组件独立 namespace，禁止共用——Impl 用 `Apace`（或 `{OpName}Impl`），frag kernel 用独立 namespace（如 `{OpName}FragImpl`，参考 AllGather `QmmMxKernelAgUdma` 的独立 namespace 形态），归约组件用独立 namespace（如 `{OpName}ReduceSumImpl`）。Impl 与 frag kernel 共用同一 namespace 会导致类型别名（LayoutA/FragTensorA 等）相互污染，新算子禁止。

### 1.3 [REUSE]/[MODIFY] 地图

| 标记 | 含义 | 验收条件 |
|:---|:---|:---|
| `[REUSE]` | 稳定共享层 | **不复制进算子目录**，经 CMake `-I` 直引 CANN 内置 apace（§1.1）；禁止以任何形式篡改（patch/overlay） |
| `[MODIFY]` | 算子层 | 本算子自有文件（§1.2 布局全部），新建并按 §3 场景食谱改造 |

**[REUSE] 直引清单**：`block/`（`aiv_comm/` 通信 API + `blaze_ext/` Blaze 扩展）、`tiling/`（`comm_tiling_data.h` + `quant_matmul_tiling_*.h`）、`basic/`（`fragment_tensor/`）、`utils/`（`comm_channel_builder.h` 等）、`kernel/` 下官方参考算子（AllToAll / AllGather，参考实现与 tiling 引擎来源）。

### 1.4 参考实现导读

| 项 | 内容 |
|:---|:---|
| 样例 | 官方 `kernel/` 已实现算子均为 PUT（通信在前）模式：AllToAll（K 轴切分）、AllGather（M 轴切分）。官网暂无 GET、compute-first 样例——GET 钩子存在于共享层 `block/aiv_comm/`（语义见 [`communication.md`](../fundamentals/communication.md)）；compute-first 设计契约见 [`fusion.md`](../fundamentals/fusion.md) §6.2 与本文 §3.5 |
| PUT 3 文件模式 | `{op}_udma_impl.h`（Impl 编排：`Init`/`Run`/`RunAllToAll`/`RunMatmul`/`SetupParams`）+ `quant_matmul_mx_kernel.h`（Blaze `BlockMmad` + `BlockScheduler` 编排 kernel）+ `{op}_tiling_data.h`（tiling 结构 + `CommContext`=`CommUdmaContext`+`CommUbmemContext`）。同目录 `*_hcomm_impl.h` 为 HCCL windows 变体，直调场景不使用 |
| include 双风格 | 官方头文件两种风格并存：impl 类文件用 `apace/...` 前缀（如 `#include "apace/block/aiv_comm/collective_comm_api.h"`），tiling_data 等用 `../../tiling/...` 相对路径（仓内自洽）。本算子源码统一用 `apace/...` 前缀 include，依赖 CMake `${OP_KERNEL_ROOT}` include 根（§4.1） |
| 验收 | `apace/...` 前缀 include 必须解析到 CANN 内置路径（§4.1 验收项）；算子目录存在任何共享层副本即 FAIL（新旧混链风险） |

---

## 2. 开发步骤

1. **确定事实源**：默认直引 CANN 内置 apace（§1.1）；仅当确需 master 新特性时才运行 `scripts/fetch_apace.sh` 并先做 diff 校验
2. **语义与 API 核对**（设计阶段完成，最易被跳过、返工成本最高）：
   - **golden 语义先行**：明确每张卡的输入/输出语义与切分轴（如 ReduceScatter 是"每卡完整 K、切 M"，写错切分轴则 golden 全错）；写入 DESIGN.md
   - **API 逐一验证**：每个选用 API 标注"验证来源（官方文件:行号）+ 在当前 CANN 版本的可用性验证状态"——生产教训：旧 CANN 版本存在符号缺失致链接失败的案例，设计阶段未发现则开发期返工
3. **直引起手**：按 §1.2 推荐布局新建本算子文件（kernel/ 4 个头文件 + src/ 4 个 host 文件）；共享层与参考 kernel 经 CMake 直引 CANN 内置路径，**禁止从零写文件，也禁止复制共享层**。**mm 内核默认 FragmentTensor 消 R 循环**；vendor 复用官方 kernel 为例外（需 SCALAR 论证，见 [`fusion.md`](../fundamentals/fusion.md) §6.2.2）
   > **已有实现迁移（通用方法，与来源底座无关）**：若目标算子已有非 apace 实现（本仓 SHMEM 版 / ops-transformer 注册版），apace 版**只替换通信层与 host 框架，计算与 golden 语义整体复用**：
   > 1. **冻结 golden 语义**：先从旧实现的 gen_data/算子规格提取每卡输入/输出分布与切分轴（输入分布轴 ≠ 输出分布轴，分别记录），golden 数值应与旧实现对齐后再迁移；
   > 2. **通信层替换**：注册版（HCCL CCU windows `GetHcclContext` / MTE）→ `CommChannelBuilder` 建链 + `CollectiveComm` 四段式 UDMA 直调；SHMEM 版（`aclshmemx_udma_*`）→ 同样替换为 CollectiveComm；
   > 3. **host 框架替换**：注册版 op_host tiling → ST 直调 main.cpp（`GetTilingData` + CommTilingData 填充 + fork 多 rank + TCP rootInfo + 建链 + `<<<>>>` launch + dtype dispatch）；
   > 4. **入口重写**：注册入口 → `__global__` dtype 变体（`KERNEL_TYPE_MIX_AIC_1_1`）；
   > 5. **分层验收**：同 shape 与旧实现 golden 对拍 → 多 rank 精度矩阵 → 与旧路径性能对标归档（R15 三路径对标中旧实现路径天然是对标基线）。
   > 工程验证：SHMEM→apace 迁移后首轮精度即通过且性能更优。
   >
   > ⚠️ **现状事实（迁移评估前提）**：ops-transformer 仓 mc2/ 下现有算子（`matmul_reduce_scatter_v2`、`quant_reduce_scatter`、`matmul_all_reduce`、`all_gather_matmul` 等）当前**均未使用 apace**（官方基准 grep 实证：用 HCCL CCU windows 或 MTE 通信，注册形态）——"以 apace 替代/优化已有算子"是目标态；已有算子仅作语义/golden/性能对标参考，不是 apace 路线代码起点。
4. **定点改造**：按 §3 改造场景食谱逐项修改，每处改造对照验收条件自查
5. **编译 + 冒烟**：编译无错误/警告后，单 rank 运行冒烟，输出非全 0。禁止跳过冒烟编译直接全量测试
6. **分阶段 bring-up**：引入新链路（bias、新通信对象、新归约路径）时先退化对照（如 bias=0、T=1、rank=2）隔离验证，再逐步铺开——每步只放大一个变量，出问题域立即可定位
7. **精度验证**：多 rank + 多 shape `verify_result.py` PASS（精度标准见 §4.6）。**精度全量达标后才进入性能阶段**
8. **性能调优**：先归档串行基线（tileCnt=1）作为对比锚点，再按"流水重叠 → tiling 调优 → 热点模块重写 → 微优化"的顺序推进；每轮单变量改动 + 绑定精度验证 + 不达标量化回退。完整方法论见 [`optimization-playbook.md`](../troubleshooting/optimization-playbook.md)
9. **文档同步**：PLAN.md 更新改造内容和测试结果；DESIGN.md 与代码一致（重构多发期主动做一次设计文档 vs 代码一致性核对，防止文档腐化）

---

## 3. 改造场景食谱

### 3.1 改造场景速查

以官方 PUT 算子（如 AllToAll 的 `QuantMatmulMxKernel` + `AllToAllMxQuantMatmulUdmaImpl`）为锚点：

| 场景 | udma_impl.h | quant_matmul_mx_kernel.h | tiling_data.h | main.cpp | scripts | 共享层 |
|:---|:---|:---|:---|:---|:---|:---|
| 换 dtype | 改模板参数 | 改模板参数 | 无 | 改 kernel 名 + tilingEngine 模板 | 改 dtype | 不动 |
| 换 shape | 无（`InitBaseParams` 为 impl 方法，随 tiling 自动推导） | 无 | 无 | 改参数解析 | 改 gen_data | 不动 |
| 改 localMatmul (PUT) | 无（`Run()` 已含 `RunLocalMatmul` 分支）；用 localMatmul=1 时加 PipeBarrier 补丁（推荐修复） | 无（LocalParams 已含 localMatmul） | 已有 localMatmul 字段 | 设 localMatmul=0/1/2 | 不动 | 不动 |
| 加 bias | SetupParams 补 mmadParams.biasGmAddr + qbmmParams.isBias | 无（bias 通道已现成） | 加 bias 字段 | 加 bias GM 分配 + 传参 | 改 gen_data | 不动 |
| 去 Scale (非 MX) | 改 DispatchPolicy | 去 scale 字段与 Tensor | 去 scale 字段 | 去 scale 参数 | 改 gen_data | 不动 |
| 改流水深度 | 无 | 无 | 调 headMSize/tileCnt（CommTilingData 字段） | ⚠️ 官方 ST 的 headMSize 命令行参数为**死参数**（仅日志打印，tiling 用重算值 `headMSize = CeilDiv(usedCoreNum, nTile) × baseM`）；须先在 main.cpp 接线（用入参覆盖重算值）再可调 | 不动 | 不动 |
| 调通算并行 | 无 | 无 | 改 CommTilingData | 同上：先接线 headMSize（tileCnt 派生） | 不动 | 不动 |
| 换卡数 | 无 | 无 | 无 | 改 rankNum 参数 | 改 gen_data | 不动 |
| **新增计算在前算子（示例：ReduceScatter，见 §3.5）** | 重写 Run：AIC 自研 FragmentTensor kernel 全量 mm（消 R 循环，T>1 按轮拆子区间）+ SetFlag；AIV 逐轮 WaitFlag 门控 + Commit/Wait + SyncAll + 增量归约 | 自研 FragmentTensor mm kernel（默认，fusion.md §6.2.2；vendor 复用为例外须 SCALAR 论证，且 `QuantMatmulMxKernel.cGmAddr` 为 `GM_ADDR` 类型与 FragmentTensor C 输出不兼容）；独立归约文件（手动 UB + guard TBuf 隔离通信区，fusion.md §6.2.6） | 单份完整 mm tiling + commTilingData + 通信派生字段（每卡行数/chunk 字节/tile 字节/staging 大小/归约粒度） | host 校验清单（§3.5 共 9 项）+ T 派生（T 整除每卡行数，无尾块）+ staging 分配（完整输出大小）+ dtype dispatch（4 变体入口） | 改 gen_data（golden 切分轴核对） | 不动 |

> 完整实现契约见 [`fusion.md`](../fundamentals/fusion.md) §6.2 与 [`operator-anatomy.md`](operator-anatomy.md) §7；场景步骤见 §3.5。

### 3.2 udma_impl.h 改造

#### 场景：换 dtype

**验收条件**：Impl 模板参数（如官方 AllToAll 的 `<AType, BType, CType, TransA, TransB>`）和 dtype 入口变体（官方 AllToAll ST 的 4 变体：E4M3E4M3 / E5M2E5M2 / E4M3E5M2 / E5M2E4M3）的模板参数均改为目标 dtype。

#### PUT 编排验收标准（以官网 `AllToAllMxQuantMatmulUdmaImpl` 为基准）

> 完整 PUT 编排详见 [`communication.md`](../fundamentals/communication.md)（四段式契约）和 [`operator-anatomy.md`](operator-anatomy.md)（Impl 骨架）。

**验收条件**（impl 侧特有项，均可在 `Init()` 中验证；逐轮编排验收见 [`fusion.md`](../fundamentals/fusion.md) §4.1，本表不重复）：

| # | 验收项 | 官网锚点 |
|:---|:---|:---|
| 1 | 通信对象数 = 2（A data + A scale），共享同一 channel | `Init()` 中 `allToAllA_` + `allToAllScaleA_` 两个通信对象 |
| 2 | scale 对象 winOffset = `rankSize × rankDataBytes` | `Init()` 中 `allToAllScaleA_.Init(...)` 末参 |
| 3 | UB 需求：双 commBuf（各 `COMM_WORKSPACE_SIZE`=512B，`block/aiv_comm/collective_comm_context.h`）+ barrierBuf | `Init()` 中 ubOffset 累加 |
| 4 | AIV 先 → AIC 后 | `Run()` 中 `ASCEND_IS_AIV` 分支在前 |

#### 场景：改 localMatmul（PUT 模式）

> 完整的 localMatmul 模式选择决策和 AtomicAdd 时序分析见 [`fusion.md`](../fundamentals/fusion.md)。

**验收条件**：

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | Run() 分支 | `localMatmul==1` 时推荐含 `RunLocalMatmul()` → `PipeBarrier<PIPE_ALL>()` → `RunMatmul()`（官网 `Run()` 当前未实现 PipeBarrier，为推荐修复） |
| 2 | PipeBarrier 必要性 | 不加 PipeBarrier 可能触发 MTE 异常（aclError:507015），详见 [`fusion.md`](../fundamentals/fusion.md) |
| 3 | kernel LocalParams | 含 `localMatmul` + `matmulMode` 字段（`quant_matmul_mx_kernel.h` `LocalParams`）；`localMatmul==1 && REMOTE` 时 `isAtomicAdd_` 置位（`Init()`） |
| 4 | tiling_data.h | 含 `uint32_t localMatmul` 字段 |
| 5 | main.cpp | 显式设置 `localMatmul` 值（官方 AllToAll ST 显式 `tilingData.localMatmul = 0`，不依赖默认值） |
| 6 | splitKNum 规则 | `localMatmul==1` 时 REMOTE 只有 `rankSize-1` 个远程卡参与，否则为 `rankSize`（官网 `SetupParams()`） |
| 7 | CMakeLists | 参考：官方 AllToAll 用例含 `-DASC_DEVKIT_MAJOR=9`，AllGather 用例当前不含 |

### 3.3 quant_matmul_mx_kernel.h 改造

#### 场景：去 Scale（非 MX 量化）

官网基线为 MX 量化：`DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<NONE_FULL_LOAD_MODE, false>`（官方 AllToAll impl）。去 scale 需替换为非 MX 策略（替代类型以 ops-tensor `blaze/gemm/policy/dispatch_policy.h` 为准）并联动清理 scale 链路。

**验收条件**（5 处修改）：

| # | 修改位置 | 验收条件 |
|:---|:---|:---|
| 1 | DispatchPolicy 定义（udma_impl.h） | 类型由 `MatmulWithScaleMx` 改为非 MX scale 策略 |
| 2 | mmadParams 赋值（udma_impl.h `SetupParams()`） | 不含 `scaleBGmAddr` 赋值 |
| 3 | kernel `ResetGmAddr()` | 不含 `scaleAGmAddr_` / `scaleBGmAddr_` 赋值 |
| 4 | kernel `ProcessSingleBatch()` | 不含 ScaleA/ScaleB 的 Layout/Tensor 创建和 Slice；`mmadOp_()` 调用不含 scale 参数 |
| 5 | tiling_data.h + main.cpp | 去 scaleCommTilingData 相关字段与 scale GM 分配 |

#### 场景：加 bias

官网 kernel 的 bias 通道已现成：`BiasType`（`BlockMmad::BiasType`）、`biasGmAddr_`（`ResetGmAddr()` 中由 `isBias_` 门控赋值）、`layoutBias`/`gmBias`（`ProcessSingleBatch()`）、`QBMMTiling::isBias` + `BIAS_ENABLED`、`mmadOp_.Init(..., isBias_, ...)`。**kernel 文件无需修改**。

**验收条件**（3 处修改，均在 impl / host 侧）：

| # | 修改位置 | 验收条件 |
|:---|:---|:---|
| 1 | udma_impl.h `SetupParams()` | 补 `mmadParams.biasGmAddr = ...`（官方 UDMA impl 当前未设置；HCCL 变体已设 `mmadParams.biasGmAddr`，可对照）；`QBMMTiling` 构造末参 `isBias` 由 `false` 改为按输入置位 |
| 2 | tiling_data.h | 含 bias 相关字段（如需） |
| 3 | main.cpp `LaunchKernel` | 含 bias GM 分配 + biasGM 参数透传 |

#### 场景：改 B 切分维度（N→K）

> 参考官方 PUT 实现：B 按 K 段拼接通过张量布局表达，无显式 B 指针累加。

**验收条件**：

| # | 修改位置 | 验收条件 |
|:---|:---|:---|
| 1 | kernel `LocalParams` | 含 `splitKNum` 字段（`quant_matmul_mx_kernel.h`），按参与 rank 数切分 K |
| 2 | B 张量布局 | 用 `MakeLayoutB{}(rankSize * K, N)` 张量布局拼接各 rank K 段（`ProcessSingleBatch()`），无显式 B 指针累加 |
| 3 | ProblemShape | N = 完整 `axisN`（非 `axisNPerRank`） |
| 4 | CommTilingData | data 通道 `nonSplitAxisSize` = ka（per-rank K）；scale 通道 = ka/32（`tiling/comm_tiling_data.h` `CommTilingData::nonSplitAxisSize`，与 [`operator-anatomy.md`](operator-anatomy.md) 一致） |

> K 轴切分通常用于 PUT 模式，GET 模式一般用 N 轴切分。

### 3.4 tiling_data.h 改造

#### 场景：加字段

**验收条件**：新字段在 tiling 结构体中正确声明，main.cpp 中正确赋值。

#### 场景：改 CommContext

**验收条件**：
- UDMA 模式：tiling_data.h 含 `CommContext` 结构体（`CommUdmaContext udmaCtx` + `CommUbmemContext ubmemCtx`，官网 `all_to_all_matmul_tiling_data.h`）
- HCCL windows 模式（非直调）：tiling_data.h 不含 `CommContext`，入口去掉 `__gm__ CommContext*` 参数

> 注意：CommContext 判据针对**具体使用的 tiling 结构体**，而非整个文件——官网 `all_to_all_matmul_tiling_data.h` 中 `CommContext`（UDMA 用）与 `ccuAllToAllMatmulTilingData`（CCU 用）共存。

### 3.5 场景：计算在前（compute-first）算子

> 架构原理见 [`fusion.md`](../fundamentals/fusion.md) §6.2；文件级契约见 [`operator-anatomy.md`](operator-anatomy.md) §7；场景完整指导（设计合同 + 实现细节模板 + 合规映射）见 [`scenarios/compute-first-reduce-scatter/`](../scenarios/compute-first-reduce-scatter/)。本节模式对一切计算在前（compute-first）算子通用。

**关键步骤**：

| # | 步骤 | 要点 |
|---|------|------|
| 1 | tiling_data.h | 通信切分 `commTilingData` + mm tiling（T>1 时**必须**三套：全量 `mmTilingData` + head 子问题 `subMmTilingData` + tail `tailMmTilingData`；T=1 退化仅全量）+ 就地定义 `CommContext` 与 flag 常量；无 `localMatmul` 字段 |
| 2 | mm 内核 | **默认自研 FragmentTensor kernel 消 R 循环**（R16）：FragTensorA/FragScaleA/FragTensorC 打包 R 个 rank 段地址 + `QmmMxBlockMmadFragment` per-fragment L1 隔离，一次调用覆盖 `R × curTileM` 行；参数结构与骨架以 [`fusion.md`](../fundamentals/fusion.md) §6.2.2/§6.2.7 契约和参考实现为准。vendor 复用官方 mm kernel 为例外（LOCAL 模式 + rank 退化 `rankId=0/rankSize=1/splitKNum=1` ⇒ `isAtomicAdd` 恒 false），必须论证 SCALAR 占比（[`fusion.md`](../fundamentals/fusion.md) §6.2.2） |
| 3 | Impl Run 编排 | AIC 统一 `for t` 循环（无 if/else，T=1 自然退化），**每轮 problem M = `R × GetTileM(t)`，地址带 tile 偏移**（见下方 per-tile 契约）。AIV 严格分离（默认）——编排骨架、分核映射公式（`jobIndex = GetBlockNum()-1-GetBlockIdx()`）、`Wait<BARRIER_NONE>`+手动 CrossDevice 序列以 [`fusion.md`](../fundamentals/fusion.md) §6.2.1 为唯一事实源，本节不重复 |
| 4 | 增量归约 | 独立文件 `reduce_sum_ref.h`；手动 UB 批量形态（6-slot 布局 + src 双缓冲 + FP32 中间累加 + N 分段 + T≤1 批量退化）以 [`fusion.md`](../fundamentals/fusion.md) §6.2.6 为唯一事实源；禁止 TPipe/TQue 逐行模型 |
| 5 | host 侧 | **前置校验清单**（见下）→ T/headMSize 派生（默认 `T \| mSeg` 无尾块；自适应 headMSize 决策与 flag 峰值联合约束见 [`fusion.md`](../fundamentals/fusion.md) §6.2.7；策略 A padding + 多套 tiling 为合法替代）→ staging 分配（`M×N×sizeof(CType)`）→ **dtype dispatch**（见下）→ fork 多进程 + TCP rootInfo 交换 + 建链 |
| 6 | flag 编排 | flagId 按 [`fusion.md`](../fundamentals/fusion.md) §3.3 规则选取；T=1 单次 / T>1 逐轮配对（Set/Wait 严格配对，计数深度按 R9 口径不设数值拒绝）；零 tile 核无条件 Set；SyncAll 在分核守卫外 |
| 7 | Win 区偏移 | compute-first 单通信对象 PUT：**winOffset 必须显式设置，禁止 0 偏移**——Win 数据区偏移按 host 建链布局确定、PUT 写入/归约读取/host 建链三处同源（布局规则与"假通过"失败案例以 [`fusion.md`](../fundamentals/fusion.md) §6.2.4 为唯一事实源——官方布局 barrier 在独立 BARRIER_BUF、数据区从 0 可用；共享布局须按约定偏移跳过头部，一种已验证实现为 96B→128） |

**dtype dispatch（禁止硬编码单入口）**：

> ⚠️ 官方 A2A ST `main.cpp` 当前硬编码调用 E4M3E4M3 入口（4 个 dtype 入口已定义但 host 无分派）——运行期 dispatch 是本 skill 生产纪律（R3），不是官方 ST 现状，禁止以"官方如此"为由照抄硬编码。

dtype 合同含 E4M3/E5M2 双组合的 FP8 量化算子需要 **4 个 dtype 变体入口**（E4M3E4M3/E5M2E5M2/E4M3E5M2/E5M2E4M3，见 [`operator-anatomy.md`](operator-anatomy.md) §7.2）；其他 dtype 合同按组合数覆盖入口。host 侧必须按 `dtypeA`/`dtypeB` 参数**运行期分派**到对应 `__global__` 入口，禁止硬编码单一入口——硬编码单入口时，异 dtype 字节流被错误模板解释（E4M3/E5M2 指数/尾数位宽与 bias 均不同），精度全元素不通过。**全元素不通过 + 误差量级稳定是"系统性解释错误"的特征信号，区别于精度累积问题**（后者误差不收敛但 matched_ratio 非零）。dispatch 宏模板（宏名按算子命名，勿残留旧算子名）：

```cpp
#define LAUNCH_{OP_NAME}_KERNEL(dA, dB, ...)                                    \
    do {                                                                          \
        if ((dA) == "e4m3" && (dB) == "e4m3")      KernelE4M3E4M3_Udma<<<...>>>(__VA_ARGS__); \
        else if ((dA) == "e5m2" && (dB) == "e5m2") KernelE5M2E5M2_Udma<<<...>>>(__VA_ARGS__); \
        else if ((dA) == "e4m3" && (dB) == "e5m2") KernelE4M3E5M2_Udma<<<...>>>(__VA_ARGS__); \
        else                                       KernelE5M2E4M3_Udma<<<...>>>(__VA_ARGS__); \
    } while (0)
```

> 注意：SWAT tiling 引擎的 dtype 模板参数可统一用 `DT_FLOAT8_E4M3FN`——CANN `mm::DataType` 枚举无 `DT_FLOAT8_E5M2`，E4M3/E5M2 均为 1 字节类型，tiling 参数仅依赖字节布局故完全一致（已实测验证）；dtype 差异只在 kernel 模板实例化层体现。`parseArguments` 需对 dtype 字符串做值域校验（仅允许 `e4m3`/`e5m2`）。

**per-tile 子区间契约（AIC RunMatmul，T>1 实现红线——违反即 T>1 精度全错）**：

代码模板见 [`scenarios/compute-first-reduce-scatter/development.md`](../scenarios/compute-first-reduce-scatter/development.md) §5.10；不变量如下：

| # | 不变量 | 违反后果（工程实证） |
|---|--------|---------------------|
| 1 | 每轮 mm 只算本轮子区间（`R × curTileM` 行），禁止每轮全量 `[M, N]` mm | 全量 mm 重复覆写 staging，T 次冗余 + AIV 读到未完成数据 → T>1 精度系统性失败（raw_max 达万级 ULP） |
| 2 | 归约每轮只处理 `turn` 对应行区间 `[turn×tileM, (turn+1)×tileM)`，禁止 `(void)turn` 处理全量 mSeg | 中间轮读取未填充 Win 区 → 精度 FAIL |
| 3 | PUT 钩子 src 偏移与 mm 写入偏移同源（`tileMOffset × axisN × sizeof(CType)`） | 通信搬运错位数据 |

**host 前置校验清单**（通用项如下；compute-first 完整 9 项清单（含 flag 峰值/单轮 PUT 阈值/FragmentTensor 上限/归约核数）见 [`scenarios/compute-first-reduce-scatter/development.md`](../scenarios/compute-first-reduce-scatter/development.md) §3.1。在 fork/建链前拒绝非法输入并给出可操作报错，main.cpp 与 gen_data.py 双侧校验）：

| # | 校验项 | 违反后果 |
|---|--------|---------|
| 1 | `M % rankSize == 0`（按算子切分语义选择被整除轴） | 切分错位 |
| 2 | 对齐约束（如 `K%32`、`N%16`，按 dtype/fixpipe 写侧 32B 行对齐推导） | 跨卡 Win 槽污染 |
| 3 | `usedCoreNum ≥ rankSize` | TeamBarrier rendezvous 永不齐 → 无超时挂死 |
| 4 | Win 容量 ≤ `HcclGetHcclBuffer` 实测值 | 通信越界 |

> **约束联合推导原则**：单条约束各自满足 ≠ 联合可行。正确性类约束（m%R、scale 偶数、tailM 16 对齐、Win 容量、usedCoreNum≥R+1、`R ≤ 32` FragmentTensor 容量）联合作用派生出 shape 可行域；风险提示类数值（单轮 PUT 大小、flag 计数深度）不进强制拒绝清单，按 case 复核（详见场景文档 §3.1 修订版）。host 校验与 T 派生必须覆盖派生边界，cases.csv 必须包含约束边界用例（strided 场景 N>redUbN、尾块边界、R 边界）——**硬件隐式上限类缺陷（静默丢零、无报错）只在边界用例下暴露**，缺失边界用例 = 缺陷逃逸到生产。

**验收清单**：compute-first 场景的逐项验收以 [`scenarios/compute-first-reduce-scatter/development.md`](../scenarios/compute-first-reduce-scatter/development.md) §6 合规映射（本场景最易踩中项 → 落点）+ [`review-checklist.md`](../review-checklist.md) 场景约束表为唯一事实源，本节不重复。

---

## 4. 工程构建

### 4.1 独立 CMake 工程（直调主场景）

> 当算子不在 ops-transformer 仓内开发时（CANNBot Kernel 直调工作流），需建独立 CMake 工程。文件级构建模式即本节内容。

**与仓内 ST 工程的关键差异**：

| 维度 | 仓内 ST | 直调独立工程 |
|:---|:---|:---|
| ops-tensor 依赖 | `cmake/third_party/ops-tensor.cmake` FetchContent | 直接引用 CANN 内置 tensor_api/blaze（路径随 CANN 打包形态实测定位，同 apace 根） |
| apace 框架 | 仓内源码树 | **直接引用 CANN 内置 apace（零复制）** |
| include 路径 | 主仓提供 | 手动配置 include 根（blaze/ tensor_api/ apace/） |
| 编译选项 | 主仓统一 | `-xasc --npu-arch=dav-3510 -DASC_DEVKIT_MAJOR=9 -O3` |
| hccl_fwk | 主仓链接 | `-Wl,--no-as-needed hccl_fwk -Wl,--as-needed` |

**CMake 骨架（已验证范式）**：

```cmake
set(ASCEND_DIR $ENV{ASCEND_HOME_PATH})
# CANN 内置 apace 公共根（路径按 Step 1 实测登记值；两种已验证形态见 §1.1：opp/built-in/... 或 vendors/custom_transformer/...）
set(OP_KERNEL_ROOT "${ASCEND_DIR}/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common")
set(APACE_ROOT "${OP_KERNEL_ROOT}/apace")

# blaze / tensor_api 三个 include 根（INTERFACE 库）
add_library(apace_blaze_api INTERFACE)
target_include_directories(apace_blaze_api INTERFACE
    "${OP_KERNEL_ROOT}" "${OP_KERNEL_ROOT}/tensor_api/include" "${OP_KERNEL_ROOT}/tensor_api")

# kernel 模式 include（apace 共享层 + 参考 kernel 全部直引，零复制）
set(KERNEL_INCLUDES
    ${OP_KERNEL_ROOT}                                        # apace/... 前缀 include 根
    ${APACE_ROOT}
    ${APACE_ROOT}/kernel/all_to_all_quant_matmul             # 官方参考实现（quant_matmul_mx_kernel.h 等）
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel                       # 本算子 kernel 头文件
    ${CMAKE_CURRENT_SOURCE_DIR}/src                          # kernel_launcher.h, utils.h
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include               # basic_api/ adv_api/ kernel_operator.h
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/ascendc/include/highlevel_api
    ${ASCEND_DIR}/compiler/tikcpp/tikcfw
    ${ASCEND_DIR}/include)
set(KERNEL_OPTS "-xasc" "--npu-arch=dav-3510" "-DASC_DEVKIT_MAJOR=9" "-w" "-O3")
set(COMMON_LIBS dl platform tiling_api ascendcl runtime hccl stdc++)
# hccl_fwk 必须 -Wl,--no-as-needed 包裹放最后
```

**验收条件**：

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | ASCEND_HOME_PATH | 已设置且指向 CANN 安装路径 |
| 2 | apace 直引 | `APACE_ROOT` 指向 CANN 内置 apace；算子目录无 `apace/` 复制副本（含 block/tiling/basic/utils/core 任何形态） |
| 3 | KERNEL_OPTS | 含 `-xasc` + `--npu-arch=dav-3510` + `-DASC_DEVKIT_MAJOR=9` |
| 4 | include 路径 | 覆盖 `blaze/` `tensor_api/` `apace/` + 本算子 kernel/src |
| 5 | hccl_fwk | 用 `--no-as-needed` 包裹 |
| 6 | COMMON_LIBS | `dl platform tiling_api ascendcl runtime hccl stdc++` + `apace_blaze_api` |

### 4.2 仓内 ST 工程（非直调场景，参考）

算子在 ops-transformer 仓内开发时，构建由主仓统一提供，独立工程无需关注：

- 顶层 `tests/st/CMakeLists.txt`：定位 bisheng 与 CANN 工具链、经 `cmake/third_party/ops-tensor.cmake` 拉取 blaze/tensor_api、创建 `apace_blaze_api` INTERFACE 库、`add_subdirectory` 接入用例
- 用例 `tests/st/{op}/CMakeLists.txt`：target 名改为新算子名；KERNEL_OPTS 含 `--npu-arch=dav-3510`；include 含 `${APACE_ROOT}/kernel/{op_name}`；链接 `COMMON_LIBS` + `apace_blaze_api`；`hccl_fwk` 用 `--no-as-needed` 包裹放最后

### 4.3 常见构建失败对照

| 现象 | 根因 |
|:---|:---|
| Ascend toolkit not found | ASCEND_HOME_PATH 未设置 |
| bisheng not found | CANN 安装不完整或架构不匹配 |
| blaze not found | CANN 内置 ops_transformer 路径缺失（检查 §4.1 路径） |
| tensor_api/tensor.h not found | 同上，tensor_api include 根未配置 |
| HcclChannelAcquire undefined | hccl_fwk 链接顺序错误（需 `--no-as-needed`） |

### 4.4 run.sh / cases.csv [MODIFY]

**验收条件**：cases.csv 中的 shape 和参数与新算子匹配。官网两份 ST 列名不同：all_to_all 为 `m,k,n,rank_num,head_m_size`，all_gather 为 `m,k,n,rank_num`（无 head_m_size，tileM 固定 `min(m, 512)`）；run.sh 支持按行号/`all`/`--cli`/`--perf` 运行（`--perf` 模式产出经 `scripts/parse_prof.py` 解析）。

### 4.5 main.cpp [MODIFY]（`src/main.cpp`）

**验收条件**：

| 改造场景 | 验收条件 |
|:---|:---|
| 换算子名 | 全局无旧算子名残留（含 ctxTag） |
| 换 kernel 名 | `LaunchKernel` 调用使用新 kernel 类名 |
| 换 dtype | `tilingEngine` 模板参数为目标 dtype（参考算子为 `QuantMatmulTilingSwat<DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN>`） |
| 换 tiling 结构 | `tilingData` 字段与 tiling_data.h 一致 |
| 加 bias | 含 bias GM 分配 + `LaunchKernel` 传 biasGM |
| localMatmul | 显式赋值（参考算子 `tilingData.localMatmul = 0`） |

> main.cpp 完整结构详见 [`host-and-testing.md`](host-and-testing.md)

### 4.6 scripts [MODIFY]（`scripts/`）

**验收条件**：
- `gen_data.py`：dtype、shape、golden 计算逻辑与新算子一致
- `verify_result.py`：dtype、容差、输出形状与新算子一致
- 精度标准：all_to_all 用 `rtol=atol=1e-2`（`ERROR_TOL`），all_gather 用 bit-exact 或 ≤1 ULP（bf16 raw uint16 比较）——各算子容差不同，以对应 `verify_result.py` 为准

**gen_data 三条通用原则**：

1. **golden 语义先行**：写脚本前先明确每张卡的输入/输出语义与切分轴（设计阶段产出，见 §2 步骤 2）——切分轴写错则 golden 全错且误差分布不指向真实 bug
2. **golden 走高精度路径**：CPU golden 的累加/反量化在 float32 下完成、最后才转输出 dtype，避免 golden 自身引入低精度误差（否则容差比对失去意义）；多段截断的算子，golden 的截断点要与 kernel 精度路径一一对应
3. **固定随机种子**：保证用例可复现，精度回归可对比

**cases.csv 用例覆盖维度**（设计用例矩阵时逐项自查）：

| 维度 | 目的 |
|:---|:---|
| 同 shape × 不同 rankNum 对照 | 暴露通信切分与 rank 数相关问题 |
| 通信 tile 粒度参数扫描（如 headMSize） | 覆盖流水深度边界（含 tileCnt=1 退化） |
| **T=1 与 T>1 双路径** | 退化路径最容易藏布局/初始化 bug；**只测 T=1 会掩盖多 tile 问题**（典型形态：多 tile 下输出布局非 row-major 时 reducer 读错位，T=1 恰好 row-major 全部 PASS） |
| 小 rank（如 rank=2） | 退化通信路径（self 合并、单 peer）单独验证 |
| tail 非整除 shape | 暴露 tail tile 对齐/padding 问题（"仅非对齐 shape 失败"是 tail 路径的特征信号） |
| 极小 shape | 覆盖退化路径（单 tile、零 tail） |
| 极端长宽比 | 覆盖 tiling 引擎边界分支 |
| 大 shape | 验证累加精度随 K/M 放大的行为与性能达标性 |

> host 侧约束（整除/对齐校验）应在 main.cpp 与 gen_data.py **双侧重复校验**，任一侧遗漏都会把非法输入送进 kernel 变成难查的精度/死锁问题。

---

## 5. 验收清单

### 工程验收条件

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | 工程结构 | 符合 §1 推荐布局（kernel/ + src/ + scripts/ + cases.csv + run.sh + profiling/）；算子目录 camelCase |
| 2 | 共享层零复制 | 算子目录无 `block/` `tiling/` `basic/` `utils/` `core/` 任何 apace 共享层副本；CMake `APACE_ROOT` 指向 CANN 内置路径 |
| 3 | 命名一致性 | 全局无旧算子名残留（类名、文件名、namespace、ctxTag） |
| 4 | 编译通过 | 无编译错误/警告 |
| 5 | 冒烟测试 | 单 rank 运行输出非全 0 |
| 6 | 精度验证 | 多 rank + 多 shape `verify_result.py` PASS |
| 7 | 文档同步 | PLAN.md 已更新改造内容和测试结果；DESIGN.md 与代码一致 |
| 8 | tiling 结构体契约 | `#pragma pack(push, 8)` + `alignas(8)`，字段顺序即 host-device 契约不可乱序 |
| 9 | 场景红线合规 | 编排/通信/归约/host 校验等场景红线逐项通过——以 [`review-checklist.md`](../review-checklist.md)（全局红线 + 场景约束表）为唯一事实源，本清单不重复 |
| 10 | API 环境验证 | 设计阶段选用的每个 API 已在当前 CANN 版本验证可用（来源文件:行号记录于 DESIGN.md），无符号缺失/链接失败残留 |

### 禁止行为

- 禁止从零写文件（参考 CANN 内置 apace `kernel/` 下官方算子起手；本算子文件按 §1 布局新建）
- **禁止整包复制 apace 共享层到算子目录**（`block/` `tiling/` `basic/` `utils/` `core/` 任何形态）——一律 CMake 直引 CANN 内置路径（§1.1/§4.1）
- 禁止跳过冒烟编译直接全量测试
- 禁止添加 `__schedmode__(1)` 或 `[[bisheng::core_ratio(1,1)]]`
- 禁止修改 `localMatmul` 等关键参数后不同步更新 DESIGN.md
- 禁止通信 UB（静态偏移区）与归约 buffer 混用重叠（[`communication.md`](../fundamentals/communication.md) 陷阱 #9）
- mm 内核默认 FragmentTensor 消 R 循环（[`fusion.md`](../fundamentals/fusion.md) §6.2.2）；选 vendor R×T 子调用为例外，必须在 DESIGN.md 论证 SCALAR 占比可接受（R×T 小 + 大 shape），否则按 FragmentTensor 实现
- 禁止把 host 侧非法输入（不满足整除/对齐/容量约束的 shape）放进 kernel 触发难查的死锁/精度问题——校验必须前置
- 禁止 compute-first 场景用 `Wait<BARRIER_DEVICE>`（totalJobs=rankSize 时内建 CrossDevice 不轮询 remote → 跨设备同步失效）与 winOffset=0（覆盖 barrier flag → 假通过）
- 禁止归约使用 TPipe/TQue 逐行模型（必须手动 UB + 2D DataCopyPad 批量，[`fusion.md`](../fundamentals/fusion.md) §6.2.6）

### tileCnt 策略

| 阶段 | tileCnt | 目的 |
|:---|:---|:---|
| 精度调试阶段 | 1（串行基线） | 消除流水干扰，flag 出错直接 deadlock 便于排查 |
| 性能调优阶段 | 扫描 {1,2,4,8,16,32} | 找 Task Duration 最小值 |

**验收条件**：切换 tileCnt 必须重新调 `QuantMatmulTilingSwat::GetTilingData`，不能复用旧 tiling。

> 详见 [`pipeline_tuning.md`](../../../shared/pipeline_tuning.md)（apace 用 `splitAxisTileCnt` 命名）

---

## 后续阅读

- [`architecture.md`](../fundamentals/architecture.md) — 心智模型入口（三层架构、GET/PUT、基础约束）
- [`operator-anatomy.md`](operator-anatomy.md) — 算子骨架共性模式
- [`host-and-testing.md`](host-and-testing.md) — host/ST
- [`fusion.md`](../fundamentals/fusion.md) / [`compute.md`](../fundamentals/compute.md) / [`communication.md`](../fundamentals/communication.md) — 接口与组合模式
