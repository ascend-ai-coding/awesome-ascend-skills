# MLA Prolog 前处理融合算子优化设计（mla_prolog_fusion）

> 适用架构：arch35（A5 / DAV_3510）；经验来源为 MlaPrologV3（DeepSeek MLA 注意力前处理融合算子）的设计与 arch35 kernel 实现。
> 定位：**Cube+Vector 大融合算子**的通用优化方法论——把多路 Matmul + RMSNorm + RoPE + 量化/反量化 + KV/KR Cache 更新融进单 kernel，靠 CV 流水互相掩盖。可迁移到其它"GEMM + 归一化 + 位置编码 + cache 写"形态的注意力前/后处理融合算子。
>
> 使用前置：确认目标是 A5、且算子确为 mixed AIC/AIV 融合形态（单 kernel 内既有 Cube 矩阵乘又有 Vector 归一化/量化）。若只是单一算子，改查对应算子族（matmul / reduce / conversion）。
>
> **与定量方法的衔接**：本设计给出结构、同步与形态选路的方法论，不含单算子实测数字。落地具体 case 时按以下顺序补齐定量闭环：
> ① tile 形状、缓冲深度与 CV 节拍按 [SFA 四层容量建模](cannbot-dsl/optimize/3-capacity-equations.md) 对本算子链**重推导**——第 1 层硬约束输入取当前实现的真实常量（`He`、`Dq`/`q_lora_rank`、`Nq` heads、NoPE `D`、RoPE `Dr`、cache 模式与量化模式集合、`AIC:AIV` 配比，以实际 `mla_prolog_comm.h`/manifest 为准）；本算子无 softmax 状态与 gather 环，台账住户、整除链与平衡方程逐项重推，不套用 SFA 数值；
> ② DSL 表达层惯用法（视图声明/双代形态/动态 region/unit_flag）按 [SFA 实施惯用法](cannbot-dsl/assemble/glue-idioms.md)；
> ③ 节拍/带宽验收与上板 profiling 流程按 [SFA 延迟线流水](cannbot-dsl/optimize/2-pipeline-modeling.md)；
> ④ 单个 Matmul 子块（如 `x·W^DQ`）的候选 tiling 可用 `ascendc-perf-optimize` 的 matmul 实测求解工具包扫描裁决。

## 1. 优化目标

- **减少下发与头开销**：4 个 Matmul + 2×RMSNorm + 2×RoPE + 量化/反量化 + 2×ScatterCache 融进一个 kernel，省去 N-1 次 kernel 下发与中间结果的 GM 往返。
- **Cube/Vector 并行**：Vector 侧（RMSNorm/RoPE/量化/Scatter）与 Cube 侧（Matmul）存在依赖，若不做 CV 流水会退化成串行，融合就失去意义；目标是让两侧尽量并行、互相掩盖搬运与计算。
- **多形态零运行时分支**：cacheMode / 量化模式 / 切分模式全部编译期特化。

> 效果为方法论层面（源自 MlaPrologV3 设计与实现），非单点 μs 实测；落地到具体 case 后需用 profiling 复测 CV busy、MTE2 段与 kernel 耗时。

## 2. 架构概览

### 2.1 计算流（四路 + 依赖）

整体公式（$\alpha_q,\alpha_{kv}$ 为 Query/Key 尺度矫正因子）：

```
q^N = α_q·RmsNorm(x·W^DQ)·W^UQ·W^UK       # Query（两次上采样）
q^R = ROPE(α_q·RmsNorm(x·W^DQ)·W^QR)       # Query 位置编码
k^R = ROPE(x·W^KR)                          # Key 位置编码 → KRCache
c^KV = α_kv·RmsNorm(x·W^DKV)                # KV 压缩 → KVCache
```

数据依赖图（`(C)`=Cube/AIC，`(V)`=Vector/AIV）：

```
                     ┌───────────── token_x ─────────────┐
                     │                                    ▼
                     │                          MatmulCkvKr (C)  [x·[W^DKV|W^KR]]
                     ▼                              │ FINISH_MM_CKVKR
                 MatmulCq (C) [x·W^DQ]        ┌─────┴─────┐
                     │ FINISH_MM_CQ            ▼           ▼
               RmsNormCq (V)              RmsNormCkv(V)  RopeKr(V)
                     │ FINISH_VEC_RMSNORM_CQ   │           │
      ┌─── MatmulQcQr (C) ───┐   [c_norm^Q·[W^UQ|W^QR]]   │
      │ FINISH_MM_QCQR       │    ScatterCkv(V)     ScatterKr(V)
      ▼                      │       ▼                  ▼
  DequantQc (V, 量化)        │   kv_cache_out       kr_cache_out
      ▼                      │ FINISH_MM_QR
  MatmulQn (C) [q^C·W^UK]    ▼
      ▼                   RopeQr (V)
  query_out               query_rope_out
```

**关键跨 Cube/Vector 依赖**：`MatmulQcQr` 依赖 `RmsNormCq`；`MatmulQn` 依赖 `MatmulQcQr`。这些边就是必须插同步的地方。

### 2.2 存储层级与内存布局（arch35 常量见 `mla_prolog_comm.h`）

```
ALIGN_BLOCK_SIZE = 32B ; MAX_UB_SIZE = 192KB
L1_A = L1_B = 128KB ; L0A = L0B = 32KB ; L0C = 64KB（均按 *2 开 Double Buffer）
```

- **Cube L1/L0**：A/B/C 一律开 `*2` 做 Ping-Pong；事件 `A_EVENT0/1`、`B_EVENT0/1`、`L0C_EVENT0/1` 配迭代计数器奇偶翻转选页，一页算、一页预取下一段 K。
- **Vector UB = 固定区 + ShareBuffer**：先按需固定分配 gamma / 各 scale / sincos（跨轮复用的常量），再用 `MAX_UB_SIZE - usedBytes` 整块开 `shareBuffer_` 吃满余量给算子临时用；cos/sin 存同一 buffer 用两个 LocalTensor 指针偏移复用。
- **GM workspace = 中间结果链**：放不下 UB 或需跨 Cube/Vector 阶段传递的中间结果落 workspace，按 `workspaceOffset` 顺序累加；类型一致的相邻结果共段（如 `mmCqOutputType==rmsNormCqOutputType` 时 `mmCqRes` 与 `rmsNormCqRes` 复用同段），受 `queryNormFlag`/量化模式控制的段按需分配。

### 2.3 事件同步模型

```cpp
SYNC_MODE_CUBE_VEC = 0x2;   // Cube ↔ Vector 点对点
SYNC_MODE_ALL_CUBE = 0x0;   // Cube 内全核 barrier
SYNC_MODE_ALL_VEC  = 0x0;   // Vector 内全核 barrier
MAX_SYNC_FLAG_COUNT = 15;   // 同一 flagId 计数器上限
```

- 点对点：`CrossCoreSetFlag<mode,pipe>(flagId)` / `CrossCoreWaitFlag(flagId)`；Cube 完成用 `PIPE_FIX`（FixPipe 搬出后），Vector 完成用 `PIPE_MTE3`（DMA 写出后），保证 flag 置起时数据真就绪。
- 全核 barrier：`WaitAllCore<mode,pipe>(flagId)` = SetFlag + WaitFlag。

## 3. 关键手段与修改点

| # | 手段 | 原理 | MlaProlog 落地 |
|---|------|------|----------------|
| 1 | **算子融合** | 省 N-1 次下发 + 中间 GM 往返 | 4×MM + 归一化/位编/量化/scatter 融进单 kernel |
| 2 | **权重拼接** | `x·[W^A\|W^B]=[x·W^A\|x·W^B]`，一次 MM 出两路 | `[W^DKV\|W^KR]`、`[W^UQ\|W^QR]`；加载 B 时按 K 偏移拼进同一 L1，算完 split |
| 3 | **反量化融进 Matmul** | L0 加载权重带 A-scale，Mmad 内部反量化后累加 | fp8/mxfp8 走 `LoadDataL1ToL0Mxfp8`，省独立反量化 pass |
| 4 | **量化融进 RMSNorm** | 归一化算完紧接 rowmax 动态量化，中间 fp32 不落 UB 外 | `RmsNormDynamicQuant` |
| 5 | **CV 流水 + 细粒度同步** | Cube 算完一段即通知 Vector | 切 N 时 MatmulQcQr 每算完一 N 块发 `FINISH_MM_QCQR_SPLIT_N`，Vector 提前做 DequantQc |
| 6 | **稀疏同步** | 绕开 flag 计数上限(15) | `subLoopTimes>15` 时隔次同步、末轮必同步 |
| 7 | **Double Buffer** | 搬运藏进计算 | L1/L0 A/B/C 全 `*2` |
| 8 | **NZ 格式 + stride 搬运** | 免格式转换 / 一次重排写 cache | 权重 FRACTAL_NZ 走 `CopyNZGmToL1`；ScatterCache NZ 用 `stride=32B*BlockSize` + `DataCopyPad` |
| 9 | **向量微指令用满 256B** | 一拍处理满寄存器宽度 | RMSNorm `Reduce<SUM>`+`Sqrt`+`Duplicate` 广播、一条 `Mul` 融合除法与 gamma；RoPE `DeInterleave` 奇偶拆分 + mask 低/高半一拍两路 |
| 10 | **TilingKey 位域编译期特化** | 运行时零分支，只编译用到的路径 | 见 §5 |
| 11 | **分核负载均衡** | 核数用满、每核均匀 | `blockDim=aicNum`；切 M 时前 `mSubCoreNum` 核多担 1 token |

### 核心计算循环（伪码，切 N 场景）

```cpp
void Process() {
  for (i = 0; i < loopsThisCore; i++) {
    updateCurrentStepSize(...);
    if (ASCEND_IS_AIC) {                 // ===== Cube =====
      MatmulCq(...);        CrossCoreSetFlag<0x2,PIPE_FIX>(FINISH_MM_CQ);
      MatmulCkvKr(...);     CrossCoreSetFlag<0x2,PIPE_FIX>(FINISH_MM_CKVKR);
      CrossCoreWaitFlag(FINISH_VEC_RMSNORM_CQ);          // 等 V: RmsNormCq
      MatmulQcQr(...);      CrossCoreSetFlag<0x2,PIPE_FIX>(FINISH_MM_QCQR);
      // MatmulQn 分核策略≠QcQr 且依赖它 → Cube 全核 barrier
      WaitAllCore<0x0,PIPE_FIX>(FINISH_MM_ALL);
      MatmulQn(...);                                     // 直接写 queryOut
    }
    if (ASCEND_IS_AIV) {                 // ===== Vector =====
      GetSinCos(...);
      CrossCoreWaitFlag(FINISH_MM_CQ);
      WaitAllCore<0x0,PIPE_MTE3>(FINISH_VEC_ALL);        // RmsNormCq 按行切
      RmsNormCq(...);
      WaitAllCore<0x0,PIPE_MTE3>(FINISH_VEC_ALL);        // QcQr 按 N 切 → 全核对齐
      CrossCoreSetFlag<0x2,PIPE_MTE3>(FINISH_VEC_RMSNORM_CQ);
      CrossCoreWaitFlag(FINISH_MM_CKVKR);
      RmsNormRopeScatterCkvKr(...);                      // RMSNorm+Scatter / Rope+Scatter
      CrossCoreWaitFlag(FINISH_MM_QCQR);
      RopeQr(...);
    }
  }
  // 收尾：Cube 侧回收全部 L1/L0 event（A/B/L0A/L0B/L0C EVENT0/1），否则挂死后续 kernel
}
```

**为什么要全核 barrier**：当上下游分核策略不同（`RmsNormCq` 按 token 行切、`MatmulQcQr` 按 N/head 切），一个下游核要消费多个上游核的结果，必须等所有上游核完成 → 用 `WaitAllCore`。规律：**点对点依赖用单 flag；跨分核策略的依赖用全核 barrier**。设计时先尽量对齐上下游分核策略，能省 barrier 就省。

## 4. 分核与 CV 比例

| 维度 | 切 N（默认） | 切 M |
|------|-------------|------|
| 主轴 | N（head） | M（batch/token） |
| 同步粒度 | 可细粒度（每 NL1 块） | 粗粒度（整 M 块） |
| workspace | 各核只读同一份 MM 结果 | 各核独占写，需先清零上一 step 残留（AIV 清零后放行 Cube） |
| 适用 | 通用 / 低时延 | 大 BS（≥8K）、mxfp8 全量化等高吞吐 |

- **CV 比例**：`aivNum==aicNum`→1:1（仅非量化 + PA_BSND/PA_NZ）；`aivNum==2*aicNum`→1:2（默认，支持全部量化）。
- MatmulCkvKr 在 `aicNum≥9` 固定 `baseN=64` 直接均分，避免过度切割。
- 切 M 触发（示例）：mxfp8 全量化 且 `T≥8192` 且 cvRatio==2 且 `He∈{7168,7680}` 且 `N==128`。

## 5. TilingKey 位域（编译期特化）

uint64 TilingKey，Host `GenTilingKey` 生成，kernel 用 `if constexpr`/模板参数展开，运行时零分支：

| bit | 变量 | 取值 |
|-----|------|------|
| 0-3 | CACHE_MODE | 1-PA_BSND 2-PA_NZ 3-PA_BLK_BSND 4-PA_BLK_NZ |
| 4-5 | SCENARIO | 1-非量化 2-量化 |
| 6-9 | QUANT_MODE | partial/full/mxfp8 × kv 量化方式组合 |
| 10 | ENABLE_DEQUANT_OPTIONAL | 反量化流水优化 |
| 11 | ENABLE_GROUP_COMPUTE_OPTIONAL | 量化算力分组（与 bit10 互斥） |
| 12-13 | EMPTY_TENSOR_MODE | 空 tensor |
| 14-15 | ACTUAL_SEQ_LEN_MODE | actualSeqLen 开关 |
| 16-17 | SPLIT_M_MODE | 0-切N 1-切M |
| 18-25 | CV_MODE | 6-MIX_AIC_1:1 7-MIX_AIC_1:2 |

## 6. 典型场景选路

| 场景 | quant(weight/kv/query) | 优化开关 | CV | 切分 |
|------|------------------------|---------|----|----|
| 非量化(baseline) | 0/0/0 | — | 可 1:1 | 切 N |
| 部分量化·kv 非量化 | 1/0/0 | 反量化融合 | 1:2 | 切 N |
| 部分量化·kv perchannel | 1/2/0 | +cache 量化写 | 1:2 | 切 N |
| int8/fp8/hif8 全量化 | 2·4·5/…/… | enableDequantOpt | 1:2 | 切 N |
| mxfp8 全量化·大 BS | 3/…/… | enableDequantOpt+splitM | 1:2 | 切 M |
| 低时延分组(decode) | 部分量化 | enableGroupComputeOpt | 1:2 | N 按 head 分组 |

落地顺序建议：先跑通**非量化切 N**（把计算流与 CV 同步跑对）→ 叠加**部分量化**（量化/反量化融进已有流水）→ **全量化 + enableDequantOpt**（反量化被 MM 流水掩盖）→ 针对特定大 shape 加**切 M** 或**低时延分组**。全程用 TilingKey 位域 + `if constexpr` 承载所有形态。

## 7. 约束与注意事项

- `enableDequantOpt` 与 `enableGroupComputeOpt` **互斥**，不能同开。
- CV 1:1 仅在非量化 + PA_BSND/PA_NZ 下合法。
- 长循环（>15 次同步）必须稀疏同步，否则 flag 计数溢出。
- Cube 侧 Process 结束必须回收所有 L1/L0 event，否则残留 flag 挂死后续 kernel。
- 切 M 下各核独占写 workspace，务必先清零上一 step 残留再放行 Cube 写入。
- fp8 场景 scale（e8m0）在 MatmulQcQr↔MatmulQn 间需用独立 `SCALE_EVENT` 管理，避免竞争。

## 8. 迁移优先级

算法层（融合 §3.1 / 拼接 §3.2 / 量化融合 §3.3-4）> 流水层（CV 并行 §3.5 / Double Buffer §3.7）> 搬运层（NZ + stride §3.8）> 指令层（MicroAPI §3.9）> 工程层（TilingKey §5 / 分核 §4）。先吃算法层大头，再逐层压榨。
