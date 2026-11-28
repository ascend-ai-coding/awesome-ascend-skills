# Compressor KV 压缩融合算子优化设计（compressor_fusion）

> 适用架构：arch35（A5 / DAV_3510 / ASCEND910_95）；经验来源为 xllm-ops `attention/compressor`（MLA 风格 KV 压缩融合算子）的 arch35 kernel 实现与 6 轮性能调优台账。arch32（910B，`__CCE_AICORE__==220`）为独立计算核，结论按 arch35 采集。
> 定位：**Cube+Vector 大融合、decode 单 token(T=1) varlen、权重载入访存受限**算子的优化方法论。与 [`mla_prolog_fusion`](mla_prolog_fusion_design.md) 同族（GEMM + 归一化 + 位置编码 + cache 写），但瓶颈相反：MLA Prolog 是 prefill 大 BS 算力/流水主导，Compressor 是 decode T=1 **访存带宽 + 核欠订 + 跨核规约同步**主导。可迁移到其它"投影 MM + 打分聚合 + 归一化 + 位编 + KV cache 写"形态的注意力前处理融合算子在 **小 batch / decode** 场景。
>
> 使用前置：先用 profiling 确认 `cost_bound`（是否 Memory-Bound / LATENCY-BOUND）、`usedCoreNum` 是否欠订、权重载入有效带宽、`aic_mte2_ratio`。**先判 bound 类型再选手段**，不要按 shape 名或算子名直接套。若非融合形态或非小 batch，改查对应算子族（matmul / reduce）或 [`mla_prolog_fusion`](mla_prolog_fusion_design.md)。

## 1. 优化目标

- **消除空转核与空转同步**：T=1 退化 varlen 下 D×M 切分只需少量核，默认满核 BATCH_MODE 启动会让多数核空转还要陪跑 SyncAll → 先把 blockDim 降到最小覆盖。
- **把权重载入推向峰值带宽**：本类算子权重（~16.78MB ND）由 Cube MTE2 载入，是主要访存量、决定 roofline（16.78MB / 4TB/s ≈ 4.19µs）；碎块载入有效带宽 <20%，目标是合并载入 + 用空闲核沿 K 轴并行摊薄。
- **逼近跨核规约的同步/GM 中转下界**：CV-UB 直通架构下跨核 UB 不可见，跨核规约只能走 GM + SyncAll；目标是静态化 tiling、削减冗余屏障，把 wall 推到该架构下界。
- **公开契约零改动**：算子 def/proto、ND 权重格式、7 输出布局不变，优化只落 kernel 内部 + host tiling 的 arch35 分支。

> 效果为实测（compressor，Ascend950，T=1 varlen decode）：`34.012µs → 29.8µs`，累计 **-12.38% / 1.1413×**，strict_pass（cmp_kv 逐字节一致）。落地到其它 shape 需 profiling 复测。

## 2. 架构概览

### 2.1 计算流（融合链 + 依赖）

整体语义：对每个压缩块(Tc) 把 seq 内 token 经投影 + 打分后加权聚合成一个压缩 KV，再归一化 + 位编写回 cache。

```
x ──MM1(wKv, wGate)──► 投影 ──softmax 打分──► 加权和 ──RMSNorm──► 半旋 RoPE ──► cmpKv
```

数据依赖图（`(C)`=Cube/AIC，`(V)`=Vector/AIV）：

```
             token_x + 权重(wKv/wGate, ND, 全 K=hiddenSize)
                          │
                    ComputeMm1 (C)   [D 轴分片, 每核读 D 分片 × 全 K]
                          │ CrossCoreSetFlag(SYNC_C1_V1)
                          ▼
                    ComputeVec1 (V)  [softmax 打分 + 加权和] ──► vec1ResGm(GM)
                          │ SyncAll (跨核可见性屏障, data-before-reduce)
                          ▼
                    ComputeVec2 (V)  [RMSNorm + 半旋 RoPE] ──► cmpKv
                          │ SyncAll (跨核 WAR: 下一轮 Vec1 写 vs 本轮 Vec2 读)
                          ▼
                      (下一 Tc 块)
```

**关键跨 Cube/Vector 依赖**：`Vec1` 依赖配对 `Mm1`（点对点 `C1_V1` flag）；`Vec2` 依赖本组所有 `Vec1` 对 `vec1ResGm` 的写完成（跨核 RAW → 全核 SyncAll）。这两处是必须插同步的地方，也是同步开销的来源。

### 2.2 存储层级与内存布局

- **权重（主访存量）**：ND 格式，~16.78MB，由 **Cube MTE2** 载入。碎块载入（小 dValue、stride=4096）有效带宽仅 ~14.7% —— 优化核心对象。
- **Workspace GM = 中间结果链 + 跨核规约区**：
  - `preMm1ResGm` / `curMm1ResGm`：MM1 输出（双缓冲，`dbWorkspaceRatio` 控制）。
  - `vec1ResGm` / `vec2InputGm`：Vec1→Vec2 中转，**同一 GM 别名**复用，省 workspace 但产生跨核 RAW，需 barrier 保护。
  - `ksplitMm1ResGm`：K 切分 partial-K 结果，**每核一槽**（`curGroupIdx*mm1ResSize` 定址，**无 AtomicAdd**），从 workspace 尾部 carve；槽布局与 UB(`mm1ResUb`) 同形，便于 `FromWorkSpaceToUb`→`AddMultiDataToUb` 直搬直加。
- **UB**：`mm1ResUb` 等临时区，双缓冲。跨核之间 UB 不可见。
- **分配原则**：① 跨核规约用独立 GM 槽 + 事后 fp32 求和，不用 AtomicAdd（T=1 规模求和更省且确定性好）；② 中转结果用 GM 别名复用换 workspace，代价是显式 barrier；③ partial 槽与 UB 同形省重排。

### 2.3 事件同步模型

```
CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX/PIPE_V>(SYNC_C1_V1_FLAG)  // Cube→Vector 点对点(配对)
CrossCoreWaitFlag<SYNC_MODE2, PIPE_V>(SYNC_C1_V1_FLAG)          // Vector 等配对 Cube
SyncAll()                                                        // 全核 barrier (跨核 GM 可见性)
```

- **点对点**：配对 AIC/AIV 用 `C1_V1` / `V1_C1` flag 维持活性握手；即使某核本轮无活也要归还 flag 保持配对计数平衡。
- **全核 barrier**：跨核规约的 GM 可见性、以及 Vec1→Vec2 的跨核 RAW，用 `SyncAll()`。**这是 CV-UB 直通架构跨核聚合的物理下界，删除会触发 507015。**
- **计数平衡铁律**：全核（含空闲组核）必须一致地参与每个 SyncAll（`blockDim==usedCoreNum`）；省屏障必须是**核间一致的循环常量条件**（如 `isLastVec2Batch`），否则计数错位卡死/507015。

## 3. 关键手段与修改点

| # | 手段 | 原理 | Compressor 落地 | 实测收益 |
|---|------|------|-----------------|---------|
| K1 | **shape-adaptive core**（host） | 小 token D×M 用不满核，满核启动=空转+空转同步 | host 按实际覆盖降 `usedCoreNum/blockDim`（28→8） | -1.845µs / 1.0574× |
| K2 | **MTE2 权重合并载入**（cube） | 碎块 ND 载入有效带宽 <20% | 合并 64 次 `CopySingleMatrixNDToNZ` 碎块为大 K 基块 | MTE2 -10.22% / -1.006µs |
| K3 | **FULL_LOAD K 轴核并行**（cube+vec+host） | 空闲核沿 hidden(K) 切分，各核只读 `hiddenSize/kBaseNum` 权重（仍走 Cube 满带宽），partial-K 落各自 GM 槽 vec 侧求和 | 移植参考 FULL_LOAD；甜点 kBaseNum=2/16 核 | 累计 -11.06% / 1.1244× |
| K4 | **静态 tiling + 低同步**（host+kernel） | host 预算每核 `[kStart,kEnd)` 入 TilingData 免运行期 SplitK 重算；末轮无后续 GM 写省冗余 WAR 屏障 | `splitCoreParam[aiCoreIdx]` 直读 + `isLastVec2Batch` 省屏障 | 累计 -12.38% / 1.1413× |
| — | **双缓冲** | 搬运藏进计算 | MM1 输出 workspace `dbWorkspaceRatio`、UB 双缓冲 | — |
| — | **编译期特化** | 运行时零分支 | TilingKey 位域（见 §5），`kBaseNum_==1` 退回 A/E 路径字节不变 | — |

### 核心计算循环（K 切分伪码，`ProcessKSplit`，`kBaseNum_>1`）

```cpp
void ProcessKSplit() {
  AllocEventID();  GetCurCoreStartIdx();
  myKGroupIdx   = aiCoreIdx / dBasicBlockNum;      // 本核 K 组序
  isDownstream  = (myKGroupIdx == 0);              // 仅 K 组 0 的 AIV 做跨组求和
  for (i = 0; i < singleCoreDealTcBasicNum; i++) {
    CalcParams(info);
    if (ASCEND_IS_AIC) {                            // ===== Cube =====
      if (doThisIter) ComputeMm1(info);            // 本核 K 段 partial → GM 组槽 (含配对 AIV 活性 C1_V1)
    } else if (doThisIter) {                        // ===== Vector =====
      CrossCoreWaitFlag<..,PIPE_V>(SYNC_C1_V1);    // 与配对 AIC 握手
    }
    SyncAll();                                      // 跨组 GM partial 写完, 对 K 组 0 AIV 可见 (防 507015)
    if (ASCEND_IS_AIV) {
      if (isDownstream && doThisIter) KSplitSumToUbAndVec1(info);  // 逐组 fp32 求和 → Vec1
      if (doThisIter) CrossCoreSetFlag<..,PIPE_V>(SYNC_V1_C1);     // 归还配对 flag (计数平衡)
    }
    if (batchEnd) {
      SyncAll();                                    // barrier A: 本组 Vec1 写完, Vec2 才读别名 GM (跨核 RAW, 不可减)
      if (ASCEND_IS_AIV && isDownstream) ComputeVec2(vec2Info);    // RMSNorm + 半旋 RoPE → cmpKv
      // barrier B: 仅非末轮需要 (保护下一轮 Vec1 写 vs 本轮 Vec2 读的 WAR); 末轮省
      if (!isLastVec2Batch) SyncAll();
    }
  }
  FreeEventID();                                    // 回收 event, 否则挂死后续 kernel
}
```

**为什么全核都进 SyncAll**：K 切分下 Vec1/Vec2 只在 K 组 0 的 AIV 执行，其余组 AIV 空闲，但**全核仍须一致参与每个 SyncAll** 保持计数平衡；省屏障（barrier B 末轮）用的是**全核一致的循环常量** `isLastVec2Batch`，全体核在同一轮一致地省，计数不错位。规律同 MLA Prolog：**点对点依赖用配对 flag；跨核规约/RAW 用全核 barrier；能省的屏障必须核间一致地省。**

## 4. 分核与 CV 比例

| 维度 | 无 K 切分（A/E，kBaseNum=1） | K 轴核并行（J/K，kBaseNum>1） |
|------|------------------------------|------------------------------|
| 主轴 | D 轴分片 + Tc 分组 | D 轴分片 × K 组（`aiCoreIdx = kGroupIdx*dBasicBlockNum + dIdx`）|
| 核数 | shape-adaptive 降至最小覆盖（如 8） | `usedCoreNum = dBasicBlockNum * kBaseNum`（甜点 16）|
| Tc 切分 | 切 Tc | **不切 Tc**（`coreGroupNum=1`，每核跑全部 Tc），改切 K |
| 规约 | 无跨核规约 | 各核 partial-K → GM 槽，K 组 0 AIV 逐组 fp32 求和 |
| 适用 | 权重载入不是瓶颈 / 核已够用 | 核欠订 + 权重载入主导 |

- **CV 形态**：mixed AIC/AIV，`KERNEL_TYPE_MIX_AIC_1_2`（AIV:AIC = 2:1）。
- **甜点核数须实测**：核越多单核 MTE2 越低，但跨核 GM 聚合 + SyncAll 开销随核数上升 → 存在最优点。compressor 实测 kBaseNum=3(24 核) 收益被同步吃光（jitter），kBaseNum=2(16 核) 最优。

## 5. 编译期特化（TilingKey / 模板参数）

kernel 入口按模板参数展开，运行时零分支（`compressor.cpp` / `compressor_template_tiling_key.h`）：

| 模板参数 | 含义 |
|---------|------|
| `XLayout` | 输入布局（BSH 等）|
| `XDType` | 输入 dtype |
| `Coff` | 压缩配置 |
| `RotaryMode` | RoPE 模式（半旋等）|
| `TemplateId` | `EMPTY_X`（空 tensor 直接 return）/ `PERF`（仅 arch32）/ 默认 |

- arch 分派：`__CCE_AICORE__==220` → arch32 计算核（+PERF 变体）；else → arch35 计算核（block_cube/block_vec + `compressor_ksplit_kernel.h` + `vf/` 微指令）。
- K 切分由 **TilingData 驱动**（`kBaseNum/kBaseSize/splitCoreParam`），不新增 TilingKey 模板实例；`kBaseNum_==1` 退回 A/E 路径逐字节不变，保证 fallback（大 token/BSH/arch32/PERF）零回归。

## 6. 典型场景选路

| 场景 | bound / 核占用 | 选手段 | 说明 |
|------|---------------|--------|------|
| decode T=1 / 小 batch，权重载入主导 | Memory-Bound + 核欠订 | **K1 → K2 → K3 → K4** | 本文主线；先消空转核，再合并载入，再 K 轴并行摊薄，最后静态化减同步 |
| 权重碎块但核已订满 | Memory-Bound + 核饱和 | K2（K3 无空闲核可用）| 只做载入合并 |
| 大 batch / 长 seq | 算力/流水主导 | 不适用本文 | 权重可跨 token 复用，回 [`mla_prolog_fusion`](mla_prolog_fusion_design.md) 或 matmul/FA 族的 pingpong/streamk/fullload |
| 需量级提升（放开约束） | — | 见 §7 契约外杠杆 | 量化 > ND→NZ > 合批 > 跨算子融合 |

落地顺序建议：先 **K1**（shape-adaptive 消空转，最低风险）→ **K2**（载入合并，访存受限核心杠杆）→ **K3**（K 轴核并行，需实测甜点核数）→ **K4**（静态 tiling + 减屏障，须 sync-check + 序列压测防 507015）。每步都以 **wall/用户目标**判定收益，不以子指标（MTE2 单降）判定。

## 7. 约束与注意事项

- **跨核 K 聚合的 GM 中转 + 每迭代 SyncAll 是架构物理下界**（CV-UB 直通），删则 507015；这是下界不是可优化项。
- **权重必须留 Cube**：迁到 AIV 做 GEMV 实测回归 6×（31→187µs，AIV 载入 109µs）—— Cube DMA 0.655 TB/s ≫ Vector DMA 0.154 TB/s。
- **只看 MTE2 下降会误判**：MTE2 降但 wall 落进 3% 抖动阈值即无效（如单缓冲 K=512：MTE2 -7.5% 但 wall -0.283µs < 阈值）。
- **契约红线**：标量前移需给 def 加 `.ValueDepend()`、消碎块需改 ND→NZ 权重格式 —— 均动公开契约，算子内不可行；满 burst K=4096 连续载入需 L1 2MB 物理不可达。
- **减屏障须核间一致**：只有全核一致的循环常量条件（`isLastVec2Batch`）可省屏障，且须 sync-check + 序列压测。
- **event 回收**：Process 结束 `FreeEventID`，否则残留挂死后续 kernel。
- **T=1 无跨迭代复用对象**：单核权重恰读一次，不做 weight-residency/深流水，转 K 轴并行摊薄。

### 契约外的量级空间（仅在放开约束时）

算子内挖尽后量级提升只在约束外，按杠杆排序：① **量化**（权重 bf16→int8/fp8，访存受限最大杠杆，roofline 减半）；② **权重 ND→NZ 格式**（直击碎块载入根因）；③ **抬高 T / 跨请求合批**（分摊载入、提核占用，shape/服务层决定）；④ **跨算子融合**（图层隐藏延迟 / 省 GM 往返）。

## 8. 迁移优先级

工程层（**K1 shape-adaptive core §3**，小 batch 第一手）> 搬运层（**K2 载入合并 §3**，访存受限大头）> 并行层（**K3 K 轴核并行 §3-4**，摊薄载入）> 同步层（**K4 静态 tiling + 减屏障 §3**，逼近下界）> 天花板判定（§7 契约外杠杆，需放开约束）。先吃"消空转 + 合并载入"两块大头，再逐层压榨同步；到架构下界即据实收尾，不为凑 roofline 硬做落进抖动的改动。

> 本文属 `flash_attention` 族；新增其它 KV 压缩 / decode 融合变体的扩展规则见 [guide.md](guide.md) 的「扩展新 FA 变体」。
