# 场景：put-all-gather-quant-matmul（AllGather + QuantMatmul 变体扩展）

> **状态**：已实现（官方参考算子 `apace/kernel/all_gather_quant_matmul/`）。
> **定位**：官方 AllGather 范式的**变体扩展**指导——标准 AllGather+QuantMatmul 需求走 `apace_native`（直接调用/参考复用官方 kernel），**不读本表**；本场景服务于官方 kernel 未覆盖的变体（新 dtype、bias 融合、rankSize>8、编排改造）与 AG 专属事实汇总。

## 1. 准入条件（语义判据）

通信原语为 AllGather；localMatmul 为 QuantMatmul（含 MX scale）；PUT 方向；**核间数据沿 M 轴聚合**（每卡持本卡 A 分片 `[m, K]`，聚合后逻辑形状 `[R×m, K]`）；UDMA；**且存在官方 kernel 未覆盖的 native gap**。

与 `put-all-to-all-quant-matmul` 的语义区分：A2A 是输入 K 轴切分（每卡持全部 M 行 × 本卡 K 段，输出 M 轴分布）；AG 是输入 M 轴切分（每卡持本卡 M 行 × 全部 K 列，每卡输出**全量 C** `[R×m, N]`，各卡 B 独立故 C 不同）。

## 2. 数据分布与 golden 语义（区别于 A2A 的关键）

| 项目 | 合同 |
|:---|:---|
| 输入分布 | 每 rank 独立 seed（`BASE_SEED + rankId`）生成本卡 A `[m, K]`（MX 量化：data fp8 + scale e8m0 per-64-group）；B 为本卡独立的 `[K, N]`（DN 列主：gen_data 写 `[N,K]` C-order 不转置，scaleB `[N, K/32]` C-order） |
| 通信语义 | AllGather = `np.concatenate` 各 rank 反量化 A（M 轴聚合，`[R×m, K]`） |
| 输出分布 | **每卡全量 C `[R×m, N]`**：`C_r = matmul(A_all_dequant, B_r_dequant) → bf16`——各卡 B 不同故输出不同（区别于 A2A 的 `[m, N]` 分片输出） |
| 精度标准 | bit-exact 或 raw uint16 差 ≤ 1 ULP（`verify_result.py`：bf16 raw uint16 比较，`raw_max ≤ 1` 即 PASS；大 K 下 fp32 累加非结合性产生百个量级 1-ULP 元素属正常）——**严于 A2A 的 1e-2 allclose**，因无跨卡部分和累加序问题 |

## 3. tiling 与切分（AG 专属公式）

| 项目 | 公式/结论 | 锚点 |
|:---|:---|:---|
| 切分轴 | M（`tileM = min(m, 512)`，`TILE_M=512` 常量，m < tileM 时收缩）；K 不切 | `tests/st/all_gather_quant_matmul/src/main.cpp:47,111-114` |
| tiling 入参形状 | **聚合后总 M**：`totalLogicalM = rankNum × (tileCnt × tileM + tailCnt × paddedTailM)` 后调 `GetTilingData(totalLogicalM, n, k)` | 同上 `main.cpp:119-127` |
| 尾块对齐 | `paddedTailM = (tailM + 15) / 16 × 16`（kPaddingLength=16）；FragmentTensor 以 `paddedTailM` 为 fragmentSize、`tailM` 为 realFragmentSize 表达 padding | `all_gather_mx_matmul_udma_impl.h:78,189`；`qmm_mx_kernel_ag_udma.h:523-528` |
| SWAT 开关 | AG 用 4 参重载 + **`SetOptimizeEnable(false)` + `SetMTailAlignEnable(true)`**（关闭边缘块合并、开启 M 尾 16 对齐——与 A2A 默认相反，移植时勿照抄 A2A） | `main.cpp:124-125` |
| host 校验 | `K % 32 == 0`、`CeilDiv(K, 64) % 2 == 0`（gen_data 与 main.cpp 双侧） | `main.cpp:81-86` |
| commTurn | `tileCnt + tailCnt`；AG 的 flagId 整体 **+1**（见 §4） | — |

## 4. Flag 编排（AG 专属：预触发模式）

与 A2A 的差异：AG 的 HEAD 区（本卡自身数据）**恒就绪**，AIV 在循环前**预触发 `CrossCoreSetFlag<0x2, PIPE_MTE3>(0)`**；循环内第 round 轮完成后 `SetFlag(round + 1)`。AIC 侧三区依赖映射：

| 区域 | M 范围 | dependTileIdx | 数据源 |
|:---|:---|:---|:---|
| HEAD | `[0, headRows)` | 0（预触发，立即返回） | 本地 aGM |
| MAIN | head + `[round × tileM × (R-1), ...)` | round + 1 | 各 rank 窗口（FragmentTensor，本 rank 除外） |
| TAIL | head + mainSection 之后 | commTurn（最后一轮） | 全部 rank（含本地，paddedTailM 对齐） |

AIC 统一经位掩码去重路径 wait（含 id 0，因已预触发不阻塞），收尾 drain 循环到 `commTurn` **含端点**。锚点：`qmm_mx_kernel_ag_udma.h:44（RegionTag）,557-592（ResolveTileCtx）,370-373（去重）,410-414（收尾）`；`all_gather_mx_matmul_udma_impl.h:246-263（AIV 循环）`。

## 5. FragmentTensor 三区调度（AG 核心机制）

- HEAD/MAIN/TAIL 三区各维护 FragmentTensor（`ResolveTileCtx` 按调度 tile 的 mPos 分区）
- MAIN 区**延迟构建**：首次命中 `BuildMainFragment`，换轮 `UpdateMainRoundAddrs`（只换地址表不重建对象），构建开销被通信 wait 掩盖
- matmul 走 blaze_ext `QmmMxBlockMmadFragment`（A/scaleA/C 三侧 FragmentTensor，B/scaleB/bias 普通 GM tensor；调用契约见 [`compute.md`](../../fundamentals/compute.md) §7）
- AG 的 `SetL2Cache` 与 A2A 差异：无 isAtomicAdd 分支；scaleB 受 `fullMBlock && align` **双 gate**（`qmm_mx_kernel_ag_udma.h:246-273`）

## 6. 硬约束（AG 专属）

| 约束 | 说明 | 锚点 |
|:---|:---|:---|
| rankSize ≤ 8 | `cFragAddrs_[8]` / `winDataRankBase_[8]` / `winScaleRankBase_[8]` 为**定长 8 数组、循环无界检查**——R>8 静默越界；变体扩展到 R>8 必须先扩数组（建议按 `MAX_FRAGMENT_COUNT=32` 扩） | `qmm_mx_kernel_ag_udma.h:203-205,437-439` |
| barrierBuf/scale 区布局 | 窗口 rank-major：`[rank r][m 行 data][m 行 scale]`（data 段在前，scale 段 offset = dataRegionBytes_）；UB 静态 512×2+32 | `all_gather_mx_matmul_udma_impl.h:135-142,166-171` |
| BarrierMode | data `Init<BARRIER_NONE>` + scale `Init<BARRIER_DEVICE>`（与 A2A 的 data NONE + scale 默认不同——AG 的 scale Wait 承担跨卡 fence） | 同上 `:166-171` |
| 单入口 | 官方仅 1 个 `__global__`（E4M3×E4M3→bf16，impl 头文件末尾）；变体加 dtype 按 A2A 的 4 变体 launcher 模式补 | `all_gather_mx_matmul_udma_impl.h:267-278` |
| `localMatmul` 字段 | AG tiling **无** localMatmul（单 REMOTE 编排）；加该能力须重设计 | `all_gather_mx_matmul_udma_tiling_data.h:36-39` |

## 7. ST 工程事实（AG 与 A2A 差异）

| 项目 | AG | A2A |
|:---|:---|:---|
| perf 计时 | `aclrtEvent` × 20 次（`BENCHMARK_ITERATIONS=20`）取平均 | `steady_clock` × 10 次去首帧 |
| `ASC_DEVKIT_MAJOR=9` | **无**（CMakeLists 无该宏） | 有 |
| cases.csv 列 | `m,k,n,rank_num`（无 head_m_size） | `m,k,n,rank_num,head_m_size` |
| devContext 处置 | `aclrtFree(devContext)`（与 A2A 不 free 的处置不同；新算子统一按 host-and-testing 口径处理） | 不 free |
| run.sh | 同构四阶段；`--check` 静默模式（供 perf 分支捕获 PASS/FAIL） | 无 `--check` |
| 用例特征 | 17 行含非对齐 M（2061/3085/4109/8205 触发尾块路径） | 8 行 |

## 8. 变体扩展改造点速查

| 变体 | 改动落点 |
|:---|:---|
| 新 dtype（E5M2/混合） | impl 模板参数 + 按 A2A launcher 模式加入口变体 + host dispatch（R3） |
| 加 bias | `QBMMTiling.isBias` + bias GM 传参（mmad 通道现成，参照 hcomm impl 已设 biasGmAddr 的写法） |
| rankSize > 8 | §6 硬约束：扩三数组 + 核数守卫重验（须 `usedCoreNum ≥ rankSize`） |
| M 轴聚合 + 输出分片混合语义 | 越界本场景——重新走场景注册表判据 |

## 9. 验证合同摘要

golden = 每 rank 独立 seed 反量化 A/B → `np.concatenate` A → matmul → bf16（§2）。精度 bit-exact/1-ULP（§2）；先 tileCnt=1 串行基线（T=1 全 PASS 不能外推 T>1——三区布局与 flag 配对问题只在多 tile 暴露）。用例矩阵：rank 端点 × 对齐/非对齐 M（尾块路径）× 边界 shape（m<512 收缩、K=32 最小对齐）。
