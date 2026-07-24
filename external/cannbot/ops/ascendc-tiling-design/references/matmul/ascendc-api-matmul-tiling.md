# Ascend C Matmul 高阶 API Tiling 策略

> **适用路径**：Ascend C **Matmul 高阶 API**（`MatmulImpl` + Host 侧 `MatmulApiTiling::GetTiling`）。
> **适用平台**：Atlas A2 / A3（Ascend910B、Ascend910_93，NpuArch `DAV_2201`）。
> **不适用**：Ascend 950（`DAV_3510`）→ 使用 `ascendc-blaze-best-practice` skill（Blaze / tensor_api 路径）。
> **扩展策略**：当新架构出现时，按新架构 NpuArch 新增 reference 文件或更新现有平台适配说明。
>
> 适用算子：MatMul、BatchMatMul、MatMulBias 等基于 `MatmulImpl` 的单组矩阵乘。
> 本文档聚焦 **Tiling 设计**；Kernel Ascend C Matmul 高阶 API 用法见 `ascendc-api-best-practices` skill → `references/api-matmul.md`。

## 三级 Tiling 架构

```
GM (Global Memory)
  │  A[M,K] × B[K,N] → C[M,N]
  ▼
┌─────────────────────────────────────┐
│         L2 级 Tiling                 │  M/N 方向切分，serpentine 遍历
│   mTileBlock × nTileBlock 个 base 块 │
└─────────────────────────────────────┘
  │  ┌──────────┐  ┌──────────┐
  ▼  ▼          ▼  ▼          ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ Core 0  │ │ Core 1  │ │ Core N-1│   Block 级（核间切分 + 错位）
│singleM×N│ │singleM×N│ │singleM×N│
└─────────┘ └─────────┘ └─────────┘
  │            │            │
  ▼ MATMUL    ▼ MATMUL    ▼ MATMUL
┌─────────┐ ┌─────────┐ ┌─────────┐
│ BaseBlk │ │ BaseBlk │ │ BaseBlk │   BaseBlock 级（cube 一次发射）
│M0N0K    │ │M1N1K    │ │MnNnK    │
└─────────┘ └─────────┘ └─────────┘
```

> 与 Vector 类算子的关键差异：计算单元从 AIV（Vector）变为 **AIC（Cube）**，片上存储从 UB 变为 **L1(A1/B1) + L0A/L0B/L0C**。

---

## Tiling 数据结构

> **字段来源**：`TCubeTiling` 字段定义来自 CANN 9.0.0 / asc-devkit 9.0.0（MatmulApiTiling 头文件），具体版本以本地 `$ASCEND_HOME_PATH/include/` 为准；版本差异请用 `ascendc-docs-search` skill 查询。

```cpp
#pragma pack(push, 8)

// CANN 标准 cube 参数（~196 字节，由 MatmulApiTiling 自动填充）
struct alignas(8) TCubeTiling {
    uint32_t M, N, Ka, Kb;
    uint32_t singleCoreM, singleCoreN, singleCoreK;
    uint32_t baseM, baseN, baseK;
    uint32_t depthA1, depthB1;           // L1 双缓冲深度
    uint32_t stepM, stepN, stepKa, stepKb;
    uint32_t isBias;
    uint32_t usedCoreNum;
    uint32_t iterateOrder;
    uint32_t dbL0A, dbL0B, dbL0C;        // L0 双缓冲
    uint32_t shareMode, shareL1Size, shareL0CSize, shareUbSize;
    // ...
};

// 自定义算子的轻量 TilingHeader
struct TilingHeader {
    TCubeTiling cubeTiling;    // CANN 标准 cube tiling（POD，~196B）
    int32_t mTotalCnt;         // ceil(M / singleCoreM)
    int32_t nTotalCnt;         // ceil(N / singleCoreN)
    int32_t totalBlock;        // mTotalCnt * nTotalCnt
    int32_t reserved;          // 8 字节对齐
};

// L2 Cache 切分参数（大算子用，小算子置 1）
struct L2cacheTilePara {
    uint32_t mTileCntL2, nTileCntL2;  // L2 Tile 计数
    uint32_t mTileBlock, nTileBlock;  // 每个 L2 Tile 的 base 块数
    uint32_t calOrder;                // ROW_FIRST=1 / COL_FIRST=2
};

// BatchMatMul 叠加 MultiBatchInfo
struct MultiBatchInfo {
    uint32_t batchUsedCoreNum;
    uint32_t iterBatch;        // 单次 IterateBatch 处理 batch 数
    uint32_t biasWithBatch;
    uint32_t mOri, batchTileBlock, aBatch, bBatch;
    // ...
};

#pragma pack(pop)
```

---

## L2 级 Tiling（M/N 方向切分）

### 目的

| 目的 | 收益 |
|------|------|
| 减少 L2 Cache 抖动 | 每个 L2 Tile 内的 A/B 子块尽量驻留 L2 |
| 提高数据局部性 | 同一行/列 base 块共享 L2 中的 A 行或 B 列 |
| 配合 serpentine 遍历 | 反向 N 遍历时下一个 L2 Tile 直接复用本 tile 末端的 B 行 |

### 切分公式

```cpp
mTileCntL2 = ceil(mTotalCnt, mTileBlock);
nTileCntL2 = ceil(nTotalCnt, nTileBlock);
mCnt = ceil(mTotalCnt, mTileCntL2);   // 每个 L2 Tile 的 M 方向 base 块数
nCnt = ceil(nTotalCnt, nTileCntL2);   // 每个 L2 Tile 的 N 方向 base 块数
totalTileCnt = mCnt * nCnt;           // 单 L2 Tile 内的 base 块总数
```

> **小算子退化**：若 `M * N * dtypeSize ≤ L2 / 2`，置 `mTileCntL2 = nTileCntL2 = 1`，跳过 L2 切分。

### Serpentine 反向遍历

```cpp
bool reverse = true;
for (uint64_t mTile = 0; mTile < mTileCntL2; ++mTile) {
    reverse = !reverse;
    for (uint64_t t = 0; t < nTileCntL2; ++t) {
        uint64_t nTile = reverse ? (nTileCntL2 - t - 1) : t;
        // ProcessL2Tile(mTile, nTile);
    }
}
```

### 行优先 vs 列优先

| calOrder | 适用条件 | 收益 |
|----------|---------|------|
| ROW_FIRST (1) | A 矩阵较大 / M ≫ N | 优先复用 A 行 |
| COL_FIRST (2) | B 矩阵较大 / N ≫ M | 优先复用 B 列 |

---

## Block 级 Tiling（核间切分）

### 整核/尾核策略

```cpp
round       = ceil(totalTileCnt, usedCoreNum);
preCoreNum  = totalTileCnt % usedCoreNum;
if (preCoreNum == 0) preCoreNum = usedCoreNum;
// 其余 (usedCoreNum - preCoreNum) 个核处理 round-1 次
```

负载均衡：所有核之间最多差 1 个 base 块。

### 错位分核（avoid bank conflict）

直接用 `blockIdx % nCntUse` 映射会让相邻核同时访问同一行 A 或同一列 B，造成 L2/L1 bank 冲突。CANN 用 LCM 错位：

```cpp
newBlockIdx = (GetCurrentBlockIdx() + usedCoreNum - blockIdxStart) % usedCoreNum
            + roundIdx * usedCoreNum;
mIdx = newBlockIdx % mCntUse;
nIdx = (newBlockIdx + newBlockIdx / Lcm(mCntUse, nCntUse)) % nCntUse;
```

### 单核处理大小与尾块

```cpp
singleCoreM / singleCoreN / singleCoreK 由 MatmulApiTiling 自动给出
mTotalCnt   = ceil(M, singleCoreM);
nTotalCnt   = ceil(N, singleCoreN);
mBaseTail   = M - (mTotalCnt - 1) * singleCoreM;   // M 方向尾行实际大小
nBaseTail   = N - (nTotalCnt - 1) * singleCoreN;   // N 方向尾列实际大小
```

Kernel 内分发：

```cpp
const int32_t mIdx = blockIdx / nTotalCnt;
const int32_t nIdx = blockIdx % nTotalCnt;
const int32_t curSingleM = (mIdx == mTotalCnt - 1) ? mBaseTail : singleCoreM;
const int32_t curSingleN = (nIdx == nTotalCnt - 1) ? nBaseTail : singleCoreN;
```

---

## BaseBlock 级（cube 一次发射）

### 经验值

| 数据类型 | 推荐 baseM | 推荐 baseN | 推荐 baseK |
|---------|-----------|-----------|-----------|
| float16 | 128 / 256 | 128 / 256 | 32 / 64 |
| bfloat16 | 128 / 256 | 128 / 256 | 32 / 64 |
| float32 | 64 / 128 | 64 / 128 | 16 / 32 |

> **优先使用 `MatmulApiTiling::GetTiling(TCubeTiling&)`** 自动给出最优组合。手算只在该 API 不可用时使用。

### 对齐约束

```cpp
ALIGNED_H = 16;       // M、K 维度对齐
c0Size    = 16;       // fp16/bf16 NZ 的 N 维（fp32 为 8）

baseM, baseN, singleCoreM, singleCoreN 必须为 ALIGNED_H 的整数倍
baseK 通常为 ALIGNED_H 的整数倍
```

### 多核切分扩展

`MatmulApiTiling::GetTiling(TCubeTiling&)` 可能返回 `singleCoreM = M`（将整个 M 分配给一个核），导致 `totalBlock = 1`，Block Dim = 1，多核完全未生效。**当 `totalBlock < aicCoreNum` 时，必须手动缩小 `singleCoreM / singleCoreN` 以充分利用多核并行。**

> ⚠️ **对齐约束**：缩小后的 `singleCoreM / singleCoreN` 仍必须为 `ALIGNED_H = 16` 的整数倍，且不得小于 `baseM / baseN`。

**算法**：交替对半缩小 M/N 方向的 singleCore 尺寸，直到 `totalBlock ≥ aicCoreNum` 或已达到下限。

```cpp
if (totalBlock < aicCoreNum) {
    // 缩小 M 方向：singleCoreM 减半，但不低于 baseM，且保持 ALIGNED_H 对齐
    while (totalBlock < aicCoreNum && singleCoreM > baseM) {
        singleCoreM = std::max(baseM, (singleCoreM / 2) / ALIGNED_H * ALIGNED_H);
        mTotalCnt = CeilDiv(M, static_cast<int64_t>(singleCoreM));
        totalBlock = mTotalCnt * nTotalCnt;
    }
    // 缩小 N 方向：singleCoreN 减半，但不低于 baseN，且保持 ALIGNED_H 对齐
    while (totalBlock < aicCoreNum && singleCoreN > baseN) {
        singleCoreN = std::max(baseN, (singleCoreN / 2) / ALIGNED_H * ALIGNED_H);
        nTotalCnt = CeilDiv(N, static_cast<int64_t>(singleCoreN));
        totalBlock = mTotalCnt * nTotalCnt;
    }
}
```

缩小完成后，必须同步更新 `cubeTiling` 和派生字段：

```cpp
// 覆写 cubeTiling
cubeTiling.singleCoreM = singleCoreM;
cubeTiling.singleCoreN = singleCoreN;
cubeTiling.usedCoreNum = std::min(totalBlock, aicCoreNum);

// 计算尾块（供 Kernel 端 SetSingleShape 使用）
mBaseTail = M - (mTotalCnt - 1) * singleCoreM;
nBaseTail = N - (nTotalCnt - 1) * singleCoreN;
```

---

## Batch 维度处理

### 通用规则

| A Batch | B Batch | C Batch | 模式 |
|---------|---------|---------|------|
| N | 1 | N | B 广播 |
| 1 | N | N | A 广播 |
| N | N | N | 一对一 |
| N | M | LCM(N,M) | 多维广播 |

### Batch 合并到 M

当 `batchB == 1 && !transA` 时，把 A 的 batch 维度合并到 M：

```cpp
if (batchB == 1 && !transA) {
    M = batchA * M;
    batchA = 1;
}
```

### Batch 负载均衡

```cpp
iterBatch = min(batchC, coreNum);
loopTimes = ceil(batchC, iterBatch * coreNum);
batchUsedCoreNum = min(batchC, coreNum);
```

---

## Host 端 Tiling 计算流程

```
1. 输入校验（dtype、shape、K 维度匹配）
        ↓
2. 通过 PlatformAscendC 获取硬件参数（coreNumAic / l1Size / l0A / l0B / l0C / ubSize）
        ↓
3. （可选）batch 合并、shape 规范化
        ↓
4. 调用 MatmulApiTiling::GetTiling(TCubeTiling&)
   ├─ 自动决定 baseM/baseN/baseK、singleCoreM/N、L1 深度
   ├─ 自动给出 usedCoreNum
   └─ 校验 K = Ka = Kb 一致
        ↓
5. 强制覆写 M/N/Ka/Kb（不同 CANN 版本策略有差异，确保一致）
   header.cubeTiling.M  = M;
   header.cubeTiling.N  = N;
   header.cubeTiling.Ka = K;
   header.cubeTiling.Kb = K;
        ↓
6. 计算派生计数 mTotalCnt / nTotalCnt / totalBlock
   mTotalCnt  = (M + singleCoreM - 1) / singleCoreM;
   nTotalCnt  = (N + singleCoreN - 1) / singleCoreN;
   totalBlock = mTotalCnt * nTotalCnt;
        ↓
7. 多核切分扩展（当 totalBlock < aicCoreNum 时可选择执行）
        ↓
7. （大算子）计算 L2 级 mTileCntL2 / nTileCntL2 / mTileBlock / nTileBlock
        ↓
8. （BatchMatMul）填充 MultiBatchInfo
        ↓
9. 把 TilingHeader 序列化到 device 端 uint8 tensor（blocking copy）
        ↓
10. EXEC_KERNEL_CMD(<op_name>, blockDim, A, B, C, [Bias], tilingTensor)
    blockDim = min(usedCoreNum, totalBlock)
```

---

## Tiling 关键参数总结

| 参数类别 | 字段 | 来源 |
|---------|------|------|
| 矩阵维度 | M, N, Ka, Kb | host 输入 |
| 单核切分 | singleCoreM, singleCoreN | `MatmulApiTiling` |
| 基础块 | baseM, baseN, baseK | `MatmulApiTiling` |
| L1 深度 | depthA1, depthB1, stepM/N/Ka/Kb | `MatmulApiTiling` |
| L0 双缓冲 | dbL0A, dbL0B, dbL0C | `MatmulApiTiling` |
| L2 切分 | mTileCntL2, nTileCntL2, mTileBlock, nTileBlock, calOrder | host 自定义计算 |
| 核数 | usedCoreNum | `≤ GetCoreNumAic()` |
| 转置 | transA, transB | host 输入 |
| 精度 | isHf32 | host 选项 |
| Bias | isBias | host 输入 |
| Batch | MultiBatchInfo | BatchMatMul host |

---

## 设计检查清单

- [ ] 维度匹配：`Ka == Kb`；BatchMatMul 满足广播规则
- [ ] 对齐：`baseM, baseN, singleCoreM, singleCoreN` 是 `ALIGNED_H=16` 的整数倍
- [ ] 核数：`usedCoreNum ≤ GetCoreNumAic()`
- [ ] blockDim 取 `min(usedCoreNum, totalBlock)` 防止富余核越界
- [ ] 多核切分扩展：当 `totalBlock < aicCoreNum` 时，可缩小 `singleCoreM/N`
- [ ] kernel 入口有 `if (GetBlockIdx() >= totalBlock) return;` 越界守卫
- [ ] kernel 端按 `blockIdx → (mIdx, nIdx)` 计算偏移和尾块尺寸，调用 `SetSingleShape`
- [ ] TilingHeader → device tensor 必须 blocking copy（non_blocking=false）
- [ ] GetTiling 后强制覆写 M/N/Ka/Kb
- [ ] CubeMathType host/kernel 配对：NORMAL ↔ SetHF32(false, 0)；HF32 ↔ SetHF32(true, 1)
- [ ] Batch 维度 ≤ 4
- [ ] 小算子（M*N*dtype ≤ L2/2）跳过 L2 切分

## 算子差异化

| 算子类型 | 特殊特性 | Tiling 差异 |
|---------|---------|------------|
| MatMul | 基础 | 无 batch；可选 L2/Split-K |
| BatchMatMul | 多 batch | 叠加 `MultiBatchInfo`；`IterateBatch`；跨调用 `indexInit_` 避免尾核空转 |
| MatMulBias | 带 bias | `isBias=1` |
| MatMulAdd | 与额外加法融合 | 在 Iterate 后/前插入 vector 加法（规划中） |
| MatMulQuant | INT8 量化 | A/B INT8、C FP16/INT32；deqScale fixpipe（规划中） |

## 性能优化要点

| 优化点 | 说明 | 详见 |
|--------|------|------|
| 多核切分扩展 | MatmulApiTiling 可能返回 singleCoreM=M 导致 Block Dim=1，手动缩小 singleCoreM/N | [多核切分扩展](#多核切分扩展) |
| L2 Cache | 合理设置 mTileBlock×nTileBlock ≤ L2/2 | [L2 级 Tiling](#l2-级-tilingmn-方向切分) |
| 负载均衡 | 整核/尾核误差 ≤ 1 个 base 块 | [Block 级 Tiling](#block-级-tiling核间切分) |
| 错位分核 | LCM 错位避免 bank 冲突 | [错位分核](#错位分核avoid-bank-conflict) |
| Serpentine | N 方向偶/奇行交替反向遍历 | [Serpentine](#serpentine-反向遍历) |
| enUnitFlag | 99% 自定义算子需要 enUnitFlag=true | `ascendc-api-best-practices` skill → `api-matmul.md` |
| L1 Full Load | M*Ka 或 Kb*N 能塞进 L1 时启用 MM_CFG_L1_FULL_LOAD | `ascendc-api-best-practices` skill → `api-matmul.md` |
| Split-K | K ≫ M*N 时优先 Multi Core Split-K + ATOMIC_ADD | [Split-K 策略](#split-k-策略补充) |
| Batch 合并 | batchB==1 && !transA 时合并到 M | [Batch 合并到 M](#batch-合并到-m) |
| DB 与流水 | 默认 dbL0A=dbL0B=dbL0C=2，由 GetTiling 给出 | [TCubeTiling](#tiling-数据结构) |
| ND→NZ | 维度对齐到 16 时不要转换；非对齐时考虑 MM_CFG_VEC_ND2NZ | [ND2NZ 格式对齐](#nd2nz-格式对齐) |

## Split-K 策略（补充）

当 K ≫ M*N 时，可将 K 维度切分到多个核上并行处理：

| 类型 | 触发条件 | 实现 |
|------|---------|------|
| Single Core Split-K | K 中等 | 单核 K 切分，串行累加 |
| Multi Core Split-K | K ≫ M*N | 多核并行计算 K 片，最后 Atomic Add 到同一 C 块 |
| Deterministic Split-K | 需要可复现 | 多核计算 + workspace 累加 + 最终归约 |

```cpp
// Multi Core Split-K 模式：多核并行不同 K 片，Atomic Add 写回
splitK = ceil(K, splitKSize);
blockDim = splitK * mTotalCnt * nTotalCnt;
for (int splitKIdx = 0; splitKIdx < splitK; ++splitKIdx) {
    kStart = splitKIdx * splitKSize;
    kEnd   = min((splitKIdx + 1) * splitKSize, K);
    mm_.SetSingleShape(curSingleM, curSingleN, kEnd - kStart);
    mm_.SetTensorA(aGm[offsetA + kStart * (transA ? 1 : aStride)], transA);
    mm_.SetTensorB(bGm[offsetB + kStart * (transB ? bStride : 1)], transB);
    mm_.Iterate();
    mm_.GetTensorC(cGm[offsetC], 2);  // ATOMIC_ADD
}
```

## ND2NZ 格式对齐

当输入为 ND 格式但 cube 内部需要 NZ 时，M/K 维度需对齐到 c0Size：

```cpp
constexpr uint32_t c0Size = (sizeof(T) == 4) ? 8 : 16;
alignedM = ceil(M, c0Size) * c0Size;
alignedK = ceil(K, ALIGNED_H) * ALIGNED_H;
// NZ 格式必须 SetOrgShape，使用对齐后的维度
mm_.SetOrgShape(alignedM, N, alignedK, K, N);
```
> 格式转换方式：`MM_CFG_VEC_ND2NZ` 让 Vector 在 kernel 内自动转换（详见 `api-matmul.md`）。
