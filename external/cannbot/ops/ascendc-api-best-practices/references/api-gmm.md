# Ascend C GMM 高阶 API 最佳实践

> **适用路径**：Ascend C **Matmul 高阶 API** 上的 GroupedMatmul（`MatmulImpl` + 分组调度）。
> **适用平台**：Atlas A2 / A3（Ascend910B、Ascend910_93，NpuArch `DAV_2201`）。
> **不适用**：Ascend 950（`DAV_3510`）GMM 实现 → 950 侧有独立实现路线，勿套用本文档。
> **扩展策略**：当新架构出现时，按新架构 NpuArch 新增 reference 文件或更新现有平台适配说明。
>
> 适用算子：GroupedMatmul（分组矩阵乘）。
> MatMul 通用 API（`MatmulImpl`、`MatmulConfig`、`IterateAll`、`SetHF32`、`TilingHeader` 加载等）见 [api-matmul.md](api-matmul.md)。本文档仅覆盖 GMM 差异化 API 和设计要点。

---

## 1. 分组 Shape 管理

GMM 核心特征是**多组矩阵乘的 M/K/N 可能各不相同**，kernel 侧需要动态获取每组的 shape。

### 1.1 Tiling 侧传递

> **字段来源**：`GMMArray[128]` 数组规格来自 CANN 9.0.0 / asc-devkit 9.0.0（GroupedMatmul 算子 Tiling 头文件），具体版本以本地 `$ASCEND_HOME_PATH/include/` 为准；版本差异请用 `ascendc-docs-search` skill 查询。

Host 端在 `GMMArray` 中填好 3 个数组（最多 128 组）：

| 数组 | 字段 | 含义 | 特殊值 |
|------|------|------|--------|
| mList[128] | 每组 M 大小 | 各组 x 的有效行数 | `mList[0] == -1` 表示 M 由 groupList 提供 |
| kList[128] | 每组 K 大小 | 各组的 K 轴大小 | 单 tensor 时仅 `kList[0]` 有效 |
| nList[128] | 每组 N 大小 | 各组的 N 轴大小 | 单 tensor 时仅 `nList[0]` 有效 |

### 1.2 Kernel 侧访问

```cpp
// 通过 tiling 数据获取
GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);
GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);

// 遍历分组时按组索引读取 shape
for (uint32_t groupIdx = 0; groupIdx < gmmBaseParams_.groupNum; ++groupIdx) {
    int32_t m = gmmArrayAddr_.mList[groupIdx];
    int32_t k = gmmArrayAddr_.kList[groupIdx];
    int32_t n = gmmArrayAddr_.nList[groupIdx];
    // 按 (m, k, n) 执行该组的 MatMul ...
}
```

**单 tensor 场景**（s-s-s / s-m-s / m-m-s）：`mList[0] == -1`，M 需从 groupList 动态读取；K/N 只用 `kList[0]` / `nList[0]`。

**多 tensor 场景**（m-m-m）：`mList[i] / kList[i] / nList[i]` 三值均有效。

### 1.3 GM 偏移累加

多 tensor 场景下，各组数据在 GM 中拼接存放，需按组累加偏移：

```cpp
int64_t gmOffsetA = 0;  // x 的每组起始偏移
int64_t gmOffsetB = 0;  // weight 的每组起始偏移
int64_t gmOffsetC = 0;  // y 的每组起始偏移

for (uint32_t groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
    int32_t m = /* ... */, k = /* ... */, n = /* ... */;
    // 当前组使用偏移 gmOffsetA/gmOffsetB/gmOffsetC
    // ...
    // 下一组累加
    gmOffsetA += m * k;
    gmOffsetB += k * n;
    gmOffsetC += m * n;
}
```

---

## 2. 分块索引计算

GMM 以 `baseM × baseN` 为基本块进行核间分配，各组统一按最大 shape 划分：

```
mDim = ceil(maxM / baseM)        // M 方向块数
nDim = ceil(maxN / baseN)        // N 方向块数
totalBlocks = mDim × nDim × groupNum   // 总任务块数
```

Kernel 内部将 blockIdx 映射到具体的 (groupIdx, mIdx, nIdx)：

```cpp
const int32_t blockIdx = GetBlockIdx();
if (blockIdx >= totalBlocks) return;  // 越界守卫

// 块坐标 → 组坐标
const int32_t blockPerGroup = mDim * nDim;
const int32_t groupIdx = blockIdx / blockPerGroup;
const int32_t innerIdx  = blockIdx % blockPerGroup;
const int32_t mIdx = innerIdx / nDim;
const int32_t nIdx = innerIdx % nDim;

// 当前 base 块的实际大小（考虑尾块）
const int32_t curM = (mIdx == mDim - 1) ? (M - mIdx * baseM) : baseM;
const int32_t curN = (nIdx == nDim - 1) ? (N - nIdx * baseN) : baseN;
```

> 对角线分核时，blockIdx 到 (groupIdx, mIdx, nIdx) 的映射需按蛇形/对角线模式重新编排，本质不变。

---

## 3. 量化 Epilogue — Per-Token Dequant

GMM 的量化场景（A8W8O16）在 MatMul 输出 int32 结果后，需要执行反量化流程。与通用 MatMul 直接用 `IterateAll` 写回不同，GMM 的量化 Epilogue 由 AIV 侧完成。

### 3.1 数据流

```
GM_C(int32) → UB_C(int32)    // MatMul Cube 输出，写入 workspace
GM_Scale(fp16/bf16) → UB_Scale(fp16/bf16)   // per-channel scale
GM_PerTokenScale(fp16/bf16) → UB_PerTokenScale(fp16/bf16)  // per-token scale

UB_C(int32) → Cast(fp32)
    → Mul(RowBroadcast: ×UB_Scale_fp32)       // 逐列放缩
    → Mul(ColumnBroadcast: ×UB_PerTokenScale_fp32)  // 逐行放缩
    → Cast(fp16/bf16) → GM_Y
```

### 3.2 AscendC API 映射

```cpp
// === 1. 搬运 MatMul 输出 (int32) 到 UB ===
// UB 由静态偏移分配，不通过 TPipe
LocalTensor<int32_t> ubC = ubBuf.Get<int32_t>(offsetC);
LocalTensor<int32_t> gmC;
gmC.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspacePtr));
DataCopyExtParams copyParam;
copyParam.blockCount = 1;
copyParam.blockLen   = tileRows * tileCols * sizeof(int32_t);  // 32B 对齐
copyParam.rsv = 0;
DataCopyPad(ubC, gmC[gmOffsetC], copyParam);

// === 2. 搬运 Scale 到 UB ===
LocalTensor<half> ubScale = ubBuf.Get<half>(offsetScale);
LocalTensor<half> gmScale;
gmScale.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(scalePtr));
copyParam.blockLen = tileCols * sizeof(half);
DataCopyPad(ubScale, gmScale[scaleOffset], copyParam);

// === 3. 搬运 PerTokenScale 到 UB ===
LocalTensor<half> ubPerTokenScale = ubBuf.Get<half>(offsetPTS);
LocalTensor<half> gmPerTokenScale;
gmPerTokenScale.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(ptsPtr));
copyParam.blockLen = tileRows * sizeof(half);
DataCopyPad(ubPerTokenScale, gmPerTokenScale[ptsOffset], copyParam);

// === 4. 计算：int32 → fp32 ===
LocalTensor<float> ubCfp32 = ubBuf.Get<float>(offsetTmp);
Cast(ubCfp32, ubC, RoundMode::CAST_NONE, tileRows * tileCols);

// === 5. Scale cast fp16 → fp32 ===
LocalTensor<float> ubScaleFp32 = ubBuf.Get<float>(offsetTmp2);
Cast(ubScaleFp32, ubScale, RoundMode::CAST_NONE, tileCols);

// === 6. RowBroadcast Mul: float[tileRows * tileCols] × float[tileCols] ===
Mul(ubMulFp32, ubCfp32, ubScaleFp32, tileRows * tileCols);

// === 7. PerTokenScale cast + ColumnBroadcast ===
LocalTensor<float> ubPtsFp32 = ubBuf.Get<float>(offsetTmp3);
Cast(ubPtsFp32, ubPerTokenScale, RoundMode::CAST_NONE, tileRows);
// 广播到 tileRows×tileCols
// 方式一：逐行 Mul(repeatTimes=1, dstRepStride=tileCols, srcRepStride=0)
Mul(ubResultFp32, ubMulFp32, ubPtsFp32, tileRows * tileCols);

// === 8. Cast fp32 → fp16/bf16 写回 ===
LocalTensor<half> ubResult = ubBuf.Get<half>(offsetOut);
Cast(ubResult, ubResultFp32, RoundMode::CAST_RINT, tileRows * tileCols);
DataCopyPad(gmY[gmOffsetY], ubResult, copyParam);
```

### 3.3 UB Buffer 分配

量化场景 UB 需分块复用，以 per-token A8W8O16 为例：

```
ubCalSize = ubSize / ubDivideBlkNum              // 每块元素数
ubCalSize = AlignDown(ubCalSize, ubBlockAlign)    // 对齐

ubBaseK = perTokenOrPerGroupSize                  // K 方向分块粒度
ubBaseN = min(BEST_UB_BASEN, AlignUp(ubCalSize / ubBaseK, MIN_UB_BASEN))
```

各 buffer 以 `ubCalcSize = ubBaseM × ubBaseN` 为基本单元份数分配：

| Buffer | 份数 | 类型 | 用途 |
|--------|------|------|------|
| MatMul 输出 | 8 (DB×2) | int32 | Cube 结果暂存 |
| 最终输出 | 4 | fp16/bf16 | dequant 后结果 |
| 中间临时 | 16 | fp32 | Cast + broadcast + Mul 复用 |
| **合计** | 28 份 | — | `28 × ubCalcSize × sizeof(type)` 需 ≤ ubSize |

---

## 4. Cube MatMul 接口（与通用 MatMul 共享）

GMM 的 Cube 计算部分复用标准 `MatmulImpl`，差异在于：

| 项目 | 通用 MatMul | GMM |
|------|-----------|-----|
| `MatmulImpl` 实例数 | 1 个 | 1 个（各组共用 Tiling 参数） |
| `SetSingleShape` | 单组 (M,N,K) | 各组共用 (baseM, baseN, K) |
| 分组管理 | 无 | 通过 blockIdx 映射到 groupIdx |
| 多组间 offset | 无 | 需累加各组 A/B/C 的 GM offset |
| Epilogue | `IterateAll` 直接写回 | AIV 侧 dequant 后写回 |

### 4.1 纯 Cube 场景（非量化）

```cpp
// 纯 Cube 模式（AIC only），与通用 MatMul 几乎一致
// 仅需 ADD 分组循环和 GM offset 计算
if ASCEND_IS_AIV { return; }                              // 只有 Cube
MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG> mm;
mm.Init(&mmTilingData_, &tPipe);

for (auto blockIdx = GetBlockIdx(); blockIdx < totalBlock; blockIdx += blockDim) {
    // 块坐标映射到 (groupIdx, mIdx, nIdx)
    // 计算 curM/curN, offsetA/offsetB/offsetC
    mm.SetSingleShape(curM, curN, K);
    mm.SetTensorA(gmA[offsetA]);
    mm.SetTensorB(gmB[offsetB]);
    mm.IterateAll(gmC[offsetC], /*enAtomic=*/false);
}
mm.End();
```

### 4.2 量化 C+V 场景

AIC 侧仅完成 MatMul 计算，不写回 GM；AIV 侧完成 dequant 后写回。两者通过 **workspace + cross-core sync flag** 通信：

```
AIC:  MatMul → workspace[coreIdx]
      SetFlag → AIV
AIV:  WaitFlag ← AIC
      Dequant(workspace[coreIdx]) → GM_Y
      SetFlag → AIC (释放 workspace)
```

---

## 5. AIC/AIV 分职模式

| 模式 | AIC 任务 | AIV 任务 | 触发条件 |
|------|---------|---------|---------|
| **Cube Only** | MatMul + 写回 GM | 无（直接返回） | 非量化 fp16/bf16/fp32 |
| **C:V = 1:1** | MatMul → workspace | Dequant → GM | A8W8 perTensor / perToken 小 token |
| **C:V = 2:1** | MatMul → workspace | Dequant → GM（双 Vector 核） | perToken + (K≤1024 或 K≥2048) + token≥128 |

---

## 6. 检查清单

- [ ] GMMArray 的 mList/kList/nList 与 tensor 组合匹配（单/多 tensor 语义）
- [ ] blockIdx 到 (groupIdx, mIdx, nIdx) 映射正确
- [ ] 各组 GM offset 累加正确（注意数据类型和 NZ 格式 stride 差异）
- [ ] 量化场景 UB 分块参数非零（ubDivideBlkNum / ubBlockAlign）
- [ ] workspace 大小为 `usedCoreNum × baseM × baseN × sizeof(int32_t)` 的倍数
- [ ] AIC/AIV cross-core sync flag 正确配对（SetFlag / WaitFlag）
- [ ] `PipeBarrier<PIPE_ALL>()` 在 Cube 完成后、AIV 开始时
- [ ] A8W8 perToken 场景的 Dequant API 顺序：Cast→Mul(RowBroadcast)→Mul(ColBroadcast)→Cast
