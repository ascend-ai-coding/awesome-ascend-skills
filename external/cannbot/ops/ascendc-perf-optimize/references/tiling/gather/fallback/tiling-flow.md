# Gather 族 — 通用 Tiling 流程

> 代际限定：本文 RegBase / Reg / MicroAPI / VF 相关实现仅适用于 arch35（A5 / DAV_3510）；通用 Tiling 建模不受此限制。

> Gather 类算子的模板决策树与 UB 预算推导。所有 Gather 算子共享同一套模板语言；
> 不同算子不仅 UB 预算不同，索引语义和路径路由也不同。
>
> **重要**：本决策树区分 `gatherMode`（PER_ELEMENT vs PER_SLICE）。PER_SLICE 可用 SIMD 行拷贝/两维；PER_ELEMENT 的非末轴路径必须在物理 Transpose、kernel 内 Transpose、Scalar/SIMT 之间做能力与成本门控。`dim != last` 和 UB 足够都不是进入内部 Transpose 的充分条件。

---

## 1. 核心概念

| 术语 | 含义 |
|------|------|
| **合轴** | 将原始 shape + axis 合并为 `(outerSize, gatherDimSize, gatherSize, innerSize)` 四元组 |
| **outerSize** | gather 轴 **之前** 所有维度乘积（batch/outer 维） |
| **gatherDimSize** | x 沿 gather 轴的长度（被索引维度大小） |
| **gatherSize** | index 沿 gather 轴的长度（索引数量 per outer×inner） |
| **innerSize** | gather 轴 **之后** 所有维度乘积（inner 维，一个索引取的 slice 大小） |
| **Dtype Improve** | 小 dtype 的 innerSize×dtypeSize 若 32B 对齐，提升为更大 dtype 减少 MTE2 次数 |
| **gatherMode** | 算子的索引语义：`PER_ELEMENT`（torch.gather/GatherElementsV2，index 与 output 同形，每位置独立索引取 1 元素）或 `PER_SLICE`（GatherV2/Embedding，每索引取 innerSize 个连续元素）。**决定哪些模板分支可用** |
| **模板** | 确定的 UB 切分与多核策略 |

### 合轴公式（Step 0）

```
给定: x.shape, index.shape, axis(dim), batch_dims=0

batchSize  = prod(x.shape[0:batch_dims])     // 默认 1
outerSize  = prod(x.shape[batch_dims:axis])  // gather 轴之前
gatherDimSize = x.shape[axis]                // 被索引维度
innerSize  = prod(x.shape[axis+1:])          // gather 轴之后
若 gatherMode == PER_SLICE:
  gatherSize = index.numel() / batchSize
  ySize      = batchSize * outerSize * gatherSize * innerSize

若 gatherMode == PER_ELEMENT:
  idxPreDim     = prod(index.shape[0:axis])
  idxGatherDim  = index.shape[axis]
  idxPostDim    = prod(index.shape[axis+1:])
  gatherSize    = idxGatherDim  // 仅供 gather 轴长度/复用比判断
  ySize         = index.numel()
  // 不再乘 outerSize/innerSize；否则会重复计算输出元素数
```

### Dtype Improve

```
lastAxisBytes = innerSize * dtypeSize
若 dtypeSize < 8 且 lastAxisBytes % 8 == 0 → improveDtypeSize = 8
若 dtypeSize < 4 且 lastAxisBytes % 4 == 0 → improveDtypeSize = 4
若 dtypeSize < 2 且 lastAxisBytes % 2 == 0 → improveDtypeSize = 2
否则 improveDtypeSize = dtypeSize
innerSize_improved = innerSize * dtypeSize / improveDtypeSize
```

---

## 2. 决策树（Step 1–2）

> **gatherMode 是决策树的关键输入**：`PER_ELEMENT`（torch.gather/GatherElementsV2，每位置独立索引取 1 元素）与 `PER_SLICE`（GatherV2/Embedding，每索引取 innerSize 个连续元素）的模板选择不同。SIMD 两维和 SIMD 行拷贝模板**仅适用于 PER_SLICE**——它们假设一个索引取整行 slice，可用 DataCopyPad 连续搬。PER_ELEMENT 中每个输出位置有独立索引，无连续 slice 可搬，走这两个模板会导致逐元素标量读退化。

```
输入: (batchSize, outerSize, gatherDimSize, gatherSize, innerSize, improveDtypeSize, ub_size, core_num, gatherMode)

  gatherMode = PER_ELEMENT (torch.gather/GatherElementsV2: index 与 output 同形，每位置独立索引，取 1 元素)
             | PER_SLICE   (GatherV2/Embedding: index 通常低维，每索引取 innerSize 个连续元素)

前置: gatherDimSize == 0 → EMPTY 模板（输出清零）

┌─ innerSize == 1（末轴 gather，所有 gatherMode）
│   && gatherDimSize × improveDtypeSize ≤ UB/2（可全载）
│   && batchSize == 1
│   && gatherSize/gatherDimSize > 1（索引复用）
│   && improveDtypeSize ≤ 8
│     → 【SIMD 末轴 Gather】x 行全载 UB，Gather 向量化
│
├─ gatherMode == PER_ELEMENT && dim != last（innerSize > 1）
│   ├─ 内部能力门控：dtype/SoC 支持，shape 可合轴连续搬运，postDim>32，
│   │                  x/idx gatherDim≥16，x元素数<INT32_MAX，xGatherDim*dtypeSize<120000
│   │   && Transpose UB 足够
│   │   && workspace 搬运成本门控通过
│   │     → 【kernel 内 Transpose + Gather】
│   │        阶段1: stride DMA + TransDataTo5HD 转置 x/index → workspace GM
│   │        阶段2: workspace 顺序读 + Gather 指令 → workspace GM
│   │        阶段3: ReTranspose + stride DMA 写回 y
│   ├─ 否则，物理 Transpose 平台可用 && 成本/实测收益门控通过
│   │     → 【物理 Transpose×3 + 末轴 Gather】
│   └─ 否则 → 【Scalar/SIMT】A/B
│
├─ gatherMode == PER_SLICE && batchSize==1 && outerSize==1（2D 场景）
│   && innerSize × improveDtypeSize ≥ SIMD 阈值（按 dtype 分级）
│   && batchSize*outerSize*gatherSize ≥ coreNum/2
│     → 【SIMD 两维】gatherSize 分核，innerSize 行向量化搬运
│        注: 仅 PER_SLICE — 一个索引取整行 slice，DataCopyPad 连续搬
│
├─ innerSize 小（< 2048）且 gatherDimSize × innerSize 可全载 UB
│   && batchSize == 1 && gatherSize/gatherDimSize > 1
│     → 【GA 全载】x GA 块全载 UB，Gather 或逐行 VF gather
│
├─ gatherDimSize × innerSize × improveDtypeSize < UB - DCache
│   && batchSize == 1 && outerSize == 1
│   && ySize × improveDtypeSize > 2048 && gatherSize/gatherDimSize > 1
│   && innerSize × improveDtypeSize < 2048
│     → 【Full-Load】x 全载 UB，逐索引 copy 出 innerSize 元素
│
├─ gatherMode == PER_SLICE && innerSize × improveDtypeSize ≥ SIMD 阈值
│   && UB ≥ 2 × INDICES_SIZE(8KB) && batchSize*outerSize*gatherSize ≥ coreNum/2
│     → 【SIMD 行拷贝】逐输出行 DataCopyPad innerSize 元素，index 缓存
│        注: 仅 PER_SLICE — 一个索引取整行 slice
│
├─ batchSize==1 && outerSize==1（2D 场景, PER_ELEMENT 兜底或 PER_SLICE inner 小）
│     → 【SIMT 两维】gatherSize 分核，SIMT 标量并行, 1 fast-div
│
└─ 否则（兜底）
      → 【SIMT 标量并行】2048 线程，fast-division，直接 GM 读写
```

### Transpose + Gather 的 UB 判定（GatherElementsV2 方式）

```
CACHELINE = 512  // 字节
TRANSPOSE_WS_LEN = 128  // 转置窗口长度

carryNumAlign = CACHELINE / dtypeSize
availableUb = ubSize - 2048

// Transpose buffer (固定)
transInBufferSize  = TRANSPOSE_WS_LEN * CACHELINE   // 64KB
transOutBufferSize = TRANSPOSE_WS_LEN * CACHELINE   // 64KB

// Gather buffer (转置后做末轴 gather)
gatherInBufferSize  = CeilAlign(gatherDimSize, xAlign) * dtypeSize
                    + CeilAlign(minIdxSlice, idxAlign) * sizeof(int32) * 2
gatherOutBufferSize = CeilAlign(minIdxSlice, xAlign) * dtypeSize

inBufferSize  = max(gatherInBufferSize, transInBufferSize)
outBufferSize = max(gatherOutBufferSize, transOutBufferSize)

needTrans = availableUb >= (inBufferSize + outBufferSize)
```

> 以上只判断内部模板能否生成 UB tiling。调用前还必须通过 API 上游能力与 workspace 成本门控；`needTrans=true` 不表示内部 Transpose 全局最优。完整门控参考最佳实践库的 `reference/gather/templates/gather_elements_v2_route.template`。

### SIMD 阈值（按 dtype）

| dtype | improveDtypeSize | SIMD 阈值 (innerSize × dtypeSize) |
|-------|------------------|-----------------------------------|
| int8  | 1 (or improved)  | 218                               |
| int16 | 2 (or improved)  | 360                               |
| int32 | 4 (or improved)  | 620                               |
| int64 | 8                | 2048                              |

### 模板速查

| 模板 | gatherMode | 维度条件 | UB 策略 | 关键技术 |
|------|-----------|---------|---------|---------|
| **kernel 内 Transpose + Gather** | **PER_ELEMENT only** | **dim!=last + 能力/UB/成本门控全部通过** | **transpose buffer + gather buffer + workspace GM** | **stride DMA + TransDataTo5HD + Gather指令 + ReTranspose** |
| **物理 Transpose + 末轴 Gather** | **PER_ELEMENT only** | **非末轴 + 平台能力/成本门控通过** | 各独立算子自行 tiling | Transpose(x/index/y) + LastDim Gather |
| SIMD 末轴 Gather | 通用 | inner==1 + GD 全载 | x 行全载 + index + out | Gather MicroAPI |
| GA 全载 | 通用 | inner 小 + GA 全载 | x GA 块全载 + index + out | Gather / VF gather |
| Full-Load | 通用 | GD×inner 全载 + 2D | x 全载 + out | 逐索引 DataCopyPad |
| SIMD 两维 | **PER_SLICE only** | 2D + inner 大 | index 缓存 + x 行 tile | DataCopyPad 逐行 |
| SIMD 行拷贝 | **PER_SLICE only** | inner 大 | index 缓存 + x 行 tile | DataCopyPad 逐行 + index 滑窗 |
| SIMT 两维 | 通用 | 2D + inner 小 | 无 UB（直接 GM） | SIMT 2048 线程 |
| SIMT 标量 | 通用 | 兜底 | 无 UB | SIMT 2048 线程 + fast-div |

> **PER_ELEMENT vs PER_SLICE 模板选择要点**：torch.gather（PER_ELEMENT）中每个输出位置有独立索引，无法用 DataCopyPad 一次搬 innerSize 个连续元素（SIMD 两维/行拷贝的前提）。当 dim!=last 时把物理/内部 Transpose 列为候选，按上述门控选择；不要把内部 Transpose 当默认路径。当前内部模板只支持 2B/4B 数据，int8 必须走物理路径或 Scalar/SIMT。

---

## 3. UB 预算公式

### 3.1 Transpose + Gather（GatherElementsV2 Transpose 模板）

```
条件: gatherMode == PER_ELEMENT, dim != last, 上游能力/成本门控通过，且 UB 够容纳 transpose + gather buffer

原理: torch.gather y[o, g, c] = x[o, index[o,g,c], c]
      对固定 (o, c), x 沿 gather 轴的 GD 个元素 stride=innerSize 不连续
      用 stride DMA (srcStride=innerSize) 加载到 UB → TransDataTo5HD 转置为连续 → workspace GM
      再从 workspace 顺序读 → Gather 指令向量化 → workspace GM
      最后 ReTranspose → stride DMA 写回 y

UB 分配:
  transInBufferSize  = TRANSPOSE_WS_LEN(128) × CACHELINE(512B) = 64KB
  transOutBufferSize = TRANSPOSE_WS_LEN(128) × CACHELINE(512B) = 64KB
  gatherInBufferSize = CeilAlign(GD, xAlign) × dtypeSize + idxSlice × sizeof(int32) × 2
  gatherOutBufferSize = CeilAlign(idxSlice, xAlign) × dtypeSize
  inBufferSize  = max(gatherInBufferSize, transInBufferSize)
  outBufferSize = max(gatherOutBufferSize, transOutBufferSize)
  门控: ubSize - 2048 >= inBufferSize + outBufferSize

workspace GM: 每核独立 workspace 区域, 存转置后的 x/index/y
  workspacePerBlock = carryNumAlign × (CeilAlign(GD×dtypeSize, 4) + idxGatherDim × 4)

多核: 按 idxPreDim × idxPostDim 分核 (行+列 2D 切分)
      usedCoreNum = min(idxPreDim × idxPostDim, coreNum)
```

### 3.2 SIMD 末轴 Gather

```
条件: innerSize==1, gatherDimSize × improveDtypeSize ≤ UB/2

UB 分配（双缓冲）:
  inputUb  = ubRows × gatherDimSize × improveDtypeSize
  indiceUb = ubCols × indicesDtypeSize
  outUb    = ubRows × ubCols × improveDtypeSize
  indiceCastUb = ubCols × 4 (int64→int32 cast, 若需)

门控: 2×(inputUb + outUb + indiceUb + MIN_OUT_UB) + indiceCastUb ≤ UB

多核: blockFactor = ceil(outerSize, coreNum)
      可选 coreInCols 切 gatherSize 增加并行度
```

### 3.3 GA 全载

```
条件: batchSize==1, gatherSize/gatherDimSize > 1, aSize < 2048

aSize = innerSize × improveDtypeSize
aSizeAligned = ceil(aSize, ubBlockSize)
gaSizeInUb = aSizeAligned × (gatherDimSize + 1)

UB 分配:
  xBuf = maxGaNum × gaSizeInUb
  indicesBuf = min(gatherSize, MIN_INDICES_UB/idxDtypeSize) × idxDtypeSize
  yBuf = (UB - xBuf - indicesBuf - 4×MIN_OUT_UB) / 2

多核: 2D [pOuter(outer切), indicesOuter(gather切)]
```

### 3.4 Full-Load

```
条件: gatherDimSize × innerSize × improveDtypeSize < UB - DCache(128KB)

UB 分配:
  maxElement = (UB - DCache) / improveDtypeSize

多核: perCoreElements = ceil(outerSize×gatherSize, coreNum)
      threadNum = min(perCoreElements × innerSize, 2048)
```

### 3.5 SIMD 行拷贝（PER_SLICE only）

```
条件: innerSize × improveDtypeSize ≥ SIMD 阈值, UB ≥ 2×INDICES_SIZE

UB 分配:
  maxElement = (UB - INDICES_SIZE) / (improveDtypeSize × BUFFER_NUM)
  indiceUbSize = INDICES_SIZE (8KB 固定)

多核: blockFactor = ceil(batchSize×outerSize×gatherSize, coreNum)
```

### 3.6 SIMT 标量 / 两维

```
无 UB 预算（直接 GM 读写）

多核: blockLen = ceil(totalOut, coreNum)
      threadNum 自适应: 从 2048 降至 128，保证 ceil(totalOut, threadNum) ≥ coreNum/2
      blockLen 对齐到 threadNum 的倍数
```

---

## 4. 多核切分

| 模板 | 切分对象 | 分配方式 |
|------|---------|---------|
| Transpose + Gather | idxPreDim × idxPostDim | 行+列 2D 切分 `usedCoreNum = min(pre×post, coreNum)` |
| SIMD 末轴 | outerSize (+ gatherSize via coreInCols) | 2D `[pBlock, gBlock]` |
| GA 全载 | outerSize × gatherSize | 2D `[pOuter, indicesOuter]` |
| Full-Load | outerSize × gatherSize | `perCore = ceil(pgDataSize, coreNum)` |
| SIMD 两维 | gatherSize | `blockFactor = ceil(gatherSize, coreNum)` |
| SIMD 行拷贝 | batchSize×outerSize×gatherSize | `blockFactor = ceil(total, coreNum)` |
| SIMT 两维 | ySize | `perCore = ceil(ySize, coreNum)`, 对齐 threadNum |
| SIMT 标量 | totalOut | `blockLen = ceil(totalOut, coreNum)`, 对齐 threadNum |

---

## 5. 地址大小选择

```
若 gatherDimSize × innerSize > INT32_MAX 或 ySize > INT32_MAX:
  INDEX_SIZE_T = uint64_t
否则:
  INDEX_SIZE_T = uint32_t  // 32 位地址（更快）
```

---

## 6. 算子差异：gatherMode 与模板选择

### 6.1 算子分类

| 算子 | gatherMode | index 形态 | 每 index 取 | 对应框架 |
|------|-----------|-----------|------------|---------|
| **GatherElementsV2** | **PER_ELEMENT** | == x.ndim（同形） | 1 元素 | torch.gather |
| **GatherV2** | **PER_SLICE** | ≤ x.ndim | innerSize 元素 | TF tf.gather |
| Embedding | PER_SLICE | 1D/2D | innerSize 元素 | nn.Embedding |
| batch_dims>0 | PER_SLICE | 前缀匹配 | innerSize 元素 | TF tf.gather(batch_dims) |

### 6.2 gatherMode 对模板选择的影响

| 模板 | PER_ELEMENT (torch.gather) | PER_SLICE (GatherV2/Embedding) | 原因 |
|------|:---:|:---:|------|
| **kernel 内 Transpose + Gather** | **✅（仅能力/UB/成本门控全通过）** | ❌ | PER_ELEMENT 专属：每位置独立索引但同一 (o,c) 的 GD 列可 stride DMA 转置 |
| **物理 Transpose + 末轴 Gather** | ✅（平台能力/成本门控通过） | ❌ | 把 gather 轴交换到末轴并复用独立高性能算子 |
| SIMD 末轴 Gather | ✅ (inner==1) | ✅ (inner==1) | 末轴 gather，x 行全载 UB + Gather 指令 |
| GA 全载 | ✅ | ✅ | GD×inner 全载 UB |
| Full-Load | ✅ | ✅ | GD×inner 全载 + 2D |
| **SIMD 两维** | **❌** | **✅** | 需一个索引取整行 slice，PER_ELEMENT 每位置独立索引无 slice |
| **SIMD 行拷贝** | **❌** | **✅** | 同上，DataCopyPad 逐行搬 innerSize 元素需索引复用 |
| SIMT 标量/两维 | ✅ | ✅ | 通用兜底，直接 GM 读写 |

### 6.3 Transpose + Gather 策略详解（GatherElementsV2 Transpose 模板）

**解决的问题**：torch.gather dim!=last（如 dim=0）且数据为受支持的 2B/4B dtype 时，SIMT 逐元素随机读 GM 可能很慢。内部 Transpose 将散点读转为顺序读 + UB 内向量化 gather；是否采用仍由能力和成本门控决定。int8 不受当前内部模板支持。

**核心思路**：torch.gather `y[o, g, c] = x[o, index[o,g,c], c]` 中，对固定的 (o, c)，x 沿 gather 轴的 GD 个元素虽然物理上不连续（stride=innerSize），但可以用 `DataCopyPad(srcStride=innerSize)` 跨步 DMA 加载到 UB，再用 `TransDataTo5HD` 在 UB 内做矩阵转置使其连续，然后写到 workspace GM。之后从 workspace 顺序读 x 行，用 `Gather` 指令在 UB 内向量化 gather。最后将结果 ReTranspose 写回 y。

**与物理 Transpose 的区别**：不在 framework 层做 3 次独立 Transpose kernel，而是在 **1 个 kernel 内的 3 个阶段**完成（Transpose → Gather → ReTranspose），用 workspace GM 中转，无额外 kernel launch 开销。

**3 阶段流程**：
1. **TransposeProcess**: `DataCopyPad(srcStride=postDim)` stride DMA → `TransDataTo5HD` UB 内转置 → workspace GM
2. **GatherProcess**: 从 workspace 顺序读 x 行 → `Gather` 指令向量化 → workspace GM
3. **ReTransposeProcess**: 从 workspace 读 → `TransDataTo5HD` 转置回 → `DataCopyPad(dstStride=postDim)` 写回 y

**适用条件**：
- `gatherMode == PER_ELEMENT`（torch.gather/GatherElementsV2）
- `dim != last`（innerSize > 1，否则直接走 SIMD 末轴 Gather）
- dtype/SoC、shape 合轴、postDim/gatherDim、元素数门控通过；当前 dtype 仅 fp16/bf16/fp32/int32
- UB 够容纳 transpose buffer (128KB) + gather buffer
- workspace 成本代理优于物理 Transpose
- 任一条件失败后，继续评估物理 Transpose，再回退 Scalar/SIMT

**关键 API**：
- `DataCopyPad(blockCount, blockLen, srcStride, dstStride)` — stride DMA 跨步加载
- `TransDataTo5HD` — UB 内矩阵转置（支持 4B/2B dtype）
- `Gather(yLocal, xLocal, offsetLocal, 0, count)` — UB 内向量化 gather 指令
- workspace GM — 每核独立区域，存转置中间结果

**Agent 推导步骤**：
1. 确定算子数学流程 → 判定 gatherMode（PER_ELEMENT / PER_SLICE）
2. 合轴为四元组 (outerSize, gatherDimSize, gatherSize, innerSize)
3. 计算 dtype improve
4. 按 gatherMode 进入决策树对应分支
5. PER_ELEMENT + dim!=last → 先做内部能力/UB/成本门控，再做物理 Transpose 能力/收益门控
6. 仅内部路径命中时，代入 §3.1 预算公式计算 UB 分配；否则选择物理 Transpose 或 Scalar/SIMT
