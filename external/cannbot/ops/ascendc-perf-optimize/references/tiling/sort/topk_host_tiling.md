# DAV_3510 TopK Host Tiling 复原规格

> 本文与 [script/topk_tiling.py](script/topk_tiling.py) 共同替代旧的 TopK `op_host` C++ 快照。
> 本文固定 CANN 接线、ABI 和复原约束；Python 固定可脱离 CANN 执行的选路、切分、UB 与 workspace
> 数学。两者必须一起读取，不能只按本文概述手写一个简化 tiler。

## 目录

1. [真值源与适用范围](#1-真值源与适用范围)
2. [输入、平台与归一化](#2-输入平台与归一化)
3. [一级路由](#3-一级路由)
4. [Common 内部路由](#4-common-内部路由)
5. [常量、tiling key 与 mode](#5-常量tiling-key-与-mode)
6. [TilingData 精确 ABI](#6-tilingdata-精确-abi)
7. [Special route 参数契约](#7-special-route-参数契约)
8. [UB 与 API tmp 建模](#8-ub-与-api-tmp-建模)
9. [Workspace 与 launch 契约](#9-workspace-与-launch-契约)
10. [目标 CANN SDK 查询点](#10-目标-cann-sdk-查询点)
11. [C++ 复原步骤](#11-c-复原步骤)
12. [Python 使用与保真边界](#12-python-使用与保真边界)
13. [复原验收清单](#13-复原验收清单)

---

## 1. 真值源与适用范围

TopK 的 Host Tiling 属于 Tiling 策略层，唯一知识源是本目录：

- `topk.md`：case 级性能路由、算法原因和禁止外推边界；
- 本文：Host/CANN/ABI 复原规格；
- `script/topk_tiling.py`：可执行的公式与切分参考。

Kernel、launch、fallback 和 TopK 私有 Transpose 的参考模板仍在
`ascendc-performance-best-practices/references/sort/arch35/templates/dav3510/op_kernel/`。复原 Host 时必须读取
这些设备侧消费者，核对每个字段和 buffer，不能在最佳实践 skill 中再维护一份 Tiling 真值。

当前行为契约：DAV_3510、连续 ND 输入、有序 TopK、value dtype 为
`int8/uint8/int32/int64/fp16/fp32/bf16`、index 通常为 `int64`，且
`1 <= k <= min(D, 2048)`。非 DAV_3510、非连续输入、无序输出或更大的 k 必须重新做能力和性能验证。

“可复原”指可生成行为与字段契约等价、能和保留的 Kernel 模板编译并通过回归的 Host 实现；不要求恢复
原文件中的注释、日志文本或函数排版。

---

## 2. 输入、平台与归一化

### 2.1 Host 输入

Host 入口至少接收：输入 shape、rank、k、dim、largest、sorted、value dtype、index dtype 和 device id。
先把负 dim 加 rank，再校验：

```text
1 <= rank <= 8
0 <= normalizedDim < rank
所有维度 > 0
1 <= k <= min(shape[normalizedDim], 2048)
value/index dtype 在支持集合内
```

计算：

```text
outer   = product(shape[:dim])
D       = shape[dim]
inner   = product(shape[dim + 1:])
rows    = outer * inner
nonLast = (dim + 1 < rank)
```

乘积、字节数、workspace offset 以及 `3*k`、`4*k` 等比例判断用 `__int128` 中间值，并在写入
`int64_t/size_t` 前检查范围。尾部维度即使全为 1，`nonLast` 仍按 dim 的 rank 位置判断；fallback 的
L/S 分支才按 `inner == 1` 判断。

### 2.2 平台信息

从 `PlatformAscendC` 读取：

```text
coreNum    = GetCoreNumAiv()
ubSize     = GetCoreMemSize(CoreMemType::UB)
ubBlock    = GetUbBlockSize()，缺失时 DAV_3510 取 32B
```

公共 tiler 和 fixed small-axis 先从目标平台查询到的 UB 容量中预留 32 KiB SIMT 空间。fallback
从实际 UB 容量扣除 8 KiB 后搜索；查询失败时停止生成 tiling。动态 UB 必须写入 launch 属性，不能只影响 Host 计算。

---

## 3. 一级路由

入口必须按以下顺序返回第一个命中项：

```text
fixed small-axis 900001
-> fallback candidate
   -> transpose-fallback 900003（满足附加门禁时）
   -> batched fallback L/S 900002
-> common tiler
```

### 3.1 Fixed small-axis：900001

全部满足时命中：

```text
nonLast && k == D && D in {7, 17} && dtype in {fp16, bf16}
```

沿 inner 切 `tileWidth`。D=7 使用 32 元素步长，D=17 使用 16 元素步长；初值为
`alignUp(ceil(inner / max(coreNum/outer, 1)), step)`，上限 4096。若 UB 不足，每次减一个 step。
核数为 `min(coreNum, outer * ceil(inner/tileWidth))`。

### 3.2 Fallback candidate：900002/900003

以下任一成立即进入候选区：

```text
dtype == int64
or (nonLast && D > 2048)
or (dtype == int32 && nonLast && D >= 32 && rows >= 512
    && (k <= 32 || 4*k <= D))
```

### 3.3 Transpose-fallback：900003

仅在已进入 fallback candidate 后，并且全部满足时尝试：

```text
outer == 1
nonLast
dtype in {int32, fp32}
D > 2048
inner >= coreNum * 32
3*k <= D
```

递归计算 `[inner, D]`、dim=1 的尾轴 TopK，递归时禁止再次选 900003。四段 workspace 为：

```text
inputOffset          = 0
valuesOffset         = align512(D * inner * 4)
indicesOffset        = align512(valuesOffset + inner * k * 4)
innerWorkspaceOffset = align512(indicesOffset + inner * k * 8)
totalWorkspace       = innerWorkspaceOffset + innerWorkspaceBytes
```

还必须满足 `innerWorkspaceOffset <= 2 * inputBytes`、递归成功、递归 key 不是 900003、总字节不溢出。
失败时继续 900002，不能报错或放宽门槛。生成三份私有 Transpose tiling：输入 `[D,inner]` 4B、
values `[inner,k]` 4B、indices `[inner,k]` 8B，perm 均为 `[1,0]`。

### 3.4 Batched fallback：900002

`inner == 1` 为 L，否则为 S。L 的 `dChunk <= 4096`，S 的 `dChunk <= 4000`，G 最大 128。
按大到小遍历去重候选：

```text
{min(limit, align32(D)), 2048, 1024, 512, 256, 128, 64, 32}
```

对每个 dChunk 找 240 KiB 内最大的 G。成本为：

```text
chunks = ceil(D / dChunk)
groups = ceil(rowsPerCore / G)
cost   = chunks * groups           # 非 int64
cost   = 2 * chunks - 1            # int64
```

选择最小 cost；相同 cost 保留先遍历到的较大 dChunk。详细 UB 项和数据类型特判以
`topk_tiling.py::_fallback_params/_fallback_ub_total` 为可执行真值。

---

## 4. Common 内部路由

一级未命中时，按下列先后选择第一个可行模板：

| 顺序 | route | 关键门控 | modeType |
|---:|---|---|---:|
| 1 | non-last small-axis | nonLast 且 D<=2048 | 8 |
| 2 | small MergeSort | 尾轴，D<=1024，fp16/fp32/bf16 | 0 |
| 3 | single block | 完整行、批次输出和 API tmp 可同时放 UB | 0 |
| 4 | fp32 inter-core merge | `0.25<=k/D<0.5`，`ceil(D/2048)>1`，总 block 不超过核数 | 6 |
| 5 | fp32 intra-core merge | 上项未命中，rows 足够且每行 2..256 个 sort block | 7 |
| 6 | single-core radix | `rows>=coreNum` 且存在合法 tile | 1 |
| 7 | optimized multi-core radix | `tile<D` 且 `k*ceil(D/tile)<=tile` | 4 |
| 8 | general multi-core radix | D<10,000,000 且存在合法 tile | 2 |
| 9 | SortAndTopK | 超大 D 或前述 tile 均失败 | 5 |

具体 tile 搜索起点为：single block/single core 的非 64 位 15360、64 位 10240；multi-core 的非
64 位 7680、64 位 5120。Radix tile 以 256 递减，optimized 的最小步进同时受 k 和 32B 对齐约束。
这些数值是性能策略，不是架构能力边界。

non-last mode 8 按 dtype 依次尝试 innerChunk：8B `{4,2,1}`、4B `{8,4,2,1}`、2B
`{16,8,4,2,1}`、1B `{32,16,8,4,2,1}`，并取 active core 更多、chunk 更大的可行项。
fp32，或 D<=1024 的 fp16/bf16 使用 merge；其余使用 radix。

---

## 5. 常量、tiling key 与 mode

### 5.1 公共常量

| 名称 | 值 |
|---|---:|
| `ALIGN_BYTES` / `AGLIN_FACTOR` | 32B |
| SIMT UB reserve | 32 KiB |
| fallback UB budget | `GetCoreMemSize(UB) - 8 KiB`，按目标设备计算 |
| system workspace | 16 MiB |
| `BIN_NUM` | 256 |
| `TOPK_K_MAX` | 2048 |
| fixed small-axis max tileWidth | 4096 |
| fallback L/S max dChunk | 4096 / 4000 |
| fallback max G | 128 |
| small MergeSort D 上限 | 1024 |
| non-last mode 8 D 上限 | 2048 |
| SortAndTopK D 门槛 | 10,000,000 |
| fp32 merge 比例 | `[0.25, 0.50)` |
| Sort 索引 int32 范围门槛 | 1,073,741,823 |
| Sort 固定 UB（int32/int64 counter） | 4096B / 7168B |
| Sort 最小性能 tile | 1024 elements |

### 5.2 Tiling key

| dtype/route | key |
|---|---:|
| int8 | 1001 |
| uint8 | 2001 |
| int32 | 1003 |
| int64 | 1004 |
| fp16 | 3002 |
| fp32 | 3003 |
| bf16 | 4002 |
| small MergeSort | dtype key + 10000 |
| fp32 inter-core merge | 23003 |
| fp32 intra-core merge | 33003 |
| fixed small-axis / fallback / transpose-fallback | 900001 / 900002 / 900003 |

Common mode 值固定为 1 single-core、2 general multi-core、4 optimized multi-core、5 SortAndTopK、
6 fp32 inter-core merge、7 fp32 intra-core merge、8 non-last small-axis。0 表示 small MergeSort 或
single-block 的默认分支。key 和 mode 必须与 Kernel 模板实例一一对应。

---

## 6. TilingData 精确 ABI

Common Kernel 使用 `TopKTilingDataSimd`。字段顺序和 C++ 类型是 ABI，必须按下表逐项生成；禁止排序、
改宽度、删除“当前 case 未使用”的字段或把 `keyParams0` 错写成 uint32。

| 序号 | C++ 类型 | 字段 |
|---:|---|---|
| 1 | `int32_t` | `isLargest` |
| 2 | `int32_t` | `isSort` |
| 3 | `uint64_t` | `sortLoopTimes` |
| 4 | `uint32_t` | `unsortedDimParallel` |
| 5 | `uint64_t` | `unsortedDimNum` |
| 6 | `uint32_t` | `lastDimNeedCore` |
| 7 | `uint32_t` | `numTileDataSize` |
| 8 | `uint32_t` | `platformCoreNum` |
| 9 | `uint32_t` | `topkAcApiTmpBufferSize` |
| 10 | `uint32_t` | `mergSortAcApiNeedBufferSize` |
| 11 | `uint64_t` | `oneCoreRowNum` |
| 12 | `uint32_t` | `batchNumInUb` |
| 13 | `uint32_t` | `tailLoopBatchNum` |
| 14 | `uint32_t` | `tailBatchNum` |
| 15 | `uint32_t` | `tailTileNum` |
| 16 | `uint32_t` | `modeType` |
| 17 | `uint32_t` | `isInInt32Range` |
| 18 | `int64_t` | `lastAxisNum` |
| 19 | `int64_t` | `topKRealValue` |
| 20 | `int64_t` | `lastDimTileNum` |
| 21 | `int64_t` | `outputLastDimValue` |
| 22 | `int64_t` | `lastDimTileNumTimes` |
| 23 | `uint32_t` | `tilingKeyForSort` |
| 24 | `uint32_t` | `sortLoopTimesForSort` |
| 25 | `uint32_t` | `unsortedDimParallelForSort` |
| 26 | `uint64_t` | `unsortedDimNumForSort` |
| 27 | `uint32_t` | `lastDimTileNumForSort` |
| 28 | `uint32_t` | `lastDimNeedCoreForSort` |
| 29 | `uint32_t` | `numTileDataSizeForSort` |
| 30 | `uint32_t` | `sortAcApiNeedBufferSizeForSort` |
| 31 | `uint32_t` | `mergSortAcApiNeedBufferSizeForSort` |
| 32 | `uint32_t` | `oneCoreRowNumForSort` |
| 33 | `uint32_t` | `outputLastDimValueForSort` |
| 34 | `uint32_t` | `isInInt32RangeForSort` |
| 35 | `int64_t` | `lastAxisNumForSort` |
| 36 | `uint32_t` | `modeTypeForSort` |
| 37 | `uint64_t` | `keyParams0` |
| 38 | `uint32_t` | `keyParams1` |
| 39 | `uint32_t` | `keyParams2` |
| 40 | `uint32_t` | `keyParams3` |
| 41 | `uint32_t` | `keyParams4` |
| 42 | `uint32_t` | `keyParams5` |
| 43 | `uint32_t` | `tmpUbSize` |
| 44 | `uint32_t` | `sortAndTopkTileDataSize` |
| 45 | `uint32_t` | `sortAndTopkBlockTileNum` |
| 46 | `uint32_t` | `sortAndTopkTailTileNum` |

用 `BEGIN_TILING_DATA_DEF` / `TILING_DATA_FIELD_DEF` / `END_TILING_DATA_DEF` 声明，并注册：

```cpp
REGISTER_TILING_DATA_CLASS(TopK, TopKTilingDataSimd)

struct TopKCompileInfo {
    int32_t coreNum;
};
```

公共字段由 `topk_tiling.py::_base_fields` 产生。mode 专属字段以脚本每个 `_plan` 分支为真值；未设置字段
保持零初始化。`keyParams0..5` 是 mode 私有槽，不能跨 mode 统一解释。SortAndTopK 使用 37..46；普通
TopK 需要最终有序化时，SortWithIndex 使用 23..43，并按设备侧契约覆盖共享的 `keyParams0..5/tmpUbSize`。

---

## 7. Special route 参数契约

Special key 不直接消费上节的 common ABI，而是通过统一 `TopKLaunchParams` 携带下列结构。

### 7.1 `TopKSmallAxisParams`

按顺序使用 7 个 `int64_t`：`outer, axisLen, inner, isLargest, tileWidth, usedCores,
localMemoryBytes`。

UB 公式：

```text
rowBytes  = align32(tileWidth * 2)
maskBytes = align32(ceil(tileWidth / 8))
required  = D * rowBytes * 2 + rowBytes * 2 + maskBytes * 2
```

### 7.2 `TopKFallbackParams`

按顺序使用 16 个 `int64_t`：`outer, dimLen, inner, rows, k, isLargest, branch, usedCores,
rowsPerCore, g, dChunk, kPad, kIdxPad, mergePad, tmpSize`。

```text
branch=0 -> L，inner==1
branch=1 -> S，inner>1
kPad      = compute dtype 的 32B 对齐元素数
kIdxPad   = int32 index 的 32B 对齐元素数
mergePad  = 2*k candidates 的对齐长度，且 <=4096
```

fp16/bf16 的 compute dtype 是 fp32。int64 保持 8B compute，但 kernel 逐 chunk 做 int32 范围检测，
不能在 Host 仅按 D 判断窄化。

### 7.3 `TopKTransposeFallbackParams`

按顺序为 6 个 `int64_t`：`innerTilingKey, inputOffset, valuesOffset, indicesOffset,
innerWorkspaceOffset, innerWorkspaceBytes`，随后依次嵌入 input/value/index 三份私有
`TransposeTiling`。三份结构必须调用保留模板的
`calc_transpose_tiling_params_with_core_num` 生成，不能自己猜字段。

### 7.4 `TopKLaunchParams`

顺序为 5 个 `int64_t`：`tilingKey, usedCores, workspaceBytes, localMemoryBytes, scheduleMode`，随后依次
嵌入 `TopKTilingDataSimd, TopKSmallAxisParams, TopKFallbackParams,
TopKTransposeFallbackParams`。`scheduleMode` 默认并固定为 1。

---

## 8. UB 与 API tmp 建模

### 8.1 公共 RadixSelect tmp 下界

脚本 `topk_api_tmp_size` 固化 DAV_3510 已验证布局：输入 index、candidate value/index、方向相关工作区、
有序输出工作区。实现 C++ 时逐项翻译，不要用一个经验比例替代。64 位 candidate 至少按 512 元素，
index 至少按 512B；sorted 输出额外需要 `512 + alignedK*(11+dtypeBytes)`。

### 8.2 Fallback UB

必须计入 compute input、input index、chunk output、running value/index、merge value/index、API tmp 和
int64 output index。再按条件加入：

- fp16/bf16 的 raw input、fp32 cast 和 output cast；
- int64 的 range check 与 b64 Sort scratch；
- S 的 2D DMA stage、Gather offset、全块 transpose table 和输出 transpose tile；
- L/int64 生命周期别名后的真实峰值。

生产 C++ 对每个 dChunk 候选先算实现下界，再调用 SDK，取两者最大值并重新找最大 G。脚本没有 SDK
时只用实现下界探索，并在 JSON 的 `sdk_queries` 明确标记，不能直接把探索值写进生产 C++。

### 8.3 Common 模式

single-block、single-core、optimized/general multi-core 的 buffer 公式分别对应脚本
`_single_block_runtime`、`_single_core_runtime`、`_optimized_runtime`、`_multi_core_runtime`。每个候选
必须满足 `runtime buffers + API tmp <= ubSize - 32KiB`。修改 Kernel `InitBuffer` 后必须同步修改公式。

fp32 inter-core workspace 元素对齐为：

```text
alignedInputElems = align32(D * 8) / 8
```

其中 8 是 fp32 value + uint32 index 的 sort struct 字节数。intra-core 的 `alignNum` 为
`ceil(D/blockSortSize)*blockSortSize`。

### 8.4 SortAndTopK

先从 `usableUb = physicalUb - 32KiB` 计算全轴 RadixSort tile：

```text
isInt32    = D <= 1,073,741,823
ubExtra    = 4096 if isInt32 else 7168
tileFactor = (10 if isInt32 else 14) + valueDtypeBytes
tile       = floor256((usableUb - ubExtra) / tileFactor)
```

对 tile 调用 `GetSortMaxMinTmpSize(uint8,uint32)`；若 tmp 放不下则每次减 256。随后按 rows、核数和
`ceil(D/tile)` 做均匀分核调整，最小 tile 1024。精确分支由脚本 `_sort_and_topk_tile` 固化。剩余 UB
扣一个 UB block 后按 block 对齐全部并入 `tmpUbSize`。

```text
lastTiles = ceil(D / tile)
parallel  = 1                          if coreNum <= lastTiles
            min(coreNum/lastTiles,rows) otherwise
coreNeed  = parallel * min(coreNum,lastTiles)
```

以 `tmpUbSize/(4 or 8)` 为 counter 容量计算 global histogram 和 exclusive histogram 的
`keyParams0..5`。取前 k 的切分为：

```text
extractTile = floor32(usableUb - 1) / max(valueBytes,indexBytes)
total       = ceil(k / extractTile)
sortAndTopkBlockTileNum = total / coreNum
sortAndTopkTailTileNum  = total % coreNum
```

不得只填 `sortAndTopkTileDataSize`；三个 extraction 字段、六个 keyParams、tmp、核数和 workspace 是
同一份契约。

---

## 9. Workspace 与 launch 契约

Common workspace 先计算用户区，再加 16 MiB 系统区：

```text
single-core:
  user = lastDimTileNum * 256 * unsortedDimParallel * indexTypeSize

optimized multi-core && axis index 可用 int32:
  factor = lastDimTileNum * unsortedDimParallel
  user = align32(dtypeSize * factor * k) + align32(4 * factor * k)

其它 radix mode:
  user = 256 * indexTypeSize * unsortedDimParallel
       + 2 * align32(lastDimTileNum * indexTypeSize * unsortedDimParallel)

fp32 inter-core merge:
  user = rows * alignedInputElems * 8 * 2

fp32 intra-core merge:
  user = activeCores * alignNum * 8 * 2

SortAndTopK:
  user = align32(key1*key4*counterBytes)
       + align32(key3*key2*key0*counterBytes)
       + align32(D*parallel*counterBytes)
       + align32(D*parallel*indexBytes)
       + lastTiles*256*parallel*2*2
       + align32(lastTiles*tile*parallel)
       + 2*align32(D*parallel*valueBytes)
```

当有序输出需要独立 SortWithIndex 时，在加系统区前追加其用户 workspace。条件为脚本
`_common_workspace` 与 Kernel `needSortWithIndex` 一致。当前 `k<=2048` 契约下该子 tiler必定进入
SortWithIndex small-size mode 1：axis=k、`lastDimTileNumForSort=1`、每行一核、rows 沿核切分，并追加
`align32(rows*k*valueBytes) + align32(rows*k*indexBytes)`。23..36 号字段由脚本直接给出；
`sortAcApiNeedBufferSizeForSort/tmpUbSize` 仍必须用目标 SDK 结果替换。

Common 和 900003 调用含高阶 API 的 mixed kernel 时：

- `SetScheduleMode(1)`；
- `SetLocalMemorySize(dynamicUb)`；
- launch 保留 `KERNEL_TYPE_MIX_AIV_1_0`；
- attr 保留 `SCHEM_MODE=1` 和 `DYN_UBUF_SIZE`；
- kernel 先 `SetSysWorkspaceForce(workspace)`，再用 `GetUserWorkspace(workspace)` 取得用户区。

900001/900002 无用户 workspace 时可以为 0；不能因此把 common 的 16 MiB 系统 workspace 删掉。

---

## 10. 目标 CANN SDK 查询点

以下结果依赖目标 CANN，不属于可移植常量：

| API | 使用位置 | 生产规则 |
|---|---|---|
| `AscendC::GetConcatTmpSize` | small MergeSort、mode 8 merge、SortWithIndex merge | 对实际对齐长度和 compute dtype 查询 |
| `AscendC::GetTopKMaxMinTmpSize` | fallback 每个 dChunk/mergePad 候选 | 与实现推导 floor 取 max，再重跑候选搜索 |
| `AscendC::GetSortMaxMinTmpSize` | int64 fallback、SortAndTopK、SortWithIndex | 对每次调整后的 tile 使用实际 shape、dtype、index dtype 和 RadixSort config |

Python JSON 出现非空 `sdk_queries` 表示结果仍含探索估计或需要额外 SortWithIndex tiling。复原 C++ 必须完成
这些查询后才能把该 case 视为生产可用。不要把 Python 的估计值硬编码成 CANN API 返回值。

SDK 接线还需要 `gert::OpTilingContextBuilder` 构造 2 输入/2 输出 TopK context，输入 k 为 Host int64
scalar，属性顺序为 `sorted, dim, largest, indexDtype`，TilingData capacity 至少 4096B，workspace 数量为 1。

---

## 11. C++ 复原步骤

建议仍生成旧的文件边界，便于保留 Kernel include 和构建接线：

| 目标文件 | 从本规格复原的职责 |
|---|---|
| `op_host/top_k_tiling_arch35.h` | 第 6 节 ABI、`TopKCompileInfo`、注册宏 |
| `op_host/top_k_tiling_base.h/.cpp` | 常量、对齐、dtype key、API tmp 和 tile helper |
| `op_host/top_k_tiling_arch35.cpp` | common 各 mode、workspace、保存 tiling data |
| `op_host/sort_with_index_tiling.h` | 最终有序化的 radix/merge tiler及 23..46 字段 |
| `op_host/top_k_tiling_adapter.cpp` | 平台查询、一级路由、special params、context builder |

复原顺序：

1. 先生成第 6、7 节的 ABI/launch 结构，确保 `op_kernel/arch35/top_k_launch.h` 可以 include。
2. 将 Python 的常量、`normalize_request`、UB helper 和 `_find_tile` 逐函数翻译为 C++，保留整数除法和
   对齐语义；乘法改用 checked `__int128`。
3. 按第 3 节实现 adapter 一级路由。900003 递归时传 `allowTranspose=false`。
4. 按第 4 节实现 common 路由。每个 Python `_plan` 返回分支对应一个 C++ tiling 函数，并写同名字段。
5. 在第 10 节的点接入目标 CANN API；任何 Python fallback estimate 只用于找到初始候选。
6. 生成 `StorageShape`、Host scalar k、`OpTilingContextBuilder`，调用 common tiler并复制 raw tiling data。
7. 设置 tiling key、block dim、local memory、workspace 和 schedule mode；special key 必须先于 common
   key 在 launch 中分发。
8. 使用 `_GLIBCXX_USE_CXX11_ABI=0` 编译 Host tiling 源，并按目标工程适配注册入口。

函数映射：

| Python | C++ 职责 |
|---|---|
| `normalize_request` | shape/dim/overflow 校验 |
| `_small_axis_plan` | `calc_topk_small_axis_params` |
| `_fallback_params` / `_fallback_ub_total` | `calc_topk_fallback_params` |
| `_mode8_plan` | non-last mode 8 |
| `_small_merge_plan` | small MergeSort |
| `_last_axis_common_plan` | common 其余 mode |
| `_common_workspace` | common + SortWithIndex workspace |
| `compute_topk_tiling` | `calc_topk_params` 一级入口 |

---

## 12. Python 使用与保真边界

自测和单 case：

```bash
python3 script/topk_tiling.py --self-test

python3 script/topk_tiling.py \
  --shape 8,64,64 --k 8 --dim 1 --dtype fp16 \
  --core-num "$TARGET_AIV_CORES" --ub-size "$TARGET_UB_BYTES"
```

必须通过 `--core-num`、`--ub-size` 传入目标平台查询值；small merge 和单个最终 SortAndTopK tile 的 SDK 结果可通过
`--merge-tmp-bytes`、`--sort-tmp-bytes` 注入。fallback 候选和 SortWithIndex 需要多次或不同配置的查询，
由 `sdk_queries` 显式列出并在 C++ 中完成。输出字段：

- `route/tiling_key/mode_type`：选路结果；
- `block_dim/local_memory_bytes/workspace_bytes`：launch 参数；
- `tiling_data`：common ABI 的非零/显式字段；
- `launch_params`：special route 结构；
- `sdk_queries`：生成生产 C++ 前仍必须执行的目标 SDK 查询；
- `notes`：探索值的限制。

Python 对不依赖 CANN 查询的归一化、门控、候选顺序、整数公式和 workspace layout 是复原真值。
含 `sdk_queries` 的输出不是最终生产 tiling：必须在目标 CANN 中查询并重新执行受 tmp 大小影响的 UB
搜索。TransposeTiling 结构也必须调用保留模板的 helper 生成。

脚本是切分/决策参考，不生成 C++，以避免把 SDK 版本、工程注册宏和头文件布局错误地固化为文本模板。

---

## 13. 复原验收清单

- [ ] `TopKTilingDataSimd` 的 46 个字段顺序、类型和 `sizeof` 与 Kernel 消费端一致。
- [ ] fixed small-axis、fallback、transpose-fallback、common 的一级顺序不变。
- [ ] common 的九个判断顺序不变，tiling key/modeType 对应正确。
- [ ] Python `--self-test` 与 `py_compile` 通过。
- [ ] 代表 case 的 Python route、C++ tiling key、mode、block dim、UB 和 workspace 一致。
- [ ] 所有非空 `sdk_queries` 已在目标 CANN 接入，不存在生产估计值。
- [ ] Host UB 每一项能映射到 Kernel `InitBuffer`，条件 buffer 同时出现或消失。
- [ ] 900003 三次 Transpose、四段 512B offset、递归防重入和 workspace 门禁完整。
- [ ] common 保留 16 MiB 系统 workspace，并与用户 workspace 分离。
- [ ] launch 保留 mixed AIV、schedule mode 1 和动态 UB。
- [ ] largest/smallest、所有 dtype、D/k/tile 边界、int64 窄宽混合、NaN、尾块和 non-last 精度通过。
- [ ] 对路由门槛相邻 case 做同环境逐 case A/B，不从单 case 外推。
