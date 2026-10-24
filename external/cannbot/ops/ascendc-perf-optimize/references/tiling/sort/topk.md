# TopK Tiling 与性能复现

> 适用范围：Ascend 950 / DAV_3510、连续 ND 输入、有序 TopK、输出 index 为 `int64`，
> `int8 / uint8 / int32 / int64 / fp16 / fp32 / bf16`，`1 <= k <= min(D, 2048)`。
> 本文固化 TopK 的已验证策略。它是 case 级多模板路由，
> 不是一套可由单一 RadixSort、MergeSort 或通用 Sort fallback 替代的算法。

---

## 1. 性能复现的完整对象

复现时必须一起迁移和核对以下四层，缺少任一层都可能正确但明显变慢：

1. 一级路由：fixed small-axis、fallback / transpose-fallback、common 的优先级与窄门控。
2. Tiling：shape 归一化、UB 候选搜索、核数、workspace、API tmp 查询和 TilingData。
3. Kernel：固定排序网络、batched running TopK、common 的 merge/radix/SortAndTopK 模板。
4. Launch：tilingKey 分发、schedule mode、动态 UB、系统与用户 workspace 的地址契约。

不要只复制当前 kernel 后重写一个“等价”的简化 tiler；现有性能主要来自选对模板和参数。

本文是 TopK Tiling 策略的唯一知识源。生成 Host C++ 时，必须同时读取
[Host Tiling 复原规格](topk_host_tiling.md) 和可执行参考
[script/topk_tiling.py](script/topk_tiling.py)；Kernel 实施经验由性能最佳实践 skill 提供。

---

## 2. Step 0：接口校验与 shape 归一化

先归一化负 `dim`，再计算：

```text
outer = product(shape[:dim])
D     = shape[dim]
inner = product(shape[dim + 1:])
rows  = outer * inner
nonLast = (dim + 1 < rank)   # host 一级路由按 dim 的 rank 位置判断
```

校验 rank、shape、`1 <= k <= D`、dtype 和乘法溢出。shape 乘积、workspace 字节数以及
`4*k <= D`、`3*k <= D` 一类比例判断使用 128 位中间值。后续所有门控都使用归一化后的
`outer / D / inner / rows / k / dtype / coreNum / ubSize`，不能从原 rank 特判推导。注意尾部维度全为
1 时可能同时满足 `nonLast == true` 和 `inner == 1`；一级 special/fallback 候选仍看 `nonLast`，fallback
内部 L/S 布局选择看 `inner`，两个判据不能合并。

---

## 3. Step 1：一级路由，顺序不可交换

按下列顺序选择第一个命中的路径：

```text
1. fixed small-axis (tilingKey 900001)
2. fallback candidate
   2.1 transpose-fallback (tilingKey 900003)
   2.2 batched fallback L/S (tilingKey 900002)
3. common tiler
```

### 3.1 Fixed small-axis：900001

以下条件必须全部满足：

```text
nonLast
k == D
D in {7, 17}
dtype in {fp16, bf16}
```

`largest` 和 `smallest` 都由同一模板在运行时支持。该路径必须先于 fallback；例如 D=17 的整轴
排序不应因 non-last 再进入通用路径。

### 3.2 Fallback candidate

以下任一条件成立即进入 fallback 候选区：

```text
dtype == int64
or (nonLast and D > 2048)
or (dtype == int32 and nonLast and D >= 32 and rows >= 512
    and (k <= 32 or 4*k <= D))
```

这些门槛来自相邻 case A/B，不是架构能力边界。不要扩展到 fp16/fp32/bf16 的中轴场景；这些
代表 shape 在 common 中更快。`int64` 则必须提前截获，以避开不稳定的 common 64 位路径。

### 3.3 Transpose-fallback：900003

仅在已经是 fallback candidate 后，且以下条件全部满足时尝试：

```text
outer == 1
nonLast
dtype in {int32, fp32}
D > 2048
inner >= coreNum * 32
3*k <= D
```

递归计算 `[inner, D]` 尾轴 TopK 的 tiling，然后再执行 workspace 门禁：

- 内部 tiling 成功，且不得再次返回 900003；
- `inputBytes = D * inner * 4`；values 和 int64 indices 中间区按 512B 对齐；
- `innerWorkspaceOffset <= 2 * inputBytes`；
- `innerWorkspaceOffset + innerWorkspaceBytes` 不溢出 `int64`。

通过后执行 `transpose input -> 尾轴 TopK -> transpose values -> transpose indices`。任一门禁失败
必须回到 900002 S 分支，不得报错或继续放宽 workspace。该路径的本质是让足够大的
`[D, inner]` 用三次额外搬运换取连续行 TopK；“non-last”本身不是 transpose 的充分条件。

---

## 4. Step 2：固定小轴 Tiling

沿 `inner` 切 `tileWidth`，每个 block 处理一个 `(outer, inner tile)`：

1. `tilesPerOuter = max(coreNum / outer, 1)`。
2. D=7：`tileWidth = align32(ceil(inner / tilesPerOuter))`，上限 4096；UB 不足时每次减 32。
3. D=17：`tileWidth = align16(ceil(inner / tilesPerOuter))`，上限 4096；UB 不足时每次减 16。
4. `usedCores = min(coreNum, outer * ceil(inner / tileWidth))`。
5. 平台 UB 先预留 32 KiB SIMT 空间；动态 UB 传剩余值。

UB 公式必须与 kernel 的 buffer 一致：

```text
rowBytes  = align32(tileWidth * sizeof(uint16))
maskBytes = align32(ceil(tileWidth / 8))
required  = D * rowBytes * 2 + rowBytes * 2 + maskBytes * 2
```

value 使用 fp16/bf16 的 16 位 storage，index 在 UB 中用 uint16。Kernel 对每列执行固定
compare-exchange network，value/index 同步交换；D=17 为 72 次 compare-exchange、11 层依赖。

---

## 5. Step 3：Batched fallback 的 UB 搜索

### 5.1 两种布局和公共算法

- L 分支：`inner == 1`，输入行连续，`dChunk <= 4096`。
- S 分支：`inner > 1`，输入跨步，`dChunk <= 4000`，避开 2D DMA blockCount 上限。

每个核批量处理最多 G 行，`G <= 128`。轴 D 分 chunk，使用：

```text
TopK(A union B) = TopK(TopK(A) union TopK(B))
```

第一块局部 TopK 成为 running TopK；后续把 `running[k]` 与当前 `chunk[min(k,nC)]` 合并为不超过
`2k` 的候选并再次 TopK。这个等式使长轴无需完整常驻 UB，但要求 value/index 在所有阶段成对移动。

### 5.2 dChunk 与 G 联合搜索

按大到小遍历去重后的候选：

```text
{min(limit, align32(D)), 2048, 1024, 512, 256, 128, 64, 32}
limit = 4096 (L) or 4000 (S)
```

对每个候选找 240 KiB UB 预算内的最大 G。先用实现推导的 API tmp 下界粗筛，再用真实
`GetTopKMaxMinTmpSize` / Sort tmp 查询复核；查询结果用 32 项环形缓存，避免 host tiling 重复执行
数百次昂贵查询。

候选成本：

```text
chunks = ceil(D / dChunk)
groups = ceil(rowsPerCore / G)
cost(non-int64) = chunks * groups
cost(int64)     = 2 * chunks - 1
```

选择最小 cost；cost 相同保留先遍历到的较大 dChunk。int64 的成本与 G 无关，是因为安全路径需要
chunk 选择和 running merge 的两类调用。禁止用“第一个能放进 UB 的候选”替代该模型。

### 5.3 UB 逐项记账

预算至少包括：输入 compute buffer、输入 index、chunk 输出、running value/index、merge value/index、
API tmp、int64 输出 index。再按条件计入：

- fp16/bf16：raw 输入、fp32 cast、回转输出；
- int64：range-check / b64 Sort scratch；
- S：2D DMA stage、Gather offset、整块输入转置表、输出转置表和 tile；
- L int64：生命周期别名后的实际容量，而不是别名前所有 buffer 的简单相加。

Host 的每一项必须能对应 kernel `InitBuffer`；条件 buffer 必须同时出现或同时消失。关键对齐字段：

| 字段 | 约束 |
|------|------|
| `kPad` | compute dtype 的 32B 对齐行长 |
| `kIdxPad` | int32 index 的 32B 对齐行长 |
| `mergePad` | `2*k` 候选的对齐长度，且不得超过 4096 |
| `tmpSize` | radix floor、TopK API tmp、Sort scratch 的最大值，再按 32B 对齐 |

### 5.4 L / S 专属门禁

L 使用 `DataCopyPad` 搬 `[G, dChunk]` 连续块并连续写 `[G, k]`。int64 可在生命周期不重叠时复用
已死亡的 input/index buffer，但候选必须满足 `dChunk >= mergePad` 和别名容量门禁。

S 先用 2D Compact DMA 搬 `[dChunk, G]`，再 Gather 为 `[G, dChunk]`：

- 8 位 Gather source offset 不超过 65535B；16 位不超过 131071B；
- 整块输入 Gather 仅在 `dChunk <= 128`、容量和 offset 均满足时使用；旧注释中的 256 不是当前门槛；
- 否则逐行 Gather，但仍保留 batched TopK；
- 输出优先把 `[G, k]` 在 UB 中分块转成 `[k, G]` 后批量 DMA；G<8、容量或 offset 不满足才逐行写。

---

## 6. Step 4：dtype、padding 与有序性契约

### 6.1 fp16 / bf16

Fallback 内统一精确 cast 到 fp32 计算再 cast 回原类型。原因是当前高阶 TopK 的 16 位 index 收集
存在已验证缺陷；不能为了节省 UB 直接恢复 16 位计算，除非重新完成精度和性能 A/B。

### 6.2 int64

不要直接调用 vendor b64 TopK，它在目标版本存在 crash 风险。每个 chunk 动态判断实际值能否精确
落入 int32：

- 能：cast 到 int32，走 RADIX_SELECT；
- 不能：该 chunk 走 b64 `Sort(RADIX_SORT)` 全排序后截断。

range-check 必须逐 chunk 执行，覆盖全窄、全宽、窄宽混合和尾块；不能只根据用户 value range 或
首块判断整行。RADIX_SELECT 返回候选不保证有序，最终写回前必须补排序。

### 6.3 padding 与 pair 不变量

padding 必须使用在当前方向下永远不会入选的最差值：largest 填 dtype 最小值或 `-inf`，smallest
填最大值或 `+inf`。NaN 语义必须与接口契约一致。任何 compare、Gather、Sort、transpose、merge、
尾块搬运都必须同时更新 value 和原始 axis index。

---

## 7. Step 5：Common 内部路由

一级未命中时，保持下列首个可行模板顺序：

| 顺序 | 模板 | 条件摘要 |
|---:|---|---|
| 1 | non-last mode 8 | non-last 且 D<=2048；UB 内重排连续行，浮点优先 merge，其余 radix |
| 2 | 小轴 MergeSort | 尾轴、D<=1024、dtype 为 fp16/fp32/bf16；bf16 以 fp32 计算 |
| 3 | single-block radix | 完整行和 API tmp 同时放入 UB |
| 4 | fp32 merge | `0.25 <= k/D < 0.50`；先 inter-core，再 intra-core |
| 5 | single-core radix | `rows >= coreNum` 且存在合法 tile |
| 6 | optimized multi-core radix | `tile < D` 且 `k * ceil(D/tile) <= tile` |
| 7 | general multi-core radix | D<10,000,000 且存在合法 radix tile |
| 8 | SortAndTopK | 超大 D 或前述 tile 均失败；全轴多核 radix sort 后复制前 k |

fp32 inter-core merge 还要求 `splitCoreNum=ceil(D/2048)>1` 且
`rows*splitCoreNum<=coreNum`。否则在 rows 足够、block 数 2 到 256 且 UB 可容纳时尝试 intra-core。
不要交换 inter/intra 或 radix 各模式的判断顺序；modeType 1/2/4/5/6/7/8 是 host-kernel 契约。

---

## 8. Step 6：TilingData 与 workspace

本节给出性能路由要求；字段的精确 C++ 类型、顺序、special launch struct、SDK 查询与文件级复原流程
以 [Host Tiling 复原规格](topk_host_tiling.md) 为准。脚本输出含 `sdk_queries` 时，必须先在目标 CANN
完成查询再生成生产实现。

### 8.1 Special route 必需字段

- 900001：`outer / D / inner / tileWidth / usedCores / localMemoryBytes / largest`。
- 900002：`branch / rowsPerCore / G / dChunk / kPad / kIdxPad / mergePad / tmpSize` 及 shape/dtype/direction。
- 900003：内部完整 launch params、三次 transpose tiling、四个 512B 对齐 offset 和内部 workspace。

900003 内部 params 除外层 tilingKey/workspace 封装外保持递归 tiler 结果；不能重新猜测核数、modeType
或动态 UB。

### 8.2 Common route 必需字段

`modeType / numTileDataSize / lastDimTileNum / unsortedDimParallel / lastDimNeedCore / sortLoopTimes /
topKRealValue / API tmp / keyParams0..5` 必须随对应 host 与 kernel 模板整体迁移。`keyParams0..5`
不是通用扩展字段，不可跨 mode 复用语义。

---

## 9. 禁止外推的边界

- 非 DAV_3510、不同 UB 或不同高阶 API 版本：先重新做能力检查和阈值 A/B。
- 无序 TopK、k>2048、非连续输入：不满足当前契约，不能静默套用。
- Sort/ArgSort 可复用 radix/merge 原理，但不能套用只保留 k 个 running candidates 的截断逻辑。
- `sort/fallback/` 是通用 M-way Sort 模型，不能替代本文的 A5 TopK 路由。
