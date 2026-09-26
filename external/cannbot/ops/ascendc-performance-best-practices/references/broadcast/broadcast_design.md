# AscendC Broadcast 性能优化与实现选型指导（总览）

> 广播算子的性能关键在于**按场景选对实现**（前置 OneDim 快路径 + tile 内 NDDMA / DataCopyPad / UB 三类）——选错直接慢甚至 nan，选对才快。本文即该选型的最佳实践。

> 纯 **AscendC API** 指导，不依赖任何模板调度框架。提炼自 atvoss broadcast 的设计经验，只保留两块可直接迁移的内容：**Tiling 选型/切分逻辑** 与 **OneDim 快路径 + 三类 Kernel 广播写法**。
> 适用：DAV_3510（A5 / Ascend950，C310）。广播 = 某输入在某些轴上 shape=1 / stride=0，需铺开到输出 shape。

## 文档结构

| 文件 | 内容 |
|------|------|
| **broadcast_design.md**（本文） | 总览、实现速查、Tiling 选型与切分逻辑、决策树、约束 |
| [onedim_design.md](onedim_design.md) | 实现④ OneDim 合轴塌一维快路径（标量广播）★前置优先 |
| [nddma_design.md](nddma_design.md) | 实现① NDDMA 搬入即广播 |
| [datacopypad_design.md](datacopypad_design.md) | 实现② DataCopyPad 紧凑搬入 + 复用 |
| [ub_broadcast_design.md](ub_broadcast_design.md) | 实现③ UB 内 Broadcast 指令 |
| [advanced_tiling.md](advanced_tiling.md) | 进阶：通用切分（任意 ubSplitAxis / 大 shape / 尾块非对齐 / cann-bench case） |
| [code/broadcast_common.h](code/broadcast_common.h) | 常量 / TilingData / `ComputeTiling` / 切分 helper / 四类广播原语（含 OneDim） |
| [code/broadcast_add_tiling.cpp](code/broadcast_add_tiling.cpp) | 端到端样例（①②③ 多维选型）：Host（选型 + 切分） |
| [code/broadcast_add_kernel.cpp](code/broadcast_add_kernel.cpp) | 端到端样例（①②③ 多维选型）：Kernel + 算子入口 |
| [code/onedim_add_tiling.cpp](code/onedim_add_tiling.cpp) | 端到端样例（④ OneDim 标量广播）：Host（`TryOneDim` + 极简切分） |
| [code/onedim_add_kernel.cpp](code/onedim_add_kernel.cpp) | 端到端样例（④ OneDim 标量广播）：Kernel + 算子入口 |

## 0. 实现速查（一个前置快路径 + tile 内三类）

| # | 实现 | 广播在哪发生 | 核心 AscendC API | 占用单元 | 典型适用 |
|---|------|------------|-----------------|---------|---------|
| ④ | [**OneDim 合轴塌一维**](onedim_design.md) ★前置 | 标量首块 Duplicate / 满输入直搬 | `Duplicate(ub,scalar,len)` + `DataCopyPad` | Vector(标量铺) / MTE2 | **合轴后塌一维**：每输入要么满 shape 要么纯标量（标量广播、同 shape elementwise） |
| ① | [**NDDMA 搬入即广播**](nddma_design.md) | GM→UB 搬运中 | `DataCopy<T,DIM,cfg>(ub,gm,MultiCopyParams)`，广播轴 `loopSrcStride=0` | MTE2（DMA），不占 Vector | 大 dtype（FP32/INT32）等、或 B8/B16 尾轴非对齐；**切分轴本身是广播轴** |
| ② | [**DataCopyPad（外层广播，offset 寻址）**](datacopypad_design.md) | 不在搬运中，靠 GM offset（外层广播轴 stride=0） | `DataCopyPad(ub,gm,ext,pad)` 每轮搬一段连续数据 | MTE2，不占 Vector | 广播轴**全在切分轴之上（外层）**、切分轴连续 |
| ③ | [**UB 内 Broadcast 指令**](ub_broadcast_design.md) | UB 内 | `Broadcast<T,R>(dst,src,dstShape,srcShape,&tiling)` | Vector | **前提广播轴在 tile 内(brcInTile)**；且：尾轴对齐且 dtype∈{INT8,UINT8,FP16,BF16,INT16,UINT16}，或非尾轴广播且尾轴≥dcache/2（~4096B），或广播中间计算结果 |

**判断口诀（按优先级）**：**先试 ④**——合轴后能塌一维（每个输入要么满 shape、要么纯标量）就走 OneDim（标量首块 Duplicate、tiling 退化 1D、固定成本最低），命中即用。塌不成一维（部分轴广播，如 `[M,1]/[1,N]`）才进 ①②③：先看广播轴在不在 UB tile 内。**全在 tile 外（外层循环轴）→ 一律 ②**（offset 寻址每轮连续搬；UB 指令和 NDDMA 都展不了外层轴）。在 tile 内才在 ③/① 间选：尾轴对齐的小字节类型（B8/B16）或尾轴非广播的 BigNLast → ③（Vector 吞吐高），其余 → ①（DMA 不占 Vector）。

## 1. Tiling 逻辑（Host）

### 1.1 统一的切分模型

无论哪类实现，切分都是两层（与普通 elementwise 一致），对象始终是**输出 shape**：

```
多核切分：blockNum 个核，每核处理 blockFormer 个「UB 外循环单元」，尾核 blockTail
UB 切分：选 ubSplitAxis 轴，每次处理 ubFormer 行（尾块 ubTail），共 ubOuter 次外循环
```

TilingData 字段见 [`code/broadcast_common.h`](code/broadcast_common.h) 的 `BroadcastTilingData`，可直接照搬。关键字段：`inputStrides`（广播轴=0）、`outputDims/Strides`、`ubSplitAxis/ubFormer/ubOuter`、`blockNum/blockFormer`、`dimProductBeforeUbInner`（UB tile 总数）、`elemNum`。

> **`ubSplitAxis` 不要写死成 0**。尾轴常超 UB 容量（如 8192 fp32），必须按输出 shape 动态选切分轴、必要时切到尾轴内部。通用选轴 + 完整公式（含 `dimProductBeforeUbInner` 通用式、大 shape、尾块非对齐）见 [advanced_tiling.md](advanced_tiling.md) 与 `ComputeTiling`。下文 §1.2–1.4 给要点。

### 1.2 广播轴识别

**广播轴 = 该输入 `inputStrides[i][axis]==0` 且 `outputStrides[axis]!=0`。** 这是所有判断的基础（不是只看 dim==1）。**必须扫描所有轴**得到三个标志，不能只取第一个广播轴（否则同时有外层广播轴和 tile 内广播轴时会误判）：

```cpp
bool anyBrc = false, brcInTile = false, nonLastBrc = false, lastAxisBrc = false;
for (int j = 0; j < shapeLen; j++) {
    if (inputStrides[i][j] == 0 && outputStrides[j] != 0) {
        anyBrc = true;
        if (j >= ubSplitAxis) brcInTile = true;     // 广播轴落在 UB tile 内（含尾轴）
        if (j < shapeLen - 1)  nonLastBrc = true;   // 存在非尾轴广播
        else                   lastAxisBrc = true;  // 尾轴本身是广播轴
    }
}
```

### 1.3 三类选型决策（每个广播输入独立判断）

> **前置：先试 OneDim（④）。** 进入下面的逐输入 `PickBroadcastMode` 之前，先 `TryOneDim`——若合轴后塌成一维（每个输入要么满 shape 要么纯标量），整算子走 OneDim 快路径（`ComputeOneDimTiling` + 标量 `Duplicate`），不再做三类切分。详见 [onedim_design.md](onedim_design.md)。塌不成一维（部分轴广播）才执行下面的三类选型。

```cpp
int PickBroadcastMode(int i) {       // 承接 §1.2 扫描得到的 anyBrc/brcInTile/nonLastBrc/lastAxisBrc
    if (!anyBrc) return 0;                                            // 非广播输入，普通 CopyIn
    if (!brcInTile) return 2;        // ★广播轴严格在外层 → ② DataCopyPad（offset 寻址；切分轴非广播，len 公式成立）

    // ★「在 tile 内」≠「UB 能展开」：切分轴广播但 ubFormer==1 时 tile 内仅 1 行(dShape==sShape)，
    //   UB Broadcast 退化空广播 → nan。只有存在真正可展开维(dst>src)才可选 ③，否则走 ① NDDMA。
    bool inTileExpandable = false;
    for (int j = ubSplitAxis+1; j < shapeLen; j++)
        if (inputStrides[i][j]==0 && outputDims[j]>1) inTileExpandable = true;        // 切分轴以下广播
    if (inputStrides[i][ubSplitAxis]==0 && outputDims[ubSplitAxis]>1 && ubFormer>1)
        inTileExpandable = true;                                                      // 切分轴广播且多行

    int64_t lastDim = outputDims[shapeLen - 1], dt = sizeof(T_i), alignEle = 32 / dt;
    static const set<DataType> kUb = {DT_INT8,DT_UINT8,DT_FLOAT16,DT_BF16,DT_INT16,DT_UINT16};
    if (inTileExpandable) {
        if (lastDim % alignEle == 0 && kUb.count(dtype_i))                  return 3; // ①尾轴对齐+B8/B16 → ③
        if (nonLastBrc && !lastAxisBrc && lastDim*dt >= GetNddmaDcacheSize()/2) return 3; // ②BigNLast → ③
    }
    if (shapeLen - ubSplitAxis > NDDMA_DIM)                 return 3; // rank>5 without-loop NDDMA 装不下 → ③(含退化守卫兜底)
    return 1;                                                         // 其余 tile 内广播 → ① NDDMA
}
```

`brcMode` 逐输入写进 TilingData / tilingKey，kernel 据此分发。完整实现见 [`code/broadcast_add_tiling.cpp`](code/broadcast_add_tiling.cpp)。两道关键门控（cann-bench Maximum case 18 实测）：
1. **`!brcInTile`（广播轴严格在外层）→ 一律 ②**：UB/NDDMA 只在 tile 内展开，外层轴只能靠 offset 寻址。
2. **`brcInTile` 但无可展开维（切分轴广播且 `ubFormer==1`）且 `rank≤5` → ① NDDMA**：此时 UB Broadcast 会 `dShape==sShape` 空广播出 nan；NDDMA 在 `ubFormer==1` 下搬 inner 连续数据、外层靠 offset 广播仍正确。（`rank>5` 时 without-loop NDDMA 装不下 → 仍走 ③，由 kernel `UbBroadcast` 退化守卫拷贝兜底。）kernel `UbBroadcast` 的退化守卫（无可展开维则拷贝）对两种情形都做双保险。

### 1.4 切分大小与选轴（`ComputeTiling`）

```cpp
elemNum = AlignDown( ubSize / (aliveBufferNum * maxDtypeBytes * DB_LOOP), alignEle );
// 从尾轴向外累乘选切分轴：能整轴塞进 elemNum 就继续向外；放不下则切该轴；
// 连尾轴一行都放不下 → 切尾轴本身（ubFormer = elemNum）
// ubFormer/ubOuter/ubTail 沿 ubSplitAxis 算；ubTail 可非对齐（DataCopyPad 兜底）
dimProductBeforeUbInner = (∏ outputDims[0 .. ubSplitAxis-1]) × ubOuter   // UB tile 总数，通用公式
```

完整实现 `ComputeTiling` 见 [`code/broadcast_common.h`](code/broadcast_common.h)，推导与 cann-bench case 对照见 [advanced_tiling.md](advanced_tiling.md)。

`aliveBufferNum`：③ 需额外一块放广播前的紧凑源（src+dst 两块），①② 直接铺到目标块（一块）——选 ③ 的输入要把这块计入预算。

## 2. 各实现

各实现的适用、Kernel 写法、注意见分册：[④ OneDim（前置）](onedim_design.md)、[① NDDMA](nddma_design.md)、[② DataCopyPad](datacopypad_design.md)、[③ UB Broadcast](ub_broadcast_design.md)。原语统一在 [`code/broadcast_common.h`](code/broadcast_common.h)：① ② ③ 为 `BroadcastNddma` / `DataCopyPadCompact` / `UbBroadcast`（通用切分 `ComputeTiling`，helper `GetAxesIndices`/`UpdateAxesIndices`/`GetGmOffset`/`FillUbShape`/`UbSrcLen`）；④ 为 `TryOneDim` / `ComputeOneDimTiling` / `OneDimCalcCore` / `OneDimLoadInput`。

端到端串接见样例 [Host](code/broadcast_add_tiling.cpp) + [Kernel](code/broadcast_add_kernel.cpp)。

## 3. 选型决策树

```
★ 前置：合轴后塌成一维吗？(每个输入要么满 shape、要么纯标量 → TryOneDim)
├─ 是 → ④ OneDim 快路径（标量首块 Duplicate、满输入直搬；tiling 退化 1D）  ——命中即用，整算子走此路
└─ 否（部分轴广播，如 [M,1]/[1,N]）↓ 进入逐输入三类选型

该输入是广播输入吗？(存在 stride==0 轴)  ——按从上到下的优先级匹配
├─ 否 → 普通 DataCopyPad 搬入(brcMode=0)
└─ 是：
   ├─ 广播轴严格在外层(brcInTile=false) → ② DataCopyPad  ★必须先判！外层轴只能 offset 寻址
   └─ 广播轴在 UB tile 内(brcInTile=true)：
      ├─ rank=(shapeLen-ubSplitAxis) > 5 → ③（without-loop NDDMA 装不下；UB 退化守卫兜底无可展开维的情形）
      ├─ 有可展开维(inTileExpandable：切分轴以下广播，或切分轴广播且 ubFormer>1)：
      │  ├─ 尾轴对齐 且 dtype∈{INT8,UINT8,FP16,BF16,INT16,UINT16}      → ③ UB-BRC
      │  ├─ 尾轴非广播 且 存在非尾轴广播 且 尾轴字节 ≥ dcache/2(~4096B) → ③ UB-BRC (BigNLast)
      │  └─ 其余                                                       → ① NDDMA
      └─ 无可展开维(切分轴广播且 ubFormer==1，tile 内仅 1 行) → ① NDDMA  ★走 ③ 会 dShape==sShape 空广播出 nan
注：实际代码 rank>5 在 ①② 之后判（可展开时①②仍优先），但对"无可展开维"分支也生效——见 broadcast_add_tiling.cpp。
官方 atvoss rank>5 走 with-loop NDDMA(SCH_MODE_NDDMA_WITH_LOOP_CACHE_BRC)，本样例未实现故回退 ③。
```

## 4. 自检清单

- [ ] ④ OneDim 前置：`TryOneDim` 命中（每输入满 shape 或纯标量）即走快路径，标量走单块 `TBuf` 首块 `Duplicate`、不进双缓冲队列；`ComputeOneDimTiling` 只给 `dimLen/tileNum/blockNum`，切分由 kernel `OneDimCalcCore` 运行期推
- [ ] 广播轴判定基于 `inputStrides[i][axis]==0 && outputStrides[axis]!=0`，不是只看 dim==1
- [ ] `brcMode` 逐输入计算并写入 TilingData/tilingKey，kernel 分支与之一一对应
- [ ] ① NDDMA：广播轴 `loopSrcStride=0`、非广播轴填真实 stride；维数 ≤ 5（否则套循环）
- [ ] ② DataCopyPad：广播靠 `GetGmOffset`（外层广播轴 stride=0），每轮连续搬即可；复用为可选优化且判据用 offset 比较（非切分轴 stride，后者是死路径）
- [ ] 切分：用 `ComputeTiling` 动态选 ubSplitAxis，`dimProductBeforeUbInner=∏outputDims[0..split-1]×ubOuter`；尾轴超 UB 时切到尾轴；尾块非对齐由 DataCopyPad 兜底
- [ ] ③ UB-BRC：`elemNum` 预算多留一块紧凑源；rank 静态(≤4)可走编译期 `Broadcast<T,R>`；尾块修正 shape 首轴
- [ ] DoubleBuffer：`elemNum` 已按 `aliveBufferNum * dtypeBytes * 2` 反推
- [ ] 精度：与 naive 全展开实现对比，FP32 < 1e-5、FP16 < 1e-3

## 5. 约束与适用范围

- 平台：DAV_3510（C310）。`MultiCopy`/NDDMA 多维 DMA 与 `Broadcast` 指令均为该架构 AscendC API。
- 维度上限：建议 `BROADCAST_MAX_DIMS = 8`；单次 NDDMA `NDDMA_DIM = 5`。
- `BLOCK_LENGTH = 32`（字节），对齐元素数 = `32 / sizeof(dtype)`。
- 只覆盖**连续布局**广播；转置/非连续布局需搬运前做 layout 处理，不在本文范围。
- ④ OneDim 是**整算子级**的前置分流（合轴塌一维则全算子走它）；①②③ 是塌不成一维时**逐输入混用**（不同输入走不同 brcMode），均由 tiling 决定。
- 对标 atvoss 连续布局四个调度模式：① ↔ NDDMA without/with-loop（schMode 1/2）、③ ↔ UB_BROADCAST（101~109）、④ ↔ ONE_DIM_ADVANCE/ONE_DIM（202/201）；非连续布局族（transpose 等）不在本文范围。
- 代码为**范式级**：`AlignDown/CeilDiv/GET_TILING_DATA/DTYPE_X` 等按工程宏/惯例占位，落地需接 op proto / REGISTER / tiling 注册 / 对齐校验等工程件。
