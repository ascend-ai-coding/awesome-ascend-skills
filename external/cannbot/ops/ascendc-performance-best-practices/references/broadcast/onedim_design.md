# 实现④：OneDim 合轴塌一维快路径（标量广播）

> 主文档：[broadcast_design.md](broadcast_design.md)　|　原语：[`code/broadcast_common.h`](code/broadcast_common.h)（`TryOneDim` / `ComputeOneDimTiling` / `OneDimCalcCore` / `OneDimLoadInput`）
> 端到端样例：[Host](code/onedim_add_tiling.cpp) + [Kernel](code/onedim_add_kernel.cpp)（`z = x + y`，y 为标量）

## 思路

很多广播在**合轴**（把相邻、广播状态一致的轴乘到一起）后会塌成一维：此时每个输入要么是**满 shape 的连续大块**，要么是**纯标量**（一个值）。这类场景没必要套多维 NDDMA 参数装配、也不必调 UB `Broadcast` 指令——

- **满 shape 输入**：一维连续，逐块 `DataCopyPad` 搬入即可；
- **标量输入**：**首块用 `Duplicate` 把那一个值铺满 UB，后续块直接复用**，完全不走逐块 DMA。

这是 `z = x + lr`、`z = x * scale`、`alpha·x + y` 这类「标量参与 elementwise」以及同 shape elementwise 的最快写法。固定成本也最低：tiling 退化成一维（只需 1D 总长 / 块大小 / 核数），切分让 kernel 运行期自己推。

> 对标 atvoss `SCH_MODE_ONE_DIM_ADVANCE`（schMode 202）：tiling 极简（`dimLen/tileNum/blockNum/scalarFlag`）、kernel 运行期派生 `ubOuter/ubTail/blockFormer/blockTail`，省去结构体拷贝与多维下标还原。atvoss 在入出参 >4 或 DAG 带 Var 标量属性时退回普通 `SCH_MODE_ONE_DIM`（201，Host 预切分），本指导只吸收 Advance 形态——纯 AscendC 下没有 Var 概念，运行期标量属性由算子自行经 tiling 传入即可。

## 适用

合轴后能塌成一维，即**每个输入都满足「纯标量」或「满 shape 连续」之一**：

- ✅ `z = x + scalar`（标量广播）；
- ✅ `z = x op y`，x、y 同 shape（无广播，纯 elementwise）；
- ✅ 多输入混合：部分满 shape、部分标量。
- ❌ `[M,1] → [M,N]`、`[1,N] → [M,N]` 这类**部分轴广播**：合轴后仍是多维，**不属本路径**，回到 ①②③ 选型。

判定见 `TryOneDim`：逐输入算元素总数，`==1` 记为标量（置 `scalarFlag` 对应位），`==dimLen` 是满 shape，二者皆非则不能塌一维。

## Host：选型 + 极简 Tiling

OneDim 是在三类选型**之前**的前置分流：先 `TryOneDim`，命中就走 OneDim，否则才进 §1.3 的 `PickBroadcastMode`。

```cpp
// 节选，完整见 code/onedim_add_tiling.cpp。在通用切分 / PickBroadcastMode 之前先试 OneDim
int64_t dimLen; int32_t scalarFlag;
if (TryOneDim(td, dimLen, scalarFlag)) {                 // 合轴塌一维（每输入满 or 标量）
    OneDimTilingData ot;
    ComputeOneDimTiling(ot, dimLen, scalarFlag, dtSize, coreNum, ubSize, /*aliveBuf=*/IN_NUM + 1);
    // 写 ot 进 tilingData，tilingKey 标记走 OneDim 分支
    return;
}
// 否则：ComputeTiling(...) + 逐输入 PickBroadcastMode(...)（见 broadcast_design.md §1）
```

`ComputeOneDimTiling`（`code/broadcast_common.h`）只算三个量，**不预切分**：

```cpp
tileNum  = AlignDown(ubSize / (aliveBuf * dtSize * DB_LOOP), alignEle);  // 单块元素数
ubOuter  = ceil(dimLen / tileNum);                                       // 共几块
blockNum = min(coreNum, ubOuter);                                        // 块数不足核数则减核（小 shape 自适应）
// dimLen / tileNum / blockNum / scalarFlag 即全部 tiling；ubTail/blockFormer/blockTail 交给 kernel
```

> 极简 TilingData 是 Advance 形态的精髓：搬运/切分细节全在 kernel 运行期由这几个标量推出，tiling 体积小、下发开销低。

## Kernel 写法

每核循环参数由 `OneDimCalcCore` 运行期推导；标量输入用单块 `TBuf`（首块 `Duplicate` 一次、后续复用），满 shape 输入用 `TQue` 双缓冲。

```cpp
// 节选，完整见 code/onedim_add_kernel.cpp。z = x + y，x 满 shape、y 可能是标量（scalarFlag bit1）
__aicore__ inline void Process() {
    bool xScalar = ot_.scalarFlag & (1 << 0);
    bool yScalar = ot_.scalarFlag & (1 << 1);
    OneDimCoreParam cp = OneDimCalcCore(ot_.dimLen, ot_.tileNum, ot_.blockNum);  // baseOffset/loops/tailLen
    int64_t off = cp.baseOffset;
    for (int64_t loop = 0; loop < cp.loops; loop++) {
        int64_t len = (loop == cp.loops - 1) ? cp.tailLen : ot_.tileNum;         // 最后一块可非对齐

        LocalTensor<T> xl = xScalar ? xBuf_.Get<T>() : qx_.AllocTensor<T>();
        OneDimLoadInput(xl, xGm_, off, len, xScalar, /*firstTile=*/loop == 0);   // 标量仅首块 Duplicate
        if (!xScalar) qx_.EnQue(xl);
        LocalTensor<T> yl = yScalar ? yBuf_.Get<T>() : qy_.AllocTensor<T>();
        OneDimLoadInput(yl, yGm_, off, len, yScalar, loop == 0);
        if (!yScalar) qy_.EnQue(yl);

        LocalTensor<T> xv = xScalar ? xl : qx_.DeQue<T>();
        LocalTensor<T> yv = yScalar ? yl : qy_.DeQue<T>();
        LocalTensor<T> zv = qz_.AllocTensor<T>();
        AscendC::Add(zv, xv, yv, len);
        qz_.EnQue(zv);
        if (!xScalar) qx_.FreeTensor(xv);
        if (!yScalar) qy_.FreeTensor(yv);

        LocalTensor<T> zo = qz_.DeQue<T>();
        AscendC::DataCopyExtParams ext{1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(zGm_[off], zo, ext);                                // 一维连续写回
        qz_.FreeTensor(zo);
        off += ot_.tileNum;
    }
}
```

`OneDimLoadInput` 内部（`code/broadcast_common.h`）：

```cpp
if (isScalar) {
    if (firstTile) AscendC::Duplicate(dst, gm.GetValue(0), len);   // 标量值铺满 UB；之后块复用同一块
} else {
    DataCopyPadCompact(gm[offset], dst, len);                      // 满输入：一维连续搬，非对齐 Pad 兜底
}
```

## 注意

- **标量不进双缓冲队列**：标量首块铺满后整轮复用，应放单块 `TBuf`；若误塞进 depth=2 的 `TQue` 并每轮 Alloc，会丢掉"只填一次"的收益，还要处理 ping/pong 两块各填一次。
- **标量首块的 `len`**：首块用 `Duplicate(dst, val, len)` 铺当前块长度即可；后续块直接复用 `dst`，无需重铺（各块长度只有最后一块可能更短，复用块多出的尾部不参与计算/写回，无害）。
- **尾块非对齐**：一维总长 `dimLen` 常非 `alignEle` 倍数，最后一块 `tailLen` 非对齐——`DataCopyPad` 自动兜底，计算指令 `count` 传真实 `len` 即可（同 [advanced_tiling.md](advanced_tiling.md) §3）。
- **小 shape 减核**：`ubOuter < coreNum` 时 `ComputeOneDimTiling` 把 `blockNum` 降到 `ubOuter`，避免空核；`OneDimCalcCore` 的 `blockFormer/blockTail` 据此自洽。
- **UB 预算**：标量块按 1 份（非双缓冲）、满输入/输出按双缓冲计入 `aliveBuf`；示例取 `IN_NUM+1` 为保守上界。
- **double**：`GetValue`/`Duplicate` 标量对 double 需按位转 int64 处理（同 atvoss 对 `double` 的 reinterpret），其余 dtype 直接用。

## 与 ①②③ 的关系

OneDim 是**合轴塌一维时的前置快路径**，优先级高于三类选型——命中即用，命中不了才进 `PickBroadcastMode`。三类处理的是"合轴后仍多维"的部分轴广播（`[M,1]`/`[1,N]` 等）。决策树见 [broadcast_design.md](broadcast_design.md) §3。
