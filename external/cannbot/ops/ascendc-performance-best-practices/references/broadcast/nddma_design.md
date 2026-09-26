# 实现①：NDDMA 搬入即广播

> 主文档：[broadcast_design.md](broadcast_design.md)　|　参考代码：[`code/broadcast_common.h`](code/broadcast_common.h)（`MakeNddmaParams` / `BroadcastNddma`）

## 思路

用多维 DMA（NDDMA）一次搬运，把广播轴的源 stride 置 0 → DMA 引擎在 GM→UB 过程中自动复制。**不占 Vector 单元**（只走 MTE2）。

## 适用

有广播轴、广播轴落在 UB tile **内**（切分轴或其下的尾轴），且**不满足 ③ 的条件**：
- 大 dtype（FP32/INT32 等，不在 {INT8,UINT8,FP16,BF16,INT16,UINT16}）；
- 或 B8/B16 但尾轴**非对齐**。

判定走 ① 还是 ② 看 `outputStrides[ubSplitAxis] != inputStrides[ubSplitAxis]`——这表示 **UB tile 内 split 轴跨度不一致**（切分轴广播，或切分轴连续但其下尾轴广播，如 `[M,1]→[M,N]`），此时走 ① DMA 复制；tile 内完全连续（广播轴全在外层）则退化为 ②。

## Kernel 写法

```cpp
// 每核外循环（节选，完整见 code/broadcast_add_kernel.cpp）
for (int64_t loop = 0; loop < ubLoopNum; loop++) {
    if (loop != 0) UpdateAxesIndices(axes, outputDims, ubSplitAxis, ubOuter);
    int64_t rows = (axes[ubSplitAxis] == ubOuter - 1) ? ubTail : ubFormer;
    int64_t off  = GetGmOffset(axes, inputStrides[i], ubSplitAxis, ubFormer);
    LocalTensor<T> ub = que.AllocTensor<T>();
    BroadcastNddma(xGm[off], ub, outputDims, outputStrides, inputStrides[i],
                   shapeLen, ubSplitAxis, /*ubSplitSize=*/rows);   // ← code/broadcast_common.h
    que.EnQue(ub);
    // ... 计算 + CopyOut
}
```

`BroadcastNddma` 内部（`code/broadcast_common.h`）：

```cpp
if (outputStrides[ubSplitAxis] != inputStrides[ubSplitAxis]) {       // tile 内 split 轴跨度不一致 → ①
    auto params = MakeNddmaParams<T>(...);                           // 广播轴 loopSrcStride=0
    static constexpr AscendC::MultiCopyConfig cfg = {false, 0, 0, false};  // 必须 static：模板参为 const&
    AscendC::DataCopy<T, NDDMA_DIM, cfg>(ub, gm, params);            // 多维 DMA
} else {                                                             // tile 内完全连续 → 落 ②
    AscendC::DataCopyPad(ub, gm, ext, pad);
}
```

参数装配 `MakeNddmaParams`：从尾轴往前填 `NDDMA_DIM(=5)` 个轴，广播轴的 `loopSrcStride` 天然为 0（因 `inputStrides[广播轴]=0`）→ 引擎复制；非广播轴填真实 stride。高位不足补 `loopSize=1` 的轴。

## 注意

- `NDDMA_DIM = 5`：单次多维 DMA 最多 5 维。`rank = shapeLen - ubSplitAxis > 5` 时 `MakeNddmaParams` 的 `axisInsideUb` 会变负越界。官方 atvoss 走 with-loop NDDMA（按融合轴拆，`BroadcastNddmaWithLoop`）；**本样例未实现 with-loop，因此 Host 选型在 rank>5 时回退 ③ BRC_UB**（见 `broadcast_add_tiling.cpp` 的 rank 兜底）。
- 广播轴 `loopSrcStride = 0` 是复制关键；非广播轴填真实 stride。
- `MultiCopyParams` 的 `constValue` 仅 pad 场景用，正常广播取 0。
- 与 ② 共用同一段 kernel：是否进 ① 由运行期 `outputStrides[ubSplitAxis] != inputStrides[ubSplitAxis]`（tile 内 split 轴跨度不一致）决定，不等同于"切分轴是广播轴"。
- `cfg` 必须是 `static constexpr`：`DataCopy` 第三模板参为 `const MultiCopyConfig&`，需静态存储期，局部 `constexpr` 编不过。
